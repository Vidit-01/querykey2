"""Experiment 1C: surrogate-guided refinement, replication, and portfolios."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.distributed import (  # noqa: E402
    barrier,
    cleanup,
    current_shard_plan,
    is_main_process,
    launched_with_torchrun,
    resolve_sharding,
)
from common.geometry import Coefficients, coefficient_grid, coefficients_from_alpha_phi  # noqa: E402
from common.io_utils import (  # noqa: E402
    append_jsonl,
    create_manifest,
    ensure_output,
    read_jsonl,
    write_json,
)
from common.portfolio import (  # noqa: E402
    FREEZE_RULES,
    coefficient_key,
    freeze_rule_counts,
    group_by_coefficient,
    passes_freeze_rule,
    point_payload,
    select_even_coverage,
)
from exp1b.run import analyze as analyze_atlas  # noqa: E402
from exp1b.run import rows_for_cell  # noqa: E402


NUMERIC_FEATURES = [
    "log_s",
    "alpha",
    "phi",
    "inverse_d",
    "m_over_d",
    "log_n",
    "mean_vi_vj",
    "mean_cij2",
    "delta",
    "tau",
]
CATEGORICAL_FEATURES = ["mask"]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    numeric = set(NUMERIC_FEATURES) | {
        "cell_id",
        "s",
        "beta",
        "gamma",
        "d",
        "m",
        "n",
    }
    for row in rows:
        for key in numeric:
            if key in row and row[key] not in ("", "None"):
                row[key] = float(row[key])
        row["log_s"] = math.log(float(row["s"]))
        row["inverse_d"] = 1.0 / float(row["d"])
        row["m_over_d"] = float(row["m"]) / float(row["d"])
        row["log_n"] = math.log(float(row["n"]))
    return rows


def make_pipeline(seed: int) -> Pipeline:
    transform = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )
    return Pipeline([("features", transform), ("classifier", classifier)])


def calibration_error(probabilities: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    error = 0.0
    for lower, upper in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        selected = (confidence >= lower) & (confidence < upper)
        if selected.any():
            accuracy = np.mean(prediction[selected] == target[selected])
            error += selected.mean() * abs(accuracy - confidence[selected].mean())
    return float(error)


def fit_surrogate(rows: list[dict], seed: int) -> tuple[Pipeline, dict]:
    labels = np.asarray([row["label"] for row in rows])
    groups = np.asarray([int(float(row["cell_id"])) for row in rows])
    splitter = GroupShuffleSplit(n_splits=20, test_size=0.20, random_state=seed)
    train, test = next(splitter.split(rows, labels, groups))
    all_labels = set(labels)
    for candidate_train, candidate_test in splitter.split(rows, labels, groups):
        if set(labels[candidate_train]) == all_labels:
            train, test = candidate_train, candidate_test
            break
    model = make_pipeline(seed)
    train_rows = [rows[index] for index in train]
    test_rows = [rows[index] for index in test]
    model.fit(pd.DataFrame(train_rows), labels[train])
    probabilities = model.predict_proba(pd.DataFrame(test_rows))
    predictions = model.classes_[probabilities.argmax(axis=1)]
    class_to_index = {label: index for index, label in enumerate(model.classes_)}
    known = np.asarray([label in class_to_index for label in labels[test]])
    target_index = np.asarray(
        [class_to_index.get(label, 0) for label in labels[test]]
    )
    uncertainty = 1 - probabilities.max(axis=1)
    boundary = uncertainty >= np.quantile(uncertainty, 0.75)
    metrics = {
        "held_out_fraction": 0.20,
        "train_rows": len(train),
        "test_rows": len(test),
        "balanced_accuracy": float(balanced_accuracy_score(labels[test], predictions)),
        "multiclass_log_loss": float(
            log_loss(labels[test][known], probabilities[known], labels=model.classes_)
        )
        if len(model.classes_) > 1 and known.any()
        else 0.0,
        "expected_calibration_error": calibration_error(
            probabilities[known], target_index[known]
        )
        if known.any()
        else math.nan,
        "boundary_error": float(np.mean(predictions[boundary] != labels[test][boundary]))
        if boundary.any()
        else math.nan,
        "boundary_definition": "top quartile of held-out predictive uncertainty",
        "classes": model.classes_.tolist(),
    }
    # The held-out 20% is never used for fitting or adaptive-point selection.
    return model, metrics


def candidate_features(
    *,
    s: float,
    alpha: float,
    phi: float,
    d: int = 64,
    m: int = 16,
    n: int = 64,
    correlation: float = 0.25,
    mask: str = "unmasked",
) -> dict:
    tau = s**2 * math.sqrt(1 + alpha**2 * correlation**2)
    delta = alpha * s**2 * math.sqrt(m) * (1 - correlation)
    return {
        "s": s,
        "log_s": math.log(s),
        "alpha": alpha,
        "phi": phi,
        "inverse_d": 1 / d,
        "m_over_d": m / d,
        "log_n": math.log(n),
        "mask": mask,
        "mean_vi_vj": 1.0,
        "mean_cij2": correlation**2,
        "delta": delta,
        "tau": tau,
    }


def theory_screen(row: dict, n: int = 64) -> str:
    deterministic_self = 1 / (1 + (n - 1) * math.exp(-row["delta"]))
    if deterministic_self >= 0.8:
        return "self_locked"
    if row["tau"] >= math.sqrt(2 * math.log(n)):
        return "concentrated"
    if row["tau"] <= 0.35:
        return "diffuse"
    return "boundary_uncertain"


def propose_points(model: Pipeline, *, count: int, seed: int, quick: bool) -> list[dict]:
    rng = np.random.default_rng(seed)
    pool_size = 300 if quick else 50_000
    candidates = [
        candidate_features(
            s=float(2 ** rng.uniform(-2, 2)),
            alpha=float(rng.uniform(-1, 1)),
            phi=float(rng.uniform(0, math.pi / 2)),
        )
        for _ in range(pool_size)
    ]
    probabilities = model.predict_proba(pd.DataFrame(candidates))
    predicted = model.classes_[probabilities.argmax(axis=1)]
    uncertainty = 1 - probabilities.max(axis=1)
    disagreement = np.asarray(
        [prediction != theory_screen(row) for prediction, row in zip(predicted, candidates, strict=True)]
    )
    uncertain_order = np.argsort(-uncertainty)
    disagreement_order = np.argsort(-(uncertainty + disagreement.astype(float)))
    selected_indices = []
    for order in (uncertain_order, disagreement_order):
        for index in order:
            if int(index) not in selected_indices:
                selected_indices.append(int(index))
            if len(selected_indices) >= count:
                break
        if len(selected_indices) >= count:
            break
    points = []
    for point_id, index in enumerate(selected_indices):
        row = candidates[index]
        coefficients = coefficients_from_alpha_phi(row["s"], row["alpha"], row["phi"])
        points.append(
            {
                "point_id": point_id,
                **coefficients.to_dict(),
                "surrogate_label": str(predicted[index]),
                "theory_screen": theory_screen(row),
                "uncertainty": float(uncertainty[index]),
                "surrogate_theory_disagreement": bool(disagreement[index]),
            }
        )
    return points


def discovery_coefficients() -> list[Coefficients]:
    return coefficient_grid(
        log2_s_values=np.linspace(-2, 2, 17),
        alpha_values=np.linspace(-1, 1, 13),
        phi_values=np.linspace(0, math.pi / 2, 9),
    )


def make_tasks(points: list[dict], *, quick: bool, replicate_discovery: bool) -> list[dict]:
    adaptive = [
        Coefficients(row["s"], row["alpha"], row["beta"], row["gamma"], row["phi"])
        for row in points
    ]
    if quick:
        dimensions = [(32, 8, 24)]
        base = adaptive
    elif replicate_discovery:
        dimensions = [(64, 16, 64), (128, 16, 128), (128, 32, 128), (256, 32, 256)]
        base = discovery_coefficients() + adaptive
    else:
        # Lighter default: reference geometry + one transfer width.
        dimensions = [(64, 16, 64), (128, 16, 128)]
        base = adaptive
    tasks = []
    for d, m, n in dimensions:
        for source_index, coefficients in enumerate(base):
            tasks.append(
                {
                    "task_id": len(tasks),
                    "source_index": source_index,
                    "d": d,
                    "m": m,
                    "n": n,
                    "coefficients": coefficients,
                }
            )
    return tasks


def default_output(preset: str) -> Path:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return Path("/kaggle/working") / "exp1c" / "data" / preset
    return Path(__file__).parent / "data" / preset


def default_discovery_summary(preset: str) -> Path:
    if custom := os.environ.get("KAGGLE_DISCOVERY_SUMMARY"):
        return Path(custom)
    repo = Path(__file__).parents[1] / "exp1b" / "data" / preset / "atlas_summary.csv"
    if repo.exists():
        return repo
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        working = Path("/kaggle/working") / "exp1b" / "data" / preset / "atlas_summary.csv"
        if working.exists():
            return working
        input_root = Path("/kaggle/input")
        if input_root.is_dir():
            for path in sorted(input_root.rglob("atlas_summary.csv")):
                return path
    return repo


def resolve_device(args: argparse.Namespace) -> torch.device:
    if args.distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed 1C run requires CUDA (use torchrun on GPU)")
        plan = current_shard_plan()
        assert plan is not None
        return torch.device(f"cuda:{plan.local_rank}")
    return torch.device(args.device)


def freeze_portfolio(summary_path: Path, output: Path, heads: int = 8) -> None:
    rows = read_csv(summary_path)
    grouped = group_by_coefficient(rows)
    rule_counts = freeze_rule_counts(grouped)
    applied_rule = None
    candidates: list[dict] = []
    grouped_for_points: dict[tuple, list[dict]] = {}
    for name, _description in FREEZE_RULES:
        matching = [
            (key, values)
            for key, values in grouped.items()
            if passes_freeze_rule(values, name)
        ]
        if len(matching) >= heads:
            applied_rule = name
            grouped_for_points = {key: values for key, values in matching}
            candidates = [values[0] for _key, values in matching]
            break
    if applied_rule is None:
        write_json(
            output / "candidate_portfolio.json",
            {
                "frozen": False,
                "requested_heads": heads,
                "requested_rule": FREEZE_RULES[0][0],
                "rule_counts": rule_counts,
                "reason": (
                    f"need {heads} screened candidates under any fallback freeze rule; "
                    f"counts={rule_counts}"
                ),
            },
        )
        return
    selected_rows = select_even_coverage(candidates, heads)
    write_json(
        output / "candidate_portfolio.json",
        {
            "frozen": True,
            "requested_heads": heads,
            "requested_rule": FREEZE_RULES[0][0],
            "applied_rule": applied_rule,
            "applied_rule_description": dict(FREEZE_RULES)[applied_rule],
            "replication_strict": applied_rule == FREEZE_RULES[0][0],
            "n_candidates": len(candidates),
            "rule_counts": rule_counts,
            "selection_rule": (
                f"{applied_rule}; even coverage after sorting by alpha then phi; "
                "training outcomes unseen"
            ),
            "points": [
                point_payload(row, grouped_for_points[coefficient_key(row)])
                for row in selected_rows
            ],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["quick", "full"], default="quick")
    parser.add_argument(
        "--mode",
        choices=["propose", "run", "analyze", "freeze", "all"],
        default="all",
    )
    parser.add_argument(
        "--discovery-summary",
        type=Path,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--adaptive-cells", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--replicate-discovery-grid", action="store_true")
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Use torchrun process-group sharding (one disjoint task stream per GPU).",
    )
    args = parser.parse_args()
    if args.distributed and not launched_with_torchrun():
        raise RuntimeError(
            "Pass --distributed only under torchrun, e.g. "
            "torchrun --standalone --nproc_per_node=2 code/exp1c/run.py --distributed"
        )
    shard_index, num_shards, rank, world_size = resolve_sharding(
        args.shard_index, args.num_shards, distributed=args.distributed
    )
    if args.discovery_summary is None:
        args.discovery_summary = default_discovery_summary(args.preset)
    if args.mode in {"propose", "all"} and not args.discovery_summary.exists():
        raise FileNotFoundError(
            f"discovery summary not found: {args.discovery_summary} "
            "(attach 1B atlas_summary.csv as a Kaggle dataset or set KAGGLE_DISCOVERY_SUMMARY)"
        )
    output = ensure_output(args.output or default_output(args.preset))
    manifest_args = dict(vars(args))
    manifest_args.update(
        {
            "effective_shard_index": shard_index,
            "effective_num_shards": num_shards,
            "distributed_rank": rank,
            "distributed_world_size": world_size,
        }
    )
    create_manifest(
        output,
        experiment="exp1c",
        arguments=manifest_args,
        inputs=[args.discovery_summary],
    )
    quick = args.preset == "quick"
    points_path = output / "adaptive_points.json"
    try:
        if args.mode in {"propose", "all"}:
            if not args.distributed or is_main_process():
                discovery_rows = read_csv(args.discovery_summary)
                model, metrics = fit_surrogate(discovery_rows, args.seed)
                joblib.dump(model, output / "discovery_surrogate.joblib")
                write_json(output / "held_out_metrics.json", metrics)
                count = args.adaptive_cells or (3 if quick else 120)
                write_json(
                    points_path,
                    propose_points(model, count=count, seed=args.seed + 1, quick=quick),
                )
            if args.distributed:
                barrier()
        if args.mode in {"run", "all"}:
            if not points_path.exists():
                raise FileNotFoundError("run propose mode first")
            points = json.loads(points_path.read_text(encoding="utf-8"))
            tasks = make_tasks(
                points, quick=quick, replicate_discovery=args.replicate_discovery_grid
            )
            seeds = args.seeds or (2 if quick else 64)
            device = resolve_device(args)
            shard = output / f"atlas_shard_{shard_index:04d}.jsonl"
            completed = {row["cell_id"] for row in read_jsonl(shard)}
            selected = tasks[shard_index::num_shards]
            print(
                f"resume: {len(completed)} tasks on shard {shard_index}, "
                f"{len(selected)} assigned to this worker (rank {rank}/{world_size})"
            )
            for position, task in enumerate(selected):
                if task["task_id"] in completed:
                    continue
                rows = rows_for_cell(
                    task["coefficients"],
                    task["task_id"],
                    preset="quick" if quick else "full",
                    seeds=seeds,
                    d=task["d"],
                    m=task["m"],
                    n=task["n"],
                    base_seed=args.seed,
                    device=device,
                )
                for row in rows:
                    row["source_index"] = task["source_index"]
                    row["stage"] = "adaptive_replication"
                    row["worker_rank"] = rank
                    row["worker_world_size"] = world_size
                append_jsonl(shard, rows)
                print(
                    f"[rank {rank} {position + 1}/{len(selected)}] "
                    f"task {task['task_id']}/{len(tasks) - 1}"
                )
            if args.distributed:
                barrier()
        if args.mode in {"analyze", "all"}:
            if args.distributed and not is_main_process():
                return
            analyze_atlas(output, args.bootstrap_draws)
        if args.mode in {"analyze", "freeze", "all"}:
            if args.distributed and not is_main_process():
                return
            freeze_portfolio(output / "atlas_summary.csv", output)
            print(f"Experiment 1C results written to {output}")
    finally:
        if args.distributed:
            cleanup()


if __name__ == "__main__":
    main()
