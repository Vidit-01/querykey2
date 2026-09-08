"""Experiment 1D: one-block trainability validation on two tasks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.data import (  # noqa: E402
    associative_recall_batch,
    build_tinystories_cache,
    language_model_batch,
    load_token_cache,
)
from common.distributed import (  # noqa: E402
    barrier,
    bind_cuda_device,
    cleanup,
    current_shard_plan,
    is_main_process,
    launched_with_torchrun,
    resolve_sharding,
)
from common.geometry import Coefficients, canonical_coefficients  # noqa: E402
from common.io_utils import (  # noqa: E402
    append_jsonl,
    create_manifest,
    ensure_output,
    file_sha256,
    read_jsonl,
    record_failure,
    write_json,
)
from common.metrics import cluster_bootstrap_ci  # noqa: E402
from common.models import DecoderLM, ModelConfig  # noqa: E402
from common.portfolio import (  # noqa: E402
    coefficient_key,
    infer_label_rule,
    labeled_representatives,
)


STRATUM_COUNTS = {
    "diffuse": 15,
    "concentrated": 15,
    "self_locked": 15,
    "screened_candidate": 15,
    "boundary_uncertain": 20,
}

DEFAULT_WORKERS_PER_GPU = 8


def read_summary(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("s", "alpha", "beta", "gamma", "phi", "cell_id"):
            row[key] = float(row[key])
    return rows


def default_output(preset: str) -> Path:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return Path("/kaggle/working") / "exp1d" / "data" / preset
    return Path(__file__).parent / "data" / preset


def default_discovery_summary(preset: str) -> Path:
    return Path(__file__).parents[1] / "exp1b" / "data" / preset / "atlas_summary.csv"


def bundled_selected_points(preset: str) -> Path:
    return Path(__file__).parent / "data" / preset / "selected_points.json"


def load_selected_points(output: Path, preset: str) -> list[dict]:
    source = output / "selected_points.json"
    if not source.exists():
        source = bundled_selected_points(preset)
    if not source.exists():
        raise FileNotFoundError("selected_points.json not found; run --mode select first")
    return json.loads(source.read_text(encoding="utf-8"))["points"]


def resolve_device(args: argparse.Namespace) -> torch.device:
    if args.distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed 1D run requires CUDA (use torchrun on GPU)")
        plan = current_shard_plan()
        assert plan is not None
        index = bind_cuda_device(plan.local_rank)
        return torch.device(f"cuda:{index}")
    return torch.device(args.device)


def atlas_tag(path: Path, fallback: str) -> str:
    parts = {part.lower() for part in path.parts}
    if "exp1b" in parts:
        return "exp1b"
    if "exp1c" in parts:
        return "exp1c"
    return fallback


def freeze_selection(
    summary: Path,
    output: Path,
    *,
    quick: bool,
    seed: int,
    discovery_summary: Path | None = None,
) -> list[dict]:
    rows = read_summary(summary)
    primary_atlas = atlas_tag(summary, "primary")
    primary_rule = infer_label_rule(rows)
    by_label = labeled_representatives(rows, primary_rule, atlas=primary_atlas)
    discovery_rule = None
    discovery_used = None
    if discovery_summary is not None and discovery_summary.exists():
        discovery_rows = read_summary(discovery_summary)
        discovery_rule = infer_label_rule(discovery_rows)
        fill = labeled_representatives(
            discovery_rows,
            discovery_rule,
            atlas=atlas_tag(discovery_summary, "discovery"),
        )
        discovery_used = str(discovery_summary)
        primary_keys = {coefficient_key(row) for values in by_label.values() for row in values}
        for label, candidates in fill.items():
            extra = [row for row in candidates if coefficient_key(row) not in primary_keys]
            by_label[label].extend(extra)
    rng = np.random.default_rng(seed)
    selected = []
    stratum_sources: dict[str, dict[str, int]] = {}
    for label, full_count in STRATUM_COUNTS.items():
        candidates = list(by_label.get(label, []))
        if not quick and len(candidates) < full_count:
            raise RuntimeError(
                f"cannot preregister {full_count} {label} cells; only {len(candidates)} available"
            )
        primary = [row for row in candidates if row.get("atlas") == primary_atlas]
        filled = [row for row in candidates if row.get("atlas") != primary_atlas]
        rng.shuffle(primary)
        rng.shuffle(filled)
        ordered = primary + filled
        count = min(len(ordered), 1 if quick else full_count)
        chosen = ordered[:count]
        stratum_sources[label] = {
            "primary": sum(row.get("atlas") == primary_atlas for row in chosen),
            "discovery": sum(row.get("atlas") != primary_atlas for row in chosen),
        }
        for row in chosen:
            selected.append(
                {
                    "point_id": len(selected),
                    "stratum": label,
                    "atlas": row.get("atlas"),
                    "label_rule": row.get("label_rule"),
                    **{name: row[name] for name in ("s", "alpha", "beta", "gamma", "phi")},
                }
            )
    screened_scales = [row["s"] for row in selected if row["stratum"] == "screened_candidate"]
    matched_scale = float(np.median(screened_scales)) if screened_scales else 1.0
    for name, coefficients in canonical_coefficients(matched_scale).items():
        selected.append(
            {
                "point_id": len(selected),
                "stratum": f"corner_{name}",
                **coefficients.to_dict(),
            }
        )
    payload = {
        "frozen_before_training": True,
        "source": str(summary),
        "discovery_source": discovery_used,
        "primary_label_rule": primary_rule,
        "discovery_label_rule": discovery_rule,
        "selection_seed": seed,
        "requested_counts": STRATUM_COUNTS,
        "stratum_sources": stratum_sources,
        "points": selected,
    }
    write_json(output / "selected_points.json", payload)
    return selected


def coefficient_from_row(row: dict) -> Coefficients:
    return Coefficients(
        float(row["s"]),
        float(row["alpha"]),
        float(row["beta"]),
        float(row["gamma"]),
        None if row.get("phi") is None else float(row["phi"]),
    )


def qk_gradient_stats(model: DecoderLM, learning_rate: float) -> dict[str, float]:
    q_norm = k_norm = q_weight = k_weight = 0.0
    for block in model.blocks:
        q = block.attention.q_proj.weight
        k = block.attention.k_proj.weight
        q_norm += float(torch.linalg.matrix_norm(q.grad)) ** 2
        k_norm += float(torch.linalg.matrix_norm(k.grad)) ** 2
        q_weight += float(torch.linalg.matrix_norm(q.detach())) ** 2
        k_weight += float(torch.linalg.matrix_norm(k.detach())) ** 2
    q_norm, k_norm = math.sqrt(q_norm), math.sqrt(k_norm)
    q_weight, k_weight = math.sqrt(q_weight), math.sqrt(k_weight)
    return {
        "gradient_q_norm": q_norm,
        "gradient_k_norm": k_norm,
        "update_to_weight_q": learning_rate * q_norm / q_weight,
        "update_to_weight_k": learning_rate * k_norm / k_weight,
    }


def train_one(
    row: dict,
    *,
    task: str,
    paired_seed: int,
    steps: int,
    batch_size: int,
    context_length: int,
    learning_rate: float,
    device: torch.device,
    token_cache: np.ndarray | None,
    output: Path,
) -> dict:
    torch.manual_seed(paired_seed)
    config = ModelConfig(
        vocab_size=256,
        context_length=context_length,
        width=128,
        layers=1,
        heads=4,
        dropout=0.0,
    )
    model = DecoderLM(config).to(device)
    coefficients = coefficient_from_row(row)
    model.apply_query_key_geometry([[coefficients] * config.heads], seed=paired_seed + 50_000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    trace_path = output / f"trace_point{row['point_id']:03d}_{task}_seed{paired_seed}.jsonl"
    traces = []
    losses = []
    tokens_seen = 0
    clipped = 0
    started = time.perf_counter()
    for step in range(steps):
        if task == "associative_recall":
            inputs, targets, target_mask = associative_recall_batch(
                batch_size=batch_size,
                pairs=max(2, (context_length - 4) // 2),
                seed=paired_seed,
                step=step,
                device=device,
            )
            logits = model(inputs)["logits"]
            loss = F.cross_entropy(logits[target_mask], targets[target_mask])
            step_tokens = int(target_mask.sum())
        else:
            if token_cache is None:
                raise RuntimeError("TinyStories token cache was not prepared")
            inputs, targets = language_model_batch(
                token_cache,
                batch_size=batch_size,
                context_length=context_length,
                seed=paired_seed,
                step=step,
                device=device,
            )
            loss = model(inputs, targets)["loss"]
            step_tokens = targets.numel()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        stats = qk_gradient_stats(model, learning_rate)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        clipped += int(float(total_norm) > 1.0)
        optimizer.step()
        value = float(loss.detach())
        losses.append(value)
        tokens_seen += step_tokens
        traces.append(
            {
                "step": step,
                "tokens_seen": tokens_seen,
                "loss": value,
                "total_gradient_norm": float(total_norm),
                **stats,
            }
        )
        if not math.isfinite(value):
            break
    if traces:
        append_jsonl(trace_path, traces)
    x = np.arange(len(losses), dtype=float) * (tokens_seen / max(1, len(losses)))
    slope = float(np.polyfit(x, np.asarray(losses), 1)[0]) if len(losses) >= 2 else math.nan
    return {
        "point_id": row["point_id"],
        "stratum": row["stratum"],
        **{name: row[name] for name in ("s", "alpha", "beta", "gamma", "phi")},
        "task": task,
        "paired_seed": paired_seed,
        "steps_completed": len(losses),
        "tokens_seen": tokens_seen,
        "initial_loss": losses[0] if losses else math.nan,
        "final_loss": losses[-1] if losses else math.nan,
        "loss_decrease_per_token": (losses[0] - losses[-1]) / max(1, tokens_seen)
        if losses
        else math.nan,
        "loss_slope_per_token": slope,
        "diverged": not losses or not all(math.isfinite(value) for value in losses),
        "gradient_clipping_frequency": clipped / max(1, len(losses)),
        "elapsed_seconds": time.perf_counter() - started,
    }


def analyze(output: Path, bootstrap_draws: int) -> None:
    rows = []
    for path in sorted(output.glob("training_results_shard_*.jsonl")):
        rows.extend(read_jsonl(path))
    if not rows:
        raise FileNotFoundError("no training results found")
    comparisons = []
    for task in sorted({row["task"] for row in rows}):
        task_rows = [row for row in rows if row["task"] == task and not row["diverged"]]
        per_seed: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in task_rows:
            group = (
                "screened"
                if row["stratum"] == "screened_candidate"
                else "pathology"
                if row["stratum"] in {"diffuse", "concentrated", "self_locked", "gradient_starved"}
                else "other"
            )
            per_seed[int(row["paired_seed"])][group].append(row["loss_decrease_per_token"])
        paired = [
            np.mean(groups["screened"]) - np.mean(groups["pathology"])
            for groups in per_seed.values()
            if groups["screened"] and groups["pathology"]
        ]
        estimate, lower, upper = cluster_bootstrap_ci(
            paired, statistic=np.mean, draws=bootstrap_draws, seed=20260829
        )
        comparisons.append(
            {
                "task": task,
                "paired_seeds": len(paired),
                "screened_minus_pathology_loss_decrease_per_token": estimate,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "atlas_useful_gate": lower > 0,
            }
        )
    write_json(
        output / "trainability_analysis.json",
        {
            "comparisons": comparisons,
            "divergence_rate": float(np.mean([row["diverged"] for row in rows])),
            "interpretation": "A task passes only when the paired CI for screened minus pathology loss decrease excludes zero above.",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["quick", "full"], default="quick")
    parser.add_argument("--mode", choices=["select", "run", "analyze", "all"], default="all")
    parser.add_argument(
        "--atlas-summary",
        type=Path,
    )
    parser.add_argument(
        "--discovery-summary",
        type=Path,
        help="1B atlas used to fill pathology strata that 1C adaptive cells do not contain.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tinystories-file", type=Path)
    parser.add_argument("--include-text", action="store_true")
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Use torchrun process-group sharding (disjoint training jobs, packed onto GPUs).",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=int(os.environ.get("WORKERS_PER_GPU", DEFAULT_WORKERS_PER_GPU)),
        help="Independent trainings to pack on each GPU (launchers set nproc_per_node accordingly).",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()
    if args.distributed and not launched_with_torchrun():
        raise RuntimeError(
            "Pass --distributed only under torchrun, e.g. "
            "torchrun --standalone --nproc_per_node=2 code/exp1d/run.py --distributed"
        )
    shard_index, num_shards, rank, world_size = resolve_sharding(
        args.shard_index, args.num_shards, distributed=args.distributed
    )
    if args.atlas_summary is None:
        args.atlas_summary = (
            Path(__file__).parents[1]
            / "exp1c"
            / "data"
            / args.preset
            / "atlas_summary.csv"
        )
    if args.discovery_summary is None:
        args.discovery_summary = default_discovery_summary(args.preset)
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
        experiment="exp1d",
        arguments=manifest_args,
        inputs=[args.atlas_summary, args.discovery_summary],
    )
    quick = args.preset == "quick"
    selection_path = output / "selected_points.json"
    try:
        existing_selection = (
            selection_path if selection_path.exists() else bundled_selected_points(args.preset)
        )
        should_select = args.mode in {"select", "all"} or not existing_selection.exists()
        if should_select:
            if not args.distributed or is_main_process():
                points = freeze_selection(
                    args.atlas_summary,
                    output,
                    quick=quick,
                    seed=args.seed,
                    discovery_summary=args.discovery_summary,
                )
            if args.distributed:
                barrier()
                if not is_main_process():
                    points = load_selected_points(output, args.preset)
        else:
            points = load_selected_points(output, args.preset)
        include_text = args.include_text or not quick
        token_cache = None
        cache_path = output / "tinystories_train.uint8"
        if include_text:
            if not args.distributed or is_main_process():
                built = build_tinystories_cache(
                    cache_path,
                    split="train",
                    local_path=args.tinystories_file,
                    max_stories=100 if quick else None,
                )
                write_json(
                    output / "data_hashes.json",
                    {"tinystories_train_uint8_sha256": file_sha256(built)},
                )
            if args.distributed:
                barrier()
            if args.mode in {"run", "all"}:
                token_cache = load_token_cache(cache_path)
        if args.mode in {"run", "all"}:
            seeds = args.seeds or (1 if quick else 8)
            steps = args.steps or (10 if quick else 2000)
            batch_size = args.batch_size or (4 if quick else 64)
            context_length = 32 if quick else 128
            tasks = ["associative_recall"] + (["tinystories"] if include_text else [])
            jobs = [
                (point, task, args.seed + seed_index)
                for point in points
                for task in tasks
                for seed_index in range(seeds)
            ]
            completed_rows = []
            for path in output.glob("training_results_shard_*.jsonl"):
                completed_rows.extend(read_jsonl(path))
            completed = {
                (row["point_id"], row["task"], row["paired_seed"])
                for row in completed_rows
            }
            assigned = jobs[shard_index::num_shards]
            device = resolve_device(args)
            results_path = output / f"training_results_shard_{shard_index:04d}.jsonl"
            device_name = (
                torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
            )
            packed = (
                world_size // max(1, torch.cuda.device_count())
                if device.type == "cuda"
                else 1
            )
            print(
                f"rank {rank}/{world_size} device={device} ({device_name}) "
                f"{packed} workers/GPU shard {shard_index}/{num_shards}: "
                f"{len(assigned)} jobs ({len(completed)} already on disk)"
            )
            for position, (point, task, paired_seed) in enumerate(assigned):
                key = (point["point_id"], task, paired_seed)
                if key in completed:
                    continue
                try:
                    result = train_one(
                        point,
                        task=task,
                        paired_seed=paired_seed,
                        steps=steps,
                        batch_size=batch_size,
                        context_length=context_length,
                        learning_rate=3e-4,
                        device=device,
                        token_cache=token_cache,
                        output=output,
                    )
                    result["worker_rank"] = rank
                    result["worker_world_size"] = world_size
                    append_jsonl(results_path, [result])
                except (RuntimeError, ValueError) as error:
                    record_failure(
                        output, {"point": point, "task": task, "rank": rank}, error
                    )
                    raise
                print(
                    f"[rank {rank} {position + 1}/{len(assigned)}] "
                    f"point={point['point_id']} task={task} seed={paired_seed}"
                )
            if args.distributed:
                barrier()
        if args.mode in {"analyze", "all"}:
            if args.distributed and not is_main_process():
                return
            analyze(output, args.bootstrap_draws)
            print(f"Experiment 1D results written to {output}")
    finally:
        if args.distributed:
            cleanup()


if __name__ == "__main__":
    main()
