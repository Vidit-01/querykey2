"""Experiment 1B: discovery sweep over the constrained initialization atlas."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.geometry import (  # noqa: E402
    Coefficients,
    coefficient_grid,
    draw_query_key,
    make_generator,
)
from common.io_utils import (  # noqa: E402
    append_jsonl,
    create_manifest,
    ensure_output,
    read_jsonl,
    record_failure,
    write_json,
)
from common.metrics import (  # noqa: E402
    attention_metrics,
    classify_cell,
    cluster_bootstrap_ci,
    gram_features,
    kernel_metrics,
)
from common.models import DecoderLM, ModelConfig  # noqa: E402


METRICS_FOR_INTERVALS = (
    "normalized_entropy",
    "distance_uniform",
    "max_mass",
    "self_mass",
    "jacobian_op",
    "update_to_weight_min",
    "attention_effective_rank",
)

_ACTIVATION_CACHE: dict[tuple[int, int, int, int, str], torch.Tensor] = {}


def discovery_grid(preset: str) -> list[Coefficients]:
    if preset == "quick":
        return coefficient_grid(
            log2_s_values=np.linspace(-1, 1, 3),
            alpha_values=np.linspace(-1, 1, 3),
            phi_values=np.linspace(0, math.pi / 2, 3),
        )
    cells = coefficient_grid(
        log2_s_values=np.linspace(-2, 2, 17),
        alpha_values=np.linspace(-1, 1, 13),
        phi_values=np.linspace(0, math.pi / 2, 9),
    )
    assert len(cells) == 1717
    return cells


def synthetic_inputs(
    d: int, n: int, correlation: float, *, seed: int, device: torch.device
) -> torch.Tensor:
    generator = make_generator(seed, device)
    common = torch.randn((1, d), generator=generator, device=device)
    noise = torch.randn((n, d), generator=generator, device=device)
    x = math.sqrt(correlation) * common + math.sqrt(1 - correlation) * noise
    return F.layer_norm(x, (d,))


def baseline_activations(
    d: int, m: int, n: int, *, seed: int, device: torch.device
) -> torch.Tensor:
    """Capture the tensor immediately after LayerNorm in a baseline decoder."""
    cache_key = (d, m, n, seed, str(device))
    if cache_key in _ACTIVATION_CACHE:
        return _ACTIVATION_CACHE[cache_key]
    torch.manual_seed(seed)
    config = ModelConfig(
        width=d,
        heads=d // m,
        layers=1,
        context_length=n,
        vocab_size=256,
    )
    model = DecoderLM(config).to(device).eval()
    generator = make_generator(seed + 1, device)
    tokens = torch.randint(0, 256, (1, n), generator=generator, device=device)
    positions = torch.arange(n, device=device)
    with torch.no_grad():
        hidden = model.token_embedding(tokens) + model.position_embedding(positions)
        result = model.blocks[0].norm1(hidden)[0].detach()
    _ACTIVATION_CACHE[cache_key] = result
    return result


def measure_cell(
    coefficients: Coefficients,
    *,
    cell_id: int,
    d: int,
    m: int,
    n: int,
    seed: int,
    input_kind: str,
    correlation: float | None,
    causal: bool,
    device: torch.device,
) -> dict:
    generator = make_generator(seed + 17_171 * cell_id, device)
    if input_kind == "synthetic":
        assert correlation is not None
        x = synthetic_inputs(d, n, correlation, seed=seed, device=device)
    else:
        x = baseline_activations(d, m, n, seed=seed, device=device)
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    x = x.to(dtype)
    wq, wk, _ = draw_query_key(
        d, m, coefficients, generator=generator, device=device, dtype=dtype
    )
    wq.requires_grad_(True)
    wk.requires_grad_(True)
    wv = torch.randn((m, d), generator=generator, device=device, dtype=dtype) / math.sqrt(d)
    q, k, values = x @ wq.T, x @ wk.T, x @ wv.T
    logits = q @ k.T / math.sqrt(m)
    mask = (
        torch.ones((n, n), dtype=torch.bool, device=device).tril()
        if causal
        else torch.ones((n, n), dtype=torch.bool, device=device)
    )
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    probabilities = F.softmax(masked_logits, dim=-1)
    output = probabilities @ values
    target = torch.roll(values.detach(), shifts=1, dims=0)
    loss = F.mse_loss(output, target)
    loss.backward()
    metrics = attention_metrics(masked_logits, probabilities, mask, values=values)
    metrics.update(kernel_metrics(wq.detach(), wk.detach()))
    gram = gram_features(x)
    token_gram = x @ x.T / d
    energies = torch.diagonal(token_gram)
    vi_vj = torch.outer(energies, energies)
    g = d * ((d + 1) * vi_vj - 2 * token_gram.square()) / ((d - 1) * (d + 2))
    predicted_mean = coefficients.alpha * coefficients.s**2 * math.sqrt(m) * token_gram
    predicted_variance = coefficients.s**4 * (
        coefficients.alpha**2 * (vi_vj + token_gram.square())
        + coefficients.beta**2 * g
        + coefficients.gamma**2 * vi_vj
    )
    active = mask
    weight_q = float(torch.linalg.matrix_norm(wq.detach()))
    weight_k = float(torch.linalg.matrix_norm(wk.detach()))
    gradient_q = float(torch.linalg.matrix_norm(wq.grad))
    gradient_k = float(torch.linalg.matrix_norm(wk.grad))
    metrics.update(
        {
            "loss": float(loss.detach()),
            "gradient_q_norm": gradient_q,
            "gradient_k_norm": gradient_k,
            "update_to_weight_q": 1e-3 * gradient_q / weight_q,
            "update_to_weight_k": 1e-3 * gradient_k / weight_k,
            "update_to_weight_min": min(
                1e-3 * gradient_q / weight_q,
                1e-3 * gradient_k / weight_k,
            ),
            "predicted_logit_mean": float(predicted_mean[active].mean()),
            "predicted_logit_variance": float(predicted_variance[active].mean()),
            **gram,
        }
    )
    return {
        "cell_id": cell_id,
        "seed": seed,
        "d": d,
        "m": m,
        "n": n,
        **coefficients.to_dict(),
        "input_kind": input_kind,
        "correlation": correlation,
        "mask": "causal" if causal else "unmasked",
        **metrics,
    }


def rows_for_cell(
    coefficients: Coefficients,
    cell_id: int,
    *,
    preset: str,
    seeds: int,
    d: int,
    m: int,
    n: int,
    base_seed: int,
    device: torch.device,
) -> list[dict]:
    strata = [("synthetic", r) for r in ((0.25,) if preset == "quick" else (0.0, 0.25, 0.5))]
    strata.append(("real_activation", None))
    rows = []
    for seed_index in range(seeds):
        paired_seed = base_seed + seed_index
        for input_kind, correlation in strata:
            for causal in (False, True):
                rows.append(
                    measure_cell(
                        coefficients,
                        cell_id=cell_id,
                        d=d,
                        m=m,
                        n=n,
                        seed=paired_seed,
                        input_kind=input_kind,
                        correlation=correlation,
                        causal=causal,
                        device=device,
                    )
                )
    return rows


def analyze(output: Path, bootstrap_draws: int) -> None:
    rows: list[dict] = []
    for path in sorted(output.glob("atlas_shard_*.jsonl")):
        rows.extend(read_jsonl(path))
    if not rows:
        raise FileNotFoundError("no atlas shard files found")
    baseline = [
        row
        for row in rows
        if abs(row["s"] - 1) < 1e-9
        and abs(row["alpha"]) < 1e-9
        and abs(row["gamma"] - 1) < 1e-9
    ]
    source = baseline or rows
    thresholds = {
        "jacobian_floor": float(np.nanpercentile([r["jacobian_op"] for r in source], 5)),
        "update_floor": float(
            np.nanpercentile([r["update_to_weight_min"] for r in source], 5)
        ),
        "rank_floor": float(
            np.nanpercentile([r["attention_effective_rank"] for r in source], 5)
        ),
        "derived_from": "independent_s1_baseline" if baseline else "all_quick_rows",
    }
    write_json(output / "pilot_thresholds.json", thresholds)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    keys = ("cell_id", "input_kind", "correlation", "mask", "d", "m", "n")
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    summaries = []
    for group_key, group_rows in groups.items():
        intervals = {}
        for metric in METRICS_FOR_INTERVALS:
            intervals[metric] = cluster_bootstrap_ci(
                [row[metric] for row in group_rows],
                draws=bootstrap_draws,
                seed=1_000_003 + int(group_key[0]),
            )
        first = group_rows[0]
        summary = {key: value for key, value in zip(keys, group_key, strict=True)}
        summary.update(
            {
                "s": first["s"],
                "alpha": first["alpha"],
                "beta": first["beta"],
                "gamma": first["gamma"],
                "phi": first["phi"],
                "sample_count": len(group_rows),
                "mean_vi_vj": float(np.mean([r["mean_vi_vj"] for r in group_rows])),
                "mean_cij2": float(np.mean([r["mean_cij2"] for r in group_rows])),
                "delta": float(np.mean([r["delta"] for r in group_rows])),
                "tau": float(np.mean([r["tau"] for r in group_rows])),
                "label": classify_cell(
                    intervals,
                    jacobian_floor=thresholds["jacobian_floor"],
                    update_floor=thresholds["update_floor"],
                    rank_floor=thresholds["rank_floor"],
                ),
                "label_sensitivity_minus_0p05": classify_cell(
                    intervals,
                    jacobian_floor=thresholds["jacobian_floor"],
                    update_floor=thresholds["update_floor"],
                    rank_floor=thresholds["rank_floor"],
                    probability_shift=-0.05,
                ),
                "label_sensitivity_plus_0p05": classify_cell(
                    intervals,
                    jacobian_floor=thresholds["jacobian_floor"],
                    update_floor=thresholds["update_floor"],
                    rank_floor=thresholds["rank_floor"],
                    probability_shift=0.05,
                ),
            }
        )
        for metric, (estimate, lower, upper) in intervals.items():
            summary[metric] = estimate
            summary[f"{metric}_lower"] = lower
            summary[f"{metric}_upper"] = upper
        summaries.append(summary)
    fieldnames = sorted({key for row in summaries for key in row})
    with (output / "atlas_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    plot_outputs(rows, summaries, output)
    write_json(
        output / "analysis_summary.json",
        {
            "raw_rows": len(rows),
            "stratified_cells": len(summaries),
            "unique_coefficient_cells": len({row["cell_id"] for row in rows}),
            "label_counts": {
                label: sum(row["label"] == label for row in summaries)
                for label in sorted({row["label"] for row in summaries})
            },
        },
    )


def plot_outputs(rows: list[dict], summaries: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    label_values = {
        "diffuse": 0,
        "gradient_starved": 1,
        "screened_candidate": 2,
        "boundary_uncertain": 3,
        "concentrated": 4,
        "self_locked": 5,
    }
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for row in summaries:
        key = (
            row["input_kind"],
            row["correlation"],
            row["mask"],
            round(float(row["phi"] or 0), 6),
            int(row["d"]),
            int(row["m"]),
            int(row["n"]),
        )
        strata[key].append(row)
    for (input_kind, correlation, mask, phi, d, m, n), values in strata.items():
        fig, axis = plt.subplots(figsize=(6.2, 4.6))
        colors = [label_values[row["label"]] for row in values]
        scatter = axis.scatter(
            [math.log2(float(row["s"])) for row in values],
            [float(row["alpha"]) for row in values],
            c=colors,
            cmap="turbo",
            vmin=0,
            vmax=5,
            s=34,
        )
        axis.set(
            xlabel="log2(s)",
            ylabel="alpha",
            title=f"{input_kind}, {mask}, r={correlation}, phi={phi}, (d,m,n)=({d},{m},{n})",
        )
        colorbar = fig.colorbar(scatter, ax=axis, ticks=list(label_values.values()))
        colorbar.ax.set_yticklabels(list(label_values))
        fig.tight_layout()
        safe_r = "real" if correlation is None else str(correlation).replace(".", "p")
        fig.savefig(
            figure_dir
            / f"atlas_d{d}_m{m}_n{n}_{input_kind}_{mask}_r{safe_r}_phi{phi:.6f}.png",
            dpi=160,
        )
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(
        [row["predicted_logit_mean"] for row in rows],
        [row["logit_mean"] - row["predicted_logit_mean"] for row in rows],
        s=8,
        alpha=0.5,
    )
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set(xlabel="predicted mean", ylabel="empirical - predicted", title="Logit mean residual")
    axes[1].scatter(
        [row["predicted_logit_variance"] for row in rows],
        [row["logit_variance"] - row["predicted_logit_variance"] for row in rows],
        s=8,
        alpha=0.5,
    )
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set(
        xlabel="predicted variance",
        ylabel="empirical - predicted",
        title="Logit variance residual",
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "exact_moment_residuals.png", dpi=160)
    plt.close(fig)


def symmetry_check(
    cells: list[Coefficients],
    *,
    count: int,
    seeds: int,
    d: int,
    m: int,
    n: int,
    base_seed: int,
    device: torch.device,
    output: Path,
) -> None:
    selected = np.linspace(0, len(cells) - 1, min(count, len(cells)), dtype=int)
    results = []
    for cell_id in selected:
        original = cells[int(cell_id)]
        variants = {
            "positive": original,
            "negative_beta": Coefficients(
                original.s, original.alpha, -original.beta, original.gamma, original.phi
            ),
            "negative_gamma": Coefficients(
                original.s, original.alpha, original.beta, -original.gamma, original.phi
            ),
        }
        aggregates = {}
        for name, coefficients in variants.items():
            measured = [
                measure_cell(
                    coefficients,
                    cell_id=int(cell_id),
                    d=d,
                    m=m,
                    n=n,
                    seed=base_seed + seed_index,
                    input_kind="synthetic",
                    correlation=0.25,
                    causal=False,
                    device=device,
                )
                for seed_index in range(seeds)
            ]
            aggregates[name] = {
                metric: float(np.mean([row[metric] for row in measured]))
                for metric in ("logit_mean", "logit_variance", "normalized_entropy", "self_mass")
            }
        results.append({"cell_id": int(cell_id), "aggregates": aggregates})
    write_json(
        output / "sign_symmetry_check.json",
        {
            "cells": len(results),
            "seeds_per_variant": seeds,
            "interpretation": "Differences should shrink with seed count; signs are distributionally, not draw-wise, redundant.",
            "results": results,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["quick", "full"], default="quick")
    parser.add_argument("--mode", choices=["run", "analyze", "all"], default="all")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    output = ensure_output(args.output or Path(__file__).parent / "data" / args.preset)
    create_manifest(output, experiment="exp1b", arguments=vars(args))
    if args.mode in {"run", "all"}:
        d, m, n = (32, 8, 24) if args.preset == "quick" else (64, 16, 64)
        seeds = args.seeds or (2 if args.preset == "quick" else 64)
        cells = discovery_grid(args.preset)
        shard_path = output / f"atlas_shard_{args.shard_index:04d}.jsonl"
        completed = {row["cell_id"] for row in read_jsonl(shard_path)}
        for position, cell_id in enumerate(range(args.shard_index, len(cells), args.num_shards)):
            if cell_id in completed:
                continue
            started = time.perf_counter()
            try:
                rows = rows_for_cell(
                    cells[cell_id],
                    cell_id,
                    preset=args.preset,
                    seeds=seeds,
                    d=d,
                    m=m,
                    n=n,
                    base_seed=args.seed,
                    device=torch.device(args.device),
                )
                for row in rows:
                    row["cell_elapsed_seconds"] = time.perf_counter() - started
                append_jsonl(shard_path, rows)
            except (RuntimeError, ValueError) as error:
                record_failure(output, {"cell_id": cell_id}, error)
                raise
            print(f"[{position + 1}] cell {cell_id}/{len(cells) - 1}")
        if args.shard_index == 0:
            symmetry_check(
                cells,
                count=3 if args.preset == "quick" else 50,
                seeds=seeds,
                d=d,
                m=m,
                n=n,
                base_seed=args.seed + 30_000_000,
                device=torch.device(args.device),
                output=output,
            )
    if args.mode in {"analyze", "all"}:
        analyze(output, args.bootstrap_draws)
        print(f"Experiment 1B results written to {output}")


if __name__ == "__main__":
    main()
