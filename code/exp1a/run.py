"""Experiment 1A: finite-width identities and implementation gate."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ks_2samp, norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.geometry import (  # noqa: E402
    Coefficients,
    canonical_coefficients,
    coefficients_from_alpha_phi,
    common_bias_ratio,
    draw_query_key,
    make_generator,
    predicted_logit_moments,
)
from common.io_utils import (  # noqa: E402
    append_jsonl,
    create_manifest,
    ensure_output,
    jsonl_unique_values,
    read_jsonl,
    write_json,
)
from common.metrics import holm_rejections  # noqa: E402
from common.models import (  # noqa: E402
    DecoderLM,
    ModelConfig,
    query_key_state,
)


def coefficient_points() -> list[tuple[str, Coefficients]]:
    points = list(canonical_coefficients().items())
    for index in range(20):
        alpha = -0.9 + 1.8 * index / 19
        phi = (index % 10 + 0.5) * math.pi / 20
        points.append((f"interior_{index:02d}", coefficients_from_alpha_phi(1.0, alpha, phi)))
    return points


def token_pair(d: int, correlation: float) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.zeros(d, dtype=torch.float64)
    y = torch.zeros(d, dtype=torch.float64)
    x[0] = math.sqrt(d)
    y[0] = correlation * math.sqrt(d)
    if abs(correlation) < 1:
        y[1] = math.sqrt(1 - correlation**2) * math.sqrt(d)
    return x, y


def configuration_list(preset: str) -> list[dict]:
    if preset == "quick":
        dimensions = [(32, 4), (32, 16)]
        correlations = [-0.5, 0.0, 1.0]
        points = coefficient_points()[:6]
    else:
        dimensions = [
            (d, int(d * ratio))
            for d in (32, 64, 128)
            for ratio in (1 / 8, 1 / 4, 1 / 2)
        ]
        correlations = [-0.5, 0.0, 0.5, 1.0]
        points = coefficient_points()
    return [
        {
            "config_id": f"d{d}_m{m}_c{correlation:+.1f}_{name}",
            "d": d,
            "m": m,
            "correlation": correlation,
            "name": name,
            "coefficients": coefficients,
        }
        for d, m in dimensions
        for correlation in correlations
        for name, coefficients in points
    ]


def run_moment_configuration(config: dict, draws: int, seed: int) -> dict:
    d, m = config["d"], config["m"]
    coefficients = config["coefficients"]
    x, y = token_pair(d, config["correlation"])
    generator = make_generator(seed)
    values = np.empty(draws, dtype=np.float64)
    q_energy = b_energy = c_energy = k_energy = 0.0
    entry_q_sum = entry_k_sum = entry_qq = entry_kk = entry_qk = 0.0
    entries = 0
    maximum_orthogonality = 0.0
    for draw in range(draws):
        wq, wk, (a, b, c) = draw_query_key(
            d, m, coefficients, generator=generator
        )
        values[draw] = float((wq @ x) @ (wk @ y) / math.sqrt(m))
        q_energy += float(wq.square().sum())
        k_energy += float(wk.square().sum())
        b_energy += float(b.square().sum())
        c_energy += float(c.square().sum())
        orthogonality = float(
            torch.linalg.matrix_norm(a @ b.T)
            / (torch.linalg.matrix_norm(a) * torch.linalg.matrix_norm(b))
        )
        maximum_orthogonality = max(maximum_orthogonality, orthogonality)
        q_flat, k_flat = wq.flatten(), wk.flatten()
        entry_q_sum += float(q_flat.sum())
        entry_k_sum += float(k_flat.sum())
        entry_qq += float(q_flat @ q_flat)
        entry_kk += float(k_flat @ k_flat)
        entry_qk += float(q_flat @ k_flat)
        entries += q_flat.numel()

    vi = float(x @ x / d)
    vj = float(y @ y / d)
    cij = float(x @ y / d)
    predicted_mean, predicted_variance = predicted_logit_moments(
        d, m, coefficients, vi, vj, cij
    )
    empirical_mean = float(values.mean())
    empirical_variance = float(values.var(ddof=1))
    mean_se = math.sqrt(empirical_variance / draws)
    centered = values - empirical_mean
    fourth = float(np.mean(centered**4))
    variance_se = math.sqrt(
        max(0.0, fourth - ((draws - 3) / (draws - 1)) * empirical_variance**2)
        / draws
    )
    q_mean, k_mean = entry_q_sum / entries, entry_k_sum / entries
    covariance = entry_qk / entries - q_mean * k_mean
    q_variance = entry_qq / entries - q_mean**2
    k_variance = entry_kk / entries - k_mean**2
    entry_correlation = covariance / math.sqrt(q_variance * k_variance)
    mean_precision_ratio = mean_se / math.sqrt(float(predicted_variance))
    variance_precision_ratio = variance_se / float(predicted_variance)
    return {
        "config_id": config["config_id"],
        "d": d,
        "m": m,
        "cij": cij,
        **coefficients.to_dict(),
        "draws": draws,
        "empirical_mean": empirical_mean,
        "predicted_mean": float(predicted_mean),
        "mean_se": mean_se,
        "mean_p": float(2 * norm.sf(abs(empirical_mean - predicted_mean) / mean_se)),
        "empirical_variance": empirical_variance,
        "predicted_variance": float(predicted_variance),
        "variance_se": variance_se,
        "mean_precision_ratio": mean_precision_ratio,
        "variance_precision_ratio": variance_precision_ratio,
        "precision_converged": (
            mean_precision_ratio <= 0.01 and variance_precision_ratio <= 0.01
        ),
        "variance_p": float(
            2
            * norm.sf(
                abs(empirical_variance - predicted_variance)
                / max(variance_se, 1e-30)
            )
        ),
        "q_energy": q_energy / draws,
        "k_energy": k_energy / draws,
        "b_energy": b_energy / draws,
        "c_energy": c_energy / draws,
        "energy_target": coefficients.s**2 * m,
        "entry_variance_q": q_variance,
        "entry_variance_k": k_variance,
        "entry_variance_target": coefficients.s**2 / d,
        "entry_correlation": entry_correlation,
        "maximum_relative_orthogonality": maximum_orthogonality,
    }


def gradient_checks(count: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    results = []
    for index in range(count):
        d, m, n = 16, 4, 5
        coefficients = coefficients_from_alpha_phi(
            float(2 ** rng.uniform(-1, 1)),
            float(rng.uniform(-0.95, 0.95)),
            float(rng.uniform(0, math.pi / 2)),
        )
        generator = make_generator(seed + index)
        wq, wk, _ = draw_query_key(d, m, coefficients, generator=generator)
        wq.requires_grad_(True)
        wk.requires_grad_(True)
        x = torch.randn((d, n), generator=generator, dtype=torch.float64)
        upstream = torch.randn((n, n), generator=generator, dtype=torch.float64)
        logits = (wq @ x).T @ (wk @ x) / math.sqrt(m)
        loss = (logits * upstream).sum()
        loss.backward()
        analytic_q = wk.detach() @ x @ upstream.T @ x.T / math.sqrt(m)
        analytic_k = wq.detach() @ x @ upstream @ x.T / math.sqrt(m)
        identity_error = max(
            float((wq.grad - analytic_q).abs().max()),
            float((wk.grad - analytic_k).abs().max()),
        )
        epsilon = 1e-6
        relative_errors = []
        for which, weight, other, analytic in (
            ("q", wq.detach(), wk.detach(), analytic_q),
            ("k", wk.detach(), wq.detach(), analytic_k),
        ):
            for _ in range(4):
                row, column = int(rng.integers(m)), int(rng.integers(d))
                plus, minus = weight.clone(), weight.clone()
                plus[row, column] += epsilon
                minus[row, column] -= epsilon
                if which == "q":
                    lp = (((plus @ x).T @ (other @ x) / math.sqrt(m)) * upstream).sum()
                    lm = (((minus @ x).T @ (other @ x) / math.sqrt(m)) * upstream).sum()
                else:
                    lp = (((other @ x).T @ (plus @ x) / math.sqrt(m)) * upstream).sum()
                    lm = (((other @ x).T @ (minus @ x) / math.sqrt(m)) * upstream).sum()
                numeric = float((lp - lm) / (2 * epsilon))
                expected = float(analytic[row, column])
                relative_errors.append(
                    abs(numeric - expected) / max(1.0, abs(numeric), abs(expected))
                )
        results.append(
            {
                "configuration": index,
                "identity_max_absolute_error": identity_error,
                "finite_difference_max_relative_error": max(relative_errors),
                "passed": identity_error < 1e-10 and max(relative_errors) < 1e-5,
            }
        )
    return results


def integration_checkpoint_check(seed: int) -> dict:
    torch.manual_seed(seed)
    config = ModelConfig(width=32, heads=4, layers=1, context_length=16)
    model = DecoderLM(config).double()
    before = query_key_state(model)
    coefficients = [
        coefficients_from_alpha_phi(1.0, alpha, phi)
        for alpha, phi in [(-0.6, 0.2), (0.0, 1.0), (0.4, 0.7), (1.0, 0.0)]
    ]
    model.apply_query_key_geometry([coefficients], seed=seed + 1)
    initialized = query_key_state(model)
    changed = all(not torch.equal(before[key], initialized[key]) for key in before)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "initial.pt"
        torch.save(model.state_dict(), checkpoint)
        reloaded = DecoderLM(config).double()
        reloaded.load_state_dict(torch.load(checkpoint, weights_only=True))
        after = query_key_state(reloaded)
    bitwise = all(torch.equal(initialized[key], after[key]) for key in initialized)
    sample = torch.randint(0, 256, (2, 8), generator=make_generator(seed))
    model.eval()
    with torch.no_grad():
        forward_ok = torch.isfinite(model(sample)["logits"]).all().item()
    return {
        "initializer_changed_qk": changed,
        "checkpoint_bitwise_equal": bitwise,
        "first_forward_finite": bool(forward_ok),
        "passed": changed and bitwise and bool(forward_ok),
    }


def common_bias_check(draws: int, seed: int) -> dict:
    d, m, alpha1, alpha2 = 32, 8, -0.45, 0.7
    c1 = coefficients_from_alpha_phi(1.0, alpha1, 0.3)
    c2 = coefficients_from_alpha_phi(1.0, alpha2, 1.1)
    generator = make_generator(seed)
    numerator = norm1 = norm2 = 0.0
    for _ in range(draws):
        q1, k1, _ = draw_query_key(d, m, c1, generator=generator)
        q2, k2, _ = draw_query_key(d, m, c2, generator=generator)
        kernel1, kernel2 = q1.T @ k1, q2.T @ k2
        numerator += float((kernel1 * kernel2).sum())
        norm1 += float(kernel1.square().sum())
        norm2 += float(kernel2.square().sum())
    empirical = numerator / math.sqrt(norm1 * norm2)
    predicted = common_bias_ratio(d, m, alpha1, alpha2)
    absolute_error = abs(empirical - predicted)
    return {
        "draws": draws,
        "empirical_ratio_of_expectations": empirical,
        "predicted_ratio_of_expectations": predicted,
        "absolute_error": absolute_error,
        "passed": absolute_error < 0.02,
    }


def baseline_initializer_check(draws: int, seed: int) -> dict:
    d, m = 32, 8
    coefficients = canonical_coefficients(1.0)["independent"]
    x, y = token_pair(d, 0.25)
    generator = make_generator(seed)
    custom = np.empty(draws)
    baseline = np.empty(draws)
    for index in range(draws):
        wq, wk, _ = draw_query_key(d, m, coefficients, generator=generator)
        baseline_q = torch.randn((m, d), generator=generator, dtype=torch.float64) / math.sqrt(d)
        baseline_k = torch.randn((m, d), generator=generator, dtype=torch.float64) / math.sqrt(d)
        custom[index] = float((wq @ x) @ (wk @ y) / math.sqrt(m))
        baseline[index] = float((baseline_q @ x) @ (baseline_k @ y) / math.sqrt(m))
    test = ks_2samp(custom, baseline)
    variance_ratio = float(custom.var(ddof=1) / baseline.var(ddof=1))
    return {
        "draws": draws,
        "ks_statistic": float(test.statistic),
        "ks_p_value": float(test.pvalue),
        "variance_ratio": variance_ratio,
        "passed": test.pvalue > 0.01 and 0.9 <= variance_ratio <= 1.1,
    }


def analyze(output: Path) -> None:
    rows = []
    for path in sorted(output.glob("moments_shard_*.jsonl")):
        rows.extend(read_jsonl(path))
    if not rows:
        raise FileNotFoundError("no moment shards found")
    p_values = [value for row in rows for value in (row["mean_p"], row["variance_p"])]
    rejected = holm_rejections(p_values, alpha=0.01)
    for row, mean_rejected, variance_rejected in zip(
        rows, rejected[::2], rejected[1::2], strict=True
    ):
        row["holm_mean_rejected"] = mean_rejected
        row["holm_variance_rejected"] = variance_rejected
        row["requirements_passed"] = (
            not mean_rejected
            and not variance_rejected
            and row["precision_converged"]
            and row["maximum_relative_orthogonality"] < 1e-6
            and abs(row["q_energy"] / row["energy_target"] - 1) <= 0.01
            and abs(row["k_energy"] / row["energy_target"] - 1) <= 0.01
            and abs(row["b_energy"] / row["m"] - 1) <= 0.01
            and abs(row["c_energy"] / row["m"] - 1) <= 0.01
            and abs(row["entry_variance_q"] / row["entry_variance_target"] - 1) <= 0.02
            and abs(row["entry_variance_k"] / row["entry_variance_target"] - 1) <= 0.02
            and abs(row["entry_correlation"] - row["alpha"]) <= 0.02
        )
    write_json(output / "moment_results.json", rows)
    checks = json.loads((output / "checks.json").read_text(encoding="utf-8"))
    gate = all(row["requirements_passed"] for row in rows)
    gate = gate and all(row["passed"] for row in checks["gradients"])
    gate = gate and checks["integration"]["passed"]
    gate = gate and checks["common_bias"]["passed"]
    gate = gate and checks["baseline_initializer"]["passed"]
    write_json(
        output / "gate_summary.json",
        {
            "passed": gate,
            "moment_configurations": len(rows),
            "moment_failures": sum(not row["requirements_passed"] for row in rows),
            "gradient_failures": sum(not row["passed"] for row in checks["gradients"]),
            "note": "Downstream confirmatory claims are blocked unless passed is true.",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["quick", "full"], default="quick")
    parser.add_argument("--mode", choices=["run", "analyze", "all"], default="all")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--draws", type=int)
    parser.add_argument("--max-draws", type=int, default=200_000)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--rebalance",
        action="store_true",
        help="Split remaining configs across the current shard workers instead of the original slice.",
    )
    args = parser.parse_args()
    output = ensure_output(args.output or Path(__file__).parent / "data" / args.preset)
    create_manifest(output, experiment="exp1a", arguments=vars(args))
    configurations = configuration_list(args.preset)
    completed_ids = jsonl_unique_values(output, "moments_shard_*.jsonl", "config_id")
    remaining = [config for config in configurations if config["config_id"] not in completed_ids]
    if args.rebalance:
        selected = remaining[args.shard_index :: args.num_shards]
    else:
        selected = [
            config
            for config in configurations[args.shard_index :: args.num_shards]
            if config["config_id"] not in completed_ids
        ]
    if args.mode in {"run", "all"}:
        minimum_draws = args.draws or (200 if args.preset == "quick" else 20_000)
        if args.preset == "full":
            minimum_draws = max(20_000, minimum_draws)
            if args.max_draws < minimum_draws:
                raise ValueError("--max-draws must be at least 20000 in full mode")
        shard_path = output / f"moments_shard_{args.shard_index:04d}.jsonl"
        print(
            f"resume: {len(completed_ids)}/{len(configurations)} configs done, "
            f"{len(remaining)} remaining globally, this worker {len(selected)}"
        )
        write_json(
            output / "progress.json",
            {
                "completed_configs": len(completed_ids),
                "total_configs": len(configurations),
                "remaining_configs": len(remaining),
                "this_worker_assigned": len(selected),
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "rebalance": args.rebalance,
            },
        )
        for index, config in enumerate(selected):
            started = time.perf_counter()
            draws = minimum_draws
            while True:
                result = run_moment_configuration(
                    config, draws, args.seed + configurations.index(config) * 10_007
                )
                if (
                    args.preset != "full"
                    or result["precision_converged"]
                    or draws >= args.max_draws
                ):
                    break
                draws = min(args.max_draws, max(draws + 1_000, math.ceil(draws * 1.5)))
            result["elapsed_seconds"] = time.perf_counter() - started
            append_jsonl(shard_path, [result])
            print(f"[{index + 1}/{len(selected)}] {config['config_id']}")
        checks_path = output / "checks.json"
        if checks_path.exists():
            print(f"keeping existing {checks_path}")
        else:
            checks = {
                "gradients": gradient_checks(
                    3 if args.preset == "quick" else 20, args.seed + 8_000_000
                ),
                "common_bias": common_bias_check(
                    max(200, minimum_draws), args.seed + 9_000_000
                ),
                "baseline_initializer": baseline_initializer_check(
                    max(500, minimum_draws), args.seed + 9_500_000
                ),
                "integration": integration_checkpoint_check(args.seed + 10_000_000),
            }
            write_json(checks_path, checks)
    finished_ids = jsonl_unique_values(output, "moments_shard_*.jsonl", "config_id")
    complete = len(finished_ids) >= len(configurations) and (output / "checks.json").exists()
    if args.mode in {"analyze", "all"}:
        if args.mode == "all" and not complete:
            print(
                f"skipping analyze: {len(finished_ids)}/{len(configurations)} configs "
                f"on disk; rerun with --mode analyze after every shard finishes"
            )
        else:
            analyze(output)
            print(f"Experiment 1A results written to {output}")


if __name__ == "__main__":
    main()
