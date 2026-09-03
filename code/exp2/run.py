"""Experiment 2: controlled real-language-model treatment comparison."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm, shapiro, ttest_1samp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.data import (  # noqa: E402
    build_tinystories_cache,
    language_model_batch,
    load_token_cache,
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
from common.metrics import cluster_bootstrap_ci, effective_rank, holm_rejections  # noqa: E402
from common.models import (  # noqa: E402
    DecoderLM,
    ModelConfig,
    parameter_count,
    query_key_state,
)


PRIMARY_CONTRASTS = [
    ("diverse_portfolio", "architecture_baseline"),
    ("diverse_portfolio", "matched_layer_homogeneous"),
]


def all_training_rows(directory: Path) -> list[dict]:
    rows: list[dict] = []
    for path in directory.glob("training_results_shard_*.jsonl"):
        rows.extend(read_jsonl(path))
    return rows


def freeze_treatments(
    *,
    portfolio_path: Path,
    trainability_dir: Path,
    output: Path,
    quick: bool,
) -> dict:
    portfolio_payload = json.loads(portfolio_path.read_text(encoding="utf-8"))
    training_rows = all_training_rows(trainability_dir)
    screened = [
        row
        for row in training_rows
        if row["stratum"] == "screened_candidate" and not row["diverged"]
    ]
    if not screened and not quick:
        raise RuntimeError("no successful screened candidate from Experiment 1D")
    if screened:
        by_point: dict[int, list[dict]] = defaultdict(list)
        for row in screened:
            by_point[int(row["point_id"])].append(row)
        best_rows = max(
            by_point.values(),
            key=lambda values: np.mean([row["loss_decrease_per_token"] for row in values]),
        )
        best = best_rows[0]
        best_point = {
            key: float(best[key]) for key in ("s", "alpha", "beta", "gamma", "phi")
        }
    else:
        best_point = {
            "s": 1.0,
            "alpha": 0.35,
            "beta": math.sqrt(1 - 0.35**2) / math.sqrt(2),
            "gamma": math.sqrt(1 - 0.35**2) / math.sqrt(2),
            "phi": math.pi / 4,
        }
    portfolio_points = portfolio_payload.get("points", [])
    if not portfolio_points:
        if not quick:
            raise RuntimeError("Experiment 1C did not freeze a candidate portfolio")
        portfolio_points = [
            {
                "s": 1.0,
                "alpha": alpha,
                "beta": math.sqrt(max(0.0, 1 - alpha**2)) * math.cos(phi),
                "gamma": math.sqrt(max(0.0, 1 - alpha**2)) * math.sin(phi),
                "phi": phi,
            }
            for alpha, phi in [(-0.5, 0.2), (-0.2, 1.2), (0.2, 0.9), (0.5, 0.4)]
        ]
    s_star = float(best_point["s"])
    # Treatments 2-5 use one common scale; 4 and 5 therefore have an exactly
    # matched model-wide per-head scale and coefficient histogram.
    normalized_portfolio = [
        {**point, "s": s_star} for point in portfolio_points
    ]
    orthogonal_rows = [
        row
        for row in training_rows
        if row["stratum"] == "corner_orthogonal" and not row["diverged"]
    ]
    orthogonal_passed = bool(orthogonal_rows) and all(
        row["loss_decrease_per_token"] > 0 for row in orthogonal_rows
    )
    payload = {
        "frozen_before_confirmatory_training": True,
        "s0": 1.0,
        "s_star": s_star,
        "best_structured_point": {**best_point, "s": s_star},
        "portfolio": normalized_portfolio,
        "orthogonal_endpoint_included": orthogonal_passed,
        "treatments": [
            "architecture_baseline",
            "scale_only",
            "homogeneous_structured",
            "diverse_portfolio",
            "matched_layer_homogeneous",
        ]
        + (["orthogonal_endpoint"] if orthogonal_passed else []),
    }
    write_json(output / "frozen_treatments.json", payload)
    return payload


def coefficient(point: dict) -> Coefficients:
    return Coefficients(
        float(point["s"]),
        float(point["alpha"]),
        float(point["beta"]),
        float(point["gamma"]),
        float(point["phi"]) if point.get("phi") is not None else None,
    )


def treatment_coefficients(
    name: str,
    frozen: dict,
    config: ModelConfig,
    *,
    seed: int,
) -> tuple[list[list[Coefficients]] | None, list[dict]]:
    if name == "architecture_baseline":
        return None, []
    independent = canonical_coefficients(float(frozen["s_star"]))["independent"]
    structured = coefficient(frozen["best_structured_point"])
    portfolio = [coefficient(point) for point in frozen["portfolio"]]
    records = []
    layers: list[list[Coefficients]] = []
    rng = np.random.default_rng(seed)
    matched_layer_order = rng.permutation(len(portfolio))
    for layer in range(config.layers):
        if name == "scale_only":
            assigned = [independent] * config.heads
        elif name == "homogeneous_structured":
            assigned = [structured] * config.heads
        elif name == "orthogonal_endpoint":
            assigned = [canonical_coefficients(float(frozen["s_star"]))["orthogonal"]] * config.heads
        elif name == "diverse_portfolio":
            repeated = [portfolio[index % len(portfolio)] for index in range(config.heads)]
            permutation = rng.permutation(config.heads)
            assigned = [repeated[int(index)] for index in permutation]
        elif name == "matched_layer_homogeneous":
            selected = portfolio[int(matched_layer_order[layer % len(portfolio)])]
            assigned = [selected] * config.heads
        else:
            raise ValueError(f"unknown treatment {name}")
        layers.append(assigned)
        records.extend(
            {"layer": layer, "head": head, **item.to_dict()}
            for head, item in enumerate(assigned)
        )
    return layers, records


def build_model(
    treatment: str,
    frozen: dict,
    config: ModelConfig,
    paired_seed: int,
    device: torch.device,
) -> tuple[DecoderLM, list[dict]]:
    torch.manual_seed(paired_seed)
    model = DecoderLM(config)
    layers, records = treatment_coefficients(
        treatment, frozen, config, seed=paired_seed + 700_000
    )
    if layers is not None:
        model.apply_query_key_geometry(layers, seed=paired_seed + 900_000)
    return model.to(device), records


def quick_token_cache(path: Path, split: str) -> Path:
    if not path.exists():
        text = (
            "Once upon a time a small fox learned to share. "
            "The fox and the bird became friends.\n"
        )
        payload = (text * (300 if split == "train" else 80)).encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return path


def model_config(quick: bool, variant: str = "main") -> ModelConfig:
    if quick:
        return ModelConfig(
            vocab_size=256,
            context_length=32,
            width=64,
            layers=2,
            heads=4,
            ff_multiplier=2,
        )
    if variant == "second_width":
        return ModelConfig(
            vocab_size=256,
            context_length=512,
            width=512,
            layers=8,
            heads=16,
            ff_multiplier=4,
        )
    if variant == "second_context":
        return ModelConfig(
            vocab_size=256,
            context_length=1024,
            width=576,
            layers=8,
            heads=8,
            ff_multiplier=4,
        )
    if variant == "deeper":
        return ModelConfig(
            vocab_size=256,
            context_length=512,
            width=512,
            layers=12,
            heads=8,
            ff_multiplier=4,
        )
    config = ModelConfig(
        vocab_size=256,
        context_length=512,
        width=576,
        layers=8,
        heads=8,
        ff_multiplier=4,
    )
    return config


def winning_treatment(output: Path) -> str:
    confirm = [
        row
        for row in all_training_rows(output)
        if row["study"] == "confirm" and not row["diverged"]
    ]
    by_treatment: dict[str, list[float]] = defaultdict(list)
    for row in confirm:
        if row["treatment"] != "architecture_baseline":
            by_treatment[row["treatment"]].append(row["validation_cross_entropy"])
    if not by_treatment:
        raise RuntimeError("complete the confirmatory study before transfer tests")
    return min(by_treatment, key=lambda name: np.mean(by_treatment[name]))


def attention_diagnostics(model: DecoderLM, maps: list[torch.Tensor]) -> dict:
    entropy_values = []
    maximum_values = []
    self_values = []
    rank_values = []
    js_values = []
    output_diversity = []
    for layer, maps_batch in enumerate(maps):
        probabilities = maps_batch[0].double()
        heads, length, _ = probabilities.shape
        active = torch.ones((length, length), dtype=torch.bool, device=probabilities.device).tril()
        for head in range(heads):
            p = probabilities[head]
            rows = []
            row_maxima = []
            for index in range(1, length):
                active_p = p[index, : index + 1]
                rows.append(-(active_p * active_p.clamp_min(1e-30).log()).sum() / math.log(index + 1))
                row_maxima.append(active_p.max())
            entropy_values.append(float(torch.stack(rows).median()))
            maximum_values.append(float(torch.stack(row_maxima).median()))
            self_values.append(float(torch.diagonal(p)[1:].median()))
            rank_values.append(effective_rank(p[: min(length, 128), : min(length, 128)]) / min(length, 128))
        for first in range(heads):
            for second in range(first + 1, heads):
                p, q = probabilities[first][active], probabilities[second][active]
                mixture = 0.5 * (p + q)
                js = 0.5 * (
                    (p * (p.clamp_min(1e-30) / mixture.clamp_min(1e-30)).log()).mean()
                    + (q * (q.clamp_min(1e-30) / mixture.clamp_min(1e-30)).log()).mean()
                )
                js_values.append(float(js))
        head_output = model.blocks[layer].attention.last_head_output
        if head_output is not None:
            flattened = torch.nn.functional.normalize(head_output[0].flatten(1), dim=1)
            similarities = flattened @ flattened.T
            offdiag = ~torch.eye(heads, dtype=torch.bool, device=similarities.device)
            output_diversity.append(float(1 - similarities[offdiag].mean()))
    return {
        "normalized_entropy": float(np.median(entropy_values)),
        "maximum_mass": float(np.median(maximum_values)),
        "self_mass": float(np.median(self_values)),
        "attention_effective_rank": float(np.median(rank_values)),
        "attention_map_js_divergence": float(np.mean(js_values)),
        "head_output_diversity": float(np.mean(output_diversity)),
    }


def centered_kernel_alignment(model: DecoderLM) -> float:
    similarities = []
    d, m = model.config.width, model.config.head_width
    for block in model.blocks:
        q = block.attention.q_proj.weight.view(model.config.heads, m, d)
        k = block.attention.k_proj.weight.view(model.config.heads, m, d)
        kernels = torch.stack([q[head].T @ k[head] for head in range(model.config.heads)])
        kernels = kernels - kernels.mean(dim=0, keepdim=True)
        flat = torch.nn.functional.normalize(kernels.flatten(1), dim=1)
        similarity = flat @ flat.T
        offdiag = ~torch.eye(model.config.heads, dtype=torch.bool, device=similarity.device)
        similarities.append(float(similarity[offdiag].mean()))
    return float(np.mean(similarities))


def geometry_persistence(model: DecoderLM, initial: dict[str, torch.Tensor]) -> float:
    current = query_key_state(model)
    similarities = []
    for key, initial_value in initial.items():
        now = current[key]
        similarities.append(
            float(
                torch.nn.functional.cosine_similarity(
                    initial_value.flatten().double(), now.flatten().double(), dim=0
                )
            )
        )
    return float(np.mean(similarities))


@torch.no_grad()
def evaluate(
    model: DecoderLM,
    validation_tokens: np.ndarray,
    *,
    seed: int,
    batches: int,
    batch_size: int,
    deep: bool,
) -> dict:
    model.eval()
    losses = []
    first_batch = None
    first_output = None
    for index in range(batches):
        inputs, targets = language_model_batch(
            validation_tokens,
            batch_size=batch_size,
            context_length=model.config.context_length,
            seed=seed,
            step=index,
            device=next(model.parameters()).device,
        )
        output = model(inputs, targets, return_attention=index == 0)
        losses.append(float(output["loss"]))
        if index == 0:
            first_batch = (inputs, targets)
            first_output = output
    result = {"validation_cross_entropy": float(np.mean(losses))}
    if first_output is not None:
        result.update(attention_diagnostics(model, first_output["attention"]))
        result["centered_kernel_alignment"] = centered_kernel_alignment(model)
    if deep and first_batch is not None:
        inputs, targets = first_batch
        baseline = float(model(inputs, targets)["loss"])
        ablations = []
        for layer in range(model.config.layers):
            for head in range(model.config.heads):
                ablated = float(model(inputs, targets, ablate=(layer, head))["loss"])
                ablations.append(ablated - baseline)
        result["mean_head_ablation_loss"] = float(np.mean(ablations))
        result["max_head_ablation_loss"] = float(np.max(ablations))
    model.train()
    return result


@torch.no_grad()
def integration_check(frozen: dict, output: Path, quick: bool, device: torch.device) -> None:
    config = model_config(True if quick else False)
    # Use a smaller shape for this exact check even in full mode.
    if not quick:
        config = ModelConfig(width=64, heads=4, layers=2, context_length=16)
    model, records = build_model("diverse_portfolio", frozen, config, 1234, device)
    tokens = torch.randint(0, 256, (2, min(8, config.context_length)), device=device)
    positions = torch.arange(tokens.shape[1], device=device)
    hidden = model.token_embedding(tokens) + model.position_embedding(positions)
    normalized = model.blocks[0].norm1(hidden)
    module = model.blocks[0].attention
    q_framework = module.q_proj(normalized).view(
        2, tokens.shape[1], config.heads, config.head_width
    ).transpose(1, 2)
    q_manual = torch.einsum(
        "btd,hmd->bhtm",
        normalized,
        module.q_proj.weight.view(config.heads, config.head_width, config.width),
    )
    k_manual = torch.einsum(
        "btd,hmd->bhtm",
        normalized,
        module.k_proj.weight.view(config.heads, config.head_width, config.width),
    )
    logits_manual = q_manual @ k_manual.transpose(-2, -1) / math.sqrt(config.head_width)
    _, maps = module(normalized, return_attention=True)
    one_hot = torch.eye(config.width, device=device)
    one_hot_q = module.q_proj(one_hot).view(
        config.width, config.heads, config.head_width
    )
    expected_one_hot_q = module.q_proj.weight.T.view(
        config.width, config.heads, config.head_width
    )
    one_hot_layout_error = float((one_hot_q - expected_one_hot_q).abs().max())
    mask = torch.ones(tokens.shape[1], tokens.shape[1], dtype=torch.bool, device=device).tril()
    maps_manual = torch.softmax(logits_manual.masked_fill(~mask, float("-inf")), dim=-1)
    initial = query_key_state(model)
    checkpoint = output / "integration_checkpoint.pt"
    torch.save(model.state_dict(), checkpoint)
    reloaded, _ = build_model("diverse_portfolio", frozen, config, 9999, device)
    reloaded.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    reloaded_state = query_key_state(reloaded)
    result = {
        "manual_query_max_error": float((q_framework - q_manual).abs().max()),
        "manual_attention_max_error": float((maps - maps_manual).abs().max()),
        "one_hot_head_layout_max_error": one_hot_layout_error,
        "checkpoint_bitwise_equal": all(
            torch.equal(initial[key], reloaded_state[key]) for key in initial
        ),
        "head_assignments": records,
    }
    result["passed"] = (
        result["manual_query_max_error"] < 1e-6
        and result["manual_attention_max_error"] < 1e-6
        and result["one_hot_head_layout_max_error"] < 1e-6
        and result["checkpoint_bitwise_equal"]
    )
    write_json(output / "integration_check.json", result)


def train_job(
    *,
    treatment: str,
    study: str,
    paired_seed: int,
    frozen: dict,
    config: ModelConfig,
    train_tokens: np.ndarray,
    validation_tokens: np.ndarray,
    token_budget: int,
    batch_size: int,
    learning_rate: float,
    eval_interval: int,
    validation_batches: int,
    target_validation_loss: float,
    device: torch.device,
    output: Path,
) -> dict:
    model, assignments = build_model(treatment, frozen, config, paired_seed, device)
    initial_qk = query_key_state(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    tokens_per_step = batch_size * config.context_length
    steps = math.ceil(token_budget / tokens_per_step)
    trace_path = output / f"trace_{study}_{treatment}_seed{paired_seed}.jsonl"
    checkpoint = output / f"checkpoint_{study}_{treatment}_seed{paired_seed}.pt"
    run_signature = {
        "study": study,
        "treatment": treatment,
        "paired_seed": paired_seed,
        "model": config.to_dict(),
        "token_budget": token_budget,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "target_validation_loss": target_validation_loss,
    }
    start_step = tokens_seen = clipping = 0
    prior_trace = read_jsonl(trace_path)
    eval_tokens = [int(row["tokens_seen"]) for row in prior_trace]
    eval_training_losses = [float(row["training_loss"]) for row in prior_trace]
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if state.get("run_signature") != run_signature:
            raise RuntimeError(
                f"checkpoint configuration mismatch at {checkpoint}; use a new output directory"
            )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        tokens_seen = int(state["tokens_seen"])
        clipping = int(state["clipping"])
        initial_qk = state["initial_qk"]
        assignments = state["assignments"]
    started = time.perf_counter()
    last_eval = {}
    last_loss = math.nan
    step = start_step - 1
    if start_step >= steps:
        last_eval = evaluate(
            model,
            validation_tokens,
            seed=paired_seed + 80_000_000,
            batches=validation_batches,
            batch_size=batch_size,
            deep=True,
        )
        last_eval["geometry_persistence_cosine"] = geometry_persistence(model, initial_qk)
    for step in range(start_step, steps):
        inputs, targets = language_model_batch(
            train_tokens,
            batch_size=batch_size,
            context_length=config.context_length,
            seed=paired_seed,
            step=step,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = model(inputs, targets)["loss"]
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        clipping += int(float(total_norm) > 1.0)
        optimizer.step()
        last_loss = float(loss.detach())
        tokens_seen += targets.numel()
        if (step + 1) % eval_interval == 0 or step + 1 == steps:
            last_eval = evaluate(
                model,
                validation_tokens,
                seed=paired_seed + 80_000_000,
                batches=validation_batches,
                batch_size=batch_size,
                deep=step + 1 == steps,
            )
            last_eval["geometry_persistence_cosine"] = geometry_persistence(model, initial_qk)
            append_jsonl(
                trace_path,
                [
                    {
                        "study": study,
                        "treatment": treatment,
                        "paired_seed": paired_seed,
                        "step": step + 1,
                        "tokens_seen": tokens_seen,
                        "training_loss": float(loss.detach()),
                        "gradient_norm": float(total_norm),
                        "elapsed_seconds": time.perf_counter() - started,
                        **last_eval,
                    }
                ],
            )
            eval_tokens.append(tokens_seen)
            eval_training_losses.append(float(loss.detach()))
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step + 1,
                    "tokens_seen": tokens_seen,
                    "clipping": clipping,
                    "initial_qk": initial_qk,
                    "assignments": assignments,
                    "run_signature": run_signature,
                },
                checkpoint,
            )
        if not math.isfinite(last_loss):
            break
    early_count = max(2, math.ceil(len(eval_tokens) / 4))
    early_slope = (
        float(
            np.polyfit(
                np.asarray(eval_tokens[:early_count], dtype=float),
                np.asarray(eval_training_losses[:early_count], dtype=float),
                1,
            )[0]
        )
        if len(eval_tokens) >= 2
        else math.nan
    )
    final_trace = read_jsonl(trace_path)
    reached = [
        row
        for row in final_trace
        if row["validation_cross_entropy"] <= target_validation_loss
    ]
    return {
        "study": study,
        "treatment": treatment,
        "paired_seed": paired_seed,
        "tokens_seen": tokens_seen,
        "steps_completed": step + 1 if steps else 0,
        "diverged": start_step < steps and not math.isfinite(last_loss),
        "gradient_clipping_frequency": clipping / max(1, step + 1),
        "early_training_loss_slope_per_token": early_slope,
        "target_validation_loss": target_validation_loss,
        "tokens_to_target_validation_loss": (
            int(reached[0]["tokens_seen"]) if reached else None
        ),
        "seconds_to_target_validation_loss": (
            float(reached[0].get("elapsed_seconds", math.nan)) if reached else None
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "parameter_count": parameter_count(model),
        "head_assignments": assignments,
        "run_signature": run_signature,
        **last_eval,
    }


def sign_permutation_paired(values: np.ndarray, seed: int, draws: int = 100_000) -> float:
    rng = np.random.default_rng(seed)
    observed = abs(values.mean())
    signs = rng.choice([-1.0, 1.0], size=(draws, len(values)))
    null = abs((signs * values).mean(axis=1))
    return float((np.sum(null >= observed) + 1) / (draws + 1))


def analyze_study(output: Path, study: str, smallest_effect: float, max_seeds: int) -> dict:
    rows = [
        row
        for row in all_training_rows(output)
        if row["study"] == study and not row["diverged"]
    ]
    by_treatment_seed = {
        (row["treatment"], int(row["paired_seed"])): row for row in rows
    }
    contrasts = []
    for first, second in PRIMARY_CONTRASTS:
        seeds = sorted(
            {
                seed
                for treatment, seed in by_treatment_seed
                if treatment == first and (second, seed) in by_treatment_seed
            }
        )
        differences = np.asarray(
            [
                by_treatment_seed[(first, seed)]["validation_cross_entropy"]
                - by_treatment_seed[(second, seed)]["validation_cross_entropy"]
                for seed in seeds
            ],
            dtype=float,
        )
        if not len(differences):
            continue
        estimate, lower, upper = cluster_bootstrap_ci(
            differences, statistic=np.mean, draws=20_000, seed=20260829
        )
        normal_enough = len(differences) < 3 or shapiro(differences).pvalue >= 0.05
        p_value = (
            float(ttest_1samp(differences, 0).pvalue)
            if normal_enough and len(differences) >= 2
            else sign_permutation_paired(differences, 20260829)
        )
        sd = float(differences.std(ddof=1)) if len(differences) >= 2 else math.nan
        required = (
            max(8, math.ceil(((norm.ppf(0.975) + norm.ppf(0.8)) * sd / smallest_effect) ** 2))
            if math.isfinite(sd)
            else None
        )
        contrasts.append(
            {
                "contrast": f"{first}_minus_{second}",
                "paired_seeds": len(differences),
                "mean_difference_nats": estimate,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "standardized_effect": estimate / sd if sd and math.isfinite(sd) else math.nan,
                "p_value": p_value,
                "test": (
                    "paired_t"
                    if normal_enough and len(differences) >= 2
                    else "paired_sign_permutation"
                ),
                "equivalent_within_0.01_nats": lower >= -0.01 and upper <= 0.01,
                "required_confirmatory_seeds": required,
            }
        )
    rejected = holm_rejections([row["p_value"] for row in contrasts], alpha=0.05)
    for row, reject in zip(contrasts, rejected, strict=True):
        row["holm_rejected"] = reject
    required_counts = [
        row["required_confirmatory_seeds"]
        for row in contrasts
        if row["required_confirmatory_seeds"] is not None
    ]
    required = max([8, *required_counts])
    pilot_sufficient = study != "pilot" or (
        len(contrasts) == len(PRIMARY_CONTRASTS)
        and all(row["paired_seeds"] >= 3 for row in contrasts)
        and all(row["required_confirmatory_seeds"] is not None for row in contrasts)
    )
    return {
        "study": study,
        "contrasts": contrasts,
        "smallest_practical_effect": smallest_effect,
        "confirmatory_seeds": required,
        "max_available_seeds": max_seeds,
        "pilot_sufficient_for_power": pilot_sufficient,
        "confirmatory_feasible": pilot_sufficient and required <= max_seeds,
    }


def analyze_transfer(output: Path, study_id: str, winner: str) -> dict:
    rows = [
        row
        for row in all_training_rows(output)
        if row["study"] == study_id and not row["diverged"]
    ]
    lookup = {(row["treatment"], int(row["paired_seed"])): row for row in rows}
    seeds = sorted(
        seed
        for treatment, seed in lookup
        if treatment == winner and ("architecture_baseline", seed) in lookup
    )
    differences = [
        lookup[(winner, seed)]["validation_cross_entropy"]
        - lookup[("architecture_baseline", seed)]["validation_cross_entropy"]
        for seed in seeds
    ]
    estimate, lower, upper = cluster_bootstrap_ci(
        differences, statistic=np.mean, draws=20_000, seed=20260829
    )
    return {
        "study": study_id,
        "winner_frozen_from_main_confirmation": winner,
        "paired_seeds": len(seeds),
        "winner_minus_baseline_nats": estimate,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "equivalent_within_0.01_nats": lower >= -0.01 and upper <= 0.01,
    }


def plot_analysis(analysis: dict, output: Path, study_id: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "contrasts" in analysis:
        rows = analysis["contrasts"]
        labels = [row["contrast"] for row in rows]
        means = [row["mean_difference_nats"] for row in rows]
        lowers = [row["ci95_lower"] for row in rows]
        uppers = [row["ci95_upper"] for row in rows]
    else:
        rows = [analysis]
        labels = [analysis["winner_frozen_from_main_confirmation"] + " minus baseline"]
        means = [analysis["winner_minus_baseline_nats"]]
        lowers = [analysis["ci95_lower"]]
        uppers = [analysis["ci95_upper"]]
    if not rows:
        return
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    positions = np.arange(len(rows))
    fig, axis = plt.subplots(figsize=(8, max(3, 1.2 * len(rows))))
    axis.errorbar(
        means,
        positions,
        xerr=[np.asarray(means) - np.asarray(lowers), np.asarray(uppers) - np.asarray(means)],
        fmt="o",
        capsize=4,
    )
    axis.axvline(0, color="black", linewidth=1)
    axis.axvspan(-0.01, 0.01, color="gray", alpha=0.15, label="+/-0.01 nat equivalence")
    axis.set(yticks=positions, yticklabels=labels, xlabel="paired validation cross-entropy difference (nats)")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / f"{study_id}_primary_contrasts.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["quick", "full"], default="quick")
    parser.add_argument(
        "--mode",
        choices=["freeze", "integration", "run", "analyze", "all"],
        default="all",
    )
    parser.add_argument("--study", choices=["pilot", "confirm", "transfer"], default="pilot")
    parser.add_argument(
        "--variant",
        choices=["main", "second_width", "second_context", "deeper", "second_corpus"],
        default="main",
    )
    parser.add_argument(
        "--portfolio",
        type=Path,
    )
    parser.add_argument(
        "--trainability-dir",
        type=Path,
    )
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--validation-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--token-budget", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--validation-batches", type=int)
    parser.add_argument("--target-validation-loss", type=float, default=2.0)
    parser.add_argument("--smallest-effect", type=float, default=0.01)
    parser.add_argument("--max-seeds", type=int, default=32)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.portfolio is None:
        args.portfolio = (
            Path(__file__).parents[1]
            / "exp1c"
            / "data"
            / args.preset
            / "candidate_portfolio.json"
        )
    if args.trainability_dir is None:
        args.trainability_dir = (
            Path(__file__).parents[1] / "exp1d" / "data" / args.preset
        )
    output = ensure_output(args.output or Path(__file__).parent / "data" / args.preset)
    create_manifest(
        output,
        experiment=f"exp2_{args.study}",
        arguments=vars(args),
        inputs=[args.portfolio, args.train_file, args.validation_file],
    )
    quick = args.preset == "quick"
    if args.study == "transfer" and args.variant == "main":
        raise ValueError("transfer study requires a non-main --variant")
    if args.study != "transfer" and args.variant != "main":
        raise ValueError("non-main variants belong to --study transfer")
    study_id = args.study if args.study != "transfer" else f"transfer_{args.variant}"
    frozen_path = output / "frozen_treatments.json"
    if args.mode in {"freeze", "all"}:
        frozen = freeze_treatments(
            portfolio_path=args.portfolio,
            trainability_dir=args.trainability_dir,
            output=output,
            quick=quick,
        )
    else:
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if args.mode in {"integration", "all"}:
        integration_check(frozen, output, quick, device)
    if args.mode in {"run", "all"}:
        if quick:
            train_path = quick_token_cache(output / "quick_train.uint8", "train")
            validation_path = quick_token_cache(output / "quick_validation.uint8", "validation")
        else:
            train_path = build_tinystories_cache(
                output / "tinystories_train.uint8",
                split="train",
                local_path=args.train_file,
            )
            validation_path = build_tinystories_cache(
                output / "tinystories_validation.uint8",
                split="validation",
                local_path=args.validation_file,
            )
        train_tokens = load_token_cache(train_path)
        validation_tokens = load_token_cache(validation_path)
        write_json(
            output / "data_hashes.json",
            {
                "train_uint8_sha256": file_sha256(train_path),
                "validation_uint8_sha256": file_sha256(validation_path),
            },
        )
        config = model_config(quick, args.variant)
        if not quick and len(frozen["portfolio"]) != config.heads:
            raise RuntimeError(
                "exact Treatment 4-5 histogram matching requires one portfolio point per head/layer"
            )
        if (
            not quick
            and args.study in {"pilot", "confirm"}
            and not (30_000_000 <= parameter_count(DecoderLM(config)) <= 60_000_000)
        ):
            raise RuntimeError("confirmatory model is outside the preregistered 30M-60M range")
        if args.study == "confirm" and not quick:
            plan_path = output / "pilot_analysis.json"
            if not plan_path.exists():
                raise RuntimeError("analyze the external pilot before confirmation")
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            if not plan["confirmatory_feasible"]:
                raise RuntimeError("power calculation exceeds the available seed budget")
            default_seeds = int(plan["confirmatory_seeds"])
        elif args.study == "transfer" and not quick:
            default_seeds = 8
        else:
            default_seeds = 1 if quick else 3
        seeds = args.seeds or default_seeds
        if args.study == "confirm" and not quick and seeds < 8:
            raise RuntimeError("confirmatory study requires at least 8 paired seeds")
        token_budget = args.token_budget or (
            20_000 if quick else 2_000_000 if args.study == "pilot" else 100_000_000
        )
        batch_size = args.batch_size or (2 if quick else 16)
        eval_interval = args.eval_interval or (5 if quick else 250)
        validation_batches = args.validation_batches or (2 if quick else 50)
        active_treatments = frozen["treatments"]
        if args.study == "transfer":
            winner = winning_treatment(output)
            active_treatments = ["architecture_baseline", winner]
            write_json(
                output / f"{study_id}_plan.json",
                {
                    "winner_frozen_from_main_confirmation": winner,
                    "variant": args.variant,
                    "second_corpus": args.variant == "second_corpus",
                },
            )
        seed_offset = {
            "pilot": 0,
            "confirm": 10_000_000,
            "transfer": 20_000_000,
        }[args.study]
        jobs = [
            (treatment, args.seed + seed_offset + seed_index)
            for treatment in active_treatments
            for seed_index in range(seeds)
        ]
        completed_rows = all_training_rows(output)
        completed = {
            (row["study"], row["treatment"], int(row["paired_seed"]))
            for row in completed_rows
        }
        results_path = output / f"training_results_shard_{args.shard_index:04d}.jsonl"
        for position, (treatment, paired_seed) in enumerate(
            jobs[args.shard_index :: args.num_shards]
        ):
            key = (study_id, treatment, paired_seed)
            if key in completed:
                continue
            try:
                result = train_job(
                    treatment=treatment,
                    study=study_id,
                    paired_seed=paired_seed,
                    frozen=frozen,
                    config=config,
                    train_tokens=train_tokens,
                    validation_tokens=validation_tokens,
                    token_budget=token_budget,
                    batch_size=batch_size,
                    learning_rate=args.learning_rate,
                    eval_interval=eval_interval,
                    validation_batches=validation_batches,
                    target_validation_loss=args.target_validation_loss,
                    device=device,
                    output=output,
                )
                append_jsonl(results_path, [result])
            except (RuntimeError, ValueError) as error:
                record_failure(
                    output,
                    {"study": study_id, "treatment": treatment, "seed": paired_seed},
                    error,
                )
                raise
            print(f"[{position + 1}] {study_id} {treatment} seed={paired_seed}")
    if args.mode in {"analyze", "all"}:
        if args.study == "transfer":
            winner = winning_treatment(output)
            analysis = analyze_transfer(output, study_id, winner)
        else:
            analysis = analyze_study(
                output, args.study, args.smallest_effect, args.max_seeds
            )
        write_json(output / f"{study_id}_analysis.json", analysis)
        plot_analysis(analysis, output, study_id)
        print(f"Experiment 2 {study_id} results written to {output}")


if __name__ == "__main__":
    main()
