"""Attention, spectrum, bootstrap, and multiple-testing utilities."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import numpy as np
import torch
from torch import Tensor


def effective_rank(matrix: Tensor, eps: float = 1e-12) -> float:
    singular = torch.linalg.svdvals(matrix).double()
    singular = singular[singular > eps]
    if singular.numel() == 0:
        return 0.0
    probabilities = singular / singular.sum()
    return float(torch.exp(-(probabilities * probabilities.log()).sum()))


def stable_rank(matrix: Tensor, eps: float = 1e-12) -> float:
    singular = torch.linalg.svdvals(matrix).double()
    if singular.numel() == 0 or float(singular[0]) <= eps:
        return 0.0
    return float((singular.square().sum() / singular[0].square()))


def _active_rows(mask: Tensor) -> list[Tensor]:
    return [torch.nonzero(mask[i], as_tuple=False).flatten() for i in range(mask.shape[0])]


def attention_metrics(
    logits: Tensor,
    probabilities: Tensor,
    mask: Tensor,
    *,
    values: Tensor | None = None,
) -> dict[str, float]:
    """Compute proposal metrics for one n-by-n attention map."""
    logits = logits.detach()
    probabilities = probabilities.detach()
    values = values.detach() if values is not None else None
    n = logits.shape[-1]
    rows = _active_rows(mask)
    centered_scales: list[Tensor] = []
    entropies: list[Tensor] = []
    supports: list[Tensor] = []
    maxima: list[Tensor] = []
    self_mass: list[Tensor] = []
    jac_fro: list[Tensor] = []
    jac_op: list[Tensor] = []
    uniform_sq = logits.new_zeros(())

    for i, indices in enumerate(rows):
        active_logits = logits[i, indices]
        active_p = probabilities[i, indices]
        k = indices.numel()
        uniform_sq = uniform_sq + ((active_p - 1.0 / k) ** 2).sum()
        if k <= 1:
            continue
        centered_scales.append(active_logits.var(unbiased=False).sqrt())
        entropy = -(active_p * active_p.clamp_min(1e-30).log()).sum()
        entropies.append(entropy / math.log(k))
        supports.append(entropy.exp() / k)
        maxima.append(active_p.max())
        self_mass.append(probabilities[i, i])
        jacobian = torch.diag(active_p) - torch.outer(active_p, active_p)
        jac_fro.append(torch.linalg.matrix_norm(jacobian, ord="fro"))
        jac_op.append(torch.linalg.eigvalsh(jacobian.double())[-1].to(logits.dtype))

    diagonal = torch.diagonal(logits)
    offdiag_mask = mask & ~torch.eye(n, dtype=torch.bool, device=logits.device)
    offdiag = logits[offdiag_mask]
    result = {
        "logit_mean": float(logits[mask].mean()),
        "logit_variance": float(logits[mask].var(unbiased=True)),
        "tau": _median(centered_scales),
        "delta": float(diagonal.mean() - offdiag.mean()) if offdiag.numel() else math.nan,
        "normalized_entropy": _median(entropies),
        "effective_support": _median(supports),
        "max_mass": _median(maxima),
        "self_mass": _median(self_mass),
        "distance_uniform": float(torch.sqrt(uniform_sq / n)),
        "attention_effective_rank": effective_rank(probabilities) / n,
        "attention_stable_rank": stable_rank(probabilities) / n,
        "jacobian_fro": _median(jac_fro),
        "jacobian_op": _median(jac_op),
    }
    if values is not None:
        output = probabilities @ values
        normalized = torch.nn.functional.normalize(output, dim=-1)
        similarity = normalized @ normalized.T
        keep = ~torch.eye(n, dtype=torch.bool, device=logits.device)
        result["output_token_cosine"] = float(similarity[keep].mean())
    return result


def kernel_metrics(wq: Tensor, wk: Tensor) -> dict[str, float]:
    kernel = wq.T @ wk
    singular = torch.linalg.svdvals(kernel).double()
    nonzero = singular[singular > 1e-10]
    total = nonzero.sum().clamp_min(1e-30)
    probabilities = nonzero / total
    erank = float(torch.exp(-(probabilities * probabilities.log()).sum()))
    return {
        "kernel_operator_norm": float(singular[0]),
        "kernel_effective_rank": erank,
        "kernel_stable_rank": stable_rank(kernel),
        "kernel_spectral_concentration": float(singular[0] / total),
    }


def gram_features(x: Tensor) -> dict[str, float]:
    d = x.shape[-1]
    gram = x @ x.T / d
    energies = torch.diagonal(gram)
    vi_vj = torch.outer(energies, energies)
    eye = torch.eye(gram.shape[0], dtype=torch.bool, device=gram.device)
    return {
        "mean_vi_vj": float(vi_vj[~eye].mean()),
        "mean_cij2": float(gram[~eye].square().mean()),
        "mean_cij": float(gram[~eye].mean()),
    }


def cluster_bootstrap_ci(
    values: Iterable[float],
    *,
    statistic: Callable[[np.ndarray], float] = np.median,
    confidence: float = 0.95,
    draws: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    values_array = np.asarray(list(values), dtype=np.float64)
    values_array = values_array[np.isfinite(values_array)]
    if values_array.size == 0:
        return math.nan, math.nan, math.nan
    estimate = float(statistic(values_array))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values_array.size, size=(draws, values_array.size))
    samples = np.apply_along_axis(statistic, 1, values_array[indices])
    alpha = 1.0 - confidence
    lower, upper = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return estimate, float(lower), float(upper)


def classify_cell(
    intervals: dict[str, tuple[float, float, float]],
    *,
    jacobian_floor: float,
    update_floor: float,
    rank_floor: float,
    probability_shift: float = 0.0,
) -> str:
    """Mutually exclusive label using the proposal's precedence order."""
    entropy = intervals["normalized_entropy"]
    distance = intervals["distance_uniform"]
    maximum = intervals["max_mass"]
    self_mass = intervals["self_mass"]
    jacobian = intervals["jacobian_op"]
    update = intervals["update_to_weight_min"]
    rank = intervals["attention_effective_rank"]
    if self_mass[1] >= 0.80 + probability_shift:
        return "self_locked"
    if entropy[2] <= 0.50 + probability_shift or maximum[1] >= 0.80 + probability_shift:
        return "concentrated"
    if entropy[1] >= 0.95 + probability_shift and distance[2] <= 0.10:
        return "diffuse"
    if jacobian[2] < jacobian_floor or update[2] < update_floor:
        return "gradient_starved"
    if (
        entropy[1] >= 0.60 + probability_shift
        and entropy[2] <= 0.95 + probability_shift
        and rank[1] >= rank_floor
    ):
        return "screened_candidate"
    return "boundary_uncertain"


def holm_rejections(p_values: Iterable[float], alpha: float = 0.01) -> list[bool]:
    p = np.asarray(list(p_values), dtype=float)
    order = np.argsort(p)
    rejected = np.zeros(len(p), dtype=bool)
    for rank, index in enumerate(order):
        threshold = alpha / (len(p) - rank)
        if p[index] <= threshold:
            rejected[index] = True
        else:
            break
    return rejected.tolist()


def benjamini_hochberg(p_values: Iterable[float], alpha: float = 0.05) -> list[bool]:
    p = np.asarray(list(p_values), dtype=float)
    order = np.argsort(p)
    ordered = p[order]
    valid = ordered <= alpha * np.arange(1, len(p) + 1) / len(p)
    rejected = np.zeros(len(p), dtype=bool)
    if valid.any():
        rejected[order[: np.nonzero(valid)[0][-1] + 1]] = True
    return rejected.tolist()


def _median(values: list[Tensor]) -> float:
    if not values:
        return math.nan
    return float(torch.stack(values).median())
