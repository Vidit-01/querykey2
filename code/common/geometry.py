"""Shared query-key initialization and exact finite-width formulas."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class Coefficients:
    s: float
    alpha: float
    beta: float
    gamma: float
    phi: float | None = None

    def validate(self, atol: float = 1e-6) -> None:
        if self.s <= 0:
            raise ValueError("s must be positive")
        norm = self.alpha**2 + self.beta**2 + self.gamma**2
        if abs(norm - 1.0) > atol:
            raise ValueError(f"coefficients must lie on the unit sphere; got {norm}")

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


def coefficients_from_alpha_phi(s: float, alpha: float, phi: float) -> Coefficients:
    if not -1.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [-1, 1]")
    if not 0.0 <= phi <= math.pi / 2 + 1e-12:
        raise ValueError("phi must be in [0, pi/2]")
    radius = math.sqrt(max(0.0, 1.0 - alpha * alpha))
    result = Coefficients(
        s=float(s),
        alpha=float(alpha),
        beta=radius * math.cos(phi),
        gamma=radius * math.sin(phi),
        phi=float(phi),
    )
    result.validate()
    return result


def canonical_coefficients(s: float = 1.0) -> dict[str, Coefficients]:
    return {
        "independent": Coefficients(s, 0.0, 0.0, 1.0, math.pi / 2),
        "orthogonal": Coefficients(s, 0.0, 1.0, 0.0, 0.0),
        "tied": Coefficients(s, 1.0, 0.0, 0.0, 0.0),
        "anti_tied": Coefficients(s, -1.0, 0.0, 0.0, 0.0),
    }


def make_generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def draw_components(
    d: int,
    m: int,
    *,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[Tensor, Tensor, Tensor]:
    """Draw A, B, C without constructing a d-by-d projector."""
    if not 0 < m < d:
        raise ValueError(f"expected 0 < m < d, got m={m}, d={d}")
    a = torch.randn((m, d), generator=generator, device=device, dtype=dtype) / math.sqrt(d)
    c = torch.randn((m, d), generator=generator, device=device, dtype=dtype) / math.sqrt(d)
    g = torch.randn((m, d), generator=generator, device=device, dtype=dtype) / math.sqrt(d - m)
    # Q spans row(A). QR sign choices do not affect the projector.
    q, _ = torch.linalg.qr(a.T, mode="reduced")
    b = g - (g @ q) @ q.T
    return a, b, c


def draw_query_key(
    d: int,
    m: int,
    coefficients: Coefficients,
    *,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor, Tensor]]:
    coefficients.validate()
    a, b, c = draw_components(
        d, m, generator=generator, device=device, dtype=dtype
    )
    wq = coefficients.s * a
    wk = coefficients.s * (
        coefficients.alpha * a
        + coefficients.beta * b
        + coefficients.gamma * c
    )
    return wq, wk, (a, b, c)


def exact_g(d: int, vi: Tensor | float, vj: Tensor | float, cij: Tensor | float):
    return d * ((d + 1) * vi * vj - 2 * cij**2) / ((d - 1) * (d + 2))


def predicted_logit_moments(
    d: int,
    m: int,
    coefficients: Coefficients,
    vi: Tensor | float,
    vj: Tensor | float,
    cij: Tensor | float,
):
    c = coefficients
    mean = c.alpha * c.s**2 * math.sqrt(m) * cij
    variance = c.s**4 * (
        c.alpha**2 * (vi * vj + cij**2)
        + c.beta**2 * exact_g(d, vi, vj, cij)
        + c.gamma**2 * vi * vj
    )
    return mean, variance


def common_bias_ratio(d: int, m: int, alpha1: float, alpha2: float) -> float:
    f1 = 1.0 + alpha1 * alpha1 * (m + 1) / d
    f2 = 1.0 + alpha2 * alpha2 * (m + 1) / d
    return alpha1 * alpha2 * m / (d * math.sqrt(f1 * f2))


def initialize_head_slices(
    q_weight: Tensor,
    k_weight: Tensor,
    head_coefficients: Sequence[Coefficients],
    *,
    seed: int,
) -> list[dict[str, float | int | None]]:
    """Initialize [output, input] query/key tensors independently by head."""
    if q_weight.shape != k_weight.shape or q_weight.ndim != 2:
        raise ValueError("query and key weights must have equal [output, input] shapes")
    output_dim, d = q_weight.shape
    heads = len(head_coefficients)
    if output_dim % heads:
        raise ValueError("output dimension must be divisible by the number of heads")
    m = output_dim // heads
    if d <= m:
        raise ValueError("the orthogonal construction requires d > head width")
    generator = make_generator(seed, q_weight.device)
    records: list[dict[str, float | int | None]] = []
    with torch.no_grad():
        for head, coefficients in enumerate(head_coefficients):
            wq, wk, _ = draw_query_key(
                d,
                m,
                coefficients,
                generator=generator,
                device=q_weight.device,
                dtype=q_weight.dtype,
            )
            sl = slice(head * m, (head + 1) * m)
            q_weight[sl].copy_(wq)
            k_weight[sl].copy_(wk)
            records.append({"head": head, **coefficients.to_dict()})
    return records


def initialize_attention_module(
    module: nn.Module,
    head_coefficients: Sequence[Coefficients],
    *,
    seed: int,
) -> list[dict[str, float | int | None]]:
    """Initialize modules exposing q_proj/k_proj Linear layers."""
    if not hasattr(module, "q_proj") or not hasattr(module, "k_proj"):
        raise TypeError("module must expose q_proj and k_proj")
    q_proj = module.q_proj
    k_proj = module.k_proj
    if not isinstance(q_proj, nn.Linear) or not isinstance(k_proj, nn.Linear):
        raise TypeError("q_proj and k_proj must be torch.nn.Linear")
    return initialize_head_slices(
        q_proj.weight, k_proj.weight, head_coefficients, seed=seed
    )


def coefficient_grid(
    *,
    log2_s_values: Iterable[float],
    alpha_values: Iterable[float],
    phi_values: Iterable[float],
) -> list[Coefficients]:
    cells: list[Coefficients] = []
    phis = list(phi_values)
    for log2_s in log2_s_values:
        for alpha in alpha_values:
            active_phis = [phis[0]] if abs(abs(alpha) - 1.0) < 1e-12 else phis
            for phi in active_phis:
                cells.append(coefficients_from_alpha_phi(2.0**log2_s, alpha, phi))
    return cells
