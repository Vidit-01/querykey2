"""Automated numerical checks for the proposal's exact identities.

Run:
    python proposal/verify_math.py
    python proposal/verify_math.py --draws 20000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass
class Config:
    d: int = 32
    m: int = 8
    n: int = 7
    s: float = 1.2
    alpha: float = 0.35
    beta: float = 0.6
    gamma: float = 0.0

    def normalize_coefficients(self) -> None:
        self.gamma = float(
            np.sqrt(max(0.0, 1.0 - self.alpha**2 - self.beta**2))
        )


def draw_components(
    rng: np.random.Generator, d: int, m: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = rng.standard_normal((m, d)) / np.sqrt(d)
    c = rng.standard_normal((m, d)) / np.sqrt(d)
    g = rng.standard_normal((m, d)) / np.sqrt(d - m)
    u, _ = np.linalg.qr(a.T, mode="reduced")
    b = g - (g @ u) @ u.T
    return a, b, c


def exact_g(d: int, vi: float, vj: float, cij: float) -> float:
    return (
        d
        * ((d + 1) * vi * vj - 2 * cij**2)
        / ((d - 1) * (d + 2))
    )


def predicted_moments(
    cfg: Config, x: np.ndarray, y: np.ndarray
) -> tuple[float, float]:
    vi = float(x @ x / cfg.d)
    vj = float(y @ y / cfg.d)
    cij = float(x @ y / cfg.d)
    mean = cfg.alpha * cfg.s**2 * np.sqrt(cfg.m) * cij
    var = cfg.s**4 * (
        cfg.alpha**2 * (vi * vj + cij**2)
        + cfg.beta**2 * exact_g(cfg.d, vi, vj, cij)
        + cfg.gamma**2 * vi * vj
    )
    return mean, var


def check_orthogonality_and_energy(
    cfg: Config, rng: np.random.Generator, draws: int
) -> None:
    energies = np.empty((draws, 2))
    max_orth_error = 0.0
    entry_second_moment = 0.0
    for k in range(draws):
        a, b, c = draw_components(rng, cfg.d, cfg.m)
        wq = cfg.s * a
        wk = cfg.s * (
            cfg.alpha * a + cfg.beta * b + cfg.gamma * c
        )
        energies[k] = [np.sum(wq**2), np.sum(wk**2)]
        entry_second_moment += np.mean(wk**2)
        max_orth_error = max(max_orth_error, np.max(np.abs(a @ b.T)))

    entry_second_moment /= draws
    energy_target = cfg.s**2 * cfg.m
    entry_target = cfg.s**2 / cfg.d
    energy_se = energies.std(axis=0, ddof=1) / np.sqrt(draws)

    assert max_orth_error < 2e-12, max_orth_error
    for observed, se in zip(energies.mean(axis=0), energy_se):
        assert abs(observed - energy_target) <= 6 * se + 2e-3
    assert abs(entry_second_moment - entry_target) < 0.02 * entry_target

    print(
        "PASS energy/orthogonality:",
        f"max|AB^T|={max_orth_error:.2e},",
        f"E||Wq||^2={energies[:, 0].mean():.4f},",
        f"E||Wk||^2={energies[:, 1].mean():.4f}",
    )


def check_logit_moments(
    cfg: Config, rng: np.random.Generator, draws: int
) -> None:
    x = rng.standard_normal(cfg.d)
    y0 = rng.standard_normal(cfg.d)
    # Include a controlled nonzero token similarity.
    x *= np.sqrt(cfg.d / (x @ x))
    y0 -= x * (x @ y0) / (x @ x)
    y0 *= np.sqrt(cfg.d / (y0 @ y0))
    rho = 0.4
    y = rho * x + np.sqrt(1 - rho**2) * y0

    values = np.empty(draws)
    for k in range(draws):
        a, b, c = draw_components(rng, cfg.d, cfg.m)
        wq = cfg.s * a
        wk = cfg.s * (
            cfg.alpha * a + cfg.beta * b + cfg.gamma * c
        )
        values[k] = (wq @ x) @ (wk @ y) / np.sqrt(cfg.m)

    mean_th, var_th = predicted_moments(cfg, x, y)
    mean_hat = values.mean()
    var_hat = values.var(ddof=1)
    mean_se = np.sqrt(var_hat / draws)
    var_se = var_hat * np.sqrt(2 / (draws - 1))

    assert abs(mean_hat - mean_th) <= 6 * mean_se
    assert abs(var_hat - var_th) <= 6 * var_se
    print(
        "PASS logit moments:",
        f"mean empirical/theory={mean_hat:.4f}/{mean_th:.4f},",
        f"var empirical/theory={var_hat:.4f}/{var_th:.4f}",
    )


def loss_from_weights(
    wq: np.ndarray,
    wk: np.ndarray,
    x: np.ndarray,
    dloss: np.ndarray,
) -> float:
    logits = (wq @ x).T @ (wk @ x) / np.sqrt(wq.shape[0])
    return float(np.sum(logits * dloss))


def check_gradients(cfg: Config, rng: np.random.Generator) -> None:
    a, b, c = draw_components(rng, cfg.d, cfg.m)
    wq = cfg.s * a
    wk = cfg.s * (
        cfg.alpha * a + cfg.beta * b + cfg.gamma * c
    )
    x = rng.standard_normal((cfg.d, cfg.n))
    dloss = rng.standard_normal((cfg.n, cfg.n))
    q = wq @ x
    k = wk @ x
    grad_q = k @ dloss.T @ x.T / np.sqrt(cfg.m)
    grad_k = q @ dloss @ x.T / np.sqrt(cfg.m)

    eps = 1e-6
    for name, weight, other, analytic in (
        ("Wq", wq, wk, grad_q),
        ("Wk", wk, wq, grad_k),
    ):
        indices = [
            (int(rng.integers(cfg.m)), int(rng.integers(cfg.d)))
            for _ in range(12)
        ]
        for row, col in indices:
            plus = weight.copy()
            minus = weight.copy()
            plus[row, col] += eps
            minus[row, col] -= eps
            if name == "Wq":
                lp = loss_from_weights(plus, other, x, dloss)
                lm = loss_from_weights(minus, other, x, dloss)
            else:
                lp = loss_from_weights(other, plus, x, dloss)
                lm = loss_from_weights(other, minus, x, dloss)
            numeric = (lp - lm) / (2 * eps)
            denom = max(1.0, abs(numeric), abs(analytic[row, col]))
            rel = abs(numeric - analytic[row, col]) / denom
            assert rel < 2e-8, (name, row, col, numeric, analytic[row, col])

    print("PASS analytic query/key gradients (float64 central differences)")


def check_rank_bound(cfg: Config, rng: np.random.Generator) -> None:
    a, b, c = draw_components(rng, cfg.d, cfg.m)
    wq = cfg.s * a
    wk = cfg.s * (
        cfg.alpha * a + cfg.beta * b + cfg.gamma * c
    )
    kernel = wq.T @ wk
    rank = np.linalg.matrix_rank(kernel, tol=1e-10)
    assert rank <= cfg.m
    assert cfg.m < cfg.d
    print(f"PASS rank(M)={rank} <= m={cfg.m}; full d-by-d condition is singular")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    cfg = Config()
    cfg.normalize_coefficients()
    assert np.isclose(
        cfg.alpha**2 + cfg.beta**2 + cfg.gamma**2, 1.0
    )
    rng = np.random.default_rng(args.seed)

    check_orthogonality_and_energy(cfg, rng, args.draws)
    check_logit_moments(cfg, rng, args.draws)
    check_gradients(cfg, rng)
    check_rank_bound(cfg, rng)
    print("ALL MATHEMATICAL CHECKS PASSED")


if __name__ == "__main__":
    main()
