"""Generate figures for the mathematically audited research proposal."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 10,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 150,
    }
)

NAVY = "#1e3a4c"
BLUE = "#2563a6"
ORANGE = "#b45309"
RED = "#b91c1c"
GREEN = "#2f7d57"
PURPLE = "#6d3f8c"
GRID = "#d6d3d1"


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=-1, keepdims=True)


def entropy(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-300, None))).sum(axis=-1)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG_DIR / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_screening_atlas() -> None:
    """Finite-n i.i.d.-Gaussian surrogate. This is not a trainability theorem."""

    n, m, r = 64, 16, 0.25
    alpha_grid = np.linspace(-1.0, 1.0, 61)
    log2s_grid = np.linspace(-2.0, 2.0, 51)
    hfrac = np.zeros((len(alpha_grid), len(log2s_grid)))
    self_mass = np.zeros_like(hfrac)
    max_mass = np.zeros_like(hfrac)

    # Common random numbers reduce Monte Carlo roughness across cells.
    rng = np.random.default_rng(20260829)
    base = rng.standard_normal((768, n))
    diag_index = np.arange(base.shape[0]) % n

    for ai, alpha in enumerate(alpha_grid):
        for si, log2s in enumerate(log2s_grid):
            s = 2.0**log2s
            tau = s**2 * np.sqrt(1.0 + alpha**2 * r**2)
            delta = alpha * s**2 * np.sqrt(m) * (1.0 - r)
            z = tau * base
            z[np.arange(len(z)), diag_index] += delta
            p = softmax(z)
            hfrac[ai, si] = entropy(p).mean() / np.log(n)
            self_mass[ai, si] = p[np.arange(len(p)), diag_index].mean()
            max_mass[ai, si] = p.max(axis=1).mean()

    extent = [log2s_grid[0], log2s_grid[-1], alpha_grid[0], alpha_grid[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), constrained_layout=True)

    im0 = axes[0].imshow(
        hfrac,
        origin="lower",
        extent=extent,
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="viridis",
    )
    axes[0].contour(
        log2s_grid,
        alpha_grid,
        hfrac,
        levels=[0.5, 0.6, 0.95],
        colors=[RED, ORANGE, "white"],
        linewidths=[1.4, 1.2, 1.2],
    )
    axes[0].set_title("(a) Normalized row entropy")
    axes[0].set_xlabel(r"$\log_2 s$")
    axes[0].set_ylabel(r"alignment $\alpha$")
    cb0 = fig.colorbar(im0, ax=axes[0], shrink=0.9)
    cb0.set_label(r"$\mathbb{E}[H]/\log n$")

    im1 = axes[1].imshow(
        self_mass,
        origin="lower",
        extent=extent,
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="magma",
    )
    axes[1].contour(
        log2s_grid,
        alpha_grid,
        self_mass,
        levels=[0.5, 0.8],
        colors=[ORANGE, "white"],
        linewidths=[1.2, 1.5],
    )
    axes[1].contour(
        log2s_grid,
        alpha_grid,
        max_mass,
        levels=[0.8],
        colors=[BLUE],
        linewidths=[1.2],
        linestyles=["--"],
    )
    axes[1].set_title("(b) Mean self-mass")
    axes[1].set_xlabel(r"$\log_2 s$")
    axes[1].set_ylabel(r"alignment $\alpha$")
    cb1 = fig.colorbar(im1, ax=axes[1], shrink=0.9)
    cb1.set_label(r"$\mathbb{E}[P_{ii}]$")

    fig.suptitle(
        "Finite-$n$ unmasked Gaussian screening surrogate "
        r"($n=64,\ m=16,\ r=0.25$)",
        color=NAVY,
        fontweight="bold",
        y=1.03,
    )
    fig.text(
        0.5,
        -0.015,
        "Contours are operational guides only: entropy 0.50/0.60/0.95, "
        "self-mass 0.50/0.80, max-mass 0.80 (blue dashed). "
        "The replicated empirical atlas replaces this surrogate.",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#44403c",
    )
    save(fig, "fig_phase.png")


def orthogonal_b(a: np.ndarray, g: np.ndarray) -> np.ndarray:
    # QR of A^T gives an orthonormal basis for row(A).
    u, _ = np.linalg.qr(a.T, mode="reduced")
    return g - (g @ u) @ u.T


def exact_g(d: int, vi: float, vj: float, cij: float) -> float:
    return d * ((d + 1) * vi * vj - 2 * cij**2) / ((d - 1) * (d + 2))


def make_moment_validation() -> None:
    d, m, draws = 64, 16, 5000
    s, cxy = 1.15, 0.35
    x = np.zeros(d)
    y = np.zeros(d)
    x[0] = np.sqrt(d)
    y[0] = np.sqrt(d) * cxy
    y[1] = np.sqrt(d) * np.sqrt(1 - cxy**2)
    vi = vj = 1.0

    rng = np.random.default_rng(314159)
    za = np.empty(draws)
    zb = np.empty(draws)
    zc = np.empty(draws)
    za_diag = np.empty(draws)
    zb_diag = np.empty(draws)
    zc_diag = np.empty(draws)
    for k in range(draws):
        a = rng.standard_normal((m, d)) / np.sqrt(d)
        c = rng.standard_normal((m, d)) / np.sqrt(d)
        g = rng.standard_normal((m, d)) / np.sqrt(d - m)
        b = orthogonal_b(a, g)
        q = a @ x
        za[k] = np.dot(q, a @ y) / np.sqrt(m)
        zb[k] = np.dot(q, b @ y) / np.sqrt(m)
        zc[k] = np.dot(q, c @ y) / np.sqrt(m)
        za_diag[k] = np.dot(q, a @ x) / np.sqrt(m)
        zb_diag[k] = np.dot(q, b @ x) / np.sqrt(m)
        zc_diag[k] = np.dot(q, c @ x) / np.sqrt(m)

    alphas = np.linspace(-1, 1, 17)
    empirical_mean = []
    theory_mean = []
    empirical_var = []
    theory_var = []
    gij = exact_g(d, vi, vj, cxy)
    for alpha in alphas:
        beta = gamma = np.sqrt(max(0.0, 1 - alpha**2) / 2)
        z = s**2 * (alpha * za + beta * zb + gamma * zc)
        empirical_mean.append(z.mean())
        empirical_var.append(z.var(ddof=1))
        theory_mean.append(alpha * s**2 * np.sqrt(m) * cxy)
        theory_var.append(
            s**4
            * (
                alpha**2 * (vi * vj + cxy**2)
                + beta**2 * gij
                + gamma**2 * vi * vj
            )
        )

    alpha0 = 0.4
    phis = np.linspace(0, np.pi / 2, 17)
    empirical_phi = []
    theory_phi = []
    component_vars = np.array(
        [
            za_diag.var(ddof=1),
            zb_diag.var(ddof=1),
            zc_diag.var(ddof=1),
        ]
    )
    for phi in phis:
        beta = np.sqrt(1 - alpha0**2) * np.cos(phi)
        gamma = np.sqrt(1 - alpha0**2) * np.sin(phi)
        empirical_phi.append(
            s**4
            * np.dot(
                np.array([alpha0**2, beta**2, gamma**2]),
                component_vars,
            )
        )
        theory_phi.append(
            s**4
            * (
                alpha0**2 * 2
                + beta**2 * d / (d + 2)
                + gamma**2
            )
        )

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.65), constrained_layout=True)
    axes[0].plot(alphas, theory_mean, color=NAVY, lw=2.2, label="exact")
    axes[0].scatter(alphas, empirical_mean, color=ORANGE, s=23, label="Monte Carlo")
    axes[0].set_title("(a) Logit mean")
    axes[0].set_xlabel(r"$\alpha$")
    axes[0].set_ylabel(r"$\mathbb{E}[L_{ij}]$")

    axes[1].plot(alphas, theory_var, color=NAVY, lw=2.2)
    axes[1].scatter(alphas, empirical_var, color=ORANGE, s=23)
    axes[1].set_title(r"(b) Variance, $\beta=\gamma$")
    axes[1].set_xlabel(r"$\alpha$")
    axes[1].set_ylabel(r"$\mathrm{Var}(L_{ij})$")

    axes[2].plot(phis / np.pi, theory_phi, color=NAVY, lw=2.2)
    axes[2].scatter(phis / np.pi, empirical_phi, color=ORANGE, s=23)
    axes[2].set_title(r"(c) Diagonal finite-$d$ effect")
    axes[2].set_xlabel(r"$\phi/\pi$")
    axes[2].set_ylabel(r"$\mathrm{Var}(L_{ij})$")

    for ax in axes:
        ax.grid(alpha=0.3, color=GRID)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(
        r"Exact finite-width moments vs. simulation "
        r"($d=64,\ m=16,\ c_{ij}=0.35,\ 5000$ draws)",
        color=NAVY,
        fontweight="bold",
        y=1.05,
    )
    save(fig, "fig_entropy.png")


def common_bias(alpha1: np.ndarray, alpha2: np.ndarray, d: int, m: int) -> np.ndarray:
    f1 = 1 + alpha1**2 * (m + 1) / d
    f2 = 1 + alpha2**2 * (m + 1) / d
    return alpha1 * alpha2 * m / (d * np.sqrt(f1 * f2))


def make_common_bias_figure() -> None:
    d, m = 64, 16
    alpha = np.linspace(-1, 1, 181)
    a1, a2 = np.meshgrid(alpha, alpha)
    r12 = common_bias(a1, a2, d, m)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.25), constrained_layout=True)
    lim = np.max(np.abs(r12))
    cmap = LinearSegmentedColormap.from_list(
        "biasdiv", ["#31578a", "#f7f5f1", "#9a3412"], N=256
    )
    im = axes[0].imshow(
        r12,
        origin="lower",
        extent=[-1, 1, -1, 1],
        cmap=cmap,
        vmin=-lim,
        vmax=lim,
        aspect="equal",
    )
    axes[0].axhline(0, color="white", lw=0.6, alpha=0.7)
    axes[0].axvline(0, color="white", lw=0.6, alpha=0.7)
    axes[0].set_title("(a) Uncentered common-bias statistic")
    axes[0].set_xlabel(r"$\alpha_1$")
    axes[0].set_ylabel(r"$\alpha_2$")
    cb = fig.colorbar(im, ax=axes[0], shrink=0.88)
    cb.set_label(r"$R_{12}$")

    diag = common_bias(alpha, alpha, d, m)
    axes[1].plot(alpha, diag, color=ORANGE, lw=2.5, label=r"$R_{12}$, $\alpha_1=\alpha_2$")
    axes[1].axhline(
        0,
        color=BLUE,
        lw=2,
        ls="--",
        label="expected centered inner product",
    )
    axes[1].fill_between(alpha, -0.004, 0.004, color=BLUE, alpha=0.10)
    axes[1].set_title("(b) Centering removes the shared mean")
    axes[1].set_xlabel(r"common alignment $\alpha$")
    axes[1].set_ylabel("population statistic")
    axes[1].grid(alpha=0.3, color=GRID)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, fontsize=8.5)
    axes[1].set_ylim(-0.012, 0.21)
    axes[1].text(
        0,
        0.012,
        "Independent centered heads have zero expected inner product\n"
        "for every alignment; functional diversity must be measured.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#44403c",
    )
    fig.suptitle(
        "Alignment creates a shared isotropic bias, not a diversity guarantee",
        color=NAVY,
        fontweight="bold",
        y=1.03,
    )
    save(fig, "fig_diversity.png")


def main() -> None:
    make_screening_atlas()
    make_moment_validation()
    make_common_bias_figure()
    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
