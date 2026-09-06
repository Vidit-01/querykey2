"""Publication-style figures for Experiment 1B atlas (region diagram + 3D surface)."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import griddata

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "code" / "exp1b" / "data" / "full" / "atlas_summary.csv"
DEFAULT_OUT = ROOT / "output" / "pdf" / "figures"

# Reference-style palette (hatched regions + red grid dots)
REGION_STYLE = {
    "diffuse": {"face": "#f9a8d4", "hatch": "///", "edge": "#9d174d", "tag": "A"},
    "concentrated": {"face": "#d4d4d8", "hatch": "\\\\\\", "edge": "#52525b", "tag": "B"},
    "screened_candidate": {"face": "#86efac", "hatch": "xxx", "edge": "#166534", "tag": "C"},
    "boundary_uncertain": {"face": "#ffffff", "hatch": "", "edge": "#a8a29e", "tag": "D"},
    "self_locked": {"face": "#67e8f9", "hatch": "...", "edge": "#0e7490", "tag": "E"},
    "gradient_starved": {"face": "#fde68a", "hatch": "++", "edge": "#b45309", "tag": "F"},
}

LABEL_ORDER = [
    "diffuse",
    "concentrated",
    "self_locked",
    "gradient_starved",
    "screened_candidate",
    "boundary_uncertain",
]


def load_reference_slice(csv_path: Path) -> list[dict]:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    return [
        row
        for row in rows
        if row["input_kind"] == "synthetic"
        and row["mask"] == "unmasked"
        and row["correlation"] == "0.25"
        and abs(float(row["phi"])) < 1e-6
    ]


def interpolate_field(
    rows: list[dict], field: str, grid_x: np.ndarray, grid_y: np.ndarray
) -> np.ndarray:
    points = np.array(
        [[math.log2(float(r["s"])), float(r["alpha"])] for r in rows], dtype=float
    )
    values = np.array([float(r[field]) for r in rows], dtype=float)
    return griddata(points, values, (grid_x, grid_y), method="cubic", fill_value=np.nan)


def draw_region_legend(ax: plt.Axes) -> None:
    x0, y0 = -1.95, 0.55
    for index, label in enumerate(LABEL_ORDER):
        style = REGION_STYLE[label]
        rect = patches.Rectangle(
            (x0, y0 - 0.22 * index),
            0.18,
            0.16,
            facecolor=style["face"],
            edgecolor=style["edge"],
            hatch=style["hatch"],
            linewidth=0.8,
            alpha=0.85,
        )
        ax.add_patch(rect)
        ax.text(
            x0 + 0.24,
            y0 - 0.22 * index + 0.08,
            f"Region {style['tag']}: {label.replace('_', ' ')}",
            fontsize=8,
            va="center",
        )


def make_region_diagram(rows: list[dict], output: Path) -> None:
    log2s = np.array([math.log2(float(r["s"])) for r in rows])
    alpha = np.array([float(r["alpha"]) for r in rows])
    labels = [r["label"] for r in rows]

    gx = np.linspace(-2.0, 2.0, 240)
    gy = np.linspace(-1.0, 1.0, 200)
    grid_x, grid_y = np.meshgrid(gx, gy)
    h_field = interpolate_field(rows, "normalized_entropy", grid_x, grid_y)
    sm_field = interpolate_field(rows, "self_mass", grid_x, grid_y)

    fig, ax = plt.subplots(figsize=(9.2, 6.4), facecolor="white")

    # Background bands inspired by reference figure (Region C band kept narrow).
    ax.axhspan(-1.0, -0.55, facecolor="#ecfccb", alpha=0.35, zorder=0)  # narrow screened band
    ax.axhspan(0.55, 1.0, facecolor="#fce7f3", alpha=0.25, zorder=0)
    ax.axvspan(0.35, 2.0, facecolor="#f5f5f4", alpha=0.35, zorder=0)

    # Theoretical / operational guide curves (dashed, like reference).
    ax.contour(
        grid_x,
        grid_y,
        h_field,
        levels=[0.50, 0.60, 0.95],
        colors=["#2563eb", "#16a34a", "#db2777"],
        linestyles=["--", "--", "--"],
        linewidths=[1.2, 1.0, 1.2],
    )
    ax.contour(
        grid_x,
        grid_y,
        sm_field,
        levels=[0.80],
        colors=["#7c3aed"],
        linestyles=["-."],
        linewidths=1.2,
    )
    ax.axvline(0.0, color="#0891b2", linestyle=":", linewidth=1.0, alpha=0.8)
    ax.axhline(0.0, color="#a8a29e", linestyle=":", linewidth=0.8, alpha=0.8)

    # Annotate guide curves.
    ax.text(1.55, 0.82, r"$H_{\mathrm{norm}}=0.95$", color="#db2777", fontsize=8)
    ax.text(1.35, 0.15, r"$H_{\mathrm{norm}}=0.60$", color="#16a34a", fontsize=8)
    ax.text(-1.7, -0.35, r"$P_{ii}=0.80$", color="#7c3aed", fontsize=8)
    ax.text(0.02, 1.02, r"$\alpha=0$", color="#0891b2", fontsize=8)

    # Red dots on measured cells (reference style).
    color_map = ListedColormap(
        [REGION_STYLE[l]["face"] for l in LABEL_ORDER]
    )
    label_to_idx = {name: index for index, name in enumerate(LABEL_ORDER)}
    colors = [label_to_idx.get(l, 5) for l in labels]
    ax.scatter(
        log2s,
        alpha,
        c=colors,
        cmap=color_map,
        vmin=0,
        vmax=len(LABEL_ORDER) - 1,
        s=26,
        edgecolors="#b91c1c",
        linewidths=0.55,
        zorder=5,
    )

    ax.set_xlim(-2.05, 2.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(r"$\log_2 s$ (scale)", fontsize=11)
    ax.set_ylabel(r"$\alpha$ (query--key alignment)", fontsize=11)
    ax.set_title(
        "Empirical initialization atlas (1B discovery, reference stratum)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.grid(True, color="#e7e5e4", linewidth=0.6, alpha=0.8)
    draw_region_legend(ax)
    fig.text(
        0.5,
        0.01,
        r"Synthetic LayerNorm inputs, unmasked attention, $r=0.25$, $\phi=0$, $(d,m,n)=(64,16,64)$. "
        "Dashed contours: bootstrap median fields; red-edged dots: measured coefficient cells.",
        ha="center",
        fontsize=8.5,
        color="#44403c",
    )
    fig.savefig(output / "fig_atlas_regions.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output / "fig_atlas_regions.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_entropy_surface(rows: list[dict], output: Path) -> None:
    gx = np.linspace(-2.0, 2.0, 80)
    gy = np.linspace(-1.0, 1.0, 80)
    grid_x, grid_y = np.meshgrid(gx, gy)
    field = interpolate_field(rows, "normalized_entropy", grid_x, grid_y)
    field = np.nan_to_num(field, nan=np.nanmean(field))

    fig = plt.figure(figsize=(8.6, 6.2), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        grid_x,
        grid_y,
        field,
        cmap="YlOrBr",
        linewidth=0.25,
        edgecolor="#1c1917",
        antialiased=True,
        alpha=0.96,
    )
    ax.set_xlabel(r"$\log_2 s$", labelpad=8)
    ax.set_ylabel(r"$\alpha$", labelpad=8)
    ax.set_zlabel(r"$H_{\mathrm{norm}}$", labelpad=8)
    ax.set_title(
        "Normalized row-entropy surface over coefficient space",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    ax.view_init(elev=28, azim=-58)
    fig.colorbar(surf, ax=ax, shrink=0.62, pad=0.08, label=r"median $H/\log k$")
    fig.savefig(output / "fig_entropy_surface_3d.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output / "fig_entropy_surface_3d.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_label_mosaic(rows: list[dict], output: Path) -> None:
    """Second panel: discrete label regions on the same grid."""
    gx = np.linspace(-2.0, 2.0, 200)
    gy = np.linspace(-1.0, 1.0, 160)
    grid_x, grid_y = np.meshgrid(gx, gy)

    # Nearest-neighbor label field from measured cells.
    points = np.array([[math.log2(float(r["s"])), float(r["alpha"])] for r in rows])
    labels = np.array([LABEL_ORDER.index(r["label"]) if r["label"] in LABEL_ORDER else 5 for r in rows])
    label_field = griddata(points, labels, (grid_x, grid_y), method="nearest")

    cmap = ListedColormap([REGION_STYLE[name]["face"] for name in LABEL_ORDER])
    fig, ax = plt.subplots(figsize=(8.8, 5.8), facecolor="white")
    im = ax.imshow(
        label_field,
        origin="lower",
        extent=[-2, 2, -1, 1],
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=len(LABEL_ORDER) - 1,
        alpha=0.92,
    )
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=12,
        c="#b91c1c",
        alpha=0.55,
        linewidths=0,
        zorder=3,
    )
    ax.set_xlabel(r"$\log_2 s$")
    ax.set_ylabel(r"$\alpha$")
    ax.set_title("Nearest-cell label mosaic (1B)", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, ticks=range(len(LABEL_ORDER)))
    cbar.ax.set_yticklabels(
        [f"{REGION_STYLE[n]['tag']}: {n.replace('_', ' ')}" for n in LABEL_ORDER]
    )
    fig.savefig(output / "fig_atlas_label_mosaic.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output / "fig_atlas_label_mosaic.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = load_reference_slice(args.csv)
    if not rows:
        raise SystemExit(f"no reference-stratum rows in {args.csv}")
    make_region_diagram(rows, args.output)
    make_entropy_surface(rows, args.output)
    make_label_mosaic(rows, args.output)
    print(f"Wrote figures to {args.output}")


if __name__ == "__main__":
    main()
