"""Honest 1A / 1B logit-moment diagnostics for the results report.

The original 1B residual figure mixed two different variance quantities and
quoted mean residuals from the first 20k (easy) rows. This script computes
RMSE, relative error, OLS calibration, and stratified residuals on the full
data, plus a law-of-total-variance correction for 1B.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "pdf" / "figures"
MOMENTS_1A = ROOT / "code" / "exp1a" / "data" / "full" / "moment_results.json"
SHARD_DIR = ROOT / "code" / "exp1b" / "data" / "full"


def _summarize(emp: np.ndarray, pred: np.ndarray, *, relative_floor: float) -> dict:
    residual = emp - pred
    abs_rel = np.abs(residual) / np.maximum(np.abs(pred), relative_floor)
    n = residual.size
    pred_c = pred - pred.mean()
    emp_c = emp - emp.mean()
    denom = float(np.dot(pred_c, pred_c))
    slope = float(np.dot(pred_c, emp_c) / denom) if denom > 0 else math.nan
    intercept = float(emp.mean() - slope * pred.mean()) if np.isfinite(slope) else math.nan
    fitted = intercept + slope * pred
    ss_res = float(np.sum((emp - fitted) ** 2))
    ss_tot = float(np.sum(emp_c**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    return {
        "n": int(n),
        "mean_residual": float(residual.mean()),
        "std_residual": float(residual.std()),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "median_abs_relative": float(np.median(abs_rel)),
        "p95_abs_relative": float(np.quantile(abs_rel, 0.95)),
        "ols_intercept": intercept,
        "ols_slope": slope,
        "ols_r2": r2,
    }


def _scatter_with_identity(ax, pred, emp, *, title, xlabel, ylabel, sample=40000, color=None, s=6):
    rng = np.random.default_rng(0)
    if pred.size > sample:
        idx = rng.choice(pred.size, size=sample, replace=False)
        pred_s, emp_s = pred[idx], emp[idx]
        color_s = None if color is None else np.asarray(color)[idx]
    else:
        pred_s, emp_s, color_s = pred, emp, color
    if color_s is None:
        ax.scatter(pred_s, emp_s, s=s, alpha=0.35, c="#2563eb", linewidths=0)
    else:
        sc = ax.scatter(pred_s, emp_s, s=s, alpha=0.45, c=color_s, cmap="viridis", linewidths=0)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=r"$\log_2 s$")
    lo = float(min(pred_s.min(), emp_s.min()))
    hi = float(max(pred_s.max(), emp_s.max()))
    pad = 0.05 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#111827", lw=1.1, label=r"$y=x$")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e5e7eb", linewidth=0.6)
    ax.legend(frameon=False, loc="upper left")


def _residual_plot(ax, pred, residual, *, title, xlabel, color=None, s=6):
    rng = np.random.default_rng(1)
    if pred.size > 40000:
        idx = rng.choice(pred.size, size=40000, replace=False)
        pred, residual = pred[idx], residual[idx]
        color = None if color is None else np.asarray(color)[idx]
    if color is None:
        ax.scatter(pred, residual, s=s, alpha=0.3, c="#2563eb", linewidths=0)
    else:
        sc = ax.scatter(pred, residual, s=s, alpha=0.4, c=color, cmap="viridis", linewidths=0)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=r"$\log_2 s$")
    ax.axhline(0.0, color="#111827", lw=1.1)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("empirical − predicted")
    ax.grid(True, color="#e5e7eb", linewidth=0.6)


def analyze_1a() -> dict:
    rows = json.loads(MOMENTS_1A.read_text(encoding="utf-8"))
    emp_m = np.array([r["empirical_mean"] for r in rows], dtype=np.float64)
    pred_m = np.array([r["predicted_mean"] for r in rows], dtype=np.float64)
    emp_v = np.array([r["empirical_variance"] for r in rows], dtype=np.float64)
    pred_v = np.array([r["predicted_variance"] for r in rows], dtype=np.float64)

    stats = {
        "mean": _summarize(emp_m, pred_m, relative_floor=1e-6),
        "variance": _summarize(emp_v, pred_v, relative_floor=1e-6),
        "holm_mean_rejected": int(sum(r["holm_mean_rejected"] for r in rows)),
        "holm_variance_rejected": int(sum(r["holm_variance_rejected"] for r in rows)),
        "n_configs": len(rows),
    }

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.6), facecolor="white")
    _scatter_with_identity(
        axes[0, 0], pred_m, emp_m,
        title="1A logit mean (864 configs)",
        xlabel="predicted mean",
        ylabel="empirical mean",
        sample=864,
        s=18,
    )
    _scatter_with_identity(
        axes[0, 1], pred_v, emp_v,
        title="1A logit variance (864 configs)",
        xlabel="predicted variance",
        ylabel="empirical variance",
        sample=864,
        s=18,
    )
    _residual_plot(axes[1, 0], pred_m, emp_m - pred_m, title="1A mean residual", xlabel="predicted mean", s=18)
    _residual_plot(axes[1, 1], pred_v, emp_v - pred_v, title="1A variance residual", xlabel="predicted variance", s=18)
    fig.suptitle("Experiment 1A: isolated-pair Monte Carlo vs exact moments", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_1a_moment_calibration.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "fig_1a_moment_calibration.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return stats


def _load_1b() -> dict[str, np.ndarray]:
    emp_m, pred_m, emp_v, pred_v = [], [], [], []
    s_vals, alpha, mask, kind, corr, log2s = [], [], [], [], [], []
    mean_cij, mean_cij2, m_vals = [], [], []
    cell_keys = []
    for path in sorted(SHARD_DIR.glob("atlas_shard_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                emp_m.append(row["logit_mean"])
                pred_m.append(row["predicted_logit_mean"])
                emp_v.append(row["logit_variance"])
                pred_v.append(row["predicted_logit_variance"])
                s = float(row["s"])
                a = float(row["alpha"])
                s_vals.append(s)
                alpha.append(a)
                log2s.append(math.log2(s))
                mask.append(row["mask"])
                kind.append(row["input_kind"])
                corr.append("real" if row["correlation"] is None else str(row["correlation"]))
                mean_cij.append(float(row.get("mean_cij") or 0.0))
                mean_cij2.append(float(row["mean_cij2"]))
                m_vals.append(float(row["m"]))
                cell_keys.append(
                    (
                        int(row["cell_id"]),
                        row["input_kind"],
                        row["mask"],
                        "real" if row["correlation"] is None else str(row["correlation"]),
                    )
                )
    arrays = {
        "emp_m": np.asarray(emp_m, dtype=np.float64),
        "pred_m": np.asarray(pred_m, dtype=np.float64),
        "emp_v": np.asarray(emp_v, dtype=np.float64),
        "pred_v": np.asarray(pred_v, dtype=np.float64),
        "s": np.asarray(s_vals, dtype=np.float64),
        "alpha": np.asarray(alpha, dtype=np.float64),
        "log2s": np.asarray(log2s, dtype=np.float64),
        "mean_cij": np.asarray(mean_cij, dtype=np.float64),
        "mean_cij2": np.asarray(mean_cij2, dtype=np.float64),
        "m": np.asarray(m_vals, dtype=np.float64),
        "mask": np.asarray(mask),
        "kind": np.asarray(kind),
        "corr": np.asarray(corr),
        "cell_keys": cell_keys,
    }
    scale = arrays["alpha"] * (arrays["s"] ** 2) * np.sqrt(arrays["m"])
    gram_var = np.maximum(arrays["mean_cij2"] - arrays["mean_cij"] ** 2, 0.0)
    arrays["pred_v_total"] = arrays["pred_v"] + (scale**2) * gram_var
    return arrays


def _cell_means(keys: list[tuple], values: np.ndarray) -> np.ndarray:
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for key, value in zip(keys, values, strict=True):
        buckets[key].append(float(value))
    return np.array([np.mean(vals) for vals in buckets.values()], dtype=np.float64)


def _stratum_table(data: dict[str, np.ndarray], pred_v: np.ndarray) -> list[dict]:
    rows = []
    residual_v = data["emp_v"] - pred_v
    residual_m = data["emp_m"] - data["pred_m"]
    for name, labels in (
        ("input_kind", data["kind"]),
        ("mask", data["mask"]),
        ("correlation", data["corr"]),
    ):
        for label in sorted(set(labels.tolist())):
            sel = labels == label
            rows.append(
                {
                    "stratum": name,
                    "value": str(label),
                    "n": int(sel.sum()),
                    "mean_rmse": float(np.sqrt(np.mean(residual_m[sel] ** 2))),
                    "var_rmse": float(np.sqrt(np.mean(residual_v[sel] ** 2))),
                    "var_mean_residual": float(residual_v[sel].mean()),
                    "var_ols_slope": _summarize(data["emp_v"][sel], pred_v[sel], relative_floor=1e-8)["ols_slope"],
                    "var_ols_r2": _summarize(data["emp_v"][sel], pred_v[sel], relative_floor=1e-8)["ols_r2"],
                }
            )
    # Discrete scale bands (the islands in the raw residual plot).
    for s in sorted(set(np.round(data["s"], 8).tolist())):
        sel = np.isclose(data["s"], s)
        if sel.sum() < 100:
            continue
        rows.append(
            {
                "stratum": "s",
                "value": f"{s:.4g}",
                "n": int(sel.sum()),
                "mean_rmse": float(np.sqrt(np.mean(residual_m[sel] ** 2))),
                "var_rmse": float(np.sqrt(np.mean(residual_v[sel] ** 2))),
                "var_mean_residual": float(residual_v[sel].mean()),
                "var_ols_slope": _summarize(data["emp_v"][sel], pred_v[sel], relative_floor=1e-8)["ols_slope"],
                "var_ols_r2": _summarize(data["emp_v"][sel], pred_v[sel], relative_floor=1e-8)["ols_r2"],
            }
        )
    return rows


def analyze_1b(data: dict[str, np.ndarray]) -> dict:
    naive = {
        "mean": _summarize(data["emp_m"], data["pred_m"], relative_floor=1.0),
        "variance_pairwise": _summarize(data["emp_v"], data["pred_v"], relative_floor=1e-6),
        "variance_total": _summarize(data["emp_v"], data["pred_v_total"], relative_floor=1e-6),
    }
    cell_emp_m = _cell_means(data["cell_keys"], data["emp_m"])
    cell_pred_m = _cell_means(data["cell_keys"], data["pred_m"])
    cell_emp_v = _cell_means(data["cell_keys"], data["emp_v"])
    cell_pred_v = _cell_means(data["cell_keys"], data["pred_v"])
    cell_pred_vt = _cell_means(data["cell_keys"], data["pred_v_total"])
    cell = {
        "mean": _summarize(cell_emp_m, cell_pred_m, relative_floor=1.0),
        "variance_pairwise": _summarize(cell_emp_v, cell_pred_v, relative_floor=1e-6),
        "variance_total": _summarize(cell_emp_v, cell_pred_vt, relative_floor=1e-6),
    }
    return {
        "raw_records": naive,
        "seed_averaged_cells": cell,
        "strata_vs_pairwise_variance": _stratum_table(data, data["pred_v"]),
        "strata_vs_total_variance": _stratum_table(data, data["pred_v_total"]),
        "note": (
            "predicted_logit_variance is the mean pairwise conditional variance given tokens. "
            "empirical logit_variance is the across-pair sample variance of one realized logit matrix. "
            "variance_total adds Var_ij(E[ell_ij|x]) ≈ (alpha s^2 sqrt(m))^2 Var(G_ij)."
        ),
    }


def plot_1b(data: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.0), facecolor="white")
    _scatter_with_identity(
        axes[0, 0],
        data["pred_m"],
        data["emp_m"],
        title="1B mean: empirical vs predicted",
        xlabel=r"predicted $\mathbb{E}[\ell]$ (pair average)",
        ylabel=r"empirical mean of realized $\ell_{ij}$",
        color=data["log2s"],
    )
    _scatter_with_identity(
        axes[0, 1],
        data["pred_v"],
        data["emp_v"],
        title="1B variance: pairwise formula (mismatched)",
        xlabel=r"mean pairwise $\mathrm{Var}(\ell_{ij}\mid x)$",
        ylabel=r"across-pair sample variance of $\ell$",
        color=data["log2s"],
    )
    _residual_plot(
        axes[1, 0],
        data["pred_m"],
        data["emp_m"] - data["pred_m"],
        title="1B mean residual",
        xlabel="predicted mean",
        color=data["log2s"],
    )
    _residual_plot(
        axes[1, 1],
        data["pred_v"],
        data["emp_v"] - data["pred_v"],
        title="1B variance residual (pairwise, structured)",
        xlabel="predicted pairwise variance",
        color=data["log2s"],
    )
    fig.suptitle(
        "Experiment 1B: full-sequence single-draw moments (color = scale)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_1b_moment_naive.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "fig_1b_moment_naive.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), facecolor="white")
    _scatter_with_identity(
        axes[0],
        data["pred_v_total"],
        data["emp_v"],
        title="Corrected: law of total variance",
        xlabel=r"$\mathbb{E}[\mathrm{Var}(\ell\mid x)]+\mathrm{Var}(\mathbb{E}[\ell\mid x])$",
        ylabel=r"empirical across-pair variance",
        color=data["log2s"],
    )
    _residual_plot(
        axes[1],
        data["pred_v_total"],
        data["emp_v"] - data["pred_v_total"],
        title="Residual after total-variance correction",
        xlabel="corrected predicted variance",
        color=data["log2s"],
    )
    fig.suptitle("1B variance after matching the empirical estimand", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_1b_variance_corrected.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "fig_1b_variance_corrected.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Stratified residual boxes by scale (explains the discrete islands).
    scales = sorted(set(np.round(data["s"], 8).tolist()))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), facecolor="white")
    groups_naive = [data["emp_v"][np.isclose(data["s"], s)] - data["pred_v"][np.isclose(data["s"], s)] for s in scales]
    groups_corr = [
        data["emp_v"][np.isclose(data["s"], s)] - data["pred_v_total"][np.isclose(data["s"], s)] for s in scales
    ]
    labels = [rf"$2^{{{math.log2(s):.0f}}}$" if abs(math.log2(s) - round(math.log2(s))) < 1e-6 else f"{s:.2f}" for s in scales]
    axes[0].boxplot(groups_naive, showfliers=False, widths=0.65)
    axes[0].axhline(0, color="#111827", lw=0.9)
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[0].set_title("Naive variance residual by $s$")
    axes[0].set_ylabel("empirical − pairwise prediction")
    axes[1].boxplot(groups_corr, showfliers=False, widths=0.65)
    axes[1].axhline(0, color="#111827", lw=0.9)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_title("Corrected variance residual by $s$")
    axes[1].set_ylabel("empirical − total-variance prediction")
    fig.tight_layout()
    fig.savefig(OUT / "fig_1b_variance_by_scale.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "fig_1b_variance_by_scale.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _fmt(stats: dict) -> dict:
    out = {}
    for key, value in stats.items():
        if isinstance(value, float):
            out[key] = round(value, 6)
        else:
            out[key] = value
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("analyzing 1A...")
    stats_1a = analyze_1a()
    print("loading 1B shards...")
    data = _load_1b()
    print(f"1B rows: {data['emp_m'].size}")
    print("analyzing 1B...")
    stats_1b = analyze_1b(data)
    plot_1b(data)
    payload = {
        "exp1a": {
            "mean": _fmt(stats_1a["mean"]),
            "variance": _fmt(stats_1a["variance"]),
            "holm_mean_rejected": stats_1a["holm_mean_rejected"],
            "holm_variance_rejected": stats_1a["holm_variance_rejected"],
            "n_configs": stats_1a["n_configs"],
        },
        "exp1b": {
            "raw_records": {k: _fmt(v) for k, v in stats_1b["raw_records"].items()},
            "seed_averaged_cells": {k: _fmt(v) for k, v in stats_1b["seed_averaged_cells"].items()},
            "note": stats_1b["note"],
            "strata_vs_pairwise_variance": stats_1b["strata_vs_pairwise_variance"],
            "strata_vs_total_variance": stats_1b["strata_vs_total_variance"],
        },
    }
    out_json = ROOT / "output" / "pdf" / "moment_diagnostics.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["exp1a"], indent=2))
    print(json.dumps(payload["exp1b"]["raw_records"], indent=2))
    print(json.dumps(payload["exp1b"]["seed_averaged_cells"], indent=2))
    print(f"wrote {out_json} and figures in {OUT}")


if __name__ == "__main__":
    main()
