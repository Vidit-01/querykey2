"""Figures and numeric diagnostics for Experiment 1D trainability."""

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
DATA = ROOT / "code" / "exp1d" / "data" / "full"
OUT = ROOT / "output" / "pdf" / "figures"
DIAG = ROOT / "output" / "pdf" / "exp1d_diagnostics.json"

STRATUM_ORDER = [
    "screened_candidate",
    "diffuse",
    "concentrated",
    "self_locked",
    "boundary_uncertain",
    "corner_independent",
    "corner_orthogonal",
    "corner_tied",
    "corner_anti_tied",
]
STRATUM_LABELS = {
    "screened_candidate": "Screened",
    "diffuse": "Diffuse",
    "concentrated": "Concentrated",
    "self_locked": "Self-locked",
    "boundary_uncertain": "Boundary",
    "corner_independent": "Corner ind.",
    "corner_orthogonal": "Corner orth.",
    "corner_tied": "Corner tied",
    "corner_anti_tied": "Corner anti",
}
COLORS = {
    "screened_candidate": "#166534",
    "diffuse": "#be185d",
    "concentrated": "#52525b",
    "self_locked": "#0e7490",
    "boundary_uncertain": "#a8a29e",
    "corner_independent": "#7c3aed",
    "corner_orthogonal": "#c2410c",
    "corner_tied": "#1d4ed8",
    "corner_anti_tied": "#b45309",
}
PATHOLOGY = {"diffuse", "concentrated", "self_locked", "gradient_starved"}
TASKS = ("associative_recall", "tinystories")
TASK_TITLES = {
    "associative_recall": "Associative recall",
    "tinystories": "TinyStories",
}


def load_points() -> dict:
    return json.loads((DATA / "selected_points.json").read_text(encoding="utf-8"))


def load_rows() -> list[dict]:
    seen: dict[tuple, dict] = {}
    for path in sorted(DATA.glob("training_results_shard_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row["point_id"]), row["task"], int(row["paired_seed"]))
            seen[key] = row
    return list(seen.values())


def group_name(stratum: str) -> str:
    if stratum == "screened_candidate":
        return "screened"
    if stratum in PATHOLOGY:
        return "pathology"
    return "other"


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(values.mean())
    if values.size < 2:
        return mean, mean, mean
    se = float(values.std(ddof=1) / math.sqrt(values.size))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def cluster_bootstrap_ci(
    values: np.ndarray, draws: int = 10000, seed: int = 20260829
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = values.size
    samples = rng.choice(values, size=(draws, n), replace=True).mean(axis=1)
    estimate = float(values.mean())
    lower = float(np.quantile(samples, 0.025))
    upper = float(np.quantile(samples, 0.975))
    return estimate, lower, upper


def summarize(rows: list[dict], points: list[dict]) -> dict:
    by_id = {int(p["point_id"]): p for p in points}
    for row in rows:
        meta = by_id[int(row["point_id"])]
        row["atlas"] = meta.get("atlas", "corner")
        row["group"] = group_name(row["stratum"])
        row["total_drop"] = float(row["initial_loss"]) - float(row["final_loss"])

    n_jobs = len(rows)
    n_div = sum(bool(r["diverged"]) for r in rows)
    seeds = sorted({int(r["paired_seed"]) for r in rows})
    point_ids = sorted({int(r["point_id"]) for r in rows})

    stratum_sources = {}
    for p in points:
        stratum_sources.setdefault(p["stratum"], {"exp1c": 0, "exp1b": 0, "corner": 0})
        atlas = p.get("atlas") or ("corner" if str(p["stratum"]).startswith("corner") else "unknown")
        if atlas not in stratum_sources[p["stratum"]]:
            stratum_sources[p["stratum"]][atlas] = 0
        stratum_sources[p["stratum"]][atlas] += 1

    scale_by_stratum = {}
    for p in points:
        scale_by_stratum.setdefault(p["stratum"], []).append(float(p["s"]))
    scale_stats = {
        k: {"n": len(v), "mean_s": float(np.mean(v)), "median_s": float(np.median(v))}
        for k, v in scale_by_stratum.items()
    }

    comparisons = []
    stratum_tables = {}
    for task in TASKS:
        task_rows = [r for r in rows if r["task"] == task and not r["diverged"]]
        per_seed: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in task_rows:
            per_seed[int(row["paired_seed"])][row["group"]].append(row["loss_decrease_per_token"])
        paired = np.array(
            [
                float(np.mean(groups["screened"]) - np.mean(groups["pathology"]))
                for groups in per_seed.values()
                if groups["screened"] and groups["pathology"]
            ],
            dtype=float,
        )
        estimate, lower, upper = cluster_bootstrap_ci(paired)
        tokens = float(np.mean([r["tokens_seen"] for r in task_rows]))
        comparisons.append(
            {
                "task": task,
                "paired_seeds": int(paired.size),
                "screened_minus_pathology_loss_decrease_per_token": estimate,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "atlas_useful_gate": bool(lower > 0),
                "implied_total_loss_gap": estimate * tokens,
                "mean_tokens_seen": tokens,
                "paired_seed_deltas": [float(x) for x in paired],
            }
        )

        table = []
        for stratum in STRATUM_ORDER:
            group = [r for r in task_rows if r["stratum"] == stratum]
            if not group:
                continue
            drop = np.array([r["total_drop"] for r in group], dtype=float)
            final = np.array([r["final_loss"] for r in group], dtype=float)
            initial = np.array([r["initial_loss"] for r in group], dtype=float)
            lpt = np.array([r["loss_decrease_per_token"] for r in group], dtype=float)
            clip = np.array([r["gradient_clipping_frequency"] for r in group], dtype=float)
            table.append(
                {
                    "stratum": stratum,
                    "n": len(group),
                    "n_points": len({int(r["point_id"]) for r in group}),
                    "mean_initial_loss": float(initial.mean()),
                    "mean_final_loss": float(final.mean()),
                    "mean_total_drop": float(drop.mean()),
                    "std_total_drop": float(drop.std(ddof=1)),
                    "mean_loss_decrease_per_token": float(lpt.mean()),
                    "mean_clip_freq": float(clip.mean()),
                    "mean_elapsed_seconds": float(np.mean([r["elapsed_seconds"] for r in group])),
                }
            )
        stratum_tables[task] = table

    return {
        "n_jobs": n_jobs,
        "n_points": len(point_ids),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "divergence_rate": n_div / max(1, n_jobs),
        "n_diverged": n_div,
        "stratum_sources": stratum_sources,
        "scale_stats": scale_stats,
        "comparisons": comparisons,
        "stratum_tables": stratum_tables,
        "rows": rows,
        "points": points,
    }


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color="#e7e5e4", linewidth=0.6, axis="y")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def fig_gate(summary: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.4), facecolor="white")
    for ax, cmp_ in zip(axes, summary["comparisons"]):
        est = cmp_["screened_minus_pathology_loss_decrease_per_token"]
        lo = cmp_["ci95_lower"]
        hi = cmp_["ci95_upper"]
        color = "#166534" if cmp_["atlas_useful_gate"] else "#b91c1c"
        ax.plot([lo, hi], [0.0, 0.0], color=color, lw=3.0, solid_capstyle="round")
        ax.plot(est, 0.0, "o", color=color, ms=9, zorder=3)
        ax.axvline(0.0, color="#111827", lw=1.0)
        ax.set_yticks([])
        ax.set_ylim(-0.8, 0.8)
        status = "PASS" if cmp_["atlas_useful_gate"] else "FAIL"
        ax.set_title(f"{TASK_TITLES[cmp_['task']]}: {status}", color=color, fontweight="bold")
        ax.set_xlabel("Screened $-$ pathology\nloss decrease / token (95% CI)")
        style_axes(ax)
    fig.suptitle("Experiment 1D atlas-usefulness gate (separate scales)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_1d_gate.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "fig_1d_gate.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_stratum_boxes(summary: dict, field: str, ylabel: str, stem: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), facecolor="white", sharey=False)
    plot_order = [
        "screened_candidate",
        "diffuse",
        "concentrated",
        "self_locked",
        "boundary_uncertain",
    ]
    for ax, task in zip(axes, TASKS):
        data = []
        labels = []
        colors = []
        task_rows = [r for r in summary["rows"] if r["task"] == task and not r["diverged"]]
        for stratum in plot_order:
            vals = [r[field] for r in task_rows if r["stratum"] == stratum]
            if not vals:
                continue
            data.append(vals)
            labels.append(STRATUM_LABELS[stratum])
            colors.append(COLORS[stratum])
        boxes = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.62)
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
            patch.set_edgecolor(color)
        for median in boxes["medians"]:
            median.set_color("#111827")
            median.set_linewidth(1.4)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_title(TASK_TITLES[task], fontweight="bold")
        ax.set_ylabel(ylabel)
        style_axes(ax)
    fig.suptitle(f"Experiment 1D: {ylabel} by frozen stratum", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_selected_points(points: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.8), facecolor="white")
    for stratum in STRATUM_ORDER:
        subset = [p for p in points if p["stratum"] == stratum]
        if not subset:
            continue
        xs = [math.log2(float(p["s"])) for p in subset]
        ys = [float(p["alpha"]) for p in subset]
        marker = "o"
        if stratum.startswith("corner"):
            marker = "D"
        elif any(p.get("atlas") == "exp1b" for p in subset):
            marker = "s"
        ax.scatter(
            xs,
            ys,
            s=42 if not stratum.startswith("corner") else 56,
            c=COLORS[stratum],
            marker=marker,
            edgecolors="#111827",
            linewidths=0.45,
            label=STRATUM_LABELS[stratum],
            zorder=3,
        )
    ax.set_xlabel(r"$\log_2 s$")
    ax.set_ylabel(r"$\alpha$")
    ax.set_title("Frozen 1D coefficient points (pre-training)", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=2)
    ax.axhline(0.0, color="#a8a29e", ls=":", lw=0.8)
    ax.axvline(0.0, color="#a8a29e", ls=":", lw=0.8)
    style_axes(ax)
    ax.grid(True, color="#e7e5e4", linewidth=0.6)
    fig.text(
        0.5,
        0.01,
        "Circles: 1C adaptive cells. Squares: 1B discovery fill (concentrated / self-locked). Diamonds: canonical corners.",
        ha="center",
        fontsize=8,
        color="#44403c",
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_1d_selected_points.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "fig_1d_selected_points.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_learning_curves(summary: dict) -> None:
    """Median loss trajectory per stratum from one representative seed."""
    seed = summary["seeds"][0]
    plot_order = [
        "screened_candidate",
        "diffuse",
        "concentrated",
        "self_locked",
        "boundary_uncertain",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), facecolor="white")
    for ax, task in zip(axes, TASKS):
        for stratum in plot_order:
            point_ids = [
                int(p["point_id"])
                for p in summary["points"]
                if p["stratum"] == stratum
            ]
            curves = []
            for pid in point_ids[:6]:
                path = DATA / f"trace_point{pid:03d}_{task}_seed{seed}.jsonl"
                if not path.exists():
                    continue
                losses = []
                for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                    if line.strip() and index % 10 == 0:
                        losses.append(json.loads(line)["loss"])
                if losses:
                    curves.append(np.asarray(losses, dtype=float))
            if not curves:
                continue
            n = min(len(c) for c in curves)
            stacked = np.stack([c[:n] for c in curves])
            median = np.median(stacked, axis=0)
            q25 = np.quantile(stacked, 0.25, axis=0)
            q75 = np.quantile(stacked, 0.75, axis=0)
            x = np.arange(n) * 10
            ax.plot(x, median, color=COLORS[stratum], lw=1.6, label=STRATUM_LABELS[stratum])
            ax.fill_between(x, q25, q75, color=COLORS[stratum], alpha=0.12, linewidth=0)
        ax.set_title(TASK_TITLES[task], fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Train loss")
        style_axes(ax)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"Median train-loss traces (seed {seed}, up to 8 points/stratum)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_1d_learning_curves.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "fig_1d_learning_curves.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = load_points()
    rows = load_rows()
    summary = summarize(rows, payload["points"])
    fig_gate(summary)
    fig_stratum_boxes(summary, "total_drop", "Initial minus final train loss", "fig_1d_loss_drop")
    fig_stratum_boxes(summary, "final_loss", "Final train loss", "fig_1d_final_loss")
    fig_selected_points(summary["points"])
    fig_learning_curves(summary)
    dump = {k: v for k, v in summary.items() if k not in {"rows", "points"}}
    dump["frozen_before_training"] = payload.get("frozen_before_training")
    dump["primary_label_rule"] = payload.get("primary_label_rule")
    dump["discovery_label_rule"] = payload.get("discovery_label_rule")
    dump["requested_counts"] = payload.get("requested_counts")
    dump["selection_stratum_sources"] = payload.get("stratum_sources")
    DIAG.write_text(json.dumps(dump, indent=2), encoding="utf-8")
    print(json.dumps({k: dump[k] for k in ("n_jobs", "n_points", "n_seeds", "divergence_rate", "comparisons")}, indent=2))
    print(f"wrote {DIAG} and figures in {OUT}")


if __name__ == "__main__":
    main()
