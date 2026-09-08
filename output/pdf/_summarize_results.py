"""Emit numeric snippets for the Experiment 1A/1B LaTeX report."""
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    moments = json.loads((ROOT / "code/exp1a/data/full/moment_results.json").read_text())
    gate = json.loads((ROOT / "code/exp1a/data/full/gate_summary.json").read_text())
    analysis = json.loads((ROOT / "code/exp1b/data/full/analysis_summary.json").read_text())
    thresholds = json.loads((ROOT / "code/exp1b/data/full/pilot_thresholds.json").read_text())
    rows = list(csv.DictReader(open(ROOT / "code/exp1b/data/full/atlas_summary.csv", encoding="utf-8")))

    ref_phi0 = [
        r
        for r in rows
        if r["input_kind"] == "synthetic"
        and r["mask"] == "unmasked"
        and r["correlation"] == "0.25"
        and abs(float(r["phi"])) < 1e-6
    ]
    ref_phi45 = [
        r
        for r in rows
        if r["input_kind"] == "synthetic"
        and r["mask"] == "unmasked"
        and r["correlation"] == "0.25"
        and abs(float(r["phi"]) - math.pi / 4) < 0.02
    ]

    def label_stats(subset: list[dict], name: str) -> None:
        print(f"\n{name} n={len(subset)}")
        for label in sorted({r["label"] for r in subset}):
            group = [r for r in subset if r["label"] == label]
            ne = statistics.mean(float(r["normalized_entropy"]) for r in group)
            mm = statistics.mean(float(r["max_mass"]) for r in group)
            sm = statistics.mean(float(r["self_mass"]) for r in group)
            print(f"  {label:20s} count={len(group):4d}  H_norm={ne:.3f}  max_mass={mm:.3f}  self_mass={sm:.3f}")

    print("GATE", gate)
    print("1A configs", len(moments), "precision_converged", sum(r["precision_converged"] for r in moments))
    print("1B", analysis)
    print("THRESHOLDS", thresholds)
    label_stats(ref_phi0, "Reference stratum synthetic unmasked r=0.25 phi=0")
    label_stats(ref_phi45, "Reference stratum synthetic unmasked r=0.25 phi=pi/4")

    for ik in ("synthetic", "real_activation"):
        sub = [r for r in rows if r["input_kind"] == ik]
        c = Counter(r["label"] for r in sub)
        print(f"\n{ik} label counts:", dict(c))

    cell_maj = Counter()
    for cell_id in {int(float(r["cell_id"])) for r in ref_phi0}:
        cell_rows = [r for r in ref_phi0 if int(float(r["cell_id"])) == cell_id]
        cell_maj[cell_rows[0]["label"]] += 1
    print("\nUnique cells labeled (phi=0 ref):", dict(cell_maj), "total", sum(cell_maj.values()))

    diag = ROOT / "output/pdf/moment_diagnostics.json"
    if diag.exists():
        payload = json.loads(diag.read_text(encoding="utf-8"))
        print("\nUse output/pdf/moment_diagnostics.json (full 879k rows), not a 20k prefix.")
        print("1A", payload["exp1a"])
        print("1B raw", payload["exp1b"]["raw_records"])
        print("1B seed-averaged", payload["exp1b"]["seed_averaged_cells"])
    else:
        print("\nNo moment_diagnostics.json; run output/pdf/make_moment_diagnostics.py")


if __name__ == "__main__":
    main()
