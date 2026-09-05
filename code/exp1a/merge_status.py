"""Inventory and merge exp1a shards; report exp1b remaining work."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from exp1a.run import configuration_list  # noqa: E402
from exp1b.run import discovery_grid  # noqa: E402


def read_shard_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def unique_by_config(rows: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    for row in rows:
        key = row["config_id"]
        current = best.get(key)
        if current is None or row.get("draws", 0) >= current.get("draws", 0):
            best[key] = row
    return list(best.values())


def shard_index(name: str) -> int:
    return int(name.split("_")[2].split(".")[0])


def collect_config_ids(base: Path) -> set[str]:
    ids: set[str] = set()
    for path in base.glob("moments_shard_*.jsonl"):
        for row in read_shard_rows(path):
            ids.add(row["config_id"])
    return ids


def merge_exp1a(exo: Path, dest: Path) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    merged: dict[int, list[dict]] = {}
    sources: list[Path] = []
    if exo.exists():
        sources.extend(sorted(exo.glob("moments_shard_*.jsonl")))
    sources.extend(sorted(dest.glob("moments_shard_*.jsonl")))
    for path in sources:
        shard = shard_index(path.name)
        merged.setdefault(shard, []).extend(read_shard_rows(path))
    actions = []
    for shard, rows in sorted(merged.items()):
        deduped = unique_by_config(rows)
        out = dest / f"moments_shard_{shard:04d}.jsonl"
        before = len(read_shard_rows(out)) if out.exists() else 0
        deduped.sort(key=lambda row: row["config_id"])
        with out.open("w", encoding="utf-8") as handle:
            for row in deduped:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        actions.append(
            {
                "shard": shard,
                "before_rows": before,
                "after_rows": len(deduped),
                "path": str(out),
            }
        )
    for name in ("checks.json", "manifest.json"):
        exo_file = exo / name
        dest_file = dest / name
        if exo_file.exists() and (
            not dest_file.exists()
            or exo_file.stat().st_mtime > dest_file.stat().st_mtime
        ):
            shutil.copy2(exo_file, dest_file)
    exo_manifests = exo / "manifests"
    if exo_manifests.exists():
        target = dest / "manifests"
        target.mkdir(parents=True, exist_ok=True)
        for path in exo_manifests.glob("*.json"):
            target_path = target / path.name
            if not target_path.exists():
                shutil.copy2(path, target_path)
    return {"shard_actions": actions}


def exp1a_report(dest: Path) -> dict:
    configs = configuration_list("full")
    expected = {row["config_id"] for row in configs}
    completed = collect_config_ids(dest)
    shard_counts = {
        path.name: len(read_shard_rows(path))
        for path in sorted(dest.glob("moments_shard_*.jsonl"))
    }
    return {
        "expected_configs": len(expected),
        "completed_configs": len(completed),
        "missing_configs": len(expected - completed),
        "shard_counts": shard_counts,
        "missing_shards": [
            index
            for index in range(16)
            if f"moments_shard_{index:04d}.jsonl" not in shard_counts
            or shard_counts[f"moments_shard_{index:04d}.jsonl"] == 0
        ],
        "checks_exists": (dest / "checks.json").exists(),
        "gate_exists": (dest / "gate_summary.json").exists(),
    }


def exp1b_report(dest: Path) -> dict:
    cells = discovery_grid("full")
    completed_cells: set[int] = set()
    shard_counts: dict[str, int] = {}
    for path in sorted(dest.glob("atlas_shard_*.jsonl")):
        ids = {json.loads(line)["cell_id"] for line in path.read_text().splitlines() if line.strip()}
        shard_counts[path.name] = len(ids)
        completed_cells |= ids
    remaining = [index for index in range(len(cells)) if index not in completed_cells]
    return {
        "expected_cells": len(cells),
        "completed_cells": len(completed_cells),
        "remaining_cells": len(remaining),
        "remaining_cell_ids_head": remaining[:20],
        "shard_counts": shard_counts,
        "missing_shards_if_32": [
            index
            for index in range(32)
            if f"atlas_shard_{index:04d}.jsonl" not in shard_counts
        ],
        "analysis_files": {
            name: (dest / name).exists()
            for name in (
                "atlas_summary.csv",
                "analysis_summary.json",
                "pilot_thresholds.json",
                "sign_symmetry_check.json",
            )
        },
        "progress_exists": (dest / "progress.json").exists(),
    }


def main() -> None:
    exo = ROOT / "exo1a1" / "exp1a" / "data" / "full"
    exp1a_dest = ROOT / "code" / "exp1a" / "data" / "full"
    exp1b_dest = ROOT / "code" / "exp1b" / "data" / "full"

    before = exp1a_report(exp1a_dest)
    merge = merge_exp1a(exo, exp1a_dest)
    after = exp1a_report(exp1a_dest)
    exp1b = exp1b_report(exp1b_dest)

    print(json.dumps({"merge": merge, "exp1a_before": before, "exp1a_after": after, "exp1b": exp1b}, indent=2))


if __name__ == "__main__":
    main()
