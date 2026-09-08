"""Coefficient grouping, freeze rules, and phenotype labels for 1C/1D."""

from __future__ import annotations

from collections import Counter, defaultdict

REFERENCE_GEOMETRY = (64, 16, 64)
SCREENED = "screened_candidate"
COEFFICIENT_KEYS = ("s", "alpha", "beta", "gamma")
PATHOLOGY_PRECEDENCE = (
    "self_locked",
    "concentrated",
    "diffuse",
    "gradient_starved",
)
LABEL_TIE_RANK = {
    "self_locked": 0,
    "concentrated": 1,
    "diffuse": 2,
    "gradient_starved": 3,
    SCREENED: 4,
    "boundary_uncertain": 5,
}

# First rule that yields enough heads is applied. The preregistered
# all-stratum intersection is kept as the top of the ladder.
FREEZE_RULES: tuple[tuple[str, str], ...] = (
    ("all_replicated_strata", "every replicated stratum is screened"),
    ("majority_all_strata", "strict majority of all replicated strata are screened"),
    (
        "majority_reference_geometry",
        "strict majority of reference-geometry strata are screened",
    ),
    ("any_reference_geometry", "at least one reference-geometry stratum is screened"),
)


def coefficient_key(row: dict) -> tuple[float, float, float, float]:
    return tuple(round(float(row[name]), 10) for name in COEFFICIENT_KEYS)


def geometry_tuple(row: dict) -> tuple[int, int, int]:
    return int(float(row["d"])), int(float(row["m"])), int(float(row["n"]))


def group_by_coefficient(rows: list[dict]) -> dict[tuple, list[dict]]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[coefficient_key(row)].append(row)
    return grouped


def reference_rows(values: list[dict]) -> list[dict]:
    geoms = {geometry_tuple(row) for row in values}
    target = REFERENCE_GEOMETRY if REFERENCE_GEOMETRY in geoms else min(geoms)
    return [row for row in values if geometry_tuple(row) == target]


def screened_count(values: list[dict]) -> int:
    return sum(row["label"] == SCREENED for row in values)


def screened_fraction(values: list[dict]) -> float:
    if not values:
        return 0.0
    return screened_count(values) / len(values)


def has_transfer_geometry(rows: list[dict]) -> bool:
    return len({geometry_tuple(row) for row in rows}) > 1


def passes_freeze_rule(values: list[dict], rule: str) -> bool:
    if rule == "all_replicated_strata":
        return bool(values) and all(row["label"] == SCREENED for row in values)
    if rule == "majority_all_strata":
        return screened_fraction(values) > 0.5
    if rule == "majority_reference_geometry":
        return screened_fraction(reference_rows(values)) > 0.5
    if rule == "any_reference_geometry":
        return screened_count(reference_rows(values)) >= 1
    raise ValueError(f"unknown freeze rule: {rule}")


def freeze_rule_counts(grouped: dict[tuple, list[dict]]) -> dict[str, int]:
    return {
        name: sum(passes_freeze_rule(values, name) for values in grouped.values())
        for name, _ in FREEZE_RULES
    }


def select_even_coverage(candidates: list[dict], heads: int) -> list[dict]:
    ordered = sorted(candidates, key=lambda row: (float(row["alpha"]), float(row["phi"])))
    if len(ordered) == 1:
        return ordered[:heads]
    import numpy as np

    indices = np.linspace(0, len(ordered) - 1, heads, dtype=int)
    return [ordered[int(index)] for index in indices]


def point_payload(row: dict, values: list[dict]) -> dict:
    ref = reference_rows(values)
    return {
        key: float(row[key]) for key in (*COEFFICIENT_KEYS, "phi")
    } | {
        "n_strata": len(values),
        "n_screened": screened_count(values),
        "n_reference_strata": len(ref),
        "n_reference_screened": screened_count(ref),
    }


def preregistered_overlap_label(values: list[dict]) -> str:
    labels = {row["label"] for row in values}
    label = next((name for name in PATHOLOGY_PRECEDENCE if name in labels), None)
    if label is not None:
        return label
    if labels == {SCREENED}:
        return SCREENED
    return "boundary_uncertain"


def majority_reference_label(values: list[dict]) -> str:
    counts = Counter(row["label"] for row in reference_rows(values))
    return min(counts, key=lambda lab: (-counts[lab], LABEL_TIE_RANK.get(lab, 99)))


def infer_label_rule(rows: list[dict]) -> str:
    return "majority_reference" if has_transfer_geometry(rows) else "preregistered"


def assign_coefficient_label(values: list[dict], rule: str) -> str:
    if rule == "preregistered":
        return preregistered_overlap_label(values)
    if rule == "majority_reference":
        return majority_reference_label(values)
    raise ValueError(f"unknown label rule: {rule}")


def labeled_representatives(rows: list[dict], rule: str, *, atlas: str) -> dict[str, list[dict]]:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for values in group_by_coefficient(rows).values():
        representative = dict(values[0])
        representative["label"] = assign_coefficient_label(values, rule)
        representative["atlas"] = atlas
        representative["label_rule"] = rule
        by_label[representative["label"]].append(representative)
    return by_label
