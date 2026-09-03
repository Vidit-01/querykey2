"""Reproducible output, manifests, and resumable JSONL helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def ensure_output(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(
        destination.suffix + f".{os.getpid()}.{time.time_ns()}.tmp"
    )
    temporary.write_text(
        json.dumps(
            _sanitize_json(value),
            indent=2,
            sort_keys=True,
            default=_json_default,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(destination)


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    _sanitize_json(row),
                    sort_keys=True,
                    default=_json_default,
                    allow_nan=False,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {source}:{line_number}") from error
    return records


def completed_keys(path: str | Path, fields: tuple[str, ...]) -> set[tuple[Any, ...]]:
    return {tuple(row.get(field) for field in fields) for row in read_jsonl(path)}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_manifest(
    output: str | Path,
    *,
    experiment: str,
    arguments: dict[str, Any],
    inputs: Iterable[str | Path] = (),
) -> dict[str, Any]:
    manifest = {
        "experiment": experiment,
        "created_unix": time.time(),
        "arguments": arguments,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "git_commit": _git_commit(),
        "installed_packages": _installed_packages(),
        "inputs": {
            str(path): file_sha256(path)
            for path in inputs
            if path is not None and Path(path).is_file()
        },
        "failures": [],
    }
    output_path = Path(output)
    run_name = f"{experiment}_{int(manifest['created_unix'] * 1000)}_{os.getpid()}.json"
    write_json(output_path / "manifests" / run_name, manifest)
    write_json(output_path / "manifest.json", manifest)
    return manifest


def record_failure(output: str | Path, context: dict[str, Any], error: BaseException) -> None:
    path = Path(output) / "failures.jsonl"
    append_jsonl(
        path,
        [
            {
                "time_unix": time.time(),
                "context": context,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        ],
    )


def elapsed_record(started: float) -> dict[str, float]:
    return {"elapsed_seconds": time.perf_counter() - started}


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _installed_packages() -> list[str]:
    try:
        output = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=15,
        )
        return sorted(line for line in output.splitlines() if line.strip())
    except (OSError, subprocess.SubprocessError):
        return []


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _sanitize_json(value: Any):
    if isinstance(value, dict):
        return {str(key): _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, torch.Tensor):
        return _sanitize_json(value.detach().cpu().tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
