"""Lightweight stderr progress reporting (stdlib only)."""

from __future__ import annotations

import sys
import time
from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")


def stage(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def progress(
    iterable: Iterable[T],
    *,
    desc: str,
    total: int | None = None,
) -> Iterator[T]:
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = None
    started = time.perf_counter()
    update_every = max(1, (total or 100) // 100)
    for index, item in enumerate(iterable):
        if total is not None:
            step = index + 1
            if step == 1 or step == total or step % update_every == 0:
                elapsed = time.perf_counter() - started
                rate = step / elapsed if elapsed > 0 else 0.0
                eta = (total - step) / rate if rate > 0 else 0.0
                bar_width = 24
                filled = int(bar_width * step / total)
                bar = "#" * filled + "-" * (bar_width - filled)
                print(
                    f"{desc} |{bar}| {step}/{total} "
                    f"({100 * step / total:5.1f}%) elapsed {elapsed:6.0f}s eta {eta:6.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
        yield item
    if total is not None:
        elapsed = time.perf_counter() - started
        stage(f"{desc}: done in {elapsed:.1f}s")
