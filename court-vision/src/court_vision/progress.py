"""Shared Rich progress bar utilities."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from typing import Generator

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)


def create_progress() -> Progress:
    """Create a Rich Progress instance with the standard Court Vision columns."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )


def make_callback(
    progress: Progress | None,
    task_id: TaskID | None,
) -> Callable[[int, int], None] | None:
    """Build a progress_callback that drives a Rich task, or None if disabled."""
    if progress is None or task_id is None:
        return None

    def cb(current: int, total: int) -> None:
        progress.update(task_id, completed=current, total=total)

    return cb


@contextmanager
def pipeline_progress(enabled: bool = True) -> Generator[Progress | None, None, None]:
    """Context manager that yields a Progress instance (or None when disabled)."""
    if not enabled:
        yield None
        return
    progress = create_progress()
    progress.start()
    try:
        yield progress
    finally:
        progress.stop()
