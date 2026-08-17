"""Bit-preserving Energy n_jobs auto-size (CPU reserve + memory guard).

Does not change the exact-route default (serial ``n_jobs=1``). Operators opt in
with ``--n-jobs auto`` on resume/launch. Explicit integers are unchanged.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Literal

import psutil

from slavv_python.analytics.parity.constants import DEFAULT_MEMORY_SAFETY_FRACTION

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

N_JOBS_AUTO: Literal["auto"] = "auto"
RESERVE_CORES = 2
DEFAULT_PER_WORKER_BYTES = 512 * 1024 * 1024
MIN_JOBS = 1

_N_JOBS_PARSE_ERROR = "n_jobs must be an integer >= 1 or 'auto'"


def recommend_energy_n_jobs(
    *,
    cpu_count: int,
    available_bytes: int,
    per_worker_bytes: int = DEFAULT_PER_WORKER_BYTES,
    reserve_cores: int = RESERVE_CORES,
    min_jobs: int = MIN_JOBS,
) -> int:
    """Return a conservative worker count from CPU and RAM budgets."""
    if cpu_count < 1:
        raise ValueError("cpu_count must be >= 1")
    if available_bytes < 0:
        raise ValueError("available_bytes must be >= 0")
    if per_worker_bytes < 1:
        raise ValueError("per_worker_bytes must be >= 1")
    if reserve_cores < 0:
        raise ValueError("reserve_cores must be >= 0")
    if min_jobs < 1:
        raise ValueError("min_jobs must be >= 1")

    cpu_cap = max(min_jobs, int(cpu_count) - int(reserve_cores))
    mem_cap = max(min_jobs, int(available_bytes) // int(per_worker_bytes))
    return min(cpu_cap, mem_cap)


def recommend_energy_n_jobs_from_host(
    *,
    memory_safety_fraction: float = DEFAULT_MEMORY_SAFETY_FRACTION,
    per_worker_bytes: int = DEFAULT_PER_WORKER_BYTES,
    reserve_cores: int = RESERVE_CORES,
) -> int:
    """Probe this host and apply the same CPU/RAM guard as the pure helper."""
    cpu_count = os.cpu_count() or 1
    available = int(psutil.virtual_memory().available * float(memory_safety_fraction))
    n_jobs = recommend_energy_n_jobs(
        cpu_count=cpu_count,
        available_bytes=available,
        per_worker_bytes=per_worker_bytes,
        reserve_cores=reserve_cores,
    )
    logger.info(
        "Energy n_jobs auto-size selected %s workers (cpu_count=%s, available_bytes=%s)",
        n_jobs,
        cpu_count,
        available,
    )
    return n_jobs


def parse_n_jobs_cli_value(value: str) -> int | Literal["auto"]:
    """Parse ``--n-jobs`` as an integer >= 1 or the token ``auto``."""
    text = str(value).strip().lower()
    if text == N_JOBS_AUTO:
        return N_JOBS_AUTO
    try:
        parsed = int(text)
    except ValueError as exc:
        raise ValueError(_N_JOBS_PARSE_ERROR) from exc
    if parsed < 1:
        raise ValueError(_N_JOBS_PARSE_ERROR)
    return parsed


def _auto_n_jobs(
    *,
    recommend: Callable[[], int] | None,
    memory_safety_fraction: float | None,
) -> int:
    if recommend is not None:
        return int(recommend())
    if memory_safety_fraction is None:
        return int(recommend_energy_n_jobs_from_host())
    return int(recommend_energy_n_jobs_from_host(memory_safety_fraction=memory_safety_fraction))


def resolve_cli_n_jobs(
    raw: object,
    *,
    recommend: Callable[[], int] | None = None,
    memory_safety_fraction: float | None = None,
) -> int | None:
    """Turn a parsed CLI value into an integer override, or None if omitted."""
    if raw is None:
        return None
    parsed: int | Literal["auto"]
    if raw == N_JOBS_AUTO:
        parsed = N_JOBS_AUTO
    else:
        parsed = parse_n_jobs_cli_value(str(raw))
    if parsed == N_JOBS_AUTO:
        return _auto_n_jobs(
            recommend=recommend,
            memory_safety_fraction=memory_safety_fraction,
        )
    return parsed
