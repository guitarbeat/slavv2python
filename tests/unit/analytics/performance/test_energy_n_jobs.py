"""CI-safe tests for Energy n_jobs auto-size (CPU reserve + memory guard)."""

from __future__ import annotations

import pytest

from slavv_python.analytics.performance.energy_n_jobs import (
    parse_n_jobs_cli_value,
    recommend_energy_n_jobs,
    recommend_energy_n_jobs_from_host,
    resolve_cli_n_jobs,
)

_MIB = 1024 * 1024
_GIB = 1024 * _MIB
_PER_WORKER = 512 * _MIB


def test_cpu_reserve_caps_eight_core_host() -> None:
    assert (
        recommend_energy_n_jobs(
            cpu_count=8,
            available_bytes=16 * _GIB,
            per_worker_bytes=_PER_WORKER,
        )
        == 6
    )


def test_memory_guard_wins_when_ram_is_tight() -> None:
    assert (
        recommend_energy_n_jobs(
            cpu_count=8,
            available_bytes=400 * _MIB,
            per_worker_bytes=_PER_WORKER,
        )
        == 1
    )


def test_two_core_host_stays_serial() -> None:
    assert (
        recommend_energy_n_jobs(
            cpu_count=2,
            available_bytes=16 * _GIB,
            per_worker_bytes=_PER_WORKER,
        )
        == 1
    )


def test_memory_cap_below_cpu_cap() -> None:
    assert (
        recommend_energy_n_jobs(
            cpu_count=16,
            available_bytes=3 * _PER_WORKER,
            per_worker_bytes=_PER_WORKER,
        )
        == 3
    )


def test_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="cpu_count"):
        recommend_energy_n_jobs(cpu_count=0, available_bytes=_GIB, per_worker_bytes=_PER_WORKER)
    with pytest.raises(ValueError, match="per_worker_bytes"):
        recommend_energy_n_jobs(cpu_count=8, available_bytes=_GIB, per_worker_bytes=0)
    with pytest.raises(ValueError, match="available_bytes"):
        recommend_energy_n_jobs(cpu_count=8, available_bytes=-1, per_worker_bytes=_PER_WORKER)


def test_parse_n_jobs_cli_value() -> None:
    assert parse_n_jobs_cli_value("auto") == "auto"
    assert parse_n_jobs_cli_value("6") == 6
    with pytest.raises(ValueError, match="auto"):
        parse_n_jobs_cli_value("0")
    with pytest.raises(ValueError, match="auto"):
        parse_n_jobs_cli_value("nope")


def test_resolve_cli_n_jobs_leaves_explicit_int() -> None:
    assert resolve_cli_n_jobs(None) is None
    assert resolve_cli_n_jobs(6) == 6
    assert resolve_cli_n_jobs("auto", recommend=lambda: 4) == 4


def test_from_host_applies_memory_safety_fraction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "slavv_python.analytics.performance.energy_n_jobs.os.cpu_count",
        lambda: 8,
    )

    class _Mem:
        available = 10 * _PER_WORKER

    monkeypatch.setattr(
        "slavv_python.analytics.performance.energy_n_jobs.psutil.virtual_memory",
        lambda: _Mem(),
    )
    assert recommend_energy_n_jobs_from_host(memory_safety_fraction=0.5) == 5
