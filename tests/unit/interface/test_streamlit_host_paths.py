from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from slavv_python.interface.streamlit.services import host_paths

if TYPE_CHECKING:
    from pathlib import Path


def test_file_manager_action_label_by_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(host_paths.sys, "platform", "darwin")
    assert host_paths.file_manager_action_label() == "Reveal in Finder"
    monkeypatch.setattr(host_paths.sys, "platform", "win32")
    assert host_paths.file_manager_action_label() == "Open in Explorer"
    monkeypatch.setattr(host_paths.sys, "platform", "linux")
    assert host_paths.file_manager_action_label() is None


def test_reveal_run_directory_uses_open_r_on_macos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "crop_M_r2024b"
    run_dir.mkdir()
    calls: list[list[str]] = []

    def fake_popen(args, **kwargs):
        calls.append(list(args))

    monkeypatch.setattr(host_paths.sys, "platform", "darwin")
    monkeypatch.setattr(host_paths.subprocess, "Popen", fake_popen)
    host_paths.reveal_run_directory(run_dir)
    assert calls == [["open", "-R", str(run_dir.resolve())]]


def test_reveal_run_directory_uses_explorer_on_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "crop_M_r2024b"
    run_dir.mkdir()
    calls: list[list[str]] = []

    def fake_popen(args, **kwargs):
        calls.append(list(args))

    monkeypatch.setattr(host_paths.sys, "platform", "win32")
    monkeypatch.setattr(host_paths.subprocess, "Popen", fake_popen)
    host_paths.reveal_run_directory(run_dir)
    assert calls == [["explorer.exe", str(run_dir.resolve())]]


def test_reveal_run_directory_rejects_unsupported_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "crop_M_r2024b"
    run_dir.mkdir()
    monkeypatch.setattr(host_paths.sys, "platform", "linux")
    with pytest.raises(OSError, match="not supported"):
        host_paths.reveal_run_directory(run_dir)
