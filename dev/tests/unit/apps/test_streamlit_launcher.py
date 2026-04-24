from __future__ import annotations

import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

from source.apps import streamlit_launcher


def test_main_reports_missing_streamlit(monkeypatch, capsys):
    monkeypatch.setattr(streamlit_launcher.util, "find_spec", lambda name: None)

    exit_code = streamlit_launcher.main([])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert 'pip install -e ".[app]"' in captured.err


def test_main_delegates_to_streamlit_cli(monkeypatch):
    @contextmanager
    def fake_resolve_web_app_path():
        yield Path("C:/tmp/web_app.py")

    commands: list[list[str]] = []

    monkeypatch.delenv("PYTHONIOENCODING", raising=False)
    monkeypatch.delenv("PYTHONUTF8", raising=False)

    def fake_run(
        command: list[str], check: bool, env: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        assert not check
        assert env["PYTHONIOENCODING"] == "utf-8"
        assert env["PYTHONUTF8"] == "1"
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(streamlit_launcher.util, "find_spec", lambda name: object())
    monkeypatch.setattr(streamlit_launcher, "_resolve_web_app_path", fake_resolve_web_app_path)
    monkeypatch.setattr(streamlit_launcher.subprocess, "run", fake_run)

    exit_code = streamlit_launcher.main(
        ["--server.headless=true", "--browser.gatherUsageStats=false"]
    )

    assert exit_code == 0
    assert commands == [
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(Path("C:/tmp/web_app.py")),
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
        ]
    ]


def test_build_env_overrides_incompatible_console_encoding(monkeypatch):
    monkeypatch.setenv("PYTHONIOENCODING", "cp1252")
    monkeypatch.setenv("PYTHONUTF8", "0")

    env = streamlit_launcher._build_env()

    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["PYTHONUTF8"] == "1"


