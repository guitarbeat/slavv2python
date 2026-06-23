"""Textual run operations console for structured SLAVV runs."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, ProgressBar, Static

from slavv_python.interface.cli.monitor_service import (
    RunMonitorView,
    load_run_monitor_view,
    render_monitor_lines,
)


class SLAVVPipelineApp(App[None]):
    """Textual TUI for monitoring SLAVV pipeline and parity runs."""

    CSS = """
    Container {
        layout: grid;
        grid-size: 2;
        grid-columns: 1fr 2fr;
        padding: 1;
    }
    Vertical {
        border: solid $accent;
        padding: 1;
        height: 100%;
    }
    Static {
        text-wrap: wrap;
    }
    .title {
        text-align: center;
        background: $accent;
        color: $text;
        font-weight: bold;
        margin-bottom: 1;
    }
    #detail_panel {
        border: solid $primary;
        padding: 1;
        height: 100%;
    }
    """

    TITLE = "SLAVV Run Monitor"
    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
    ]

    current_stage: reactive[str] = reactive("Initializing")
    progress_val: reactive[float] = reactive(0.0)

    def __init__(self, run_dir: str | Path | None = None, *, poll_seconds: float = 5.0) -> None:
        super().__init__()
        self.run_dir = Path(run_dir).expanduser().resolve() if run_dir else Path.cwd()
        self.poll_seconds = poll_seconds
        self.latest_view: RunMonitorView | None = None

    def compose(self) -> ComposeResult:
        """Compose the widget tree."""
        yield Header(show_clock=True)
        with Container():
            with Vertical():
                yield Static("SLAVV Run Operations", classes="title")
                yield Static("Current Stage: Initializing", id="stage_label")
                yield ProgressBar(total=100, show_percentage=True, id="progress_bar")
                yield Static("", id="summary_panel")
            yield Static("", id="detail_panel")
        yield Footer()

    def on_mount(self) -> None:
        """Start polling once the app is mounted."""
        self.refresh_view()
        self.set_interval(self.poll_seconds, self.refresh_view)

    def action_refresh(self) -> None:
        """Refresh immediately from disk."""
        self.refresh_view()

    def watch_current_stage(self, stage: str) -> None:
        """Called when current_stage changes."""
        with contextlib.suppress(Exception):
            self.query_one("#stage_label", Static).update(f"Current Stage: {stage}")

    def watch_progress_val(self, progress: float) -> None:
        """Called when progress_val changes."""
        with contextlib.suppress(Exception):
            self.query_one("#progress_bar", ProgressBar).progress = progress

    def update_state(self, stage: str, progress: float, log_msg: str | None = None) -> None:
        """Update core reactive values, preserving the old test-facing interface."""
        self.current_stage = stage
        self.progress_val = progress
        if log_msg:
            with contextlib.suppress(Exception):
                self.query_one("#detail_panel", Static).update(log_msg)

    def refresh_view(self) -> None:
        """Load the shared monitor view and render it into the TUI."""
        view = load_run_monitor_view(self.run_dir)
        self.latest_view = view
        snapshot = view.snapshot
        if snapshot is not None:
            self.current_stage = snapshot.current_stage or view.effective_status
            self.progress_val = max(0.0, min(100.0, snapshot.overall_progress * 100.0))
        else:
            self.current_stage = view.effective_status
            self.progress_val = 0.0

        summary = self._summary_text(view)
        detail = "\n".join(render_monitor_lines(view))
        with contextlib.suppress(Exception):
            self.query_one("#summary_panel", Static).update(summary)
            self.query_one("#detail_panel", Static).update(detail)

    def _summary_text(self, view: RunMonitorView) -> str:
        snapshot = view.snapshot
        lines = [
            f"Run: {view.run_dir.name}",
            f"Status: {view.effective_status}",
            f"Reason: {view.status_reason}",
        ]
        if snapshot is not None:
            lines.extend(
                [
                    f"Run ID: {snapshot.run_id}",
                    f"Target: {snapshot.target_stage}",
                    f"Detail: {snapshot.current_detail or '(none)'}",
                ]
            )
        if view.pid_statuses:
            pid = view.pid_statuses[0]
            lines.append(f"PID: {pid.pid or 'unknown'} ({pid.state})")
        if view.proof_statuses:
            proof = view.proof_statuses[0]
            if proof.passed is True:
                lines.append("Proof: passed")
            elif proof.passed is False:
                lines.append(
                    f"Proof: failed {proof.first_failing_stage}.{proof.first_failing_field}"
                )
        return "\n".join(lines)


__all__ = ["SLAVVPipelineApp"]
