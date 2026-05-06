"""Ink-style live pipeline tracking dashboard using Textual."""

from __future__ import annotations

import contextlib
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, ProgressBar, RichLog, Static


class SLAVVPipelineApp(App[None]):
    """Textual TUI for monitoring SLAVV pipeline execution in real-time."""

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
    RichLog {
        border: solid $primary;
        background: $boost;
        height: 100%;
    }
    .title {
        text-align: center;
        background: $accent;
        color: $text;
        font-weight: bold;
        margin-bottom: 1;
    }
    #stage_label {
        font-size: 110%;
        margin-top: 1;
        margin-bottom: 1;
        color: $text;
    }
    """

    TITLE = "SLAVV Pipeline Monitor"
    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [("q", "quit", "Quit SLAVV")]

    current_stage: reactive[str] = reactive("Initializing")
    progress_val: reactive[float] = reactive(0.0)

    def compose(self) -> ComposeResult:
        """Compose the widget tree."""
        yield Header(show_clock=True)
        with Container():
            with Vertical():
                yield Static("🩸 SLAVV Pipeline State", classes="title")
                yield Static("Current Stage: Initializing", id="stage_label")
                yield ProgressBar(total=100, show_percentage=True, id="progress_bar")
            yield RichLog(id="log_view", max_lines=1000)
        yield Footer()

    def watch_current_stage(self, stage: str) -> None:
        """Called when current_stage changes."""
        with contextlib.suppress(Exception):
            label = self.query_one("#stage_label", Static)
            label.update(f"Current Stage: [bold]{stage}[/bold]")
            log = self.query_one("#log_view", RichLog)
            log.write(f"[INFO] Transitioned to pipeline stage: {stage}")

    def watch_progress_val(self, progress: float) -> None:
        """Called when progress_val changes."""
        with contextlib.suppress(Exception):
            bar = self.query_one("#progress_bar", ProgressBar)
            bar.progress = progress

    def update_state(self, stage: str, progress: float, log_msg: str | None = None) -> None:
        """Updates the TUI widgets dynamically from pipeline threads."""
        self.current_stage = stage
        self.progress_val = progress
        if log_msg:
            with contextlib.suppress(Exception):
                self.query_one("#log_view", RichLog).write(log_msg)
