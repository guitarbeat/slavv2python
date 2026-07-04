"""Textual run operations console for structured SLAVV runs."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Footer, Header, ProgressBar, Static

from slavv_python.interface.cli.monitor_service import (
    EnergyProgress,
    RunMonitorView,
    build_stage_rows,
    compute_energy_progress,
    format_duration,
    format_energy_progress_line,
    live_overall_progress,
    load_run_monitor_view,
    status_style,
    tail_log_lines,
)

if TYPE_CHECKING:
    from textual.timer import Timer

_MIN_POLL_SECONDS = 1.0
_MAX_POLL_SECONDS = 60.0


class SLAVVPipelineApp(App[None]):
    """Textual TUI for monitoring SLAVV pipeline and parity runs."""

    CSS = """
    #body {
        layout: grid;
        grid-size: 2;
        grid-columns: 1fr 2fr;
        padding: 1;
        height: 1fr;
    }
    #left, #right {
        border: round $accent;
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
        text-style: bold;
    }
    .subtitle {
        text-style: bold;
        color: $accent;
        margin-top: 1;
    }
    #status_line {
        margin-top: 1;
    }
    #log_scroll {
        height: 1fr;
        border: round $primary;
        margin-top: 1;
        padding: 0 1;
    }
    """

    TITLE = "SLAVV Run Monitor"
    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("r", "refresh", "Refresh"),
        ("p", "toggle_pause", "Pause"),
        ("f", "faster", "Faster"),
        ("s", "slower", "Slower"),
        ("q", "quit", "Quit"),
    ]

    current_stage: reactive[str] = reactive("Initializing")
    progress_val: reactive[float] = reactive(0.0)

    def __init__(self, run_dir: str | Path | None = None, *, poll_seconds: float = 5.0) -> None:
        super().__init__()
        self.run_dir = Path(run_dir).expanduser().resolve() if run_dir else Path.cwd()
        self.poll_seconds = poll_seconds
        self.latest_view: RunMonitorView | None = None
        self._paused = False
        self._timer: Timer | None = None

    def compose(self) -> ComposeResult:
        """Compose the widget tree."""
        yield Header(show_clock=True)
        with Container(id="body"):
            with Vertical(id="left"):
                yield Static("SLAVV Run Operations", classes="title")
                yield Static("", id="status_line")
                yield Static("", id="meta_line")
                yield Static("Current Stage: Initializing", id="stage_label")
                yield Static("Overall", classes="subtitle")
                yield ProgressBar(total=100, show_percentage=True, id="progress_bar")
                yield Static("", id="energy_label")
                yield ProgressBar(total=100, show_percentage=True, id="energy_bar")
                yield Static("", id="proof_line")
            with Vertical(id="right"):
                yield Static("Stages", classes="subtitle")
                yield Static("", id="stages_panel")
                yield Static("Log tail", classes="subtitle", id="log_title")
                with VerticalScroll(id="log_scroll"):
                    yield Static("", id="log_panel")
        yield Footer()

    def on_mount(self) -> None:
        """Start polling once the app is mounted."""
        with contextlib.suppress(Exception):
            self.query_one("#energy_bar", ProgressBar).display = False
            self.query_one("#energy_label", Static).display = False
        self.refresh_view()
        self._timer = self.set_interval(self.poll_seconds, self.refresh_view)

    def action_refresh(self) -> None:
        """Refresh immediately from disk."""
        self.refresh_view()

    def action_toggle_pause(self) -> None:
        """Pause or resume the auto-refresh timer."""
        self._paused = not self._paused
        if self._timer is not None:
            self._timer.pause() if self._paused else self._timer.resume()
        self._update_subtitle()

    def action_faster(self) -> None:
        """Decrease the polling interval (refresh more often)."""
        self._set_poll_seconds(self.poll_seconds - 1.0)

    def action_slower(self) -> None:
        """Increase the polling interval (refresh less often)."""
        self._set_poll_seconds(self.poll_seconds + 1.0)

    def _set_poll_seconds(self, seconds: float) -> None:
        self.poll_seconds = max(_MIN_POLL_SECONDS, min(_MAX_POLL_SECONDS, seconds))
        if self._timer is not None:
            self._timer.stop()
        self._timer = self.set_interval(self.poll_seconds, self.refresh_view)
        if self._paused and self._timer is not None:
            self._timer.pause()
        self._update_subtitle()

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
                self.query_one("#log_panel", Static).update(Text(log_msg))

    def refresh_view(self) -> None:
        """Load the shared monitor view and render it into the TUI."""
        view = load_run_monitor_view(self.run_dir)
        self.latest_view = view
        snapshot = view.snapshot
        energy = compute_energy_progress(view)
        if snapshot is not None:
            self.current_stage = snapshot.current_stage or view.effective_status
            overall = snapshot.overall_progress
            if energy is not None:
                overall = max(overall, live_overall_progress(snapshot, energy))
            self.progress_val = max(0.0, min(100.0, overall * 100.0))
        else:
            self.current_stage = view.effective_status
            self.progress_val = 0.0

        self._update_energy_bar(energy)
        self._render_panels(view, energy)
        self._update_subtitle()

    def _update_energy_bar(self, energy: EnergyProgress | None) -> None:
        """Show a determinate energy-chunk bar while the energy stage is active."""
        with contextlib.suppress(Exception):
            bar = self.query_one("#energy_bar", ProgressBar)
            label = self.query_one("#energy_label", Static)
            if energy is None:
                bar.display = False
                label.display = False
                return
            bar.display = True
            label.display = True
            bar.total = float(energy.units_total)
            bar.progress = float(energy.live_units_completed)
            label.update(format_energy_progress_line(energy))

    def _render_panels(self, view: RunMonitorView, energy: EnergyProgress | None) -> None:
        with contextlib.suppress(Exception):
            self.query_one("#status_line", Static).update(self._status_text(view))
            self.query_one("#meta_line", Static).update(self._meta_text(view))
            self.query_one("#proof_line", Static).update(self._proof_text(view))
            self.query_one("#stages_panel", Static).update(self._stages_table(view))
            name, lines = tail_log_lines(view)
            title = f"Log tail — {name}" if name else "Log tail"
            self.query_one("#log_title", Static).update(title)
            body = "\n".join(lines) if lines else "(no run log found)"
            self.query_one("#log_panel", Static).update(Text(body, no_wrap=False))

    def _update_subtitle(self) -> None:
        state = " · PAUSED" if self._paused else ""
        self.sub_title = f"{self.run_dir.name} · refresh {self.poll_seconds:.0f}s{state}"

    def _status_text(self, view: RunMonitorView) -> Text:
        style = status_style(view.effective_status)
        text = Text()
        text.append("● ", style=style)
        text.append(view.effective_status.upper(), style=f"bold {style}")
        if view.status_reason:
            text.append(f"  {view.status_reason}", style="dim")
        return text

    def _meta_text(self, view: RunMonitorView) -> Text:
        snapshot = view.snapshot
        if snapshot is None:
            return Text("No run snapshot found.", style="yellow")
        parts = [f"Run {snapshot.run_id}", f"target {snapshot.target_stage}"]
        if snapshot.elapsed_seconds:
            parts.append(f"elapsed {format_duration(snapshot.elapsed_seconds)}")
        if snapshot.eta_seconds is not None:
            parts.append(f"ETA {format_duration(snapshot.eta_seconds)}")
        return Text(" · ".join(parts), style="dim")

    def _proof_text(self, view: RunMonitorView) -> Text:
        if not view.proof_statuses:
            return Text("")
        proof = view.proof_statuses[0]
        if proof.passed is True:
            return Text("Proof: passed", style="bold green")
        if proof.passed is False:
            failure = f"{proof.first_failing_stage}.{proof.first_failing_field}".strip(".")
            return Text(f"Proof: FAILED {failure}", style="bold red")
        return Text("Proof: (unreadable)", style="yellow")

    def _stages_table(self, view: RunMonitorView) -> Table | Text:
        snapshot = view.snapshot
        if snapshot is None:
            return Text("Waiting for run snapshot…", style="yellow")
        table = Table(expand=True, show_edge=False, pad_edge=False, box=None)
        table.add_column("Stage", style="bold")
        table.add_column("Status")
        table.add_column("Progress", justify="right")
        table.add_column("Units", justify="right")
        table.add_column("Detail", ratio=2, overflow="ellipsis", no_wrap=True)
        current = snapshot.current_stage
        for row in build_stage_rows(snapshot):
            marker = "▶ " if row.name == current else "  "
            table.add_row(
                Text(marker + row.name),
                Text(row.status, style=status_style(row.status)),
                f"{row.progress * 100:.0f}%",
                row.units_label,
                Text(row.detail or "", overflow="ellipsis", no_wrap=True),
            )
        return table


__all__ = ["SLAVVPipelineApp"]
