"""
FSI Monitor — Main Textual application.

Layout:
┌───────────────────────────────────────────────────────────────┐
│  ● path/to/csv  │  t = 0.0195 s                              │
├───────────┬──────────────┬────────────────────────────────────┤
│ ROTATION  │ FORCES       │ PERFORMANCE                        │
│ compact   │ compact      │ compact                            │
├───────────┴──────────────┴────────────────────────────────────┤
│                                                               │
│              PLOT (one metric, full history)                   │
│                                                               │
│  Omega [rad/s]  1/20  │  Y:lin  │  t₀=0  │  ←→  Ctrl+P      │
├───────────────────────────────────────────────────────────────┤
│  q Quit  l Lin/Log  ← Prev  → Next  ? Help                   │
└───────────────────────────────────────────────────────────────┘

Key bindings:
  q / ctrl+c    quit
  l             toggle log scale on plot
  left / right  previous / next metric
  ?             toggle help overlay

All other configuration via Ctrl+P command palette:
  - Select any metric by name
  - Set / reset start time
  - Toggle log scale
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

from textual.app import App, ComposeResult
from textual.command import DiscoveryHit, Hit, Hits, Provider
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Footer, Input, Static

from fem_shell.fsi_monitor.data_provider import FSIDataProvider, PLOTTABLE_METRICS
from fem_shell.fsi_monitor.tui.panels.forces import ForcesPanel
from fem_shell.fsi_monitor.tui.panels.plot_panel import PlotPanel
from fem_shell.fsi_monitor.tui.panels.power import PowerPanel
from fem_shell.fsi_monitor.tui.panels.rotation import RotationPanel


# ---------------------------------------------------------------------------
# Header status widget
# ---------------------------------------------------------------------------


class SimulationHeader(Static):
    """Top status bar: CSV path, current time, update pulse."""

    DEFAULT_CSS = """
    SimulationHeader {
        height: 1;
        width: 1fr;
        padding: 0 2;
        background: $primary-darken-3;
        color: $text;
        text-style: bold;
    }
    """

    _content: reactive[str] = reactive("")

    def __init__(self, provider: FSIDataProvider, csv_path: Path, **kwargs):
        super().__init__(**kwargs)
        self._provider = provider
        self._csv_path = csv_path
        self._pulse = False
        self._last_version: int = -1
        self._spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._spinner_idx: int = 0

    def on_mount(self) -> None:
        self.set_interval(0.5, self._tick)

    def _tick(self) -> None:
        if not self._provider.is_ready:
            self._spinner_idx = (self._spinner_idx + 1) % len(self._spinner_frames)
            spin = self._spinner_frames[self._spinner_idx]
            status = self._provider.load_status
            path_str = self._csv_path.name
            self._content = f"[bold yellow]{spin}[/] {path_str}  │  [yellow]{status}[/]"
            return

        v = self._provider.data_version
        updated = v != self._last_version
        self._last_version = v
        self._pulse = updated

        row = self._provider.get_current()
        time_s = row.get("Time [s]", "—") if row else "—"
        time_str = f"{time_s:.4f} s" if isinstance(time_s, float) else str(time_s)

        indicator = "[bold green]●[/]" if self._pulse else "[dim]○[/]"
        path_str = self._csv_path.name
        self._content = f"{indicator} {path_str}  │  [bold cyan]t = {time_str}[/]"

    def render(self) -> str:
        return self._content or "[dim]○ Initializing…[/]"


# ---------------------------------------------------------------------------
# Help overlay
# ---------------------------------------------------------------------------

HELP_TEXT = """\
[bold cyan]FSI Monitor[/]

  [bold]q[/] / [bold]Ctrl+C[/]    Quit
  [bold]l[/]              Toggle Y-axis: linear ↔ log
  [bold]\u2190[/] / [bold]\u2192[/]         Previous / next metric
  [bold]Ctrl+P[/]         Command palette (select metric, set start time)
  [bold]?[/]              Toggle this help

[dim]Reads rotor_performance.csv independently of solver.
Loads full time-series history.[/]
"""


class HelpOverlay(Static):
    """Full-screen help overlay, toggled with '?'."""

    DEFAULT_CSS = """
    HelpOverlay {
        display: none;
        layer: overlay;
        width: 54;
        height: auto;
        padding: 1 3;
        margin: 2 4;
        border: heavy $primary;
        background: $surface;
    }
    HelpOverlay.-visible {
        display: block;
    }
    """

    def render(self) -> str:
        return HELP_TEXT


# ---------------------------------------------------------------------------
# Start-time input modal
# ---------------------------------------------------------------------------


class StartTimeScreen(ModalScreen[float | None]):
    """Modal dialog to set the plot start-time filter."""

    DEFAULT_CSS = """
    StartTimeScreen {
        align: center middle;
    }
    #st-container {
        width: 50;
        height: auto;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }
    #st-label {
        height: 1;
        margin-bottom: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="st-container"):
            yield Static("[bold]Start time (seconds):[/]", id="st-label")
            yield Input(placeholder="e.g. 0.019", id="st-input")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            self.dismiss(None)
            return
        try:
            self.dismiss(float(text))
        except ValueError:
            self.dismiss(None)

    def key_escape(self) -> None:
        self.dismiss(None)


# ---------------------------------------------------------------------------
# Command palette provider
# ---------------------------------------------------------------------------


class FSICommandProvider(Provider):
    """Commands for Ctrl+P: metric selection, start time, log scale."""

    async def discover(self) -> Hits:
        app = self.app
        assert isinstance(app, _FSIApp)

        yield DiscoveryHit(
            "Toggle Log/Linear Scale",
            app.action_toggle_log,
            help="Switch Y-axis between linear and logarithmic",
        )
        yield DiscoveryHit(
            "Set Start Time\u2026",
            app.action_set_start_time,
            help="Filter plot to show data from a specific time",
        )
        yield DiscoveryHit(
            "Reset Start Time",
            app.action_reset_start_time,
            help="Show data from the beginning",
        )

    async def search(self, query: str) -> Hits:
        app = self.app
        assert isinstance(app, _FSIApp)
        matcher = self.matcher(query)

        # Static commands
        commands = [
            ("Toggle Log/Linear Scale", app.action_toggle_log, "Switch Y-axis scale"),
            ("Set Start Time\u2026", app.action_set_start_time, "Filter from time"),
            ("Reset Start Time", app.action_reset_start_time, "Show all data"),
        ]
        for label, callback, help_text in commands:
            score = matcher.match(label)
            if score > 0:
                yield Hit(score, matcher.highlight(label), callback, help=help_text)

        # Dynamic metric commands
        plot = app.query_one("#plot-panel", PlotPanel)
        for name in plot._available_metrics():
            command = f"Plot: {name}"
            score = matcher.match(command)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(command),
                    partial(plot.set_metric, name),
                    help="Switch plot to this metric",
                )


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


class _FSIApp(App):
    """Textual application: FSI simulation dashboard."""

    TITLE = "FSI Monitor"

    COMMANDS = App.COMMANDS | {FSICommandProvider}

    CSS = """
    Screen {
        layers: base overlay;
        layout: vertical;
    }

    /* -- metric panels row (top) -- */
    #metrics-row {
        height: auto;
        max-height: 50%;
    }
    RotationPanel, ForcesPanel, PowerPanel {
        width: 1fr;
        height: auto;
        min-width: 20;
    }

    /* -- plot area (bottom, fills remaining space) -- */
    PlotPanel {
        height: 1fr;
        min-height: 8;
    }
    """

    BINDINGS = [
        ("q",              "quit",             "Quit"),
        ("l",              "toggle_log",       "Lin/Log"),
        ("left",           "prev_metric",      "Prev metric"),
        ("right",          "next_metric",      "Next metric"),
        ("question_mark",  "toggle_help",      "Help"),
    ]

    def __init__(self, provider: FSIDataProvider, csv_path: Path, **kwargs):
        super().__init__(**kwargs)
        self._provider = provider
        self._csv_path = csv_path
        self._help_visible = False

    def compose(self) -> ComposeResult:
        yield SimulationHeader(self._provider, self._csv_path, id="sim-header")

        with Horizontal(id="metrics-row"):
            yield RotationPanel(self._provider, id="rotation-panel")
            yield ForcesPanel(self._provider, id="forces-panel")
            yield PowerPanel(self._provider, id="power-panel")

        yield PlotPanel(self._provider, id="plot-panel")

        yield HelpOverlay(id="help-overlay")
        yield Footer()

    # ------------------------------------------------------------------
    # Keybinding actions
    # ------------------------------------------------------------------

    def action_quit(self) -> None:
        self._provider.shutdown()
        self.exit()

    def action_toggle_log(self) -> None:
        self.query_one("#plot-panel", PlotPanel).toggle_log_scale()

    def action_prev_metric(self) -> None:
        self.query_one("#plot-panel", PlotPanel).prev_metric()

    def action_next_metric(self) -> None:
        self.query_one("#plot-panel", PlotPanel).next_metric()

    def action_set_start_time(self) -> None:
        """Open modal to set plot start-time filter."""
        def _on_result(t: float | None) -> None:
            if t is not None:
                self.query_one("#plot-panel", PlotPanel).set_start_time(t)

        self.push_screen(StartTimeScreen(), callback=_on_result)

    def action_reset_start_time(self) -> None:
        self.query_one("#plot-panel", PlotPanel).set_start_time(None)

    def action_toggle_help(self) -> None:
        overlay = self.query_one("#help-overlay", HelpOverlay)
        self._help_visible = not self._help_visible
        if self._help_visible:
            overlay.add_class("-visible")
        else:
            overlay.remove_class("-visible")


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------


class FSIMonitorApp:
    """
    Public entry point for the FSI Monitor TUI.

    Usage::

        app = FSIMonitorApp(csv_file=Path("rotor_performance.csv"))
        app.run()
    """

    def __init__(
        self,
        csv_file: Path,
        refresh_interval: float = 0.5,
    ):
        self._csv_file = csv_file
        self._refresh_interval = refresh_interval

    def run(self) -> None:
        provider = FSIDataProvider(
            csv_path=self._csv_file,
            poll_interval=self._refresh_interval,
        )
        app = _FSIApp(provider=provider, csv_path=self._csv_file)
        try:
            app.run()
        finally:
            provider.shutdown()
