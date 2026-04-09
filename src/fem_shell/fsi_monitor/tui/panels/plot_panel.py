"""
Single-metric time-series plot panel for FSI Monitor.

Uses textual-plotext (PlotextPlot) for correct auto-sizing.
One metric at a time, full history, start-time filter, log toggle.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static
from textual_plotext import PlotextPlot

from fem_shell.fsi_monitor.data_provider import (
    DEFAULT_PLOT_METRIC,
    FSIDataProvider,
    PLOTTABLE_METRICS,
)


class PlotPanel(Widget):
    """Full-width single-metric time-series plot."""

    DEFAULT_CSS = """
    PlotPanel {
        height: 1fr;
        width: 1fr;
        layout: vertical;
    }
    PlotPanel PlotextPlot {
        width: 1fr;
        height: 1fr;
    }
    PlotPanel #status-bar {
        height: 1;
        width: 1fr;
        padding: 0 1;
        background: $surface-darken-1;
    }
    """

    log_scale: reactive[bool] = reactive(False)

    def __init__(self, provider: FSIDataProvider, **kwargs: Any):
        super().__init__(**kwargs)
        self._provider = provider
        self._last_version: int = -1
        self._metric_index: int = 0
        self._start_time: float | None = None
        self._current_metric: str = DEFAULT_PLOT_METRIC

        # Resolve initial index
        avail = self._available_metrics()
        if avail and self._current_metric in avail:
            self._metric_index = avail.index(self._current_metric)
        elif avail:
            self._metric_index = 0
            self._current_metric = avail[0]

    def compose(self) -> ComposeResult:
        yield PlotextPlot(id="chart")
        yield Static("", id="status-bar", markup=True)

    def on_mount(self) -> None:
        self.set_interval(1.0, self._tick)

    def _available_metrics(self) -> list[str]:
        avail = [m for m in PLOTTABLE_METRICS if self._provider.is_plottable(m)]
        return avail or PLOTTABLE_METRICS

    def _tick(self) -> None:
        v = self._provider.data_version
        if v == self._last_version and v != 0:
            return
        self._last_version = v
        self._redraw()

    def _redraw(self) -> None:
        chart = self.query_one("#chart", PlotextPlot)
        status = self.query_one("#status-bar", Static)

        p = chart.plt  # per-widget plotext instance (no global state)
        p.clear_data()
        p.clear_figure()
        p.theme("dark")
        p.xlabel("Time [s]")

        xs, ys = self._provider.get_history_xy(self._current_metric)

        # Apply start-time filter
        if self._start_time is not None and xs:
            pairs = [(x, y) for x, y in zip(xs, ys) if x >= self._start_time]
            if pairs:
                xs, ys = [p2[0] for p2 in pairs], [p2[1] for p2 in pairs]
            else:
                xs, ys = [], []

        if xs and ys:
            if self.log_scale:
                ys = [abs(v) if v != 0 else 1e-30 for v in ys]
                p.yscale("log")
            else:
                p.yscale("linear")

            # Upsample to ~400 points so the line appears solid regardless of
            # how sparse the solver timesteps are relative to terminal columns.
            if len(xs) >= 2:
                import numpy as _np
                xi = _np.linspace(xs[0], xs[-1], max(len(xs), 400))
                yi = _np.interp(xi, xs, ys)
                xs, ys = xi.tolist(), yi.tolist()

            p.plot(xs, ys, color="cyan", marker="braille")
            # Metric name as text label in upper-right corner of the data area
            p.text(
                self._current_metric,
                x=xs[-1],
                y=max(ys),
                color="cyan+",
                style="bold",
                alignment="right",
            )

        chart.refresh()

        # Status bar
        avail = self._available_metrics()
        n = len(avail)
        idx = self._metric_index + 1
        scale = "[bold cyan]LOG[/]" if self.log_scale else "[dim]lin[/]"
        if self._start_time is not None:
            t0 = f"t\u2080={self._start_time:.4f}s"
        else:
            t0 = "t\u2080=0"
        status.update(
            f"[dim]{idx}/{n}[/]  \u2502  Y:{scale}  \u2502  {t0}  \u2502  "
            f"[dim]\u2190\u2192[/] metric  [dim]Ctrl+P[/] commands"
        )

    # ------------------------------------------------------------------
    # Public API (called by app / command palette)
    # ------------------------------------------------------------------

    def set_metric(self, name: str) -> None:
        """Switch the plot to *name*."""
        avail = self._available_metrics()
        if name in avail:
            self._metric_index = avail.index(name)
            self._current_metric = name
            self._last_version = -1
            self._redraw()

    def next_metric(self) -> None:
        avail = self._available_metrics()
        if not avail:
            return
        self._metric_index = (self._metric_index + 1) % len(avail)
        self._current_metric = avail[self._metric_index]
        self._last_version = -1
        self._redraw()

    def prev_metric(self) -> None:
        avail = self._available_metrics()
        if not avail:
            return
        self._metric_index = (self._metric_index - 1) % len(avail)
        self._current_metric = avail[self._metric_index]
        self._last_version = -1
        self._redraw()

    def toggle_log_scale(self) -> None:
        self.log_scale = not self.log_scale
        self._redraw()

    def set_start_time(self, t: float | None) -> None:
        """Set (or clear) the start-time filter."""
        self._start_time = t
        self._last_version = -1
        self._redraw()

    @property
    def current_metric(self) -> str:
        return self._current_metric

    @property
    def start_time(self) -> float | None:
        return self._start_time

