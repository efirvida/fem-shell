"""
FSI data provider: unified, solver-independent data access layer.

Source:
  - rotor_performance.csv — primary, full time-series metrics

Threading model:
  A single background daemon thread polls for file changes and updates
  shared state under a RLock. The TUI reads state through shallow-copied
  snapshots so that no render call ever blocks the polling thread.

Startup policy:
  Always starts from the *latest available row* — never backfills history
  older than the file's current last row at startup time. Historical plotting
  reads N trailing rows on-demand.
"""

import threading
from pathlib import Path
from typing import Any, Optional

from fem_shell.fsi_monitor.core import CSVReader, SafeFileReader

# ---------------------------------------------------------------------------
# Metric group definitions — mirrored from rotor.py _log_rotor_performance()
# ---------------------------------------------------------------------------

ROTATION_METRICS = [
    "Time [s]",
    "Angle [deg]",
    "Speed [RPM]",
    "Omega [rad/s]",
    "Alpha [rad/s2]",
]

FORCE_METRICS = [
    "Aero Thrust [N]",
    "Aero Torque [Nm]",
    "Inertial Torque [Nm]",
    "Gravity Torque [Nm]",
    "Total Torque [Nm]",
    "Aero Torque X [Nm]",
    "Aero Torque Y [Nm]",
    "Aero Torque Z [Nm]",
    "Total Torque X [Nm]",
    "Total Torque Y [Nm]",
    "Total Torque Z [Nm]",
]

POWER_METRICS = [
    "Aero Power [W]",
    "Total Power [W]",
    "Structural Efficiency",
    "Cp",
    "Cq",
    "Ct",
    "TSR",
]

STRUCTURE_METRICS = [
    "Max Displacement [m]",
    "Deformed Radius [m]",
]

# All plottable columns — time is always X axis
PLOTTABLE_METRICS: list[str] = (
    ROTATION_METRICS[1:]  # skip Time [s] — it's always X
    + FORCE_METRICS
    + POWER_METRICS
    + STRUCTURE_METRICS
)

# Default metric shown in the plot on startup
DEFAULT_PLOT_METRIC: str = "Omega [rad/s]"

ALL_GROUPS: dict[str, list[str]] = {
    "rotation": ROTATION_METRICS,
    "force": FORCE_METRICS,
    "power": POWER_METRICS,
    "structure": STRUCTURE_METRICS,
}


class FSIDataProvider:
    """
    Unified data access layer for FSI monitoring.

    Usage
    -----
    provider = FSIDataProvider(csv_path)
    # TUI reads:
    row   = provider.get_current()      # latest metrics dict
    hist  = provider.get_history("Cp")  # list[float] for plotting
    # Shutdown on exit:
    provider.shutdown()
    """

    def __init__(
        self,
        csv_path: Path,
        poll_interval: float = 0.5,
    ):
        self._csv_reader = CSVReader(csv_path, SafeFileReader())
        self._poll_interval = poll_interval

        # Shared state \u2014 always replaced, never mutated in-place
        self._current_row: Optional[dict[str, Any]] = None
        self._available_columns: list[str] = []

        self._lock = threading.RLock()
        self._data_version: int = 0  # monotonically incremented on update
        self._stop = threading.Event()
        # Set when the first read completes — TUI can poll this for loading state
        self._ready = threading.Event()
        self._load_status: str = "Reading CSV…"

        # Bootstrap is async — TUI starts immediately without blocking
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="fsi-monitor-poll")
        self._thread.start()

    # ------------------------------------------------------------------
    # Background polling
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        # First iteration: bootstrap (happens in background while TUI renders)
        with self._lock:
            self._load_status = "Reading CSV…"
        self._refresh_csv()
        self._ready.set()
        with self._lock:
            self._load_status = ""

        while not self._stop.is_set():
            self._refresh_csv()
            self._stop.wait(timeout=self._poll_interval)

    def _refresh_csv(self) -> None:
        """Update current row if file has changed."""
        row = self._csv_reader.get_latest_row()
        cols = self._csv_reader.columns
        if row is None:
            return
        with self._lock:
            self._current_row = row
            if cols and not self._available_columns:
                self._available_columns = list(cols)
            self._data_version += 1

    # ------------------------------------------------------------------
    # Public read API (TUI-facing, always non-blocking)
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """True once the first CSV read has completed."""
        return self._ready.is_set()

    @property
    def load_status(self) -> str:
        """Human-readable status of the current loading stage (empty when done)."""
        with self._lock:
            return self._load_status

    @property
    def data_version(self) -> int:
        """Monotonically increasing; TUI can check this to detect updates."""
        with self._lock:
            return self._data_version

    @property
    def available_columns(self) -> list[str]:
        with self._lock:
            return list(self._available_columns)

    def get_current(self) -> Optional[dict[str, Any]]:
        """Latest row as a shallow copy — safe to read without lock."""
        with self._lock:
            return dict(self._current_row) if self._current_row else None

    def get_metric(self, name: str) -> Optional[Any]:
        """Single metric value from current row."""
        with self._lock:
            if self._current_row is None:
                return None
            return self._current_row.get(name)

    def get_group(self, group: str) -> dict[str, Optional[Any]]:
        """All metrics for a named group, intersected with available columns."""
        keys = ALL_GROUPS.get(group, [])
        row = self.get_current() or {}
        return {k: row.get(k) for k in keys if k in row or k in self._available_columns}

    def get_history(self, column: str, max_points: int = 0) -> list[Any]:
        """Return values for *column*. If *max_points* > 0, tail-slice."""
        return self._csv_reader.get_column(column, max_points=max_points)

    def get_history_xy(
        self, y_column: str, max_points: int = 0
    ) -> tuple[list[Any], list[Any]]:
        """Return (time, y_values) pair for plotting."""
        return self._csv_reader.get_two_columns("Time [s]", y_column, max_points)

    def is_plottable(self, column: str) -> bool:
        cols = self.available_columns
        return column in cols

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Stop background polling thread gracefully."""
        self._stop.set()
        self._thread.join(timeout=3)



