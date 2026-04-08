"""Forces and torques metric panel."""

from typing import Any

from textual.reactive import reactive
from textual.widgets import Static

from fem_shell.fsi_monitor.data_provider import FSIDataProvider

_W = 12  # value column width


def _v(value: Any, prec: int = 3) -> str:
    if value is None:
        return "—".rjust(_W)
    if isinstance(value, float):
        return f"{value:.{prec}f}".rjust(_W)
    return str(value).rjust(_W)


def _row(label: str, val: str, unit: str = "") -> str:
    u = f" [dim]{unit}[/]" if unit else ""
    return f"  [dim]{label:<14}[/][bold]{val}[/]{u}"


def _xyz_row(label: str, x: Any, y: Any, z: Any, prec: int = 3) -> str:
    """Format a labeled xyz vector as a compact row."""
    fmt = lambda v: f"{v:.{prec}f}" if isinstance(v, float) else "—"
    return f"  [dim]{label:<14}[/][bold]{fmt(x):>8} {fmt(y):>8} {fmt(z):>8}[/]"


class ForcesPanel(Static):
    """Forces & torques panel."""

    DEFAULT_CSS = """
    ForcesPanel {
        border: tall $warning 50%;
        padding: 0 1;
        height: auto;
    }
    """

    _content: reactive[str] = reactive("", layout=True)

    def __init__(self, provider: FSIDataProvider, **kwargs: Any):
        super().__init__(**kwargs)
        self._provider = provider
        self._last_version: int = -1

    def on_mount(self) -> None:
        self.set_interval(0.5, self._tick)

    def _tick(self) -> None:
        v = self._provider.data_version
        if v == self._last_version:
            return
        self._last_version = v

        r = self._provider.get_current() or {}
        g = r.get

        lines = [
            "[bold yellow]FORCES & TORQUES[/]",
            _row("Thrust",       _v(g("Aero Thrust [N]")),          "N"),
            _row("τ aero",       _v(g("Aero Torque [Nm]")),         "Nm"),
            _row("τ inertial",   _v(g("Inertial Torque [Nm]")),     "Nm"),
            _row("τ gravity",    _v(g("Gravity Torque [Nm]")),      "Nm"),
            _row("τ total",      _v(g("Total Torque [Nm]")),        "Nm"),
            "",
            _xyz_row("Aero   xyz",
                     g("Aero Torque X [Nm]"),
                     g("Aero Torque Y [Nm]"),
                     g("Aero Torque Z [Nm]")),
            _xyz_row("Total  xyz",
                     g("Total Torque X [Nm]"),
                     g("Total Torque Y [Nm]"),
                     g("Total Torque Z [Nm]")),
        ]
        self._content = "\n".join(lines)

    def render(self) -> str:
        return self._content or "[dim]FORCES & TORQUES\n  Waiting…[/]"

