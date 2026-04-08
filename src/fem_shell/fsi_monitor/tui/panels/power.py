"""Power, efficiency, and aerodynamic coefficient panel."""

from typing import Any

from textual.reactive import reactive
from textual.widgets import Static

from fem_shell.fsi_monitor.data_provider import FSIDataProvider

_W = 14  # value column width


def _v(value: Any, prec: int = 4) -> str:
    if value is None:
        return "—".rjust(_W)
    if isinstance(value, float):
        return f"{value:.{prec}f}".rjust(_W)
    return str(value).rjust(_W)


def _row(label: str, val: str, unit: str = "") -> str:
    u = f" [dim]{unit}[/]" if unit else ""
    return f"  [dim]{label:<14}[/][bold]{val}[/]{u}"


def _eff_row(value: Any) -> str:
    """Efficiency row with color-coded percentage."""
    label = "η"
    if value is None or not isinstance(value, (int, float)):
        return f"  [dim]{label:<14}[/][bold]{'—':>{_W}}[/]"
    pct = float(value) * 100
    color = "green" if pct > 90 else "yellow" if pct > 75 else "red"
    return f"  [dim]{label:<14}[/][{color}]{pct:>{_W}.1f} %[/]"


class PowerPanel(Static):
    """Power, efficiency, and aero coefficients."""

    DEFAULT_CSS = """
    PowerPanel {
        border: tall $success 50%;
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
            "[bold green]PERFORMANCE[/]",
            _row("P aero",       _v(g("Aero Power [W]"), 2),     "W"),
            _row("P total",      _v(g("Total Power [W]"), 2),    "W"),
            _eff_row(g("Structural Efficiency")),
            "",
            _row("Cp",           _v(g("Cp"), 4),                 ""),
            _row("Cq",           _v(g("Cq"), 4),                 ""),
            _row("Ct",           _v(g("Ct"), 4),                 ""),
            _row("TSR",          _v(g("TSR"), 3),                ""),
        ]
        self._content = "\n".join(lines)

    def render(self) -> str:
        return self._content or "[dim]PERFORMANCE\n  Waiting…[/]"

