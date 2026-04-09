"""Rotation dynamics + structural deformation panel."""

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


def _angle_row(angle: Any) -> str:
    """Angle row: angle mod 360 with 1-based revolution counter."""
    if angle is None or not isinstance(angle, (int, float)):
        return f"  [dim]{'Angle':<14}[/][bold]{'—':>{_W}}[/] [dim]°[/]"
    a = float(angle)
    current_rev = int(a / 360) + 1  # 1-based: first lap = rev 1
    mod = a % 360
    val = f"{mod:>{_W}.2f}"
    return f"  [dim]{'Angle':<14}[/][bold]{val}[/] [dim]° (rev {current_rev})[/]"


class RotationPanel(Static):
    """Rotation metrics + structural deformation."""

    DEFAULT_CSS = """
    RotationPanel {
        border: tall $primary 50%;
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
        g = self._provider.get_group("rotation")

        lines = [
            "[bold cyan]ROTATION[/]",
            _row("Time",         _v(g.get("Time [s]"), 4),             "s"),
            _angle_row(g.get("Angle [deg]")),
            _row("Speed",        _v(g.get("Speed [RPM]"), 1),          "RPM"),
            _row("ω",            _v(g.get("Omega [rad/s]"), 4),        "rad/s"),
            _row("α",            _v(g.get("Alpha [rad/s2]"), 2),       "rad/s²"),
            "",
            "[bold blue]STRUCTURE[/]",
            _row("Max Disp",     _v(r.get("Max Displacement [m]"), 6), "m"),
            _row("Def Radius",   _v(r.get("Deformed Radius [m]"), 4), "m"),
        ]
        self._content = "\n".join(lines)

    def render(self) -> str:
        return self._content or "[dim]ROTATION\n  Waiting…[/]"
