"""FSI simulation real-time monitor — TUI dashboard, solver-independent."""

from fem_shell.fsi_monitor.core import CSVReader, SafeFileReader
from fem_shell.fsi_monitor.data_provider import FSIDataProvider

__all__ = ["CSVReader", "FSIDataProvider", "SafeFileReader"]
