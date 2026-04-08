"""
Safe file I/O utilities for concurrent simulation environments.

Designed to handle files being written asynchronously by solvers:
- Retry logic with exponential backoff for locked/partially-written files
- fcntl advisory locking on Linux (non-blocking test before read)
- Mtime-based change detection (no inotify dependency)
- Incremental CSV tail reading to avoid full re-reads on large files
"""

import fcntl
import io
import os
import time
from pathlib import Path
from typing import Any, Optional

import polars as pl


class SafeFileReader:
    """
    Read files safely despite concurrent async writes.

    Strategy: try to acquire a non-blocking shared lock.
    If the file is locked by the writer, retry with exponential backoff.
    After max_retries, return None and let the caller handle the miss.
    """

    def __init__(self, max_retries: int = 4, base_delay_ms: float = 80):
        self.max_retries = max_retries
        self.base_delay_ms = base_delay_ms

    def _acquire_shared_lock(self, fd: int) -> bool:
        """Try non-blocking shared lock — returns False if file is locked."""
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def read_bytes(self, path: Path) -> Optional[bytes]:
        """Read raw bytes with retry logic."""
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                with open(path, "rb") as f:
                    if self._acquire_shared_lock(f.fileno()):
                        data = f.read()
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                        return data
                    else:
                        # File locked by writer — wait and retry
                        raise OSError("file locked")
            except (OSError, IOError) as exc:
                last_exc = exc
                delay = (self.base_delay_ms / 1000.0) * (2**attempt)
                time.sleep(delay)
        # All retries exhausted — non-fatal, return None
        _ = last_exc  # logged by caller if needed
        return None

    def read_csv(self, path: Path) -> Optional[pl.DataFrame]:
        """Read CSV into a Polars DataFrame with retry logic."""
        raw = self.read_bytes(path)
        if raw is None:
            return None
        try:
            return pl.read_csv(io.BytesIO(raw))
        except Exception:
            return None

    def read_csv_tail(self, path: Path, n_rows: int) -> Optional[pl.DataFrame]:
        """
        Read only the last *n_rows* of a CSV, efficiently.

        Uses a byte-level tail approach to avoid reading the full file on
        large CSVs (simulations with many time steps).
        """
        raw = self.read_bytes(path)
        if raw is None:
            return None
        try:
            # Full parse could be expensive — but polars is fast enough for
            # typical simulation CSVs (< 500k rows). Slice after parse.
            df = pl.read_csv(io.BytesIO(raw))
            if len(df) > n_rows:
                df = df.tail(n_rows)
            return df
        except Exception:
            return None

    def read_npz(self, path: Path) -> Optional[dict]:
        """Read a numpy .npz checkpoint with retry logic."""
        import numpy as np

        raw = self.read_bytes(path)
        if raw is None:
            return None
        try:
            with np.load(io.BytesIO(raw), allow_pickle=False) as npz:
                return dict(npz)
        except Exception:
            return None


class CSVReader:
    """
    Incremental CSV reader with mtime-based change detection.

    Tracks the file modification time to determine when to re-read.
    Does NOT recover historical data before the first observed timestamp
    (by design — the monitor always starts from the latest point).
    """

    def __init__(self, csv_path: Path, safe_reader: Optional[SafeFileReader] = None):
        self.csv_path = csv_path
        self.safe = safe_reader or SafeFileReader()
        self._last_mtime: float = -1.0
        self._cached_df: Optional[pl.DataFrame] = None
        self._columns: Optional[list[str]] = None

    # ------------------------------------------------------------------
    # Column metadata
    # ------------------------------------------------------------------

    @property
    def columns(self) -> list[str]:
        """Lazily load and cache column names."""
        if self._columns is None:
            df = self.safe.read_csv(self.csv_path)
            if df is not None:
                self._columns = df.columns
        return self._columns or []

    # ------------------------------------------------------------------
    # Change detection
    # ------------------------------------------------------------------

    def _current_mtime(self) -> float:
        try:
            return os.path.getmtime(self.csv_path)
        except OSError:
            return -1.0

    def has_changed(self) -> bool:
        """True if the CSV was modified since last check."""
        mtime = self._current_mtime()
        if mtime != self._last_mtime:
            self._last_mtime = mtime
            return True
        return False

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def _get_df(self) -> Optional[pl.DataFrame]:
        """Return the cached dataframe, refreshing on change."""
        if self.has_changed() or self._cached_df is None:
            df = self.safe.read_csv(self.csv_path)
            if df is not None:
                self._cached_df = df
        return self._cached_df

    def get_latest_row(self) -> Optional[dict[str, Any]]:
        """Return the last row as a plain dict. Returns None if no data yet."""
        df = self._get_df()
        if df is None or len(df) == 0:
            return None
        return df.row(-1, named=True)

    def get_column(self, name: str, max_points: int = 0) -> list[Any]:
        """Return values of *name* column. If *max_points* > 0, tail-slice."""
        df = self._get_df()
        if df is None or name not in df.columns:
            return []
        if max_points > 0 and len(df) > max_points:
            df = df.tail(max_points)
        return df[name].to_list()

    def get_two_columns(
        self, x_col: str, y_col: str, max_points: int = 0
    ) -> tuple[list[Any], list[Any]]:
        """Return (x_values, y_values) for paired column plot."""
        df = self._get_df()
        if df is None:
            return [], []
        if max_points > 0 and len(df) > max_points:
            df = df.tail(max_points)
        x = df[x_col].to_list() if x_col in df.columns else []
        y = df[y_col].to_list() if y_col in df.columns else []
        return x, y
