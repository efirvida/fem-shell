#!/usr/bin/env python3
"""Post-process CalculiX modal analysis results.

Parses the CalculiX .dat file (eigenfrequencies) and optionally the .frd file
(mode shapes) to estimate directional dominance. Displays results in the same
Rich table format as fem-shell's modal solver for easy comparison.

Usage
-----
    # Only frequencies from .dat
    python postprocess_ccx_modal.py blade_modal.dat

    # Frequencies + directional analysis from .frd
    python postprocess_ccx_modal.py blade_modal.dat --frd blade_modal.frd

    # Compare with reference frequencies (comma-separated Hz values)
    python postprocess_ccx_modal.py blade_modal.dat --frd blade_modal.frd \\
        --ref 0.5698,0.8694,1.3925,1.7032,2.0793,4.2935

    # Compare with fem-shell csv export
    python postprocess_ccx_modal.py blade_modal.dat --frd blade_modal.frd \\
        --fem-csv modal_results.csv

Examples
--------
Run CalculiX first::

    cd simulations/blade/solid
    ccx blade_modal
    python /path/to/tools/postprocess_ccx_modal.py blade_modal.dat \\
        --frd blade_modal.frd \\
        --ref 0.5698,0.8694,1.3925,1.7032,2.0793,4.2935
"""

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# .dat parser — full modal results
# ---------------------------------------------------------------------------

# Numeric token (scientific or decimal notation, possibly negative)
_NUM = r"[+\-]?\d+[\d.]*(?:[EeDd][+\-]?\d+)?"

# Data row: integer mode followed by 4 or 6 floats
_ROW4 = re.compile(
    r"^\s*(\d+)\s+({n})\s+({n})\s+({n})\s+({n})\s*$".format(n=_NUM)
)
_ROW6 = re.compile(
    r"^\s*(\d+)\s+({n})\s+({n})\s+({n})\s+({n})\s+({n})\s+({n})\s*$".format(n=_NUM)
)


def _parse_table_section(lines, start_idx, n_cols):
    """Parse a CCX tabular section starting just after its header.

    Returns a dict mapping mode_no (int) → list[float] of n_cols values,
    and the index of the line after the section.
    """
    pattern = _ROW6 if n_cols == 6 else _ROW4
    data = {}
    i = start_idx
    while i < len(lines):
        line = lines[i]
        m = pattern.match(line)
        if m:
            mode_n = int(m.group(1))
            vals = [float(m.group(k + 2)) for k in range(n_cols)]
            data[mode_n] = vals
            i += 1
            continue
        # "TOTAL" line marks end of per-mode section
        if re.search(r"\bTOTAL\b", line, re.IGNORECASE):
            break
        # Non-blank, non-numeric line ends the block
        stripped = line.strip()
        if stripped and not re.match(r"^[\s\d.EeDd+\-]+$", stripped):
            break
        i += 1
    return data, i


def parse_dat(dat_path):
    # type: (Path) -> Dict
    """Parse a CalculiX .dat file from a *FREQUENCY step.

    Handles the verbose CCX EIGENVALUE OUTPUT format::

        E I G E N V A L U E   O U T P U T

        MODE NO    EIGENVALUE      FREQUENCY
                                   REAL PART    IMAGINARY PART
                       (RAD/TIME)  (CYCLES/TIME (RAD/TIME)

             1  0.1420353E+01  0.1191786E+01  0.1896786E+00  0.0000000E+00

    Columns per data row:
      1 mode_no | 2 eigenvalue | 3 freq_rad | 4 freq_hz | 5 freq_imag_rad

    Also parses PARTICIPATION FACTORS and EFFECTIVE MODAL MASS tables plus
    TOTAL EFFECTIVE MODAL MASS to compute per-mode mass participation (%).

    Returns
    -------
    dict with keys:
        'frequencies'   : list[float] — Hz values sorted by mode
        'participation' : list[dict]  — per-mode, keys Ux%…Rz% (may be empty)
    """
    text = dat_path.read_text(errors="replace")
    lines = text.splitlines()

    frequencies = {}   # type: Dict[int, float]
    eff_mass = {}      # type: Dict[int, List[float]]  # 6 directions
    total_mass = []    # type: List[float]             # 6 directions

    DIR_LABELS = ["Ux", "Uy", "Uz", "Rx", "Ry", "Rz"]

    i = 0
    while i < len(lines):
        line = lines[i]
        upper = line.upper()

        # ---- Eigenvalue / frequency block --------------------------------
        if "E I G E N V A L U E" in upper and "O U T P U T" in upper:
            # Skip forward until we hit the header line with (CYCLES/TIME
            j = i + 1
            while j < len(lines) and "(CYCLES/TIME" not in lines[j].upper():
                j += 1
            j += 1  # skip the header line itself
            # Now parse data rows: mode  eigenval  freq_rad  freq_hz  freq_imag
            while j < len(lines):
                m = _ROW4.match(lines[j])
                if m:
                    mode_n = int(m.group(1))
                    freq_hz = float(m.group(4))  # column 4 = cycles/time
                    frequencies[mode_n] = freq_hz
                    j += 1
                    continue
                stripped = lines[j].strip()
                if stripped and not re.match(r"^[\s\d.EeDd+\-]+$", stripped):
                    break
                j += 1
            i = j
            continue

        # ---- Effective modal mass table ----------------------------------
        if "E F F E C T I V E" in upper and "M O D A L" in upper and "M A S S" in upper:
            # skip header lines until first data row
            j = i + 1
            while j < len(lines):
                m = _ROW6.match(lines[j])
                if m:
                    break
                j += 1
            data, j = _parse_table_section(lines, j, n_cols=6)
            eff_mass.update(data)

            # Parse TOTAL line right after data
            while j < len(lines):
                tline = lines[j]
                if re.search(r"\bTOTAL\b", tline, re.IGNORECASE):
                    nums = re.findall(_NUM, tline)
                    if len(nums) >= 6:
                        total_mass = [float(x) for x in nums[:6]]
                    j += 1
                    break
                j += 1
            i = j
            continue

        i += 1

    # ---- Compute mass participation percentages -------------------------
    participation = []  # type: List[Dict[str, float]]
    if eff_mass and total_mass and len(total_mass) >= 6:
        for mode_n in sorted(eff_mass):
            vals = eff_mass[mode_n]
            row = {}
            for d in range(min(6, len(vals))):
                tot = total_mass[d]
                pct = vals[d] / tot * 100.0 if tot > 1e-30 else 0.0
                row[DIR_LABELS[d] + "%"] = pct
            participation.append(row)

    return {
        "frequencies": [frequencies[k] for k in sorted(frequencies)],
        "participation": participation,
    }


def parse_dat_frequencies(dat_path):
    # type: (Path) -> List[float]
    """Convenience wrapper — returns only the frequency list."""
    return parse_dat(dat_path)["frequencies"]


# ---------------------------------------------------------------------------
# .frd parser — mode shapes → directional content
# ---------------------------------------------------------------------------

def parse_frd_mode_shapes(frd_path: Path, n_modes: int) -> List[Dict[str, float]]:
    """Parse CalculiX .frd file and compute directional RMS per mode.

    For each mode returns a dict::

        {"Ux_pct": float, "Uy_pct": float, "Uz_pct": float}

    percentages of total RMS displacement energy in each direction.

    Parameters
    ----------
    frd_path:
        Path to the .frd file.
    n_modes:
        Number of modes to parse (stops after this many DISP blocks).

    Notes
    -----
    The .frd format uses:
    - ``    1PSTEP   <mode>`` — start of a new result set (mode)
    - ``-4  DISP     4 1``   — DISP result block with 4 components (D1,D2,D3,ALL)
    - ``-1  <node>  D1 D2 D3``  — nodal data
    - ``-3``                 — end of result block
    """
    results = []  # type: List[Dict[str, float]]

    try:
        f = frd_path.open("r", errors="replace")
    except OSError as e:
        print(f"[WARNING] Cannot open .frd file: {e}", file=sys.stderr)
        return []

    mode_count = 0
    in_disp_block = False
    sum_sq = [0.0, 0.0, 0.0]
    n_nodes = 0

    node_data_pat = re.compile(
        r"^\s*-1\s+\d+\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)"
    )
    step_pat = re.compile(r"^\s*1PSTEP\s+(\d+)", re.IGNORECASE)

    # Report progress for large files
    line_count = 0
    for line in f:
        line_count += 1
        stripped = line.strip()

        # New result set / mode
        m = step_pat.match(stripped)
        if m:
            # Save previous DISP block if any
            if in_disp_block and n_nodes > 0:
                _store_mode(sum_sq, n_nodes, results)
                mode_count += 1
                if mode_count >= n_modes:
                    break
            in_disp_block = False
            sum_sq = [0.0, 0.0, 0.0]
            n_nodes = 0
            continue

        # DISP block header
        if stripped.startswith("-4") and "DISP" in stripped.upper():
            in_disp_block = True
            continue

        # End of block
        if stripped == "-3":
            if in_disp_block and n_nodes > 0:
                _store_mode(sum_sq, n_nodes, results)
                mode_count += 1
                if mode_count >= n_modes:
                    break
            in_disp_block = False
            sum_sq = [0.0, 0.0, 0.0]
            n_nodes = 0
            continue

        # Nodal data
        if in_disp_block and stripped.startswith("-1"):
            m = node_data_pat.match(stripped)
            if m:
                sum_sq[0] += float(m.group(1)) ** 2
                sum_sq[1] += float(m.group(2)) ** 2
                sum_sq[2] += float(m.group(3)) ** 2
                n_nodes += 1

    # Handle last open block (EOF without -3)
    if in_disp_block and n_nodes > 0 and mode_count < n_modes:
        _store_mode(sum_sq, n_nodes, results)

    f.close()
    return results


def _store_mode(sum_sq, n_nodes, results):
    # type: (List[float], int, List[Dict[str, float]]) -> None
    """Compute directional percentages from accumulated squared displacements."""
    rms = [math.sqrt(s / n_nodes) for s in sum_sq]
    total = math.sqrt(sum(r**2 for r in rms))
    if total < 1e-30:
        results.append({"Ux_pct": 0.0, "Uy_pct": 0.0, "Uz_pct": 0.0})
    else:
        results.append({
            "Ux_pct": (rms[0] / total) * 100.0,
            "Uy_pct": (rms[1] / total) * 100.0,
            "Uz_pct": (rms[2] / total) * 100.0,
        })


# ---------------------------------------------------------------------------
# CSV fem-shell results parser
# ---------------------------------------------------------------------------

def parse_femshell_csv(csv_path):
    # type: (Path) -> List[float]
    """Read fem-shell modal CSV export (mode, freq_hz, ...) → list of Hz."""
    import csv
    freqs = []
    with open(str(csv_path), newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = next((k for k in row if "freq" in k.lower() and "hz" in k.lower()), None)
            if key:
                freqs.append(float(row[key]))
    return freqs


def parse_femshell_csv_full(csv_path):
    # type: (Path) -> Dict
    """Read fem-shell modal CSV — returns frequencies and participation.

    Returns dict with 'frequencies' (List[float]) and 'participation'
    (List[Dict[str,float]]) matching the same structure as parse_dat().
    """
    import csv
    freqs = []
    participation = []
    with open(str(csv_path), newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            freq_key = next((k for k in row if "freq" in k.lower() and "hz" in k.lower()), None)
            if freq_key is None:
                continue
            freqs.append(float(row[freq_key]))
            # Collect eff_mass_Xx_pct columns
            pf = {}
            for k, v in row.items():
                kl = k.lower()
                if kl.startswith("eff_mass_") and kl.endswith("_pct"):
                    # e.g. eff_mass_Ux_pct  -> Ux%
                    dirname = k[len("eff_mass_"):-len("_pct")]
                    pf[dirname + "%"] = float(v)
            participation.append(pf)
    return {"frequencies": freqs, "participation": participation}


# ---------------------------------------------------------------------------
# Comparative table: CCX vs fem-shell side by side
# ---------------------------------------------------------------------------

def _dominant_type(pf_row, cols):
    # type: (Dict[str, float], List[str]) -> str
    """Return the label of the column with the highest participation value."""
    if not cols or not pf_row:
        return "-"
    vals = [pf_row.get(c, 0.0) for c in cols]
    return cols[vals.index(max(vals))].rstrip("%")


def display_comparison(
    ccx_freqs,        # type: List[float]
    ccx_pf,           # type: List[Dict[str, float]]
    fem_freqs,        # type: List[float]
    fem_pf,           # type: List[Dict[str, float]]
    ref_freqs=None,   # type: Optional[List[float]]
    ref_label="Ref",  # type: str
):
    # type: (...) -> None
    """Side-by-side comparison table: CalculiX | fem-shell | Delta."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
    except ImportError:
        _display_comparison_plain(ccx_freqs, ccx_pf, fem_freqs, fem_pf, ref_freqs, ref_label)
        return

    console = Console()
    n = min(len(ccx_freqs), len(fem_freqs))

    # Determine participation columns present in CCX results
    pf_cols = [lb + "%" for lb in _DIR_LABELS if ccx_pf and lb + "%" in ccx_pf[0]]

    table = Table(
        title="MODAL COMPARISON: CalculiX vs fem-shell",
        title_style="bold cyan",
        show_lines=False,
        pad_edge=True,
        box=box.SIMPLE_HEAVY,
    )

    table.add_column("Mode", justify="right", style="bold")
    # CalculiX columns
    table.add_column("CCX [Hz]", justify="right", style="cyan")
    table.add_column("CCX Type", justify="center", style="cyan")
    # fem-shell columns
    table.add_column("FEM [Hz]", justify="right", style="green")
    table.add_column("FEM Type", justify="center", style="green")
    # Delta
    table.add_column("CCX-FEM %", justify="right")
    # Optional reference
    if ref_freqs:
        table.add_column("%s [Hz]" % ref_label, justify="right", style="dim")
        table.add_column("CCX-Ref %", justify="right")
        table.add_column("FEM-Ref %", justify="right")

    deltas_ccx_fem = []
    deltas_ccx_ref = []
    deltas_fem_ref = []

    for i in range(n):
        fc = ccx_freqs[i]
        ff = fem_freqs[i]
        delta_cf = (fc - ff) / ff * 100.0 if ff > 0 else float("nan")
        deltas_ccx_fem.append(abs(delta_cf))

        ccx_type = _dominant_type(ccx_pf[i] if i < len(ccx_pf) else {}, pf_cols)
        fem_type = _dominant_type(fem_pf[i] if i < len(fem_pf) else {}, pf_cols)

        sign_cf = "+" if delta_cf > 0 else ""
        color_cf = "green" if abs(delta_cf) < 1 else "yellow" if abs(delta_cf) < 5 else "red"

        row = [
            str(i + 1),
            "%.6f" % fc,
            ccx_type,
            "%.6f" % ff,
            fem_type,
            "[%s]%s%.2f%%[/%s]" % (color_cf, sign_cf, delta_cf, color_cf),
        ]

        if ref_freqs and i < len(ref_freqs):
            fr = ref_freqs[i]
            d_cr = (fc - fr) / fr * 100.0 if fr > 0 else float("nan")
            d_fr = (ff - fr) / fr * 100.0 if fr > 0 else float("nan")
            deltas_ccx_ref.append(abs(d_cr))
            deltas_fem_ref.append(abs(d_fr))
            sign_cr = "+" if d_cr > 0 else ""
            sign_fr = "+" if d_fr > 0 else ""
            col_cr = "green" if abs(d_cr) < 5 else "yellow" if abs(d_cr) < 20 else "red"
            col_fr = "green" if abs(d_fr) < 5 else "yellow" if abs(d_fr) < 20 else "red"
            row += [
                "%.6f" % fr,
                "[%s]%s%.1f%%[/%s]" % (col_cr, sign_cr, d_cr, col_cr),
                "[%s]%s%.1f%%[/%s]" % (col_fr, sign_fr, d_fr, col_fr),
            ]
        elif ref_freqs:
            row += ["-", "-", "-"]

        table.add_row(*row)

    console.print()
    console.print(table)

    # Summary line
    if deltas_ccx_fem:
        mean_cf = sum(deltas_ccx_fem) / len(deltas_ccx_fem)
        console.print(
            "  CCX vs fem-shell mean |delta|: [bold cyan]%.2f%%[/bold cyan]" % mean_cf
        )
    if deltas_ccx_ref:
        mean_cr = sum(deltas_ccx_ref) / len(deltas_ccx_ref)
        mean_fr = sum(deltas_fem_ref) / len(deltas_fem_ref)
        console.print(
            "  CCX vs %s mean |delta|: [bold]%.1f%%[/bold]   "
            "fem-shell vs %s mean |delta|: [bold]%.1f%%[/bold]" % (
                ref_label, mean_cr, ref_label, mean_fr)
        )
    console.print()


def _display_comparison_plain(
    ccx_freqs, ccx_pf, fem_freqs, fem_pf, ref_freqs=None, ref_label="Ref"
):
    # type: (...) -> None
    n = min(len(ccx_freqs), len(fem_freqs))
    pf_cols = [lb + "%" for lb in _DIR_LABELS if ccx_pf and lb + "%" in ccx_pf[0]]

    print("\n" + "=" * 80)
    print("  MODAL COMPARISON: CalculiX vs fem-shell")
    print("=" * 80)

    hdr = "%5s  %12s  %8s  %12s  %8s  %10s" % (
        "Mode", "CCX[Hz]", "CCXType", "FEM[Hz]", "FEMType", "CCX-FEM%")
    if ref_freqs:
        hdr += "  %12s  %8s  %8s" % (ref_label + "[Hz]", "CCX-Ref%", "FEM-Ref%")
    print(hdr)
    print("-" * len(hdr))

    for i in range(n):
        fc = ccx_freqs[i]
        ff = fem_freqs[i]
        delta_cf = (fc - ff) / ff * 100.0 if ff > 0 else float("nan")
        ccx_type = _dominant_type(ccx_pf[i] if i < len(ccx_pf) else {}, pf_cols)
        fem_type = _dominant_type(fem_pf[i] if i < len(fem_pf) else {}, pf_cols)
        line = "%5d  %12.6f  %8s  %12.6f  %8s  %+9.2f%%" % (
            i + 1, fc, ccx_type, ff, fem_type, delta_cf)
        if ref_freqs and i < len(ref_freqs):
            fr = ref_freqs[i]
            d_cr = (fc - fr) / fr * 100.0 if fr > 0 else float("nan")
            d_fr = (ff - fr) / fr * 100.0 if fr > 0 else float("nan")
            line += "  %12.6f  %+7.1f%%  %+7.1f%%" % (fr, d_cr, d_fr)
        print(line)
    print()



def parse_femshell_csv_full(csv_path):
    # type: (Path) -> Dict
    """Read fem-shell modal CSV — returns frequencies and participation.

    Returns dict with 'frequencies' (List[float]) and 'participation'
    (List[Dict[str,float]]) matching the same structure as parse_dat().
    """
    import csv
    freqs = []
    participation = []
    with open(str(csv_path), newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            freq_key = next((k for k in row if "freq" in k.lower() and "hz" in k.lower()), None)
            if freq_key is None:
                continue
            freqs.append(float(row[freq_key]))
            # Collect eff_mass_Xx_pct columns
            pf = {}
            for k, v in row.items():
                kl = k.lower()
                if kl.startswith("eff_mass_") and kl.endswith("_pct"):
                    # e.g. eff_mass_Ux_pct  -> Ux%
                    dirname = k[len("eff_mass_"):-len("_pct")]
                    pf[dirname + "%"] = float(v)
            participation.append(pf)
    return {"frequencies": freqs, "participation": participation}


# ---------------------------------------------------------------------------
# Comparative table: CCX vs fem-shell side by side
# ---------------------------------------------------------------------------

def _dominant_type(pf_row, cols):
    # type: (Dict[str, float], List[str]) -> str
    """Return the label of the column with the highest participation value."""
    if not cols or not pf_row:
        return "-"
    vals = [pf_row.get(c, 0.0) for c in cols]
    return cols[vals.index(max(vals))].rstrip("%")


def display_comparison(
    ccx_freqs,        # type: List[float]
    ccx_pf,           # type: List[Dict[str, float]]
    fem_freqs,        # type: List[float]
    fem_pf,           # type: List[Dict[str, float]]
    ref_freqs=None,   # type: Optional[List[float]]
    ref_label="Ref",  # type: str
):
    # type: (...) -> None
    """Side-by-side comparison table: CalculiX | fem-shell | Delta."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
    except ImportError:
        _display_comparison_plain(ccx_freqs, ccx_pf, fem_freqs, fem_pf, ref_freqs, ref_label)
        return

    console = Console()
    n = min(len(ccx_freqs), len(fem_freqs))

    # Determine participation columns present in CCX results
    pf_cols = [lb + "%" for lb in _DIR_LABELS if ccx_pf and lb + "%" in ccx_pf[0]]

    table = Table(
        title="MODAL COMPARISON: CalculiX vs fem-shell",
        title_style="bold cyan",
        show_lines=False,
        pad_edge=True,
        box=box.SIMPLE_HEAVY,
    )

    table.add_column("Mode", justify="right", style="bold")
    # CalculiX columns
    table.add_column("CCX [Hz]", justify="right", style="cyan")
    table.add_column("CCX Type", justify="center", style="cyan")
    # fem-shell columns
    table.add_column("FEM [Hz]", justify="right", style="green")
    table.add_column("FEM Type", justify="center", style="green")
    # Delta
    table.add_column("CCX-FEM %", justify="right")
    # Optional reference
    if ref_freqs:
        table.add_column("%s [Hz]" % ref_label, justify="right", style="dim")
        table.add_column("CCX-Ref %", justify="right")
        table.add_column("FEM-Ref %", justify="right")

    deltas_ccx_fem = []
    deltas_ccx_ref = []
    deltas_fem_ref = []

    for i in range(n):
        fc = ccx_freqs[i]
        ff = fem_freqs[i]
        delta_cf = (fc - ff) / ff * 100.0 if ff > 0 else float("nan")
        deltas_ccx_fem.append(abs(delta_cf))

        ccx_type = _dominant_type(ccx_pf[i] if i < len(ccx_pf) else {}, pf_cols)
        fem_type = _dominant_type(fem_pf[i] if i < len(fem_pf) else {}, pf_cols)

        sign_cf = "+" if delta_cf > 0 else ""
        color_cf = "green" if abs(delta_cf) < 1 else "yellow" if abs(delta_cf) < 5 else "red"

        row = [
            str(i + 1),
            "%.6f" % fc,
            ccx_type,
            "%.6f" % ff,
            fem_type,
            "[%s]%s%.2f%%[/%s]" % (color_cf, sign_cf, delta_cf, color_cf),
        ]

        if ref_freqs and i < len(ref_freqs):
            fr = ref_freqs[i]
            d_cr = (fc - fr) / fr * 100.0 if fr > 0 else float("nan")
            d_fr = (ff - fr) / fr * 100.0 if fr > 0 else float("nan")
            deltas_ccx_ref.append(abs(d_cr))
            deltas_fem_ref.append(abs(d_fr))
            sign_cr = "+" if d_cr > 0 else ""
            sign_fr = "+" if d_fr > 0 else ""
            col_cr = "green" if abs(d_cr) < 5 else "yellow" if abs(d_cr) < 20 else "red"
            col_fr = "green" if abs(d_fr) < 5 else "yellow" if abs(d_fr) < 20 else "red"
            row += [
                "%.6f" % fr,
                "[%s]%s%.1f%%[/%s]" % (col_cr, sign_cr, d_cr, col_cr),
                "[%s]%s%.1f%%[/%s]" % (col_fr, sign_fr, d_fr, col_fr),
            ]
        elif ref_freqs:
            row += ["-", "-", "-"]

        table.add_row(*row)

    console.print()
    console.print(table)

    # Summary line
    if deltas_ccx_fem:
        mean_cf = sum(deltas_ccx_fem) / len(deltas_ccx_fem)
        console.print(
            "  CCX vs fem-shell mean |delta|: [bold cyan]%.2f%%[/bold cyan]" % mean_cf
        )
    if deltas_ccx_ref:
        mean_cr = sum(deltas_ccx_ref) / len(deltas_ccx_ref)
        mean_fr = sum(deltas_fem_ref) / len(deltas_fem_ref)
        console.print(
            "  CCX vs %s mean |delta|: [bold]%.1f%%[/bold]   "
            "fem-shell vs %s mean |delta|: [bold]%.1f%%[/bold]" % (
                ref_label, mean_cr, ref_label, mean_fr)
        )
    console.print()


def _display_comparison_plain(
    ccx_freqs, ccx_pf, fem_freqs, fem_pf, ref_freqs=None, ref_label="Ref"
):
    # type: (...) -> None
    n = min(len(ccx_freqs), len(fem_freqs))
    pf_cols = [lb + "%" for lb in _DIR_LABELS if ccx_pf and lb + "%" in ccx_pf[0]]

    print("\n" + "=" * 80)
    print("  MODAL COMPARISON: CalculiX vs fem-shell")
    print("=" * 80)

    hdr = "%5s  %12s  %8s  %12s  %8s  %10s" % (
        "Mode", "CCX[Hz]", "CCXType", "FEM[Hz]", "FEMType", "CCX-FEM%")
    if ref_freqs:
        hdr += "  %12s  %8s  %8s" % (ref_label + "[Hz]", "CCX-Ref%", "FEM-Ref%")
    print(hdr)
    print("-" * len(hdr))

    for i in range(n):
        fc = ccx_freqs[i]
        ff = fem_freqs[i]
        delta_cf = (fc - ff) / ff * 100.0 if ff > 0 else float("nan")
        ccx_type = _dominant_type(ccx_pf[i] if i < len(ccx_pf) else {}, pf_cols)
        fem_type = _dominant_type(fem_pf[i] if i < len(fem_pf) else {}, pf_cols)
        line = "%5d  %12.6f  %8s  %12.6f  %8s  %+9.2f%%" % (
            i + 1, fc, ccx_type, ff, fem_type, delta_cf)
        if ref_freqs and i < len(ref_freqs):
            fr = ref_freqs[i]
            d_cr = (fc - fr) / fr * 100.0 if fr > 0 else float("nan")
            d_fr = (ff - fr) / fr * 100.0 if fr > 0 else float("nan")
            line += "  %12.6f  %+7.1f%%  %+7.1f%%" % (fr, d_cr, d_fr)
        print(line)
    print()



# ---------------------------------------------------------------------------
# Display — Rich table
# ---------------------------------------------------------------------------

# Direction labels shared by display functions
_DIR_LABELS = ["Ux", "Uy", "Uz", "Rx", "Ry", "Rz"]


def display_results(
    ccx_frequencies,   # type: List[float]
    participation,     # type: List[Dict[str, float]]  — Ux%..Rz% from .dat
    ref_frequencies=None,  # type: Optional[List[float]]
    label="CalculiX",  # type: str
    dir_content=None,  # type: Optional[List[Dict[str, float]]]  — legacy FRD
):
    # type: (...) -> None
    """Print results in a Rich table matching fem-shell modal output."""
    # Merge FRD dir_content into participation format when dat participation
    # is unavailable but FRD data was parsed
    if not participation and dir_content:
        participation = [
            {"Ux%": d["Ux_pct"], "Uy%": d["Uy_pct"], "Uz%": d["Uz_pct"]}
            for d in dir_content
        ]

    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
    except ImportError:
        _display_plain(ccx_frequencies, participation, ref_frequencies, label)
        return

    console = Console()
    n = len(ccx_frequencies)
    has_pf = len(participation) >= n
    has_ref = ref_frequencies is not None and len(ref_frequencies) > 0

    # Determine which direction columns are present
    if has_pf:
        sample = participation[0]
        pf_cols = [lb + "%" for lb in _DIR_LABELS if lb + "%" in sample]
    else:
        pf_cols = []

    table = Table(
        title="MODAL ANALYSIS RESULTS — " + label,
        title_style="bold cyan",
        show_lines=False,
        pad_edge=True,
        box=box.SIMPLE_HEAVY,
    )

    table.add_column("Mode", justify="right", style="bold")
    table.add_column("Freq [Hz]", justify="right")
    table.add_column("Period [s]", justify="right")
    table.add_column("omega [rad/s]", justify="right")

    for col in pf_cols:
        table.add_column(col, justify="right")
    if pf_cols:
        table.add_column("Type", justify="right", style="bold green")

    if has_ref:
        table.add_column("Ref [Hz]", justify="right", style="dim")
        table.add_column("Delta%", justify="right")

    for i in range(n):
        f = ccx_frequencies[i]
        omega = 2.0 * math.pi * f
        period = 1.0 / f if f > 0 else float("inf")

        row = [
            str(i + 1),
            "%.6f" % f,
            "%.6f" % period,
            "%.4f" % omega,
        ]

        if has_pf and i < len(participation):
            d = participation[i]
            vals = [d.get(col, 0.0) for col in pf_cols]
            dom_idx = vals.index(max(vals))
            dominant = pf_cols[dom_idx].rstrip("%")
            row += ["%.1f" % v for v in vals]
            row.append(dominant)

        if has_ref and i < len(ref_frequencies):
            ref_f = ref_frequencies[i]
            delta = (f - ref_f) / ref_f * 100.0 if ref_f > 0 else float("nan")
            sign = "+" if delta > 0 else ""
            color = "green" if abs(delta) < 5 else "yellow" if abs(delta) < 20 else "red"
            row += ["%.6f" % ref_f, "[%s]%s%.1f%%[/%s]" % (color, sign, delta, color)]

        table.add_row(*row)

    console.print()
    console.print(table)

    # Summary
    if has_ref:
        valid = [
            (ccx_frequencies[i], ref_frequencies[i])
            for i in range(min(n, len(ref_frequencies)))
            if ref_frequencies[i] > 0
        ]
        if valid:
            deltas = [abs((c - r) / r * 100) for c, r in valid]
            console.print(
                "  Mean absolute error vs reference: [bold]%.1f%%[/bold]" % (sum(deltas) / len(deltas))
            )
    console.print()


def _display_plain(
    ccx_frequencies,   # type: List[float]
    participation,     # type: List[Dict[str, float]]
    ref_frequencies,   # type: Optional[List[float]]
    label,             # type: str
):
    # type: (...) -> None
    """Fallback plain-text table (no rich dependency)."""
    sep = "=" * 70
    print("\n" + sep)
    print("  MODAL ANALYSIS RESULTS -- " + label)
    print(sep)

    if participation:
        sample = participation[0]
        pf_cols = [lb + "%" for lb in _DIR_LABELS if lb + "%" in sample]
    else:
        pf_cols = []

    header = "%5s  %12s  %12s  %12s" % ("Mode", "Freq[Hz]", "Period[s]", "omega[rad/s]")
    for col in pf_cols:
        header += "  %6s" % col
    if pf_cols:
        header += "  %6s" % "Type"
    if ref_frequencies:
        header += "  %12s  %8s" % ("Ref[Hz]", "Delta%")
    print(header)
    print("-" * len(header))

    for i, f in enumerate(ccx_frequencies):
        omega = 2.0 * math.pi * f
        period = 1.0 / f if f > 0 else float("inf")
        line = "%5d  %12.6f  %12.6f  %12.4f" % (i + 1, f, period, omega)
        if i < len(participation):
            d = participation[i]
            vals = [d.get(col, 0.0) for col in pf_cols]
            dom = pf_cols[vals.index(max(vals))].rstrip("%") if vals else ""
            for v in vals:
                line += "  %6.1f" % v
            if dom:
                line += "  %6s" % dom
        if ref_frequencies and i < len(ref_frequencies):
            r = ref_frequencies[i]
            delta = (f - r) / r * 100 if r > 0 else float("nan")
            line += "  %12.6f  %+8.1f%%" % (r, delta)
        print(line)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Post-process CalculiX modal analysis output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("dat", metavar="FILE.dat", help="CalculiX .dat output file")
    p.add_argument(
        "--frd",
        metavar="FILE.frd",
        default=None,
        help="CalculiX .frd output file for directional content analysis",
    )
    p.add_argument(
        "--ref",
        metavar="F1,F2,...",
        default=None,
        help="Comma-separated reference frequencies in Hz for comparison",
    )
    p.add_argument(
        "--fem-csv",
        metavar="FILE.csv",
        default=None,
        help="CSV exported from fem-shell modal solver for comparison",
    )
    p.add_argument(
        "--n-modes",
        type=int,
        default=20,
        help="Maximum number of modes to display (default: 20)",
    )
    p.add_argument(
        "--label",
        default="CalculiX",
        help="Label for the solver column (default: CalculiX)",
    )
    return p


def main():
    args = build_parser().parse_args()

    dat_path = Path(args.dat)
    if not dat_path.exists():
        print("ERROR: .dat file not found: %s" % dat_path, file=sys.stderr)
        sys.exit(1)

    print("Parsing %s ..." % dat_path, file=sys.stderr)
    parsed = parse_dat(dat_path)
    frequencies = parsed["frequencies"]
    participation = parsed["participation"]

    if not frequencies:
        print(
            "WARNING: No frequencies found in .dat file.\n"
            "  Possible causes:\n"
            "  * CalculiX has not finished yet (empty .dat)\n"
            "  * Analysis is not a *FREQUENCY step\n"
            "  * Different output format -- check the .dat file manually",
            file=sys.stderr,
        )
        sys.exit(1)

    frequencies = frequencies[: args.n_modes]
    participation = participation[: args.n_modes]
    n = len(frequencies)
    print("Found %d mode(s)." % n, file=sys.stderr)
    if participation:
        print("Effective mass participation parsed from .dat.", file=sys.stderr)

    # FRD fallback for directional content when .dat has no participation
    dir_content = []  # type: List[Dict[str, float]]
    if not participation and args.frd:
        frd_path = Path(args.frd)
        if frd_path.exists():
            print("Parsing %s for mode shapes ..." % frd_path, file=sys.stderr)
            dir_content = parse_frd_mode_shapes(frd_path, n_modes=n)
            print("  Parsed %d mode shape(s)." % len(dir_content), file=sys.stderr)
        else:
            print("WARNING: .frd file not found: %s" % frd_path, file=sys.stderr)

    # Reference frequencies
    ref_frequencies = None  # type: Optional[List[float]]
    if args.ref:
        ref_frequencies = [float(x.strip()) for x in args.ref.split(",") if x.strip()]

    # Comparison mode: side-by-side table when --fem-csv is given
    if args.fem_csv:
        fem_path = Path(args.fem_csv)
        if not fem_path.exists():
            print("WARNING: fem-shell CSV not found: %s" % fem_path, file=sys.stderr)
        else:
            print("Parsing fem-shell CSV: %s ..." % fem_path, file=sys.stderr)
            fem_data = parse_femshell_csv_full(fem_path)
            fem_freqs = fem_data["frequencies"][: args.n_modes]
            fem_pf = fem_data["participation"][: args.n_modes]
            display_comparison(
                ccx_freqs=frequencies,
                ccx_pf=participation,
                fem_freqs=fem_freqs,
                fem_pf=fem_pf,
                ref_freqs=ref_frequencies,
                ref_label="Ref",
            )
            return

    display_results(
        ccx_frequencies=frequencies,
        participation=participation,
        dir_content=dir_content,
        ref_frequencies=ref_frequencies,
        label=args.label,
    )


if __name__ == "__main__":
    main()
