"""
Aerodynamic data layer for wind turbine blades.

Loads airfoil polar tables (Cl, Cd, Cm vs alpha) from WindIO YAML files or
NuMAD Excel files, with NeuralFoil as a fallback (YAML) or primary source
(Excel) for polar generation. Provides interpolated blade aerodynamic
properties along the span for use by BEM solvers.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml


@dataclass
class PolarData:
    """Single-Re airfoil polar table.

    Parameters
    ----------
    alpha : ndarray
        Angle of attack in radians.
    cl : ndarray
        Lift coefficient.
    cd : ndarray
        Drag coefficient.
    cm : ndarray
        Moment coefficient.
    re : float
        Reynolds number for this polar.
    """

    alpha: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    cm: np.ndarray
    re: float

    def evaluate(self, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate Cl, Cd, Cm at given angles of attack.

        Uses linear interpolation with periodic wrapping on [-pi, pi].
        """
        alpha = np.asarray(alpha, dtype=float)
        # Wrap alpha to [-pi, pi]
        alpha_wrapped = (alpha + np.pi) % (2 * np.pi) - np.pi
        cl = np.interp(alpha_wrapped, self.alpha, self.cl)
        cd = np.interp(alpha_wrapped, self.alpha, self.cd)
        cm = np.interp(alpha_wrapped, self.alpha, self.cm)
        return cl, cd, cm


@dataclass
class AirfoilAero:
    """Aerodynamic data for a single airfoil.

    Parameters
    ----------
    name : str
        Airfoil name (e.g. "FFA-W3-211").
    coordinates : ndarray
        Airfoil shape, Nx2 array of (x, y) normalised coordinates.
    relative_thickness : float
        Maximum relative thickness (t/c).
    aerodynamic_center : float
        Chordwise position of aerodynamic center (fraction of chord).
    polars : list of PolarData
        Polar tables at different Reynolds numbers.
    """

    name: str
    coordinates: np.ndarray
    relative_thickness: float
    aerodynamic_center: float
    polars: List[PolarData] = field(default_factory=list)

    def get_polar(self, re: float) -> PolarData:
        """Return polar for the closest Reynolds number."""
        if len(self.polars) == 1:
            return self.polars[0]
        re_vals = np.array([p.re for p in self.polars])
        idx = int(np.argmin(np.abs(re_vals - re)))
        return self.polars[idx]


@dataclass
class AeroStation:
    """Aerodynamic properties at a single spanwise station.

    Parameters
    ----------
    span_fraction : float
        Normalised span position [0, 1].
    r : float
        Radial position from hub center in metres.
    chord : float
        Chord length in metres.
    twist : float
        Aerodynamic twist in radians (positive nose-up).
    pitch_axis : float
        Chordwise pitch-axis position (fraction of chord).
    airfoil : AirfoilAero
        Reference airfoil for this station.
    """

    span_fraction: float
    r: float
    chord: float
    twist: float
    pitch_axis: float
    airfoil: AirfoilAero


@dataclass
class BladeAero:
    """Complete blade aerodynamic definition.

    Parameters
    ----------
    airfoils : list of AirfoilAero
        All airfoils defined in the YAML.
    stations : list of AeroStation
        Spanwise aerodynamic stations (ordered root → tip).
    blade_length : float
        Blade span from root to tip in metres.
    hub_radius : float
        Hub radius in metres.
    rotor_radius : float
        Rotor tip radius in metres.
    n_blades : int
        Number of blades.
    """

    airfoils: List[AirfoilAero]
    stations: List[AeroStation]
    blade_length: float
    hub_radius: float
    rotor_radius: float
    n_blades: int

    @property
    def r(self) -> np.ndarray:
        """Radial positions of all stations (m)."""
        return np.array([s.r for s in self.stations])

    @property
    def chord(self) -> np.ndarray:
        """Chord at each station (m)."""
        return np.array([s.chord for s in self.stations])

    @property
    def twist(self) -> np.ndarray:
        """Twist at each station (rad)."""
        return np.array([s.twist for s in self.stations])


def _parse_polars_from_yaml(af_data: dict) -> List[PolarData]:
    """Extract polar tables from a WindIO airfoil entry."""
    polars = []
    for polar_entry in af_data.get("polars", []):
        re_val = float(polar_entry["re"])
        alpha = np.array(polar_entry["c_l"]["grid"], dtype=float)
        cl = np.array(polar_entry["c_l"]["values"], dtype=float)
        cd = np.array(polar_entry["c_d"]["values"], dtype=float)
        cm = np.array(polar_entry["c_m"]["values"], dtype=float)
        # Ensure alpha is sorted
        sort_idx = np.argsort(alpha)
        polars.append(
            PolarData(
                alpha=alpha[sort_idx],
                cl=cl[sort_idx],
                cd=cd[sort_idx],
                cm=cm[sort_idx],
                re=re_val,
            )
        )
    return polars


def _viterna_extrapolation(
    alpha_attach: np.ndarray,
    cl_attach: np.ndarray,
    cd_attach: np.ndarray,
    cm_attach: np.ndarray,
    ar: float = 17.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extend a polar to [-180°, +180°] using the Viterna–Corrigan method.

    Parameters
    ----------
    alpha_attach : ndarray
        Angles of attack (radians) in the attached-flow regime (typically
        the NeuralFoil output for |alpha| <= alpha_stall_deg).
    cl_attach, cd_attach, cm_attach : ndarray
        Corresponding aerodynamic coefficients.
    ar : float
        Blade aspect ratio used in the Viterna equations. Defaults to 17
        (representative of IEA-15 MW outer sections).

    Returns
    -------
    alpha_full, cl_full, cd_full, cm_full : ndarray
        Polar tables spanning [-π, +π] at 1° resolution.
    """
    # Viterna empirical limit for maximum drag at 90°
    cd_max = max(1.11 + 0.13 * ar, 1.11)  # Viterna eq. 8, capped at ~1.40 for large AR

    # Reference point: take the stall angle from the last attached-flow point
    # (last alpha before we switch to Viterna on each side)
    alpha_stall_pos = alpha_attach[-1]   # e.g. +25°
    alpha_stall_neg = alpha_attach[0]    # e.g. -25°
    cl_stall_pos = cl_attach[-1]
    cd_stall_pos = cd_attach[-1]
    cl_stall_neg = cl_attach[0]
    cd_stall_neg = cd_attach[0]

    # Viterna coefficients for the positive (+) side
    A2p = (cl_stall_pos - cd_max * np.sin(alpha_stall_pos) * np.cos(alpha_stall_pos)) * (
        np.sin(alpha_stall_pos) / np.cos(alpha_stall_pos) ** 2
    )
    B2p = (cd_stall_pos - cd_max * np.sin(alpha_stall_pos) ** 2) / np.cos(alpha_stall_pos)

    # Viterna coefficients for the negative (-) side (mirror)
    A2n = (cl_stall_neg - cd_max * np.sin(alpha_stall_neg) * np.cos(alpha_stall_neg)) * (
        np.sin(alpha_stall_neg) / np.cos(alpha_stall_neg) ** 2
    )
    B2n = (cd_stall_neg - cd_max * np.sin(alpha_stall_neg) ** 2) / np.cos(alpha_stall_neg)

    alpha_full_deg = np.arange(-180, 181, 1, dtype=float)
    alpha_full = np.deg2rad(alpha_full_deg)

    cl_full = np.zeros_like(alpha_full)
    cd_full = np.zeros_like(alpha_full)
    cm_full = np.zeros_like(alpha_full)

    for i, a in enumerate(alpha_full):
        a_deg = alpha_full_deg[i]
        if alpha_stall_neg <= a <= alpha_stall_pos:
            # Attached-flow regime: interpolate from NeuralFoil data
            cl_full[i] = np.interp(a, alpha_attach, cl_attach)
            cd_full[i] = np.interp(a, alpha_attach, cd_attach)
            cm_full[i] = np.interp(a, alpha_attach, cm_attach)
        elif a > alpha_stall_pos:
            # Viterna equations — post-stall positive
            sa, ca = np.sin(a), np.cos(a)
            # Guard against sin(α)≈0 at α=±180° where Viterna diverges
            if abs(sa) < 1e-6:
                cl_full[i] = 0.0
                cd_full[i] = cd_max
            else:
                cl_full[i] = cd_max / 2.0 * np.sin(2.0 * a) + A2p * ca ** 2 / sa
                cd_full[i] = cd_max * sa ** 2 + B2p * ca
            cm_full[i] = np.interp(a, alpha_attach, cm_attach, left=cm_attach[-1], right=cm_attach[-1])
        else:
            # Viterna equations — post-stall negative (mirror about zero)
            sa, ca = np.sin(a), np.cos(a)
            # Guard against sin(α)≈0 at α=±180° where Viterna diverges
            if abs(sa) < 1e-6:
                cl_full[i] = 0.0
                cd_full[i] = cd_max
            else:
                cl_full[i] = cd_max / 2.0 * np.sin(2.0 * a) + A2n * ca ** 2 / sa
                cd_full[i] = cd_max * sa ** 2 - B2n * ca
            cm_full[i] = np.interp(a, alpha_attach, cm_attach, left=cm_attach[0], right=cm_attach[0])

    # Ensure cd is non-negative everywhere
    cd_full = np.maximum(cd_full, 0.0)

    return alpha_full, cl_full, cd_full, cm_full


def _generate_polars_neuralfoil(
    coordinates: np.ndarray,
    re: float = 1e7,
    model_size: str = "large",
    alpha_stall_deg: Optional[float] = None,
    ar: float = 17.0,
    confidence_threshold: float = 0.5,
) -> List[PolarData]:
    """Generate polar tables using NeuralFoil + Viterna–Corrigan extrapolation.

    NeuralFoil is used for the attached-flow regime. When ``alpha_stall_deg``
    is ``None`` (the default) the stall boundary is detected automatically
    using NeuralFoil's ``analysis_confidence`` output: the last angle where
    confidence >= ``confidence_threshold`` is used as the Viterna hand-off
    point, capped at ±30° to avoid over-extension.  Viterna–Corrigan takes
    over beyond the stall boundary to extend the polar to ±180°.

    Parameters
    ----------
    coordinates : ndarray
        Airfoil Nx2 coordinates (normalised, (x,y)).
    re : float
        Reynolds number.
    model_size : str
        NeuralFoil model size ("xlarge", "large", "medium", etc.).
    alpha_stall_deg : float or None
        Half-range (degrees) for the NeuralFoil / attached-flow region.
        ``None`` (default) triggers auto-detection via ``analysis_confidence``.
    ar : float
        Effective aspect ratio used in Viterna's cd_max formula
        (``cd_max = 1.11 + 0.13·AR``).  Set to a small value (e.g. 3–5) for
        cylindrical root sections where CD90 ≈ 1.0.
    confidence_threshold : float
        Minimum ``analysis_confidence`` to consider a NeuralFoil point
        reliable.  Only used when ``alpha_stall_deg`` is ``None``.
    """
    try:
        import neuralfoil as nf
    except ImportError:
        raise ImportError(
            "NeuralFoil is required to generate polars from coordinates. "
            "Install it with: pip install neuralfoil"
        )

    # Scan a wide range so auto-detection can observe the confidence drop-off.
    # Even when alpha_stall_deg is given we still scan widely so we hand off
    # at the user-specified angle with the correct coefficients.
    alpha_scan_deg = np.linspace(-35.0, 35.0, 281)

    aero = nf.get_aero_from_coordinates(
        coordinates=coordinates,
        alpha=alpha_scan_deg,
        Re=re,
        model_size=model_size,
    )

    cl_scan = np.asarray(aero["CL"], dtype=float)
    cd_scan = np.asarray(aero["CD"], dtype=float)
    cm_scan = np.asarray(aero["CM"], dtype=float)
    conf = np.asarray(aero["analysis_confidence"], dtype=float)

    if alpha_stall_deg is None:
        # Auto-detect: find the outermost alpha on each side where NeuralFoil
        # still claims confidence >= threshold.
        confident = conf >= confidence_threshold
        pos_idx = np.where((alpha_scan_deg > 0) & confident)[0]
        neg_idx = np.where((alpha_scan_deg < 0) & confident)[0]

        stall_pos = float(alpha_scan_deg[pos_idx[-1]]) if pos_idx.size else 20.0
        stall_neg = float(alpha_scan_deg[neg_idx[0]]) if neg_idx.size else -20.0

        # Safety cap — never extend the attached regime past ±30°
        stall_pos = min(stall_pos, 30.0)
        stall_neg = max(stall_neg, -30.0)
    else:
        stall_pos = float(alpha_stall_deg)
        stall_neg = -float(alpha_stall_deg)

    # Extract the attached-flow region (NeuralFoil data inside stall boundary)
    mask = (alpha_scan_deg >= stall_neg) & (alpha_scan_deg <= stall_pos)
    alpha_attach_rad = np.deg2rad(alpha_scan_deg[mask])
    cl_attach = cl_scan[mask]
    cd_attach = cd_scan[mask]
    cm_attach = cm_scan[mask]

    alpha_full, cl_full, cd_full, cm_full = _viterna_extrapolation(
        alpha_attach_rad, cl_attach, cd_attach, cm_attach, ar=ar
    )

    return [
        PolarData(
            alpha=alpha_full,
            cl=cl_full,
            cd=cd_full,
            cm=cm_full,
            re=re,
        )
    ]


def _load_from_excel(
    excel_path: Path,
    default_re: float,
    neuralfoil_model: str,
    hub_radius: float,
    n_blades: int,
    airfoil_dir: Optional[str],
    viterna_ar: float = 17.0,
    viterna_confidence_threshold: float = 0.5,
) -> BladeAero:
    """Load blade aerodynamic data from a NuMAD Excel file.

    Since Excel files do not contain polar data, all polars are generated
    via NeuralFoil from the airfoil coordinates.
    """
    from fem_shell.models.blade.numad.objects.blade import Blade as NumadBlade

    blade = NumadBlade()
    if airfoil_dir is not None:
        blade.read_excel(str(excel_path), airfoil_dir=airfoil_dir)
    else:
        blade.read_excel(str(excel_path))
    defn = blade.definition

    blade_length = float(defn.span[-1])
    rotor_radius = hub_radius + blade_length

    # --- Build unique AirfoilAero objects from station airfoils ---
    airfoils: List[AirfoilAero] = []
    af_name_map: dict[str, int] = {}
    af_span_locs: List[float] = []
    af_indices: List[int] = []

    for station in defn.stations:
        af = station.airfoil
        if af is None or af.coordinates is None:
            continue
        if af.name not in af_name_map:
            coords = np.asarray(af.coordinates, dtype=float)
            rel_t = float(af.percentthick / 100.0) if af.percentthick else 0.0

            polars = _generate_polars_neuralfoil(
                coords, re=default_re, model_size=neuralfoil_model,
                ar=viterna_ar, confidence_threshold=viterna_confidence_threshold,
            )

            airfoil_aero = AirfoilAero(
                name=af.name,
                coordinates=coords,
                relative_thickness=rel_t,
                aerodynamic_center=0.25,
                polars=polars,
            )
            af_name_map[af.name] = len(airfoils)
            airfoils.append(airfoil_aero)

        if station.spanlocation is None:
            continue
        af_span_locs.append(float(station.spanlocation))
        af_indices.append(af_name_map[af.name])

    af_span_arr = np.array(af_span_locs)
    af_idx_arr = np.array(af_indices)

    # --- Build stations on the definition span grid ---
    span = defn.span
    chord = defn.chord
    twist_rad = np.deg2rad(defn.degreestwist)
    aerocenter = defn.aerocenter if defn.aerocenter is not None else np.full_like(span, 0.25)

    stations: List[AeroStation] = []
    for i, s in enumerate(span):
        eta = s / blade_length
        r = hub_radius + s

        nearest = int(np.argmin(np.abs(af_span_arr - s)))
        af_idx = af_idx_arr[nearest]

        stations.append(
            AeroStation(
                span_fraction=float(eta),
                r=float(r),
                chord=float(chord[i]),
                twist=float(twist_rad[i]),
                pitch_axis=float(aerocenter[i]),
                airfoil=airfoils[af_idx],
            )
        )

    return BladeAero(
        airfoils=airfoils,
        stations=stations,
        blade_length=blade_length,
        hub_radius=hub_radius,
        rotor_radius=rotor_radius,
        n_blades=n_blades,
    )


def load_blade_aero(
    blade_file: str,
    default_re: float = 1e7,
    neuralfoil_model: str = "large",
    *,
    hub_radius: float = 0.0,
    n_blades: int = 3,
    airfoil_dir: Optional[str] = None,
    viterna_ar: float = 17.0,
    viterna_confidence_threshold: float = 0.5,
) -> BladeAero:
    """Load blade aerodynamic data from a WindIO YAML or NuMAD Excel file.

    For YAML files, parses the ``outer_shape_bem`` section for chord, twist,
    pitch axis, and reference axis, and the ``airfoils`` section for polar
    data.  When an airfoil entry has no ``polars`` key, NeuralFoil is used
    as a fallback to generate Cl/Cd/Cm from the airfoil coordinates.

    For Excel files (.xlsx/.xls), loads the blade definition using the NuMAD
    parser and generates *all* polars via NeuralFoil (Excel files do not
    contain polar data).  The ``hub_radius``, ``n_blades``, and ``airfoil_dir``
    keyword arguments are only used for the Excel path.

    Parameters
    ----------
    blade_file : str
        Path to a WindIO YAML (.yaml/.yml) or NuMAD Excel (.xlsx/.xls) file.
    default_re : float
        Reynolds number used for NeuralFoil polar generation.
    neuralfoil_model : str
        NeuralFoil model size ("xlarge", "large", "medium", "small", "xsmall").
    hub_radius : float
        Hub radius in metres (Excel only; YAML reads this from the file).
    n_blades : int
        Number of blades (Excel only; YAML reads this from the file).
    airfoil_dir : str or None
        Directory containing airfoil coordinate .txt files (Excel only).
        Defaults to an ``airfoils/`` folder next to the Excel file.
    viterna_ar : float
        Aspect ratio for the Viterna–Corrigan cd_max formula
        (``cd_max = 1.11 + 0.13·AR``).  Default 17 suits IEA-15 outer sections.
        Use a smaller value (e.g. 3–5) for cylindrical root sections.
    viterna_confidence_threshold : float
        NeuralFoil ``analysis_confidence`` threshold below which a point is
        considered outside the reliable regime and handed off to Viterna.
        Only used when the blade file does not supply pre-computed polars.

    Returns
    -------
    BladeAero
        Complete blade aerodynamic definition.
    """
    blade_path = Path(blade_file)
    ext = blade_path.suffix.lower()

    if ext in (".xlsx", ".xls"):
        return _load_from_excel(
            blade_path,
            default_re=default_re,
            neuralfoil_model=neuralfoil_model,
            hub_radius=hub_radius,
            n_blades=n_blades,
            airfoil_dir=airfoil_dir,
            viterna_ar=viterna_ar,
            viterna_confidence_threshold=viterna_confidence_threshold,
        )

    # --- YAML path ---
    with open(blade_path) as f:
        data = yaml.load(f, Loader=yaml.Loader)

    # --- Assembly info ---
    assembly = data.get("assembly", {})
    n_blades = int(assembly.get("number_of_blades", 3))
    rotor_diameter = float(assembly.get("rotor_diameter", 0))

    # --- Hub info ---
    hub_data = data.get("components", {}).get("hub", {})
    hub_osb = hub_data.get("outer_shape_bem", hub_data)
    hub_diameter = float(hub_osb.get("diameter", 0))
    hub_radius = hub_diameter / 2.0

    # --- Blade outer shape BEM ---
    blade_bem = data["components"]["blade"]["outer_shape_bem"]

    # Reference axis z → blade length
    ref_z_grid = np.array(blade_bem["reference_axis"]["z"]["grid"], dtype=float)
    ref_z_vals = np.array(blade_bem["reference_axis"]["z"]["values"], dtype=float)
    blade_length = ref_z_vals[-1]
    rotor_radius = hub_radius + blade_length if rotor_diameter == 0 else rotor_diameter / 2.0

    # Chord
    chord_grid = np.array(blade_bem["chord"]["grid"], dtype=float)
    chord_vals = np.array(blade_bem["chord"]["values"], dtype=float)

    # Twist (already in radians in WindIO)
    twist_grid = np.array(blade_bem["twist"]["grid"], dtype=float)
    twist_vals = np.array(blade_bem["twist"]["values"], dtype=float)

    # Pitch axis
    pa_grid = np.array(blade_bem["pitch_axis"]["grid"], dtype=float)
    pa_vals = np.array(blade_bem["pitch_axis"]["values"], dtype=float)

    # Airfoil position along span
    af_pos_grid = np.array(blade_bem["airfoil_position"]["grid"], dtype=float)
    af_pos_labels = blade_bem["airfoil_position"]["labels"]

    # --- Parse airfoils ---
    af_data_list = data["airfoils"]
    af_name_map: dict[str, int] = {}
    airfoils: List[AirfoilAero] = []

    for i, af_entry in enumerate(af_data_list):
        name = af_entry["name"]
        x = np.array(af_entry["coordinates"]["x"], dtype=float)
        y = np.array(af_entry["coordinates"]["y"], dtype=float)
        coords = np.column_stack((x, y))

        rel_t = float(af_entry.get("relative_thickness", 0))
        ac = float(af_entry.get("aerodynamic_center", 0.25))

        # Try parsing polars from YAML
        polars = _parse_polars_from_yaml(af_entry)

        # Fallback: generate with NeuralFoil if no polars
        if not polars:
            polars = _generate_polars_neuralfoil(
                coords, re=default_re, model_size=neuralfoil_model,
                ar=viterna_ar, confidence_threshold=viterna_confidence_threshold,
            )

        airfoil = AirfoilAero(
            name=name,
            coordinates=coords,
            relative_thickness=rel_t,
            aerodynamic_center=ac,
            polars=polars,
        )
        airfoils.append(airfoil)
        af_name_map[name] = i

    # --- Build airfoil index for each span fraction ---
    # Map airfoil labels to indices
    af_indices_at_pos = []
    for label in af_pos_labels:
        if label not in af_name_map:
            raise ValueError(
                f"Airfoil '{label}' referenced in airfoil_position but not found in airfoils section"
            )
        af_indices_at_pos.append(af_name_map[label])

    # --- Build stations at chord/twist grid points ---
    # Use the union of chord grid as the primary station grid
    station_grid = chord_grid.copy()

    # Interpolate quantities onto station grid
    station_chord = np.interp(station_grid, chord_grid, chord_vals)
    station_twist = np.interp(station_grid, twist_grid, twist_vals)
    station_pa = np.interp(station_grid, pa_grid, pa_vals)

    # For each station, find the nearest airfoil by interpolating af_pos_grid
    af_indices_float = np.interp(
        station_grid, af_pos_grid, np.arange(len(af_pos_grid), dtype=float)
    )
    af_indices_nearest = np.round(af_indices_float).astype(int)
    af_indices_nearest = np.clip(af_indices_nearest, 0, len(af_pos_labels) - 1)

    stations: List[AeroStation] = []
    for i, eta in enumerate(station_grid):
        r = hub_radius + eta * blade_length
        af_idx = af_indices_at_pos[af_indices_nearest[i]]
        stations.append(
            AeroStation(
                span_fraction=float(eta),
                r=float(r),
                chord=float(station_chord[i]),
                twist=float(station_twist[i]),
                pitch_axis=float(station_pa[i]),
                airfoil=airfoils[af_idx],
            )
        )

    return BladeAero(
        airfoils=airfoils,
        stations=stations,
        blade_length=float(blade_length),
        hub_radius=float(hub_radius),
        rotor_radius=float(rotor_radius),
        n_blades=n_blades,
    )
