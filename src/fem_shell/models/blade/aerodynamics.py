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


def _generate_polars_neuralfoil(
    coordinates: np.ndarray,
    re: float = 1e7,
    model_size: str = "large",
    n_alpha: int = 360,
) -> List[PolarData]:
    """Generate polar tables using NeuralFoil from airfoil coordinates."""
    try:
        import neuralfoil as nf
    except ImportError:
        raise ImportError(
            "NeuralFoil is required to generate polars from coordinates. "
            "Install it with: pip install neuralfoil"
        )

    alpha_deg = np.linspace(-180, 180, n_alpha)
    alpha_rad = np.deg2rad(alpha_deg)

    aero = nf.get_aero_from_coordinates(
        coordinates=coordinates,
        alpha=alpha_deg,
        Re=re,
        model_size=model_size,
    )

    return [
        PolarData(
            alpha=alpha_rad,
            cl=np.asarray(aero["CL"], dtype=float),
            cd=np.asarray(aero["CD"], dtype=float),
            cm=np.asarray(aero["CM"], dtype=float),
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

            polars = _generate_polars_neuralfoil(coords, re=default_re, model_size=neuralfoil_model)

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
            polars = _generate_polars_neuralfoil(coords, re=default_re, model_size=neuralfoil_model)

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
