"""BEM aerodynamic solver wrapping CCBlade.

Provides a thin interface between the fem-shell aerodynamic data layer
(``BladeAero``) and the NREL CCBlade BEM solver.
"""

from dataclasses import dataclass

import numpy as np

from fem_shell.models.blade.aerodynamics import AirfoilAero, BladeAero


@dataclass
class BEMResult:
    """Results from a BEM computation.

    Attributes
    ----------
    r : ndarray
        Radial positions of blade stations (m).
    Np : ndarray
        Normal force per unit length at each station (N/m).
    Tp : ndarray
        Tangential force per unit length at each station (N/m).
    alpha : ndarray
        Angle of attack at each station (deg).
    cl : ndarray
        Lift coefficient at each station.
    cd : ndarray
        Drag coefficient at each station.
    a : ndarray
        Axial induction factor at each station.
    ap : ndarray
        Tangential induction factor at each station.
    thrust : float
        Integrated rotor thrust (N).
    torque : float
        Integrated rotor torque (N*m).
    power : float
        Integrated rotor power (W).
    cm : ndarray or None
        Moment coefficient at each station.
    Mp : ndarray or None
        Pitching moment per unit span at each station (N*m/m),
        about the aerodynamic centre.  Computed from
        ``Cm * q_c * chord`` where ``q_c`` is the dynamic
        pressure per unit span back-calculated from *Np*, *Tp*,
        *Cl*, and *Cd*.
    """

    r: np.ndarray
    Np: np.ndarray
    Tp: np.ndarray
    alpha: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    a: np.ndarray
    ap: np.ndarray
    thrust: float
    torque: float
    power: float
    cm: np.ndarray | None = None
    Mp: np.ndarray | None = None


def _build_ccairfoil(airfoil: AirfoilAero):
    """Convert an ``AirfoilAero`` to a CCBlade ``CCAirfoil``."""
    from ccblade.ccblade import CCAirfoil

    polars_sorted = sorted(airfoil.polars, key=lambda p: p.re)
    re_list = [p.re for p in polars_sorted]
    alpha_deg = np.rad2deg(polars_sorted[0].alpha)

    if len(polars_sorted) == 1:
        p = polars_sorted[0]
        cl = p.cl.reshape(-1, 1)
        cd = p.cd.reshape(-1, 1)
        cm = p.cm.reshape(-1, 1)
    else:
        cl = np.column_stack([p.cl for p in polars_sorted])
        cd = np.column_stack([p.cd for p in polars_sorted])
        cm = np.column_stack([p.cm for p in polars_sorted])

    coords = airfoil.coordinates
    x = coords[:, 0] if coords is not None and len(coords) > 0 else []
    y = coords[:, 1] if coords is not None and len(coords) > 0 else []

    return CCAirfoil(alpha_deg, re_list, cl, cd, cm=cm, x=x, y=y, AFName=airfoil.name)


class BEMSolver:
    """BEM solver wrapping CCBlade.

    Parameters
    ----------
    blade_aero : BladeAero
        Complete blade aerodynamic definition.
    rho : float
        Air density (kg/m^3).
    mu : float
        Dynamic viscosity (kg/(m*s)).
    precone : float
        Hub precone angle (deg).
    tilt : float
        Nacelle tilt angle (deg).
    hub_height : float
        Hub height (m), used for wind shear calculation.
    shear_exp : float
        Wind shear exponent for power-law profile.
    """

    def __init__(
        self,
        blade_aero: BladeAero,
        rho: float = 1.225,
        mu: float = 1.81206e-5,
        precone: float = 0.0,
        tilt: float = 0.0,
        hub_height: float = 150.0,
        shear_exp: float = 0.2,
    ):
        from ccblade.ccblade import CCBlade

        self.blade_aero = blade_aero

        # Build CCAirfoil objects (one per unique airfoil)
        af_cache: dict[str, object] = {}
        for airfoil in blade_aero.airfoils:
            if airfoil.name not in af_cache:
                af_cache[airfoil.name] = _build_ccairfoil(airfoil)

        # Per-station airfoil list
        af_list = [af_cache[s.airfoil.name] for s in blade_aero.stations]

        # Extract arrays — CCBlade expects theta in degrees
        r = blade_aero.r
        chord = blade_aero.chord
        theta_deg = np.rad2deg(blade_aero.twist)

        self.rotor = CCBlade(
            r=r,
            chord=chord,
            theta=theta_deg,
            af=af_list,
            Rhub=blade_aero.hub_radius,
            Rtip=blade_aero.rotor_radius,
            B=blade_aero.n_blades,
            rho=rho,
            mu=mu,
            precone=precone,
            tilt=tilt,
            hubHt=hub_height,
            shearExp=shear_exp,
        )

    def compute(
        self,
        v_inf: float,
        omega: float,
        pitch: float,
        azimuth: float = 0.0,
    ) -> BEMResult:
        """Run BEM analysis at the specified operating conditions.

        Parameters
        ----------
        v_inf : float
            Hub-height wind speed (m/s).
        omega : float
            Rotor rotational speed (RPM).
        pitch : float
            Blade collective pitch (deg, positive feather).
        azimuth : float
            Blade azimuthal position (deg).

        Returns
        -------
        BEMResult
            Distributed and integrated aerodynamic loads.
        """
        # BEM theory is singular at omega=0 (no tangential velocity).
        # Substitute a tiny rotation so the solver converges while keeping
        # the physics essentially unchanged (tip-speed ratio ~ 0).
        _OMEGA_EPS = 1e-3  # RPM
        omega_bem = omega if abs(omega) > _OMEGA_EPS else _OMEGA_EPS

        # Distributed loads at the given azimuth
        loads, _ = self.rotor.distributedAeroLoads(v_inf, omega_bem, pitch, azimuth)

        # Integrated loads (azimuth-averaged)
        outputs, _ = self.rotor.evaluate([v_inf], [omega_bem], [pitch])

        # -- Aerodynamic pitching moment per unit span ----------------------
        # Try to get Cm directly from CCBlade output; fall back to polar
        # evaluation at the computed angle of attack.
        Np = loads["Np"]
        Tp = loads["Tp"]
        cl = loads["Cl"]
        cd = loads["Cd"]

        try:
            cm = loads["Cm"]
        except (KeyError, TypeError):
            alpha_rad = np.deg2rad(loads["alpha"])
            cm = np.array([
                st.airfoil.get_polar(re=1e7).evaluate(
                    np.atleast_1d(alpha_rad[k]),
                )[2][0]
                for k, st in enumerate(self.blade_aero.stations)
            ])

        # Back-calculate dynamic pressure per unit span:
        #   q_c = sqrt(Np² + Tp²) / sqrt(Cl² + Cd²)
        # Then: Mp = q_c · chord · Cm
        cl_cd_norm = np.sqrt(cl**2 + cd**2)
        safe = cl_cd_norm > 1e-12
        q_c = np.where(
            safe,
            np.sqrt(Np**2 + Tp**2) / np.where(safe, cl_cd_norm, 1.0),
            0.0,
        )
        Mp = q_c * self.blade_aero.chord * cm

        return BEMResult(
            r=self.blade_aero.r,
            Np=Np,
            Tp=Tp,
            alpha=loads["alpha"],
            cl=cl,
            cd=cd,
            a=loads["a"],
            ap=loads["ap"],
            thrust=float(outputs["T"][0]),
            torque=float(outputs["Q"][0]),
            power=float(outputs["P"][0]),
            cm=cm,
            Mp=Mp,
        )
