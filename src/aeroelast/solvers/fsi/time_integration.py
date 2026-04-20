"""
Newmark-β time integration coefficients for structural dynamics.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NewmarkCoefficients:
    """
    Newmark-β time integration coefficients.

    These coefficients are derived from the Newmark parameters (beta, gamma)
    and the time step (dt). They are used in the effective stiffness formulation
    and the state update equations.

    Attributes:
        a0: 1 / (β·dt²) - Mass coefficient for K_eff
        a1: γ / (β·dt) - Damping coefficient for K_eff
        a2: 1 / (β·dt) - Velocity coefficient for F_eff
        a3: 1/(2β) - 1 - Acceleration coefficient for F_eff
        a4: γ/β - 1 - Velocity coefficient for damping contribution
        a5: dt·(γ/(2β) - 1) - Acceleration coefficient for damping
        a6: dt·(1 - γ) - Previous acceleration coefficient for velocity update
        a7: γ·dt - New acceleration coefficient for velocity update
    """

    a0: float
    a1: float
    a2: float
    a3: float
    a4: float
    a5: float
    a6: float
    a7: float

    @classmethod
    def from_newmark_params(
        cls, beta: float, gamma: float, dt: float
    ) -> "NewmarkCoefficients":
        """Create coefficients from Newmark parameters and time step."""
        return cls(
            a0=1.0 / (beta * dt**2),
            a1=gamma / (beta * dt),
            a2=1.0 / (beta * dt),
            a3=1.0 / (2 * beta) - 1.0,
            a4=gamma / beta - 1.0,
            a5=dt * (gamma / (2 * beta) - 1.0),
            a6=dt * (1 - gamma),
            a7=gamma * dt,
        )
