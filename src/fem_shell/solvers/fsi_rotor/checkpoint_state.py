"""
Checkpoint state management for implicit FSI coupling.

This module provides state storage and restoration capabilities
for preCICE implicit coupling schemes.
"""

from dataclasses import dataclass
from typing import Tuple

from petsc4py import PETSc


@dataclass
class CheckpointState:
    """
    State storage for implicit coupling checkpoints.

    This class stores the complete solver state required for checkpoint/restore
    operations in preCICE implicit coupling schemes. It includes both structural
    (displacement, velocity, acceleration) and angular (omega, theta, alpha) state.

    Parameters
    ----------
    u : PETSc.Vec
        Displacement vector (reduced system).
    v : PETSc.Vec
        Velocity vector (reduced system).
    a : PETSc.Vec
        Acceleration vector (reduced system).
    t : float
        Current simulation time [s].
    omega : float
        Angular velocity [rad/s].
    theta : float
        Accumulated rotation angle [rad].
    alpha : float
        Angular acceleration [rad/s²].

    Notes
    -----
    The checkpoint mechanism is essential for implicit (iterative) coupling
    schemes where the FSI loop may not converge on the first attempt. When
    convergence fails, the solver must restore the state and retry with
    updated boundary data.

    **Important**: When using checkpoints, always COPY values FROM the
    checkpoint TO the working vectors. Do NOT assign references, as this
    would cause the checkpoint to be corrupted when the working vectors
    are modified.

    Example
    -------
    Storing a checkpoint::

        checkpoint = CheckpointState(
            u=u, v=v, a=a, t=t,
            omega=solver._omega,
            theta=solver._theta,
            alpha=solver._alpha,
        )
        adapter.store_checkpoint(checkpoint.to_tuple())

    Restoring a checkpoint::

        data = adapter.read_checkpoint()
        checkpoint = CheckpointState.from_tuple(data)

        # COPY values, don't assign references!
        checkpoint.u.copy(u)
        checkpoint.v.copy(v)
        checkpoint.a.copy(a)
        t = checkpoint.t
        solver._omega = checkpoint.omega
    """

    u: PETSc.Vec
    v: PETSc.Vec
    a: PETSc.Vec
    t: float
    omega: float
    theta: float
    alpha: float

    def to_tuple(self) -> Tuple:
        """
        Convert state to tuple for storage.

        Returns
        -------
        Tuple
            State data as (u, v, a, t, omega, theta, alpha).

        Notes
        -----
        We pass references here. The Adapter's SolverState class handles
        creating deep copies of PETSc vectors for safe storage.
        """
        return (self.u, self.v, self.a, self.t, self.omega, self.theta, self.alpha)

    @classmethod
    def from_tuple(cls, data: Tuple) -> "CheckpointState":
        """
        Restore state from tuple.

        Parameters
        ----------
        data : Tuple
            State data as (u, v, a, t, omega, theta, alpha).

        Returns
        -------
        CheckpointState
            Restored checkpoint state.
        """
        u, v, a, t, omega, theta, alpha = data
        return cls(u=u, v=v, a=a, t=t, omega=omega, theta=theta, alpha=alpha)

    def __repr__(self) -> str:
        return (
            f"CheckpointState(t={self.t:.6f}s, "
            f"omega={self.omega:.4f} rad/s, "
            f"theta={self.theta:.4f} rad)"
        )
