"""Conservative force projection from BEM stations onto 3D shell mesh.

Maps per-span-station aerodynamic loads (Np, Tp) from a BEM computation
onto the finite-element mesh nodes, preserving the total integrated force
and moment on each chordwise strip.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.linalg import lstsq

from fem_shell.core.mesh.model import MeshModel
from fem_shell.models.blade.aerodynamics import BladeAero
from fem_shell.solvers.bem.engine import BEMResult


@dataclass
class _Strip:
    """Internal: a chordwise strip of mesh nodes."""

    node_indices: np.ndarray  # indices into mesh.nodes
    r_center: float  # span-wise centre (m, from hub)
    dr: float  # strip width (m)
    centroid: np.ndarray  # 3-D centroid of strip nodes
    offsets: np.ndarray  # (n, 3) node positions relative to centroid


class ForceProjector:
    """Project BEM distributed loads onto shell mesh nodes.

    Forces are distributed so that every chordwise strip conserves the
    total BEM force vector **and** moment about its centroid.

    Parameters
    ----------
    mesh : MeshModel
        The shell finite-element mesh (all nodes considered).
    blade_aero : BladeAero
        Aerodynamic blade definition (provides radial station positions).
    span_direction : array-like
        Unit vector along the blade span in global coordinates
        (default: z-axis ``[0, 0, 1]``).
    normal_direction : array-like
        Global direction for the BEM normal force *Np*
        (default: x-axis ``[1, 0, 0]``, i.e. downwind).
    tangential_direction : array-like
        Global direction for the BEM tangential force *Tp*
        (default: y-axis ``[0, 1, 0]``, positive in rotation direction).
    hub_radius : float or None
        Override hub radius for span coordinate calculation.
        If *None*, taken from *blade_aero*.
    """

    def __init__(
        self,
        mesh: MeshModel,
        blade_aero: BladeAero,
        span_direction=None,
        normal_direction=None,
        tangential_direction=None,
        hub_radius: float | None = None,
    ):
        span_dir = np.asarray(
            span_direction if span_direction is not None else [0.0, 0.0, 1.0],
            dtype=float,
        )
        span_dir /= np.linalg.norm(span_dir)

        self._normal_dir = np.asarray(
            normal_direction if normal_direction is not None else [1.0, 0.0, 0.0],
            dtype=float,
        )
        self._normal_dir /= np.linalg.norm(self._normal_dir)

        self._tangential_dir = np.asarray(
            tangential_direction if tangential_direction is not None else [0.0, 1.0, 0.0],
            dtype=float,
        )
        self._tangential_dir /= np.linalg.norm(self._tangential_dir)

        coords = mesh.coords_array  # (N, 3)
        n_nodes = coords.shape[0]
        hub_r = hub_radius if hub_radius is not None else blade_aero.hub_radius

        # Span coordinate for every mesh node (distance from hub centre
        # measured along the span direction)
        span_coords = coords @ span_dir  # projection

        # BEM station radial positions
        r_stations = blade_aero.r  # already from hub centre

        # Build strip boundaries at midpoints between stations
        r_mid = 0.5 * (r_stations[:-1] + r_stations[1:])
        r_low = np.empty(len(r_stations))
        r_high = np.empty(len(r_stations))
        r_low[0] = r_stations[0] - (r_mid[0] - r_stations[0])
        r_high[-1] = r_stations[-1] + (r_stations[-1] - r_mid[-1])
        r_low[1:] = r_mid
        r_high[:-1] = r_mid

        # Assign each node to a strip
        node_strip = np.full(n_nodes, -1, dtype=int)
        for k in range(len(r_stations)):
            mask = (span_coords >= r_low[k]) & (span_coords < r_high[k])
            node_strip[mask] = k
        # Handle nodes exactly at the tip boundary
        node_strip[span_coords == r_high[-1]] = len(r_stations) - 1

        # Build strip objects
        self._strips: List[_Strip] = []
        self._n_nodes = n_nodes
        for k in range(len(r_stations)):
            idx = np.where(node_strip == k)[0]
            if len(idx) == 0:
                # Empty strip – create a placeholder
                self._strips.append(
                    _Strip(
                        node_indices=idx,
                        r_center=float(r_stations[k]),
                        dr=float(r_high[k] - r_low[k]),
                        centroid=np.zeros(3),
                        offsets=np.zeros((0, 3)),
                    )
                )
                continue
            strip_coords = coords[idx]
            centroid = strip_coords.mean(axis=0)
            self._strips.append(
                _Strip(
                    node_indices=idx,
                    r_center=float(r_stations[k]),
                    dr=float(r_high[k] - r_low[k]),
                    centroid=centroid,
                    offsets=strip_coords - centroid,
                )
            )

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def project(self, bem_result: BEMResult) -> np.ndarray:
        """Map BEM loads onto mesh nodes.

        Parameters
        ----------
        bem_result : BEMResult
            Output of :pymethod:`BEMSolver.compute`.

        Returns
        -------
        forces : ndarray, shape (n_nodes, 3)
            Nodal force vectors in global coordinates.
        """
        forces = np.zeros((self._n_nodes, 3))

        for k, strip in enumerate(self._strips):
            n_k = len(strip.node_indices)
            if n_k == 0:
                continue

            # Total strip force in aero frame
            F_n = float(bem_result.Np[k]) * strip.dr  # normal (N)
            F_t = float(bem_result.Tp[k]) * strip.dr  # tangential (N)

            # Global force vector for the strip
            F_strip = F_n * self._normal_dir + F_t * self._tangential_dir

            # Target moment about strip centroid – zero (no Cm projection)
            M_strip = np.zeros(3)

            # Distribute to nodes (constrained minimum-norm)
            f_nodes = self._distribute(strip, F_strip, M_strip)
            forces[strip.node_indices] = f_nodes

        return forces

    def verify(self, bem_result: BEMResult, forces: np.ndarray) -> dict:
        """Check global force and moment conservation.

        The expected force is computed using the same discrete strip
        summation (``Np[k] * dr_k``) that :meth:`project` uses, so
        the error reflects only the per-strip distribution fidelity.

        Returns
        -------
        dict
            ``force_error`` – norm of residual (N).
            ``force_bem`` – expected total BEM force vector (N).
            ``force_mesh`` – actual summed mesh force vector (N).
        """
        # Discrete strip sum — same as project()
        F_total_n = 0.0
        F_total_t = 0.0
        for k, strip in enumerate(self._strips):
            if len(strip.node_indices) == 0:
                continue
            F_total_n += float(bem_result.Np[k]) * strip.dr
            F_total_t += float(bem_result.Tp[k]) * strip.dr

        F_bem = F_total_n * self._normal_dir + F_total_t * self._tangential_dir
        F_mesh = forces.sum(axis=0)

        return {
            "force_error": float(np.linalg.norm(F_mesh - F_bem)),
            "force_bem": F_bem,
            "force_mesh": F_mesh,
        }

    # ------------------------------------------------------------------
    #  Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _distribute(
        strip: _Strip,
        F_strip: np.ndarray,
        M_strip: np.ndarray,
    ) -> np.ndarray:
        """Minimum-norm nodal forces preserving total force and moment.

        Solves::

            min  Σ |f_j|²
            s.t. Σ f_j          = F_strip        (3 eqs)
                 Σ d_j × f_j   = M_strip        (3 eqs)

        via the pseudoinverse  f = Aᵀ (A Aᵀ)⁻¹ b.
        """
        n = len(strip.node_indices)
        if n == 1:
            return F_strip.reshape(1, 3)

        # Build constraint matrix A  (6 × 3n)
        A = np.zeros((6, 3 * n))
        for j in range(n):
            c = 3 * j
            # Force balance rows
            A[0, c] = 1.0
            A[1, c + 1] = 1.0
            A[2, c + 2] = 1.0
            # Moment balance rows:  d × f
            dx, dy, dz = strip.offsets[j]
            #  (d × f)_x =  dy * fz - dz * fy
            A[3, c + 1] += -dz
            A[3, c + 2] += dy
            #  (d × f)_y =  dz * fx - dx * fz
            A[4, c] += dz
            A[4, c + 2] += -dx
            #  (d × f)_z =  dx * fy - dy * fx
            A[5, c] += -dy
            A[5, c + 1] += dx

        b = np.concatenate([F_strip, M_strip])

        # Minimum-norm solution: f = Aᵀ (A Aᵀ)⁻¹ b
        AAT = A @ A.T
        lam, _, _, _ = lstsq(AAT, b)
        f_flat = A.T @ lam

        return f_flat.reshape(n, 3)
