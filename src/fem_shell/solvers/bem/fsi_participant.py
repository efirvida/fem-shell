r"""BEM-based preCICE fluid participant with bidirectional FSI coupling.

Runs as an independent process alongside the structural solver, acting as
the "fluid" side of the preCICE coupling (equivalent to OpenFOAM in a
traditional FSI setup but using Blade Element Momentum theory instead of
solving the Navier–Stokes equations).

Bidirectional coupling overview
-------------------------------
Within each preCICE time window the implicit coupling loop (IQN-ILS)
iterates through the following sequence until force–displacement
equilibrium is achieved:

1. **Read displacements** :math:`\mathbf{u}_i` from the structural solver via
   preCICE (mapped from Solid-Mesh to Fluid-Mesh via RBF interpolation).

2. **Update the deformed blade geometry** from the displacement field.
   Two aerodynamically relevant quantities are extracted per BEM strip *k*:

   * The deformed radial position :math:`r_k^{\mathrm{def}}`, which captures
     flapwise and edgewise bending.
   * The elastic twist increment :math:`\Delta\theta_k`, which captures
     torsional deformation around the span axis.

   Chord length is assumed invariant under linear elastic deformation
   (no in-plane stretching).

3. **Rebuild the BEM solver** (CCBlade) with the deformed radii and twist
   distribution.  Airfoil polar assignment remains unchanged — each station
   retains its reference airfoil.

4. **Rebuild the force projector** on the deformed nodal positions so that
   distributed BEM loads are projected onto the *actual* deformed surface.

5. **Evaluate BEM** at the current operating conditions and project forces
   onto mesh nodes.

6. **Write forces** :math:`\mathbf{F}_i` to preCICE for the structural solver.

Steps 1–6 repeat within each time window until preCICE's convergence
criterion is satisfied (see ``precice-config.xml``,
``<relative-convergence-limit>``).

Mathematical formulation
========================

Deformed radial position
------------------------
For each BEM strip *k* with node index set :math:`\mathcal{I}_k`, the
deformed radial coordinate is the arithmetic mean of the nodal span-wise
projections on the deformed configuration:

.. math::

    r_k^{\mathrm{def}}
        = \frac{1}{|\mathcal{I}_k|}
          \sum_{i \in \mathcal{I}_k}
          \bigl(\mathbf{X}_i + \mathbf{u}_i\bigr) \cdot \hat{\mathbf{e}}_s

where :math:`\mathbf{X}_i` are the reference nodal coordinates,
:math:`\mathbf{u}_i` the displacement vector, and
:math:`\hat{\mathbf{e}}_s` the unit span direction.

This captures the radial redistribution of blade stations caused by
flapwise and edgewise bending.  For a pure flapwise deflection
:math:`\delta` at radius *r*, the effective shortening of the projected
span is of order :math:`\delta^2 / (2r)` — small but non-negligible for
flexible blades with tip deflections exceeding 10 % of span.

Elastic twist extraction
------------------------
The local aerodynamic twist at strip *k* on the deformed blade is:

.. math::

    \theta_k^{\mathrm{def}} = \theta_k^{\mathrm{ref}} + \Delta\theta_k

where the elastic twist increment :math:`\Delta\theta_k` is obtained from
the rotation of the *chord direction* of the strip nodes between the
reference and deformed configurations.

**Chord direction estimation (SVD-based PCA).**  For each strip *k*:

1. Collect the strip node positions :math:`\mathbf{x}_i` (reference or
   deformed) and compute their centroid :math:`\bar{\mathbf{x}}_k`.
2. Form the offset matrix
   :math:`\mathbf{D}_k = [\mathbf{x}_i - \bar{\mathbf{x}}_k]_{i \in \mathcal{I}_k}`.
3. Project onto the plane perpendicular to :math:`\hat{\mathbf{e}}_s`:

   .. math::

       \mathbf{D}_k^{\perp}
           = \mathbf{D}_k
             - \bigl(\mathbf{D}_k \, \hat{\mathbf{e}}_s\bigr)
               \hat{\mathbf{e}}_s^\top

4. Compute the SVD :math:`\mathbf{D}_k^{\perp} = \mathbf{U \Sigma V}^\top`.
   The first right singular vector :math:`\mathbf{v}_1` is the direction of
   maximum variance in the cross-section plane — i.e. the **chord direction**
   :math:`\hat{\mathbf{c}}_k`.

5. Orient consistently: if
   :math:`\hat{\mathbf{c}}_k \cdot \hat{\mathbf{e}}_n < 0` (where
   :math:`\hat{\mathbf{e}}_n` is the global normal direction), flip the sign.

This is equivalent to a 2-D Principal Component Analysis (PCA) of the
cross-section node distribution and is robust to:

* Non-uniform chordwise node spacing (common in FE meshes with refinement
  near the leading/trailing edge).
* Moderate out-of-plane warp of the cross-section (the projection onto the
  :math:`\hat{\mathbf{e}}_s`-perpendicular plane removes the span component).
* Large rotations — the SVD does not linearise the rotation.

**Signed angle computation.**  Given the reference chord direction
:math:`\hat{\mathbf{c}}_k^{\mathrm{ref}}` and the deformed chord direction
:math:`\hat{\mathbf{c}}_k^{\mathrm{def}}`, the elastic twist is the signed
angle between them measured about the span axis:

.. math::

    \Delta\theta_k = \arctan2\!\bigl(
        (\hat{\mathbf{c}}_k^{\mathrm{ref}} \times \hat{\mathbf{c}}_k^{\mathrm{def}})
            \cdot \hat{\mathbf{e}}_s,\;
        \hat{\mathbf{c}}_k^{\mathrm{ref}} \cdot \hat{\mathbf{c}}_k^{\mathrm{def}}
    \bigr)

The sign convention is positive nose-up (trailing edge rotates toward the
suction side), consistent with the BEM twist definition in CCBlade.

Force projection on deformed geometry
--------------------------------------
After obtaining the BEM distributed loads :math:`N_p(r_k)` and
:math:`T_p(r_k)` (normal and tangential force per unit span), the
``ForceProjector`` is reconstructed using the **deformed nodal coordinates**
:math:`\mathbf{X}_i + \mathbf{u}_i`.  This ensures that:

* Node-to-strip assignment uses actual deformed span positions (a node
  displaced by bending may migrate to an adjacent strip).
* The moment-preserving distribution (minimum-norm pseudoinverse) acts on
  the deformed strip geometry, so the resultant moment about the deformed
  centroid matches the BEM integrated moment.

Aerodynamic pitching moment
----------------------------
The BEM solver computes a pitching-moment coefficient :math:`C_m` at each
station (from the airfoil polars evaluated at the local angle of attack).
The pitching moment per unit span is recovered from the distributed loads:

.. math::

    M_p(r_k) = \frac{\sqrt{N_p^2 + T_p^2}}{\sqrt{C_l^2 + C_d^2}}
               \;\cdot\; c_k \;\cdot\; C_m(r_k)

The first factor is the dynamic pressure per unit span
:math:`q_c = \tfrac{1}{2}\rho W^2 c`, back-calculated from the force
resultant and the aerodynamic coefficients.  This avoids requiring the
relative wind speed :math:`W` as an explicit BEM output.

The total moment on BEM strip *k* is :math:`M_p \, \Delta r_k` and acts
about the span axis :math:`\hat{\mathbf{e}}_s`.  It is passed to the
``ForceProjector`` as the target moment vector per strip, so that the
minimum-norm force distribution creates nodal forces whose net moment
about the strip centroid matches the aerodynamic pitching moment.

Consistent deformed strip widths
---------------------------------
When the blade geometry is updated from the displacement field, the
``ForceProjector`` is rebuilt using a **deformed** ``BladeAero`` (the
same one passed to the BEM solver).  This ensures that the per-strip
width :math:`\Delta r_k` used to convert force-per-span into total
strip force is consistent with the deformed radial positions
:math:`r_k^{\mathrm{def}}`.

Checkpoint strategy
-------------------
BEM is memoryless: given a displacement field and operating conditions it
produces a deterministic force field.  The checkpoint therefore stores only
the last committed nodal forces :math:`\mathbf{F}^{n}`, managed by the
shared :class:`~fem_shell.solvers.fsi.base.Adapter` (which provides
checkpoint save/restore via preCICE's implicit-iteration protocol).  On
rollback the previous forces are restored and a fresh BEM evaluation is
performed using the new displacement iterate.

Physical assumptions and limitations
-------------------------------------
* **Linear elasticity**: chord length is invariant.  For geometrically
  nonlinear structural models the chord could be updated analogously to *r*
  and *twist*, but this is not implemented.
* **Rigid airfoil profiles**: deformation changes the angle of attack via
  twist but does not alter the airfoil shape itself.
* **Quasi-steady aerodynamics**: BEM evaluates steady-state loads at each
  implicit iteration; dynamic stall and unsteady wake effects are not
  captured.
* **Single-blade formulation**: the participant models one blade; multi-blade
  effects are handled through the BEM rotor model (rotor thrust factor *B*).
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from fem_shell.core.mesh.model import MeshModel
from fem_shell.models.blade.aerodynamics import (
    AeroStation,
    BladeAero,
    load_blade_aero,
)
from fem_shell.solvers.bem.engine import BEMResult, BEMSolver
from fem_shell.solvers.bem.force_projection import ForceProjector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main participant class
# ---------------------------------------------------------------------------


class BEMFSIParticipant:
    r"""preCICE fluid participant that computes aerodynamic forces via BEM
    with full bidirectional deformation feedback.

    On every implicit coupling iteration the participant:

    1. Reads the structural displacement field :math:`\mathbf{u}`.
    2. Extracts deformed strip radii :math:`r_k^{\mathrm{def}}` and elastic
       twist :math:`\theta_k^{\mathrm{def}}` (see module docstring for the
       mathematical formulation).
    3. Rebuilds ``BEMSolver`` and ``ForceProjector`` on the deformed geometry.
    4. Evaluates BEM and projects nodal forces.
    5. Writes forces to preCICE for the structural solver.

    Parameters
    ----------
    mesh : MeshModel
        Blade surface mesh.  Its nodes define the Fluid-Mesh vertices
        registered with preCICE.
    blade_aero : BladeAero
        Aerodynamic blade definition (airfoil polars, radial stations).
    bem_config : dict
        BEM operating conditions and solver options.  Recognised keys:

        * ``wind_speed``           – free-stream wind speed [m/s]
        * ``omega``                – rotor angular velocity [RPM]
        * ``pitch``                – collective pitch angle [deg]
        * ``air_density``          – ρ [kg/m³] (default 1.225)
        * ``dynamic_viscosity``    – μ [Pa·s]  (default 1.81e-5)
        * ``hub_height``           – hub centre height above ground [m]
        * ``shear_exp``            – wind shear exponent (default 0.2)
        * ``precone``              – precone angle [deg] (default 0)
        * ``tilt``                 – shaft tilt angle [deg] (default 0)
        * ``span_direction``       – unit vector along blade span (default Z)
        * ``normal_direction``     – global direction for BEM Np (default X)
        * ``tangential_direction`` – global direction for BEM Tp (default Y)
    participant : str
        preCICE participant name (must match the XML config).
    config_file : str or Path
        Path to ``precice-config.xml``.
    coupling_mesh : str
        Name of the mesh provided by this participant.
    displacement_data : str
        Name of the displacement data field to read.
    force_data : str
        Name of the force data field to write.
    output_folder : str or Path
        Directory for CSV diagnostics.
    log_interval : int
        How many time windows between diagnostic log lines.
    """

    def __init__(
        self,
        mesh: MeshModel,
        blade_aero: BladeAero,
        bem_config: dict,
        participant: str = "Fluid",
        config_file: str | Path = "precice-config.xml",
        coupling_mesh: str = "Fluid-Mesh",
        displacement_data: str = "Displacement",
        force_data: str = "Force",
        output_folder: str | Path = "bem_fsi_results",
        log_interval: int = 10,
        viz_mesh: MeshModel | None = None,
    ) -> None:
        self._mesh = mesh
        self._blade_aero = blade_aero
        self._bem_cfg = bem_config
        self._participant_name = participant
        self._config_file = str(config_file)
        self._coupling_mesh = coupling_mesh
        self._displacement_data = displacement_data
        self._force_data = force_data
        self._output_folder = Path(output_folder)
        self._log_interval = log_interval

        # Reference node coordinates (n_nodes, 3) — immutable after init
        self._ref_coords: np.ndarray = mesh.coords_array.copy()
        self._n_nodes: int = self._ref_coords.shape[0]

        # -- Direction vectors (normalised once) ----------------------------
        self._span_dir = np.asarray(
            bem_config.get("span_direction", [0.0, 0.0, 1.0]), dtype=float,
        )
        self._span_dir /= np.linalg.norm(self._span_dir)

        self._normal_dir = np.asarray(
            bem_config.get("normal_direction", [1.0, 0.0, 0.0]), dtype=float,
        )
        self._normal_dir /= np.linalg.norm(self._normal_dir)

        self._tangential_dir = np.asarray(
            bem_config.get("tangential_direction", [0.0, 1.0, 0.0]), dtype=float,
        )
        self._tangential_dir /= np.linalg.norm(self._tangential_dir)

        # -- BEM solver keyword arguments (constant across iterations) ------
        self._bem_solver_kwargs: dict = dict(
            rho=bem_config.get("air_density", 1.225),
            mu=bem_config.get("dynamic_viscosity", 1.81206e-5),
            precone=bem_config.get("precone", 0.0),
            tilt=bem_config.get("tilt", 0.0),
            hub_height=bem_config.get("hub_height", 150.0),
            shear_exp=bem_config.get("shear_exp", 0.2),
        )

        # -- Reference BEM solver and projector -----------------------------
        self._bem_solver = BEMSolver(blade_aero, **self._bem_solver_kwargs)

        ref_projector = ForceProjector(
            mesh, blade_aero,
            span_direction=self._span_dir,
            normal_direction=self._normal_dir,
            tangential_direction=self._tangential_dir,
        )
        self._projector = ref_projector

        # -- Strip-to-node mapping from the reference projector -------------
        # These index arrays define which mesh nodes belong to each BEM strip.
        # The mapping is established on the reference geometry and is also
        # used by the deformed-geometry methods (chord direction estimation,
        # radial position averaging).  The ForceProjector rebuilt on deformed
        # coordinates will independently reassign nodes to strips (a node
        # shifted by large bending may migrate to an adjacent strip).
        self._strip_node_indices: list[np.ndarray] = [
            strip.node_indices.copy() for strip in ref_projector._strips
        ]

        # -- Reference per-strip quantities ---------------------------------
        self._ref_r: np.ndarray = blade_aero.r.copy()        # (n_strips,)
        self._ref_twist: np.ndarray = blade_aero.twist.copy()  # (n_strips,) rad

        # Reference chord directions (unit vectors in the e_s-perp plane).
        # Computed via SVD-based PCA of each strip's node distribution.
        self._ref_chord_dirs: list[np.ndarray] = self._compute_strip_chord_dirs(
            self._ref_coords,
        )

        # -- Working copy of the mesh for projector reconstruction ----------
        # A single deep copy is made at init time; subsequent iterations
        # mutate its coords_array via the setter (O(n_nodes) per call)
        # instead of creating a new deep copy every iteration.
        self._working_mesh: MeshModel = copy.deepcopy(mesh)

        # -- Full mesh with elements for surface VTU output -----------------
        # When a coupling_node_set filter is applied in the CLI, the coupling
        # mesh has no elements. viz_mesh keeps the full mesh for ParaView output.
        self._viz_mesh: MeshModel = viz_mesh if viz_mesh is not None else mesh

        # -- Tip node index (max span coordinate) --------------------------
        # disp[-1] is NOT guaranteed to be the tip — node ordering in the
        # coupling mesh depends on the mesh generator + coupling_node_set filter.
        span_coords = self._ref_coords @ self._span_dir  # (n_nodes,)
        self._tip_node_idx: int = int(np.argmax(span_coords))

        # -- Counters -------------------------------------------------------
        self._last_forces = np.zeros((self._n_nodes, 3), dtype=float)
        self._window_count = 0
        self._iteration_count = 0

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run(self) -> None:
        """Execute the preCICE coupling loop (blocking until finalized)."""
        # Lazy import — the Adapter (and its precice dependency) are only
        # loaded when actually running, avoiding GLIBCXX errors on login nodes.
        from fem_shell.solvers.fsi.base import Adapter

        self._output_folder.mkdir(parents=True, exist_ok=True)

        adapter = Adapter(
            participant=self._participant_name,
            config_file=self._config_file,
            coupling_meshes={self._coupling_mesh: self._ref_coords},
        )
        adapter.initialize()

        logger.info(
            "[BEM-FSI] Initialized.  Participant=%s  mesh=%s  nodes=%d  "
            "strips=%d",
            self._participant_name,
            self._coupling_mesh,
            self._n_nodes,
            len(self._strip_node_indices),
        )

        bem_result: Optional[BEMResult] = None

        while adapter.is_coupling_ongoing:
            # preCICE 3.x requires explicit checkpoint write/read guards.
            if adapter.requires_writing_checkpoint:
                adapter.store_checkpoint((self._last_forces.copy(),))

            if adapter.requires_reading_checkpoint:
                states = adapter.retrieve_checkpoint()
                self._last_forces = states[0]
                self._iteration_count += 1

            dt = adapter.dt

            # Read displacements from the structural solver
            displacements = adapter.read_data(self._coupling_mesh, self._displacement_data)

            # BEM on (possibly deformed) geometry
            forces, bem_result = self._compute_forces(displacements)

            adapter.write_data(self._coupling_mesh, self._force_data, forces)

            adapter.advance(dt)

            if adapter.is_time_window_complete:
                self._last_forces = forces
                self._window_count += 1
                self._iteration_count = 0
                current_time = self._window_count * dt

                self._write_timestep_output(
                    forces, displacements, bem_result,
                    self._window_count, current_time,
                )

                if self._window_count % self._log_interval == 0:
                    self._log_window(
                        bem_result, displacements, current_time,
                    )

        adapter.finalize()
        logger.info("[BEM-FSI] Finalized after %d time windows.", self._window_count)

    # -----------------------------------------------------------------------
    # Deformed geometry extraction
    # -----------------------------------------------------------------------

    def _compute_deformed_geometry(
        self, displacements: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Extract deformed BEM strip radii and twist from nodal displacements.

        For the full mathematical derivation see the module-level docstring.

        **Algorithm summary** (per strip *k*):

        1. Compute deformed coordinates
           :math:`\mathbf{x}_i = \mathbf{X}_i + \mathbf{u}_i`.
        2. Radial position:
           :math:`r_k^{\mathrm{def}} = \mathrm{mean}_{i \in \mathcal{I}_k}
           (\mathbf{x}_i \cdot \hat{\mathbf{e}}_s)`.
        3. Chord direction on the deformed config via SVD-based PCA
           (``_compute_strip_chord_dirs``).
        4. Signed elastic twist:
           :math:`\Delta\theta_k = \arctan2(
           (\hat{\mathbf{c}}^{\mathrm{ref}} \times \hat{\mathbf{c}}^{\mathrm{def}})
           \cdot \hat{\mathbf{e}}_s,\;
           \hat{\mathbf{c}}^{\mathrm{ref}} \cdot \hat{\mathbf{c}}^{\mathrm{def}})`.
        5. Deformed twist:
           :math:`\theta_k^{\mathrm{def}} = \theta_k^{\mathrm{ref}} + \Delta\theta_k`.

        Parameters
        ----------
        displacements : ndarray, shape (n_nodes, 3)
            Nodal displacement vectors from the structural solver.

        Returns
        -------
        r_def : ndarray, shape (n_strips,)
            Deformed radial position of each BEM strip projected onto the
            span axis.
        twist_def : ndarray, shape (n_strips,)
            Total aerodynamic twist at each strip [rad], including the
            reference twist plus the elastic torsion increment.
        """
        deformed_coords = self._ref_coords + displacements
        n_strips = len(self._strip_node_indices)
        r_def = np.empty(n_strips)
        twist_def = np.empty(n_strips)

        # Chord directions on the deformed configuration
        def_chord_dirs = self._compute_strip_chord_dirs(deformed_coords)

        s = self._span_dir
        for k in range(n_strips):
            idx = self._strip_node_indices[k]
            if len(idx) == 0:
                r_def[k] = self._ref_r[k]
                twist_def[k] = self._ref_twist[k]
                continue

            # Deformed radial position (mean span-wise projection)
            strip_span = deformed_coords[idx] @ s
            r_def[k] = float(np.mean(strip_span))

            # Elastic twist from chord direction rotation
            c_ref = self._ref_chord_dirs[k]
            c_def = def_chord_dirs[k]
            cos_a = float(np.clip(np.dot(c_ref, c_def), -1.0, 1.0))
            sin_a = float(np.dot(np.cross(c_ref, c_def), s))
            delta_twist = np.arctan2(sin_a, cos_a)

            twist_def[k] = self._ref_twist[k] + delta_twist

        return r_def, twist_def

    def _compute_strip_chord_dirs(
        self, coords: np.ndarray,
    ) -> list[np.ndarray]:
        r"""Estimate a chord-direction unit vector for each BEM strip via
        SVD-based PCA of the cross-section node distribution.

        **Method (per strip k):**

        1. Collect strip node positions and compute centroid
           :math:`\bar{\mathbf{x}}_k`.
        2. Form the offset matrix :math:`\mathbf{D}_k` (nodes × 3).
        3. Project offsets onto the plane perpendicular to
           :math:`\hat{\mathbf{e}}_s`:
           :math:`\mathbf{D}_k^{\perp} = \mathbf{D}_k
           - (\mathbf{D}_k \hat{\mathbf{e}}_s) \hat{\mathbf{e}}_s^\top`.
        4. SVD: :math:`\mathbf{D}_k^{\perp} = \mathbf{U\Sigma V}^\top`.
           The first row of :math:`\mathbf{V}^\top` (i.e. :math:`\mathbf{v}_1`)
           is the principal axis = chord direction.
        5. Flip sign if :math:`\mathbf{v}_1 \cdot \hat{\mathbf{e}}_n < 0`
           to ensure a consistent orientation across strips and iterations.

        This is mathematically equivalent to a 2-D PCA in the airfoil
        cross-section plane.  The chord direction dominates the variance
        because the chord is typically 5–20× the airfoil thickness, making
        the leading singular value well-separated from the second.

        **Robustness:**

        * Strips with fewer than 2 nodes fall back to the global normal
          direction :math:`\hat{\mathbf{e}}_n` (no PCA possible).
        * Near-zero norm after projection (degenerate planar strip exactly
          parallel to span) also falls back to :math:`\hat{\mathbf{e}}_n`.

        Parameters
        ----------
        coords : ndarray, shape (n_nodes, 3)
            Nodal coordinates (reference or deformed).

        Returns
        -------
        chord_dirs : list of ndarray, each shape (3,)
            Unit chord-direction vector for each strip.
        """
        chord_dirs: list[np.ndarray] = []
        s = self._span_dir

        for idx in self._strip_node_indices:
            if len(idx) < 2:
                chord_dirs.append(self._normal_dir.copy())
                continue

            strip_pts = coords[idx]
            centroid = strip_pts.mean(axis=0)
            offsets = strip_pts - centroid

            # Project offsets onto the plane ⊥ span_dir
            offsets_plane = offsets - np.outer(offsets @ s, s)

            _, _, Vt = np.linalg.svd(offsets_plane, full_matrices=False)
            chord_dir = Vt[0]

            # Consistent orientation with the global normal direction
            if np.dot(chord_dir, self._normal_dir) < 0:
                chord_dir = -chord_dir

            norm = np.linalg.norm(chord_dir)
            if norm > 1e-12:
                chord_dir /= norm
            else:
                chord_dir = self._normal_dir.copy()

            chord_dirs.append(chord_dir)

        return chord_dirs

    # -----------------------------------------------------------------------
    # BEM and projector reconstruction on deformed geometry
    # -----------------------------------------------------------------------

    def _rebuild_bem_solver(
        self, r_def: np.ndarray, twist_def: np.ndarray,
    ) -> tuple[BEMSolver, BladeAero]:
        r"""Construct a new ``BEMSolver`` with deformed strip radii and twist.

        Creates a new set of ``AeroStation`` objects with:

        * :math:`r_k \leftarrow r_k^{\mathrm{def}}`
        * :math:`\theta_k \leftarrow \theta_k^{\mathrm{def}}`
        * ``chord``, ``pitch_axis``, ``airfoil`` — unchanged from reference.

        A new ``BladeAero`` is assembled from these stations and passed to
        ``BEMSolver``, which internally rebuilds the ``CCBlade`` rotor object
        with the updated geometry.

        Parameters
        ----------
        r_def : ndarray, shape (n_strips,)
            Deformed radial positions [m].
        twist_def : ndarray, shape (n_strips,)
            Deformed aerodynamic twist [rad].

        Returns
        -------
        solver : BEMSolver
            Fresh solver instance on the deformed geometry.
        deformed_aero : BladeAero
            The deformed blade definition.  Returned so that
            ``_rebuild_projector`` can use the same deformed station radii
            for strip-width computation (consistent :math:`\Delta r_k`).
        """
        ba = self._blade_aero

        new_stations = [
            AeroStation(
                span_fraction=st.span_fraction,
                r=float(r_def[k]),
                chord=st.chord,
                twist=float(twist_def[k]),
                pitch_axis=st.pitch_axis,
                airfoil=st.airfoil,
            )
            for k, st in enumerate(ba.stations)
        ]

        deformed_aero = BladeAero(
            airfoils=ba.airfoils,
            stations=new_stations,
            blade_length=ba.blade_length,
            hub_radius=ba.hub_radius,
            rotor_radius=ba.rotor_radius,
            n_blades=ba.n_blades,
        )

        return BEMSolver(deformed_aero, **self._bem_solver_kwargs), deformed_aero

    def _rebuild_projector(
        self, deformed_coords: np.ndarray, deformed_aero: BladeAero,
    ) -> ForceProjector:
        r"""Construct a ``ForceProjector`` on the deformed mesh.

        The working mesh (a deep copy created once at init) has its nodal
        coordinates updated in-place to :math:`\mathbf{X} + \mathbf{u}`,
        then a fresh projector is built.  This ensures node-to-strip
        assignment and the moment-preserving force distribution both operate
        on the actual deformed surface.

        The *deformed_aero* argument supplies the deformed station radii so
        that strip widths :math:`\Delta r_k` are consistent with the BEM
        evaluation.

        Parameters
        ----------
        deformed_coords : ndarray, shape (n_nodes, 3)
            Deformed nodal coordinates.
        deformed_aero : BladeAero
            Blade definition with deformed radii and twist (same as used
            by ``_rebuild_bem_solver``).

        Returns
        -------
        ForceProjector
            Projector on the deformed geometry.
        """
        self._working_mesh.coords_array = deformed_coords
        return ForceProjector(
            self._working_mesh,
            deformed_aero,
            span_direction=self._span_dir,
            normal_direction=self._normal_dir,
            tangential_direction=self._tangential_dir,
        )

    # -----------------------------------------------------------------------
    # Force computation (bidirectional)
    # -----------------------------------------------------------------------

    def _compute_forces(
        self, displacements: np.ndarray,
    ) -> tuple[np.ndarray, BEMResult]:
        r"""Evaluate BEM on the (possibly deformed) blade and project forces.

        **Pipeline:**

        1. If :math:`\max_i \|\mathbf{u}_i\| < 10^{-12}` (effectively zero
           displacement), skip geometry reconstruction and evaluate on the
           reference configuration.  This avoids unnecessary SVD and CCBlade
           rebuilds at the start of the simulation or on converged windows.

        2. Otherwise:

           a. ``_compute_deformed_geometry`` → :math:`(r^{\mathrm{def}},
              \theta^{\mathrm{def}})`.
           b. ``_rebuild_bem_solver`` → new ``BEMSolver`` with deformed
              geometry.
           c. ``_rebuild_projector`` → new ``ForceProjector`` on deformed
              coordinates.
           d. ``BEMSolver.compute`` → distributed loads :math:`N_p, T_p`.
           e. ``ForceProjector.project`` → nodal force array.

        Parameters
        ----------
        displacements : ndarray, shape (n_nodes, 3)
            Nodal displacements from the structural solver.

        Returns
        -------
        forces : ndarray, shape (n_nodes, 3)
            Aerodynamic nodal forces on the (deformed) blade surface.
        bem_result : BEMResult
            Full BEM output (Np, Tp, alpha, Cl, Cd, integrated loads).
        """
        v_inf = float(self._bem_cfg.get("wind_speed", 45.0))
        omega = float(self._bem_cfg.get("omega", 0.0))
        pitch = float(self._bem_cfg.get("pitch", 0.0))

        disp_max = float(np.max(np.linalg.norm(displacements, axis=1)))

        if disp_max < 1e-12:
            # Zero displacement — use pre-built reference solver and projector
            bem_result = self._bem_solver.compute(v_inf, omega, pitch)
            forces = self._projector.project(bem_result)
            return forces, bem_result

        # -- Deformed geometry pipeline ------------------------------------
        r_def, twist_def = self._compute_deformed_geometry(displacements)
        bem_solver, deformed_aero = self._rebuild_bem_solver(r_def, twist_def)

        deformed_coords = self._ref_coords + displacements
        projector = self._rebuild_projector(deformed_coords, deformed_aero)

        bem_result = bem_solver.compute(v_inf, omega, pitch)
        forces = projector.project(bem_result)
        return forces, bem_result

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def _write_timestep_output(
        self,
        forces: np.ndarray,
        displacements: np.ndarray,
        bem_result: BEMResult,
        step: int,
        time: float,
    ) -> None:
        """Write all output for one converged time window.

        Structure mirrors the solid solver::

            {output_folder}/
              results.pvd               ← PVD time collection for fields.vtu
              bem_report.csv            ← one row per window (global loads)
              {time}/
                fields.vtu              ← surface mesh: Force + Displacement
                bem_sections.vtu        ← strip centroids: BEM spanwise data
                bem_sectional.csv       ← sectional loads table
        """
        time_str = f"{time:.6g}"
        ts_dir = self._output_folder / time_str
        ts_dir.mkdir(parents=True, exist_ok=True)

        self._write_fields_vtu(forces, displacements, ts_dir, time_str, time)
        self._write_sections_vtu(bem_result, ts_dir)
        self._write_sectional_csv(bem_result, ts_dir)
        self._append_global_csv(bem_result, forces, displacements, step, time, time_str)

    def _write_fields_vtu(
        self,
        forces: np.ndarray,
        displacements: np.ndarray,
        ts_dir: "Path",
        time_str: str,
        time: float,
    ) -> None:
        """Write surface VTU with Force and Displacement nodal fields."""
        try:
            import meshio
        except ImportError:
            return

        viz = self._viz_mesh
        coords = viz.coords_array  # (N_viz, 3)

        # Build node-id → index map for the viz mesh
        node_id_to_viz_idx = {node.id: i for i, node in enumerate(viz.nodes)}

        # Map coupling-mesh forces/displacements onto the viz mesh
        # (coupling nodes are a subset of viz nodes — same Node objects)
        n_viz = len(viz.nodes)
        forces_viz = np.zeros((n_viz, 3))
        disp_viz = np.zeros((n_viz, 3))
        for i, node in enumerate(self._working_mesh.nodes):
            j = node_id_to_viz_idx.get(node.id)
            if j is not None:
                forces_viz[j] = forces[i]
                disp_viz[j] = displacements[i]

        # Build cell connectivity grouped by type
        cell_blocks: dict[str, list] = {}
        for elem in viz.elements:
            conn = [node_id_to_viz_idx[n.id] for n in elem.nodes]
            n_nds = len(conn)
            if n_nds == 3:
                ctype = "triangle"
            elif n_nds == 4:
                ctype = "quad"
            elif n_nds == 6:
                ctype = "triangle6"
            elif n_nds == 8:
                ctype = "quad8"
            else:
                continue
            cell_blocks.setdefault(ctype, []).append(conn)

        meshio_cells = [(k, np.array(v)) for k, v in cell_blocks.items()]
        if not meshio_cells:
            # Fallback to point cloud if no elements
            meshio_cells = [("vertex", np.arange(n_viz).reshape(-1, 1))]

        m = meshio.Mesh(
            points=coords,
            cells=meshio_cells,
            point_data={"Force": forces_viz, "Displacement": disp_viz},
        )
        vtu_path = ts_dir / "fields.vtu"
        m.write(str(vtu_path))

        # Update PVD with relative path
        rel_path = f"{ts_dir.name}/fields.vtu"
        self._update_pvd(self._output_folder, "results.pvd", rel_path, time)

    def _write_sections_vtu(self, bem_result: BEMResult, ts_dir: "Path") -> None:
        """Write strip centroids VTU with spanwise BEM aerodynamic data."""
        try:
            import meshio
        except ImportError:
            return

        strips = self._projector._strips
        n_strips = len(strips)

        centroids = np.array([s.centroid for s in strips])
        dr = np.array([s.dr for s in strips])

        # Integrated force vectors at each strip centroid
        F_section = np.zeros((n_strips, 3))
        for k, strip in enumerate(strips):
            F_n = float(bem_result.Np[k]) * strip.dr
            F_t = float(bem_result.Tp[k]) * strip.dr
            F_section[k] = F_n * self._normal_dir + F_t * self._tangential_dir

        point_data: dict[str, np.ndarray] = {
            "r_m":      bem_result.r.astype(float),
            "Np_N_m":   bem_result.Np.astype(float),
            "Tp_N_m":   bem_result.Tp.astype(float),
            "alpha_deg": bem_result.alpha.astype(float),
            "cl":       bem_result.cl.astype(float),
            "cd":       bem_result.cd.astype(float),
            "a":        bem_result.a.astype(float),
            "ap":       bem_result.ap.astype(float),
            "dr_m":     dr.astype(float),
            "Force_section": F_section,
        }
        # Optional fields — present when CCBlade version supports them
        if bem_result.cn is not None:
            point_data["cn"] = bem_result.cn.astype(float)
        if bem_result.ct is not None:
            point_data["ct"] = bem_result.ct.astype(float)
        if bem_result.W is not None:
            point_data["W_m_s"] = bem_result.W.astype(float)
        if bem_result.Re is not None:
            point_data["Re"] = bem_result.Re.astype(float)
        if bem_result.cm is not None:
            point_data["cm"] = bem_result.cm.astype(float)
        if bem_result.Mp is not None:
            point_data["Mp_Nm_m"] = bem_result.Mp.astype(float)
        if bem_result.chord is not None:
            point_data["chord_m"] = bem_result.chord.astype(float)
        if bem_result.twist_deg is not None:
            point_data["twist_deg"] = bem_result.twist_deg.astype(float)

        cells = [("vertex", np.arange(n_strips).reshape(-1, 1))]
        m = meshio.Mesh(points=centroids, cells=cells, point_data=point_data)
        m.write(str(ts_dir / "bem_sections.vtu"))

    def _write_sectional_csv(self, bem_result: BEMResult, ts_dir: "Path") -> None:
        """Write sectional loads CSV for this timestep."""
        strips = self._projector._strips
        dr = np.array([s.dr for s in strips])

        names = ["r[m]", "dr[m]",
                 "chord[m]", "twist[deg]",
                 "Np[N/m]", "Tp[N/m]",
                 "alpha[deg]", "cl", "cd"]
        cols = [
            bem_result.r,
            dr,
            bem_result.chord if bem_result.chord is not None else np.zeros_like(bem_result.r),
            bem_result.twist_deg if bem_result.twist_deg is not None else np.zeros_like(bem_result.r),
            bem_result.Np, bem_result.Tp,
            bem_result.alpha, bem_result.cl, bem_result.cd,
        ]
        for attr, col_name in [
            ("cn",  "cn"),
            ("ct",  "ct"),
            ("a",   "a"),
            ("ap",  "ap"),
            ("W",   "W[m/s]"),
            ("Re",  "Re"),
            ("cm",  "cm"),
            ("Mp",  "Mp[N.m/m]"),
        ]:
            val = getattr(bem_result, attr, None)
            if val is not None:
                names.append(col_name)
                cols.append(val)
        data = np.column_stack(cols)
        np.savetxt(
            ts_dir / "bem_sectional.csv",
            data, delimiter=",", header=",".join(names), comments="",
        )

    def _append_global_csv(
        self,
        bem_result: BEMResult,
        forces: np.ndarray,
        displacements: np.ndarray,
        step: int,
        time: float,
        time_str: str = "",
    ) -> None:
        """Append one row to the accumulated global loads CSV (like structural_report.csv)."""
        csv_path = self._output_folder / "bem_report.csv"
        write_header = not csv_path.exists()

        disp_mag = np.linalg.norm(displacements, axis=1)
        force_mag = np.linalg.norm(forces, axis=1)

        tip_vec = displacements[self._tip_node_idx] if len(displacements) else np.zeros(3)
        tip_disp_x, tip_disp_y, tip_disp_z = float(tip_vec[0]), float(tip_vec[1]), float(tip_vec[2])
        tip_disp_mag = float(np.linalg.norm(tip_vec))

        max_disp = float(np.max(disp_mag)) if len(disp_mag) else 0.0
        max_force = float(np.max(force_mag)) if len(force_mag) else 0.0

        CP = bem_result.CP if bem_result.CP is not None else float("nan")
        CT = bem_result.CT if bem_result.CT is not None else float("nan")
        CQ = bem_result.CQ if bem_result.CQ is not None else float("nan")
        Mb = bem_result.Mb if bem_result.Mb is not None else float("nan")

        with open(csv_path, "a") as f:
            if write_header:
                f.write(
                    "Time [s],Step,"
                    "Thrust [N],Torque [N.m],Power [W],"
                    "CT,CQ,CP,"
                    "Mb [N.m],"
                    "Max Disp [m],"
                    "Tip Disp X [m],Tip Disp Y [m],Tip Disp Z [m],Tip Disp Mag [m],"
                    "Max Nodal Force [N]\n"
                )
            f.write(
                f"{time:.6f},{step},"
                f"{bem_result.thrust:.6e},{bem_result.torque:.6e},{bem_result.power:.6e},"
                f"{CT:.6f},{CQ:.6f},{CP:.6f},"
                f"{Mb:.6e},"
                f"{max_disp:.6e},"
                f"{tip_disp_x:.6e},{tip_disp_y:.6e},{tip_disp_z:.6e},{tip_disp_mag:.6e},"
                f"{max_force:.6e}\n"
            )

    def _update_pvd(self, folder: "Path", pvd_name: str, filename: str, time: float) -> None:
        """Append a timestep entry to the PVD collection file."""
        pvd_path = folder / pvd_name

        header = (
            '<?xml version="1.0"?>\n'
            '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian"'
            ' compressor="vtkZLibDataCompressor">\n'
            "  <Collection>\n"
        )
        footer = "  </Collection>\n</VTKFile>"
        entry = f'    <DataSet timestep="{time}" group="" part="0" file="{filename}"/>\n'

        if not pvd_path.exists():
            with open(pvd_path, "w") as f:
                f.write(header + entry + footer)
        else:
            with open(pvd_path, "r") as f:
                lines = f.readlines()
            valid_lines = [
                line for line in lines
                if "</Collection>" not in line and "</VTKFile>" not in line
            ]
            with open(pvd_path, "w") as f:
                f.writelines(valid_lines)
                f.write(entry)
                f.write(footer)

    def _log_window(
        self,
        bem_result: BEMResult,
        displacements: np.ndarray,
        current_time: float,
    ) -> None:
        """Log a diagnostic summary for this time window."""
        if len(displacements) == 0:
            tip_vec = np.zeros(3)
            max_disp = 0.0
        else:
            disp_mag = np.linalg.norm(displacements, axis=1)
            tip_vec = displacements[self._tip_node_idx]
            max_disp = float(np.max(disp_mag))
        tip_x, tip_y, tip_z = float(tip_vec[0]), float(tip_vec[1]), float(tip_vec[2])
        tip_mag = float(np.linalg.norm(tip_vec))
        logger.info(
            "[BEM-FSI] t=%.4f s | window=%d | iters=%d | T=%.2f kN "
            "| Q=%.2f kNm | P=%.2f kW "
            "| tip=(%.4f, %.4f, %.4f) mag=%.4f m | max_disp=%.4f m",
            current_time,
            self._window_count,
            self._iteration_count,
            bem_result.thrust * 1e-3,
            bem_result.torque * 1e-3,
            bem_result.power * 1e-3,
            tip_x, tip_y, tip_z,
            tip_mag,
            max_disp,
        )


# ---------------------------------------------------------------------------
# Factory helper used by the CLI
# ---------------------------------------------------------------------------


def build_from_config(
    mesh: MeshModel,
    cfg: dict,
    config_file: str | Path = "precice-config.xml",
    viz_mesh: MeshModel | None = None,
) -> "BEMFSIParticipant":
    """Construct a :class:`BEMFSIParticipant` from a YAML config dict.

    The dict is expected to match the schema documented in
    ``docs/cli-reference.md`` for ``fem-shell-bem-fsi``.

    Parameters
    ----------
    mesh : MeshModel
        Pre-built blade surface mesh (caller's responsibility).
    cfg : dict
        Parsed YAML content.
    config_file : str or Path
        Override for the preCICE XML path (takes precedence over
        ``cfg["config_file"]`` when explicitly provided).
    """
    bem_cfg = cfg.get("bem", {})
    output_cfg = cfg.get("output", {})

    blade_file = cfg.get("blade_file")
    if blade_file is None:
        raise ValueError("'blade_file' is required in the BEM-FSI configuration.")

    blade_aero = load_blade_aero(
        blade_file,
        default_re=float(bem_cfg.get("default_re", 1e7)),
        neuralfoil_model=bem_cfg.get("neuralfoil_model", "large"),
        hub_radius=float(bem_cfg.get("hub_radius", 0.0)),
        n_blades=int(bem_cfg.get("n_blades", 3)),
        viterna_ar=float(bem_cfg.get("viterna_ar", 17.0)),
        viterna_confidence_threshold=float(bem_cfg.get("viterna_confidence_threshold", 0.5)),
    )

    return BEMFSIParticipant(
        mesh=mesh,
        blade_aero=blade_aero,
        bem_config=bem_cfg,
        participant=cfg.get("participant", "Fluid"),
        config_file=cfg.get("config_file", str(config_file)),
        coupling_mesh=cfg.get("coupling_mesh", "Fluid-Mesh"),
        displacement_data=cfg.get("displacement_data", "Displacement"),
        force_data=cfg.get("force_data", "Force"),
        output_folder=output_cfg.get("folder", "bem_fsi_results"),
        log_interval=int(output_cfg.get("log_interval", 10)),
        viz_mesh=viz_mesh,
    )
