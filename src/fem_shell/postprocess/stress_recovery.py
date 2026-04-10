"""
Stress and Strain Recovery Module for Shell and Solid Elements.

This module recovers nodal stress and strain fields from a displacement
solution vector and computes the derived engineering quantities (Von Mises,
principal stresses, maximum shear) that are standard outputs in commercial
FEA post-processors such as ANSYS, Abaqus and Nastran.

Two element families are supported:

* **Shell elements** (MITC3, MITC4) — Reissner–Mindlin plate/shell theory,
  plane-stress assumption (σ_zz = 0).  Three components are recovered:
  σ_xx, σ_yy, σ_xy.  The through-thickness variation is captured by
  evaluating at TOP (+h/2), MIDDLE (0) and BOTTOM (−h/2) surfaces.

* **Solid elements** (HEXA8/20, TETRA4/10, WEDGE6/15, PYRAMID5/13) —
  full 3-D elasticity.  Six Voigt components are recovered:
  σ_xx, σ_yy, σ_zz, τ_xy, τ_yz, τ_zx.

Shell Theory
------------
For Reissner–Mindlin shells, the in-plane stress at through-thickness
coordinate *z* measured from the mid-surface is:

    σ(z) = σ_m + z · κ_b

where:

* σ_m = (C / h) · (B_m · u_e)  — **membrane stress** (constant through
  thickness).  B_m is the membrane strain–displacement matrix and C is the
  plane-stress constitutive matrix (E, ν).  Note that the element's
  ``Cm()`` method returns the *integrated* membrane stiffness D = C · h;
  we divide by h to recover the actual material stress–strain matrix.
* κ_b = B_κ · u_e — **curvature** from the bending strain–displacement
  matrix.  The bending stress contribution is (C / h) · z · κ_b, varying
  linearly through the thickness.
* h — total shell thickness.

The stress is evaluated at the element's **parametric node coordinates**
(natural coordinates of the element nodes), then contributions from
adjacent elements are averaged at shared nodes (SPR-style nodal smoothing).

Solid Theory
------------
For 3-D solid elements the stress at any interior point is:

    σ = C · ε = C · B(ξ, η, ζ) · u_e

where C is the 6×6 constitutive (elasticity) tensor and B is the 6×(3·n)
strain–displacement matrix evaluated at parametric coordinates (ξ, η, ζ).

Stresses are most accurate at the **Gauss integration points**
(superconvergent points of B), not at the nodes.  To obtain nodal values
the module uses a **Gauss-to-Node extrapolation matrix**:

    E = N_gp⁻¹    (when n_gp = n_nodes — exact inverse)
    E = pinv(N_gp) (otherwise — Moore–Penrose pseudo-inverse)

where N_gp is the (n_gp × n_nodes) matrix whose row *i* contains the shape
functions evaluated at Gauss point *i*.  Multiplying E · σ_gp yields the
best nodal estimate of the stress field within each element.  Values at
nodes shared by multiple elements are then averaged.

This approach matches the *stress extrapolation* procedure used in
ANSYS Mechanical (ESHAPE), Abaqus (EXTRAPOLATE=YES) and Nastran (GPSTRESS).

Derived Quantities
------------------
Von Mises equivalent stress (3-D general form, reduces to the plane-stress
formula when σ_zz = τ_yz = τ_zx = 0):

    σ_vm = √( ½[(σ_xx−σ_yy)² + (σ_yy−σ_zz)² + (σ_zz−σ_xx)²]
              + 3[τ_xy² + τ_yz² + τ_zx²] )

Principal stresses:

* **2-D (shells):** Mohr's circle  —  σ₁,₂ = (σ_xx+σ_yy)/2
  ± √[(σ_xx−σ_yy)²/4 + τ_xy²] ;  τ_max = (σ₁ − σ₂) / 2.
* **3-D (solids):** Closed-form cubic eigenvalue of the Cauchy stress
  tensor via the three tensor invariants I₁, I₂, I₃.  The three roots
  are obtained with the trigonometric (Cardano) formula without any
  iterative solver.  τ_max = (σ₁ − σ₃) / 2.

References
----------
- Bathe, K.J. (2014). *Finite Element Procedures*, 2nd Edition.
  Chapters 5 (isoparametric elements) and 6 (shell elements).
- Cook, R.D., Malkus, D.S., Plesha, M.E., Witt, R.J. (2002).
  *Concepts and Applications of Finite Element Analysis*, 4th Edition.
- Hinton, E. & Campbell, J.S. (1974). "Local and Global Smoothing of
  Discontinuous Finite Element Functions Using a Least Squares Method",
  Int. J. Num. Meth. Eng. 8, 461–480.  (Gauss-to-node extrapolation.)
- Zienkiewicz, O.C. & Zhu, J.Z. (1992). "The Superconvergent Patch
  Recovery and a posteriori error estimates", Int. J. Num. Meth. Eng.
  33, 1331–1364.  (Theoretical basis for nodal stress smoothing.)
"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from fem_shell.elements.elements import ElementFamily

if TYPE_CHECKING:
    from fem_shell.core.assembler import MeshAssembler


class StressLocation(Enum):
    """Through-thickness coordinate for shell stress evaluation.

    In Reissner–Mindlin shell theory the stress varies linearly through
    the thickness *h*.  The through-thickness coordinate *z* is measured
    from the mid-surface:

    * ``TOP``    — z = +h/2 (outer fibre, tension under positive bending).
    * ``MIDDLE`` — z = 0    (mid-surface, membrane stress only).
    * ``BOTTOM`` — z = −h/2 (inner fibre, compression under positive bending).

    For solid elements this parameter is ignored; the full 3-D stress
    state is computed directly at the Gauss points.
    """

    TOP = "top"
    MIDDLE = "middle"
    BOTTOM = "bottom"


class StressType(Enum):
    """Stress decomposition mode for shell elements.

    Shell stresses can be decomposed into two physically distinct
    contributions:

    * ``MEMBRANE`` — σ_m = (C/h) · B_m · u_e.  Constant through the
      thickness; arises from in-plane stretching/compression.
    * ``BENDING``  — σ_b(z) = (C/h) · z · B_κ · u_e.  Varies linearly;
      arises from curvature (plate bending).
    * ``TOTAL``    — σ = σ_m + σ_b(z).  Combined stress at the
      requested ``StressLocation``.

    For solid elements the decomposition is not applicable: the full
    3-D stress is always computed regardless of this parameter.
    """

    MEMBRANE = "membrane"
    BENDING = "bending"
    TOTAL = "total"


@dataclass
class StressResult:
    """
    Container for stress computation results.

    For shell elements (plane stress), only ``sigma_xx``, ``sigma_yy``
    and ``sigma_xy`` are non-zero.  For solid elements the full 3-D
    tensor is populated (``sigma_zz``, ``tau_yz``, ``tau_zx``).

    Attributes
    ----------
    sigma_xx, sigma_yy, sigma_xy : np.ndarray
        In-plane stress components (always present).
    von_mises : np.ndarray
        Von Mises equivalent stress (general 3-D formula).
    sigma_1, sigma_2 : np.ndarray
        Maximum / minimum principal stresses (2-D Mohr circle when
        ``sigma_zz`` is absent; full eigenvalue if 3-D).
    sigma_3 : np.ndarray or None
        Third principal stress (only for 3-D solid elements).
    tau_max : np.ndarray
        Maximum shear stress.
    principal_angle : np.ndarray
        Angle (rad) of first principal direction from x-axis (2-D only;
        set to 0 for 3-D since direction is a full eigenvector).
    sigma_zz : np.ndarray or None
        Out-of-plane normal stress (solid elements).
    tau_yz : np.ndarray or None
        Transverse shear stress yz (solid elements).
    tau_zx : np.ndarray or None
        Transverse shear stress zx (solid elements).
    """

    sigma_xx: np.ndarray
    sigma_yy: np.ndarray
    sigma_xy: np.ndarray
    von_mises: np.ndarray
    sigma_1: np.ndarray
    sigma_2: np.ndarray
    tau_max: np.ndarray
    principal_angle: np.ndarray
    # 3-D fields (None when plane-stress / shell)
    sigma_zz: Optional[np.ndarray] = None
    tau_yz: Optional[np.ndarray] = None
    tau_zx: Optional[np.ndarray] = None
    sigma_3: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary for VTK output."""
        d: Dict[str, np.ndarray] = {
            "sigma_xx": self.sigma_xx,
            "sigma_yy": self.sigma_yy,
            "sigma_xy": self.sigma_xy,
            "von_mises": self.von_mises,
            "sigma_1": self.sigma_1,
            "sigma_2": self.sigma_2,
            "tau_max": self.tau_max,
            "principal_angle": np.degrees(self.principal_angle),
        }
        if self.sigma_zz is not None:
            d["sigma_zz"] = self.sigma_zz
        if self.tau_yz is not None:
            d["tau_yz"] = self.tau_yz
        if self.tau_zx is not None:
            d["tau_zx"] = self.tau_zx
        if self.sigma_3 is not None:
            d["sigma_3"] = self.sigma_3
        return d


@dataclass
class StrainResult:
    """
    Container for strain computation results.

    Attributes
    ----------
    epsilon_xx, epsilon_yy : np.ndarray
        Normal strains.
    gamma_xy : np.ndarray
        In-plane engineering shear strain (= 2·ε_xy).
    epsilon_1, epsilon_2 : np.ndarray
        Principal strains (max / min).
    gamma_max : np.ndarray
        Maximum shear strain.
    epsilon_zz : np.ndarray or None
        Out-of-plane normal strain (solid elements).
    gamma_yz : np.ndarray or None
        Transverse shear strain yz (solid elements).
    gamma_zx : np.ndarray or None
        Transverse shear strain zx (solid elements).
    epsilon_3 : np.ndarray or None
        Third principal strain (solid elements).
    """

    epsilon_xx: np.ndarray
    epsilon_yy: np.ndarray
    gamma_xy: np.ndarray
    epsilon_1: np.ndarray
    epsilon_2: np.ndarray
    gamma_max: np.ndarray
    # 3-D fields
    epsilon_zz: Optional[np.ndarray] = None
    gamma_yz: Optional[np.ndarray] = None
    gamma_zx: Optional[np.ndarray] = None
    epsilon_3: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary for VTK output."""
        d: Dict[str, np.ndarray] = {
            "epsilon_xx": self.epsilon_xx,
            "epsilon_yy": self.epsilon_yy,
            "gamma_xy": self.gamma_xy,
            "epsilon_1": self.epsilon_1,
            "epsilon_2": self.epsilon_2,
            "gamma_max": self.gamma_max,
        }
        if self.epsilon_zz is not None:
            d["epsilon_zz"] = self.epsilon_zz
        if self.gamma_yz is not None:
            d["gamma_yz"] = self.gamma_yz
        if self.gamma_zx is not None:
            d["gamma_zx"] = self.gamma_zx
        if self.epsilon_3 is not None:
            d["epsilon_3"] = self.epsilon_3
        return d


class StressRecovery:
    """
    Stress and strain recovery engine for shell and solid finite elements.

    This class takes a converged displacement solution and recovers the
    complete Cauchy stress tensor at nodes or element centres, together
    with the standard derived engineering quantities (Von Mises, principal
    stresses, maximum shear) used for structural assessment.

    Workflow
    --------
    1. **Instantiation** — store a reference to the ``MeshAssembler``
       (which owns the element map, connectivity and constitutive data)
       and a copy of the displacement vector.
    2. **Stress evaluation** — for each element the displacement DOFs are
       gathered and the stress is computed:

       * *Shell elements (MITC3, MITC4)*:  Membrane and bending
         strain–displacement matrices ``B_m``, ``B_κ`` produce a plane-
         stress triplet [σ_xx, σ_yy, τ_xy] at a chosen parametric
         point (r, s) and through-thickness location z.
       * *Solid elements (HEXA, TETRA, WEDGE, PYRAMID)*:  The 6×(3n)
         strain–displacement matrix ``B(ξ, η, ζ)`` is evaluated at each
         Gauss integration point.  The product C · B · u_e yields the
         full Voigt stress vector [σ_xx, σ_yy, σ_zz, τ_xy, τ_yz, τ_zx].

    3. **Nodal smoothing** — stresses from adjacent elements are averaged
       at shared nodes.  For solid elements the Gauss-point values are
       first **extrapolated to the element nodes** via the pseudo-inverse
       extrapolation matrix ``E = pinv(N_gp)`` before averaging.

    Supported Element Topologies
    ----------------------------
    ============  =====  =======  ===================================
    Element       Nodes  GP count Quadrature rule
    ============  =====  =======  ===================================
    MITC3           3       3     Hammer (triangle)
    MITC4           4       4     2×2 Gauss–Legendre
    TETRA4          4       1     Single centroid point
    TETRA10        10       4     4-point Hammer
    HEXA8           8       8     2×2×2 Gauss–Legendre
    HEXA20         20      27     3×3×3 Gauss–Legendre
    WEDGE6          6       6     Triangle × 2-pt Gauss
    WEDGE15        15      21     Triangle × 3-pt Gauss (higher order)
    PYRAMID5        5       8     Based on collapsed hex mapping
    PYRAMID13      13      18     Higher-order collapsed hex
    ============  =====  =======  ===================================

    Parameters
    ----------
    domain : MeshAssembler
        Mesh assembler that owns the element map, node coordinates,
        constitutive properties and DOF connectivity.
    u : np.ndarray or PETSc.Vec
        Full (unreduced) displacement solution vector.  For shell
        elements it contains 6 DOFs per node [u, v, w, θ_x, θ_y, θ_z];
        for solid elements 3 DOFs per node [u, v, w].

    Attributes
    ----------
    n_nodes : int
        Total number of mesh nodes.
    n_elements : int
        Total number of elements in the mesh.
    dofs_per_node : int
        Number of DOFs per node (6 for shells, 3 for solids).

    Notes
    -----
    The extrapolation matrices are **cached** per element topology
    (``_extrap_cache``) so that they are computed only once regardless
    of the number of elements of each type.
    """

    # Parametric node coordinates for each supported element topology.
    _SHELL_NODE_COORDS: Dict[str, List[Tuple[float, float]]] = {
        "MITC3": [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        "MITC4": [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
    }

    _SOLID_NODE_COORDS: Dict[str, List[Tuple[float, float, float]]] = {
        "TETRA4": [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ],
        "TETRA10": [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.5, 0.0, 0.0),
            (0.5, 0.5, 0.0),
            (0.0, 0.5, 0.0),
            (0.0, 0.0, 0.5),
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
        ],
        "HEXA8": [
            (-1, -1, -1),
            (1, -1, -1),
            (1, 1, -1),
            (-1, 1, -1),
            (-1, -1, 1),
            (1, -1, 1),
            (1, 1, 1),
            (-1, 1, 1),
        ],
        "HEXA20": [
            (-1, -1, -1),
            (1, -1, -1),
            (1, 1, -1),
            (-1, 1, -1),
            (-1, -1, 1),
            (1, -1, 1),
            (1, 1, 1),
            (-1, 1, 1),
            (0, -1, -1),
            (1, 0, -1),
            (0, 1, -1),
            (-1, 0, -1),
            (0, -1, 1),
            (1, 0, 1),
            (0, 1, 1),
            (-1, 0, 1),
            (-1, -1, 0),
            (1, -1, 0),
            (1, 1, 0),
            (-1, 1, 0),
        ],
        "WEDGE6": [
            (0, 0, -1),
            (1, 0, -1),
            (0, 1, -1),
            (0, 0, 1),
            (1, 0, 1),
            (0, 1, 1),
        ],
        "WEDGE15": [
            (0, 0, -1),
            (1, 0, -1),
            (0, 1, -1),
            (0, 0, 1),
            (1, 0, 1),
            (0, 1, 1),
            (0.5, 0, -1),
            (0.5, 0.5, -1),
            (0, 0.5, -1),
            (0.5, 0, 1),
            (0.5, 0.5, 1),
            (0, 0.5, 1),
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
        ],
        "PYRAMID5": [
            (-1, -1, 0),
            (1, -1, 0),
            (1, 1, 0),
            (-1, 1, 0),
            (0, 0, 1),
        ],
        "PYRAMID13": [
            (-1, -1, 0),
            (1, -1, 0),
            (1, 1, 0),
            (-1, 1, 0),
            (0, 0, 1),
            (0, -1, 0),
            (1, 0, 0),
            (0, 1, 0),
            (-1, 0, 0),
            (-0.5, -0.5, 0.5),
            (0.5, -0.5, 0.5),
            (0.5, 0.5, 0.5),
            (-0.5, 0.5, 0.5),
        ],
    }

    def __init__(self, domain: "MeshAssembler", u):
        self.domain = domain
        if hasattr(u, "array"):
            self.u = u.array.copy()
        else:
            self.u = np.asarray(u).copy()
        self.dofs_per_node = domain.dofs_per_node
        self.n_nodes = len(domain.mesh.coords_array)
        self.n_elements = len(domain._element_map)
        self._node_id_to_index = domain.mesh.node_id_to_index

        # Pre-compute and cache extrapolation matrices for solid elements
        self._extrap_cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Element family helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_shell(element) -> bool:
        """Return *True* if *element* belongs to the shell family (MITC3/4)."""
        return getattr(element, "element_family", None) == ElementFamily.SHELL

    @staticmethod
    def _is_solid(element) -> bool:
        """Return *True* if *element* belongs to the 3-D solid family."""
        return getattr(element, "element_family", None) == ElementFamily.SOLID

    # ------------------------------------------------------------------
    # Extrapolation matrix (Gauss → Node) for solid elements
    # ------------------------------------------------------------------
    def _get_extrapolation_matrix(self, element) -> np.ndarray:
        """Build (or retrieve from cache) the Gauss-to-Node extrapolation matrix.

        In isoparametric elements, a field *f* known at the Gauss points
        can be expressed in terms of the nodal shape functions:

            f(ξ_i) = Σ_j  N_j(ξ_i) · f_j^node   ⇒   f_gp = N_gp · f_nodes

        Inverting this relation gives the **extrapolation matrix**:

            f_nodes = E · f_gp ,   E = N_gp⁻¹  (or  pinv(N_gp))

        where ``N_gp`` has shape ``(n_gp, n_nodes)`` with row *i* being
        the shape functions evaluated at Gauss point *i*.

        * When ``n_gp == n_nodes`` (e.g. HEXA8 with 2×2×2 quadrature)
          the system is square and ``E`` is the exact inverse — the
          extrapolated nodal values reproduce the GP values exactly.
        * Otherwise the Moore–Penrose pseudo-inverse is used, yielding
          the least-squares (over-determined) or minimum-norm
          (under-determined) solution.

        The matrix is **cached** by element name so it is computed at
        most once per element topology in the mesh.

        Parameters
        ----------
        element : SolidElement
            Any concrete solid element instance (HEXA8, TETRA4, …).

        Returns
        -------
        np.ndarray, shape (n_nodes, n_gp)
            Extrapolation matrix such that
            ``sigma_nodes = E @ sigma_gauss_pts``.
        """
        key = element.name
        if key in self._extrap_cache:
            return self._extrap_cache[key]

        points, _weights = element.integration_points
        n_gp = len(_weights)
        n_nodes = element.node_count

        # Build matrix N_gp (n_gp × n_nodes):  row i = N(ξ_i)
        N_gp = np.zeros((n_gp, n_nodes))
        for i, pt in enumerate(points):
            N_gp[i, :] = element.shape_functions(*pt)

        if n_gp == n_nodes:
            E = np.linalg.inv(N_gp)
        else:
            E = np.linalg.pinv(N_gp)  # (n_nodes, n_gp)

        self._extrap_cache[key] = E
        return E

    # ------------------------------------------------------------------
    # Displacement extraction
    # ------------------------------------------------------------------
    def _extract_element_displacements(self, element) -> np.ndarray:
        """Gather element DOFs from the global displacement vector.

        For each node belonging to *element*, the corresponding DOFs
        are extracted from the full solution vector ``self.u`` and
        concatenated into a local element displacement vector ``u_e``.

        * Shell elements (6 DOF/node): u_e has length 6 × n_nodes
          [u₁, v₁, w₁, θx₁, θy₁, θz₁, u₂, …].
        * Solid elements (3 DOF/node): u_e has length 3 × n_nodes
          [u₁, v₁, w₁, u₂, v₂, w₂, …].

        Parameters
        ----------
        element : FemElement
            Element whose connectivity supplies the node IDs.

        Returns
        -------
        np.ndarray, shape (n_dofs_elem,)
            Local displacement vector for the element.
        """
        elem_dofs: List[int] = []
        for node_id in element.node_ids:
            node_idx = self._node_id_to_index[node_id]
            start = node_idx * self.dofs_per_node
            elem_dofs.extend(range(start, start + self.dofs_per_node))
        return self.u[elem_dofs]

    # ------------------------------------------------------------------
    # Shell: stress / strain at a single parametric point
    # ------------------------------------------------------------------
    def _compute_shell_stress(
        self,
        element,
        u_elem: np.ndarray,
        r: float,
        s: float,
        location: StressLocation,
        stress_type: StressType,
    ) -> np.ndarray:
        """Evaluate the shell stress vector at a single parametric point.

        Computes the plane-stress triplet [σ_xx, σ_yy, τ_xy] at
        parametric coordinates *(r, s)* and through-thickness location *z*
        determined by *location*.

        The computation follows Reissner–Mindlin shell theory:

            σ = (C / h) · [ B_m · u_e  +  z · B_κ · u_e ]

        where C = ``element.Cm() / h`` is the plane-stress constitutive
        matrix (in Pa), B_m the membrane strain–displacement matrix,
        B_κ the bending (curvature) strain–displacement matrix, and *z*
        the distance from the mid-surface.

        .. note::

           ``element.Cm()`` returns the *integrated* membrane stiffness
           D = C·h (units N/m for force resultants).  Dividing by *h*
           gives the true material matrix and produces stress in Pa.

        Parameters
        ----------
        element : MITC3 | MITC4
            Shell element instance.
        u_elem : np.ndarray
            Element displacement vector (length = dofs_per_node × n_nodes).
        r, s : float
            Parametric (natural) coordinates within the element.
        location : StressLocation
            Through-thickness location: TOP (+h/2), MIDDLE (0) or BOTTOM (−h/2).
        stress_type : StressType
            Which stress contribution to include: MEMBRANE, BENDING or TOTAL.

        Returns
        -------
        np.ndarray, shape (3,)
            Stress vector [σ_xx, σ_yy, τ_xy] in Pa.
        """
        h = element.thickness
        z = {StressLocation.TOP: h / 2, StressLocation.BOTTOM: -h / 2}.get(location, 0.0)

        # Material matrix WITHOUT thickness factor  →  actual stress (Pa)
        C_mat = element.Cm() / h

        sigma = np.zeros(3)

        if stress_type in (StressType.MEMBRANE, StressType.TOTAL):
            B_m = element.B_m(r, s)
            epsilon_m = B_m @ u_elem
            sigma += C_mat @ epsilon_m

        if stress_type in (StressType.BENDING, StressType.TOTAL):
            B_kappa = element.B_kappa(r, s)
            kappa = B_kappa @ u_elem
            sigma += C_mat @ (z * kappa)

        return sigma

    def _compute_shell_strain(
        self,
        element,
        u_elem: np.ndarray,
        r: float,
        s: float,
        location: StressLocation,
    ) -> np.ndarray:
        """Evaluate the shell strain vector at a single parametric point.

        Computes the in-plane engineering strain triplet
        [ε_xx, ε_yy, γ_xy] at parametric coordinates *(r, s)* and
        through-thickness location *z* (*location*).

        The strain is decomposed as:

            ε(z) = ε_m + z · κ

        where ε_m = B_m · u_e is the membrane strain (mid-surface
        stretching) and κ = B_κ · u_e is the curvature (change in
        slope).  The combined strain at depth *z* captures both
        contributions.  Note that γ_xy is the *engineering* shear
        strain (= 2 · ε_xy).

        Parameters
        ----------
        element : MITC3 | MITC4
            Shell element instance.
        u_elem : np.ndarray
            Element displacement vector.
        r, s : float
            Parametric (natural) coordinates.
        location : StressLocation
            Through-thickness location.

        Returns
        -------
        np.ndarray, shape (3,)
            Strain vector [ε_xx, ε_yy, γ_xy].
        """
        h = element.thickness
        z = {StressLocation.TOP: h / 2, StressLocation.BOTTOM: -h / 2}.get(location, 0.0)

        B_m = element.B_m(r, s)
        epsilon_m = B_m @ u_elem

        B_kappa = element.B_kappa(r, s)
        kappa = B_kappa @ u_elem

        return epsilon_m + z * kappa

    # ------------------------------------------------------------------
    # Solid: stress / strain at Gauss points → extrapolate to nodes
    # ------------------------------------------------------------------
    def _compute_solid_gauss_stresses(
        self, element, u_elem: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute stress and strain at every Gauss integration point.

        For each integration point (ξ_i, η_i, ζ_i) the 6-component
        Voigt strain and stress are obtained as:

            ε_i = B(ξ_i, η_i, ζ_i) · u_e
            σ_i = C · ε_i

        where B is the 6×(3·n_nodes) strain–displacement matrix and C
        is the 6×6 constitutive (elasticity) matrix of the element.

        These Gauss-point values are the most accurate representation
        of the stress field within the element (superconvergent points
        of the B matrix).  They serve as input to the Gauss-to-Node
        extrapolation procedure in ``_get_extrapolation_matrix``.

        Parameters
        ----------
        element : SolidElement
            Solid element instance (HEXA8, TETRA4, …).
        u_elem : np.ndarray
            Element displacement vector (length = 3 × n_nodes).

        Returns
        -------
        sigma_gp : np.ndarray, shape (n_gp, 6)
            Voigt stress [σ_xx, σ_yy, σ_zz, τ_xy, τ_yz, τ_zx] at each
            Gauss point, in Pa.
        epsilon_gp : np.ndarray, shape (n_gp, 6)
            Voigt strain [ε_xx, ε_yy, ε_zz, γ_xy, γ_yz, γ_zx] at each
            Gauss point (dimensionless).
        """
        points, _weights = element.integration_points
        n_gp = len(_weights)
        sigma_gp = np.zeros((n_gp, 6))
        epsilon_gp = np.zeros((n_gp, 6))

        C = element.C

        for i, pt in enumerate(points):
            xi, eta, zeta = pt
            B = element.compute_B_matrix(xi, eta, zeta)
            eps = B @ u_elem
            sig = C @ eps
            epsilon_gp[i, :] = eps
            sigma_gp[i, :] = sig

        return sigma_gp, epsilon_gp

    # ------------------------------------------------------------------
    # Element-level stress  (centroid / single point)
    # ------------------------------------------------------------------
    def compute_element_stresses(
        self,
        location: StressLocation = StressLocation.MIDDLE,
        stress_type: StressType = StressType.TOTAL,
        gauss_point: Tuple[float, float] = (0.0, 0.0),
    ) -> StressResult:
        """Compute a single representative stress per element (centroid value).

        This method provides one stress state per element, suitable for
        contour plots at element centres ("element results" in commercial
        codes).  The evaluation strategy differs by element family:

        * **Shell elements** — stress is evaluated at the parametric
          point *gauss_point* (default: element centre) and
          through-thickness location *z* (from *location*).
        * **Solid elements** — the Gauss-point stresses are **averaged**.
          For low-order elements (HEXA8, TETRA4) this average coincides
          with the stress evaluated at the element's parametric centroid.

        Parameters
        ----------
        location : StressLocation, default MIDDLE
            Through-thickness position (shells only).
        stress_type : StressType, default TOTAL
            Which shell stress contribution to include.
        gauss_point : tuple of float, default (0.0, 0.0)
            Parametric coordinates at which to evaluate shell stress.

        Returns
        -------
        StressResult
            One stress state per element.  For shells only the in-plane
            components (σ_xx, σ_yy, τ_xy) are populated; for solids the
            full 3-D tensor is available.
        """
        n_elem = self.n_elements
        r0, s0 = gauss_point

        # Pre-allocate for the maximum (6) components
        sigma_all = np.zeros((n_elem, 6))
        has_solid = False

        for elem_idx, element in self.domain._element_map.items():
            u_elem = self._extract_element_displacements(element)

            if self._is_shell(element):
                sig3 = self._compute_shell_stress(element, u_elem, r0, s0, location, stress_type)
                sigma_all[elem_idx, :3] = sig3

            elif self._is_solid(element):
                has_solid = True
                sig_gp, _ = self._compute_solid_gauss_stresses(element, u_elem)
                sigma_all[elem_idx, :] = sig_gp.mean(axis=0)

        return self._build_stress_result(sigma_all, is_3d=has_solid)

    # ------------------------------------------------------------------
    # Nodal-averaged stress (the primary output)
    # ------------------------------------------------------------------
    def compute_nodal_stresses(
        self,
        location: StressLocation = StressLocation.MIDDLE,
        stress_type: StressType = StressType.TOTAL,
        smoothing: str = "average",
    ) -> StressResult:
        """Compute smoothed (continuous) nodal stress fields.

        This is the **primary output** method.  It produces a nodal
        stress field that can be passed directly to a VTU writer for
        visualisation.  The smoothing mimics commercial FEA practice
        (e.g. ANSYS “Nodeal Solution”, Abaqus “field output at nodes”).

        Procedure
        ---------
        * **Shell elements** — the stress is evaluated at each element's
          **parametric node coordinates** (the natural-coordinate location
          of each node).  Each element contributes its own stress value
          to the global node; contributions from all surrounding elements
          are then averaged (or area-weighted).
        * **Solid elements** — stresses are first computed at every
          **Gauss integration point** (superconvergent locations) and
          then **extrapolated to element nodes** via the pseudo-inverse
          extrapolation matrix ``E = pinv(N_gp)`` (see
          ``_get_extrapolation_matrix``).  The extrapolated element-nodal
          values are then averaged at shared global nodes.

        This two-step procedure (Gauss → element-node → global average)
        is the standard approach described in Zienkiewicz & Zhu (1992)
        and implemented in most commercial codes.

        Parameters
        ----------
        location : StressLocation, default MIDDLE
            Through-thickness position for shell stress evaluation.
        stress_type : StressType, default TOTAL
            Membrane, bending or total (shells only).
        smoothing : str, default ``"average"``
            Smoothing mode.  ``"average"`` gives equal weight to every
            contributing element; ``"area_weighted"`` weights by element
            area (shells) or volume (solids).

        Returns
        -------
        StressResult
            Nodal stress field with shape ``(n_nodes,)`` for each
            component.  For mixed meshes (shell + solid), both in-plane
            and 3-D components are populated.
        """
        n_nodes = self.n_nodes

        sigma_sum = np.zeros((n_nodes, 6))
        weight_sum = np.zeros(n_nodes)
        has_solid = False

        for _ei, element in self.domain._element_map.items():
            u_elem = self._extract_element_displacements(element)

            if smoothing == "area_weighted":
                weight = self._compute_element_weight(element)
            else:
                weight = 1.0

            if self._is_shell(element):
                node_coords = self._SHELL_NODE_COORDS.get(element.name)
                if node_coords is None:
                    raise ValueError(f"Unsupported shell element type: {element.name}")
                for local_idx, (r, s) in enumerate(node_coords):
                    sig3 = self._compute_shell_stress(element, u_elem, r, s, location, stress_type)
                    gn = self._node_id_to_index[element.node_ids[local_idx]]
                    sigma_sum[gn, :3] += weight * sig3
                    weight_sum[gn] += weight

            elif self._is_solid(element):
                has_solid = True
                E_mat = self._get_extrapolation_matrix(element)
                sig_gp, _ = self._compute_solid_gauss_stresses(element, u_elem)
                # Extrapolate each component independently
                sig_nodes = E_mat @ sig_gp  # (n_nodes_elem, 6)

                for local_idx in range(element.node_count):
                    gn = self._node_id_to_index[element.node_ids[local_idx]]
                    sigma_sum[gn, :] += weight * sig_nodes[local_idx, :]
                    weight_sum[gn] += weight

        mask = weight_sum > 0
        sigma_avg = np.zeros((n_nodes, 6))
        sigma_avg[mask] = sigma_sum[mask] / weight_sum[mask, np.newaxis]

        return self._build_stress_result(sigma_avg, is_3d=has_solid)

    # ------------------------------------------------------------------
    # Multi-layer shell convenience (TOP / MIDDLE / BOTTOM at once)
    # ------------------------------------------------------------------
    def compute_nodal_stresses_all_layers(
        self,
        stress_type: StressType = StressType.TOTAL,
        smoothing: str = "average",
    ) -> Dict[str, StressResult]:
        """Compute nodal stresses at TOP, MIDDLE and BOTTOM shell surfaces.

        For structures with shell elements, stress varies linearly
        through the thickness.  This convenience method evaluates
        ``compute_nodal_stresses`` three times — once at each
        through-thickness location — and returns the results in a dict.

        The three layers correspond to:

        * ``"TOP"`` — z = +h/2 (outer fibre)
        * ``"MID"`` — z = 0   (mid-surface, membrane only)
        * ``"BOT"`` — z = −h/2 (inner fibre)

        For solid elements the through-thickness location is irrelevant;
        the same 3-D stress field is returned for all three keys.

        Parameters
        ----------
        stress_type : StressType, default TOTAL
            Membrane, bending or combined.
        smoothing : str, default ``"average"``
            Smoothing mode (``"average"`` or ``"area_weighted"``).

        Returns
        -------
        dict of {str: StressResult}
            Keys are ``"TOP"``, ``"MID"``, ``"BOT"``.
        """
        return {
            "TOP": self.compute_nodal_stresses(StressLocation.TOP, stress_type, smoothing),
            "MID": self.compute_nodal_stresses(StressLocation.MIDDLE, stress_type, smoothing),
            "BOT": self.compute_nodal_stresses(StressLocation.BOTTOM, stress_type, smoothing),
        }

    def compute_nodal_stresses_all_layers_dict(
        self,
        stress_type: StressType = StressType.TOTAL,
        smoothing: str = "average",
    ) -> Dict[str, np.ndarray]:
        """Return a flat dictionary of nodal stress arrays for VTU export.

        Calls ``compute_nodal_stresses_all_layers`` internally and
        flattens the per-layer ``StressResult`` objects into a single
        ``dict[str, np.ndarray]`` with prefixed keys that can be passed
        directly as ``extra_fields`` to the checkpoint writer.

        Key naming convention::

            {LAYER}_{component}

        Examples: ``TOP_von_mises``, ``MID_sigma_xx``, ``BOT_tau_max``,
        ``TOP_sigma_zz`` (3-D solids), etc.

        Parameters
        ----------
        stress_type : StressType, default TOTAL
            Membrane, bending or combined.
        smoothing : str, default ``"average"``
            Smoothing mode.

        Returns
        -------
        dict of {str: np.ndarray}
            Flat dictionary suitable for ``CheckpointManager.write(
            extra_fields=...)``.
        """
        layers = self.compute_nodal_stresses_all_layers(stress_type, smoothing)
        out: Dict[str, np.ndarray] = {}
        for prefix, result in layers.items():
            for key, arr in result.to_dict().items():
                out[f"{prefix}_{key}"] = arr
        return out

    def compute_nodal_strains_all_layers_dict(
        self,
        smoothing: str = "average",
    ) -> Dict[str, np.ndarray]:
        """Return a flat dictionary of nodal strain arrays for VTU export.

        Evaluates ``compute_nodal_strains`` at TOP, MIDDLE and BOTTOM
        through-thickness locations (shells) and flattens the resulting
        ``StrainResult`` objects into a single ``dict[str, np.ndarray]``
        with layer-prefixed keys.

        Key naming convention::

            {LAYER}_{component}

        Examples: ``TOP_epsilon_xx``, ``MID_gamma_xy``, ``BOT_epsilon_1``.

        For solid elements the through-thickness location is irrelevant;
        the same strain field is returned for all three keys.

        Parameters
        ----------
        smoothing : str, default ``"average"``
            ``"average"`` or ``"area_weighted"``.

        Returns
        -------
        dict of {str: np.ndarray}
            Flat dictionary suitable for ``CheckpointManager.write(
            extra_fields=...)``.
        """
        out: Dict[str, np.ndarray] = {}
        for prefix, loc in (
            ("TOP", StressLocation.TOP),
            ("MID", StressLocation.MIDDLE),
            ("BOT", StressLocation.BOTTOM),
        ):
            result = self.compute_nodal_strains(location=loc, smoothing=smoothing)
            for key, arr in result.to_dict().items():
                out[f"{prefix}_{key}"] = arr
        return out

    # ------------------------------------------------------------------
    # Nodal strains
    # ------------------------------------------------------------------
    def compute_element_strains(
        self,
        location: StressLocation = StressLocation.MIDDLE,
        gauss_point: Tuple[float, float] = (0.0, 0.0),
    ) -> StrainResult:
        """Compute a single representative strain per element (centroid).

        Mirrors ``compute_element_stresses`` but returns the engineering
        strain tensor instead of the Cauchy stress.

        * **Shell** — evaluates ε_m + z·κ at *gauss_point* / *location*.
        * **Solid** — averages the Gauss-point strains.

        Parameters
        ----------
        location : StressLocation, default MIDDLE
            Through-thickness position (shells).
        gauss_point : tuple of float, default (0.0, 0.0)
            Parametric coordinates for shell evaluation.

        Returns
        -------
        StrainResult
            Element-centroid strains.  Normal strains are dimensionless;
            shear strains are *engineering* values (γ = 2ε).
        """
        r0, s0 = gauss_point
        n_elem = self.n_elements

        eps_all = np.zeros((n_elem, 6))
        has_solid = False

        for elem_idx, element in self.domain._element_map.items():
            u_elem = self._extract_element_displacements(element)

            if self._is_shell(element):
                eps3 = self._compute_shell_strain(element, u_elem, r0, s0, location)
                eps_all[elem_idx, :3] = eps3

            elif self._is_solid(element):
                has_solid = True
                _, eps_gp = self._compute_solid_gauss_stresses(element, u_elem)
                eps_all[elem_idx, :] = eps_gp.mean(axis=0)

        return self._build_strain_result(eps_all, is_3d=has_solid)

    def compute_nodal_strains(
        self,
        location: StressLocation = StressLocation.MIDDLE,
        smoothing: str = "average",
    ) -> StrainResult:
        """Compute smoothed (continuous) nodal strain fields.

        Applies the same procedure as ``compute_nodal_stresses`` but for
        the strain tensor:

        * **Shell** — ε(z) = ε_m + z·κ evaluated at each element's
          parametric node coordinates, then averaged at shared nodes.
        * **Solid** — Gauss-point strains are extrapolated to element
          nodes via ``E = pinv(N_gp)`` and averaged globally.

        Parameters
        ----------
        location : StressLocation, default MIDDLE
            Through-thickness position (shells).
        smoothing : str, default ``"average"``
            ``"average"`` or ``"area_weighted"``.

        Returns
        -------
        StrainResult
            Nodal strain field with shape ``(n_nodes,)`` for each
            component.
        """
        n_nodes = self.n_nodes

        eps_sum = np.zeros((n_nodes, 6))
        weight_sum = np.zeros(n_nodes)
        has_solid = False

        for _ei, element in self.domain._element_map.items():
            u_elem = self._extract_element_displacements(element)

            if smoothing == "area_weighted":
                weight = self._compute_element_weight(element)
            else:
                weight = 1.0

            if self._is_shell(element):
                node_coords = self._SHELL_NODE_COORDS.get(element.name)
                if node_coords is None:
                    raise ValueError(f"Unsupported shell element type: {element.name}")
                for local_idx, (r, s) in enumerate(node_coords):
                    eps3 = self._compute_shell_strain(element, u_elem, r, s, location)
                    gn = self._node_id_to_index[element.node_ids[local_idx]]
                    eps_sum[gn, :3] += weight * eps3
                    weight_sum[gn] += weight

            elif self._is_solid(element):
                has_solid = True
                E_mat = self._get_extrapolation_matrix(element)
                _, eps_gp = self._compute_solid_gauss_stresses(element, u_elem)
                eps_nodes = E_mat @ eps_gp  # (n_nodes_elem, 6)

                for local_idx in range(element.node_count):
                    gn = self._node_id_to_index[element.node_ids[local_idx]]
                    eps_sum[gn, :] += weight * eps_nodes[local_idx, :]
                    weight_sum[gn] += weight

        mask = weight_sum > 0
        eps_avg = np.zeros((n_nodes, 6))
        eps_avg[mask] = eps_sum[mask] / weight_sum[mask, np.newaxis]

        return self._build_strain_result(eps_avg, is_3d=has_solid)

    # ------------------------------------------------------------------
    # Derived quantities helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _von_mises_3d(s: np.ndarray) -> np.ndarray:
        """Compute the Von Mises equivalent stress from a Voigt array.

        The Von Mises (or Huber–Mises–Hencky) yield criterion is the
        most widely used scalar measure of the stress state.  It
        represents the distortional strain energy per unit volume and
        is used to predict the onset of yielding in ductile metals.

        The general 3-D formula is:

            σ_vm = √( ½[(σ_xx−σ_yy)² + (σ_yy−σ_zz)² + (σ_zz−σ_xx)²]
                       + 3[τ_xy² + τ_yz² + τ_zx²] )

        When σ_zz = τ_yz = τ_zx = 0 (plane stress) this reduces to
        the familiar shell formula:

            σ_vm = √(σ_xx² + σ_yy² − σ_xx·σ_yy + 3·τ_xy²)

        Parameters
        ----------
        s : np.ndarray, shape (N, 6)
            Voigt stress array with columns [σ_xx, σ_yy, σ_zz, τ_xy,
            τ_yz, τ_zx].

        Returns
        -------
        np.ndarray, shape (N,)
            Von Mises equivalent stress (scalar ≥ 0 at each point).
        """
        sxx, syy, szz = s[:, 0], s[:, 1], s[:, 2]
        txy, tyz, tzx = s[:, 3], s[:, 4], s[:, 5]
        return np.sqrt(
            0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
            + 3.0 * (txy**2 + tyz**2 + tzx**2)
        )

    @staticmethod
    def _principal_2d(sxx, syy, sxy):
        """Compute 2-D principal stresses via Mohr’s circle.

        For a plane-stress state the Cauchy stress tensor has the form:

            | σ_xx  τ_xy |
            | τ_xy  σ_yy |

        The principal stresses are the eigenvalues of this 2×2 tensor:

            σ₁,₂ = (σ_xx+σ_yy)/2  ±  R

        where R = √[(σ_xx−σ_yy)²/4 + τ_xy²] is the Mohr’s circle
        radius.  The maximum in-plane shear stress is τ_max = R, and
        the principal angle (angle from the x-axis to the first
        principal direction) is:

            θ_p = ½ arctan(2τ_xy / (σ_xx − σ_yy))

        Parameters
        ----------
        sxx, syy, sxy : np.ndarray
            In-plane stress components (broadcastable).

        Returns
        -------
        sigma_1 : np.ndarray
            Maximum principal stress.
        sigma_2 : np.ndarray
            Minimum principal stress.
        tau_max : np.ndarray
            Maximum in-plane shear stress (Mohr’s circle radius).
        theta_p : np.ndarray
            Principal angle in radians.
        """
        avg = (sxx + syy) / 2
        diff = (sxx - syy) / 2
        R = np.sqrt(diff**2 + sxy**2)
        return avg + R, avg - R, R, 0.5 * np.arctan2(2 * sxy, sxx - syy)

    @staticmethod
    def _principal_3d(s: np.ndarray):
        """Compute 3-D principal stresses via closed-form cubic eigenvalues.

        The Cauchy stress tensor in 3-D is a real symmetric 3×3 matrix
        whose eigenvalues are the principal stresses.  Rather than
        using ``np.linalg.eigvalsh`` (which involves QR iterations and
        is costly per-point), this method solves the characteristic
        cubic analytically using the three stress-tensor invariants:

            I₁ = σ_xx + σ_yy + σ_zz
            I₂ = σ_xxσ_yy + σ_yyσ_zz + σ_zzσ_xx − τ_xy² − τ_yz² − τ_zx²
            I₃ = det(σ)

        The cubic characteristic equation  λ³ − I₁λ² + I₂λ − I₃ = 0
        always has three real roots (since the tensor is symmetric).
        They are obtained with the **Cardano trigonometric formula**:

            σ_k = I₁/3 + 2√(p/3) · cos(φ/3 − 2πk/3),  k = 0,1,2

        where p = I₁²/3 − I₂ and φ = arccos(−q / (2(p/3)^{3/2})).

        The results are sorted so that σ₁ ≥ σ₂ ≥ σ₃, and the absolute
        maximum shear stress is τ_max = (σ₁ − σ₃) / 2.

        Parameters
        ----------
        s : np.ndarray, shape (N, 6)
            Voigt stress array [σ_xx, σ_yy, σ_zz, τ_xy, τ_yz, τ_zx].

        Returns
        -------
        sigma_1 : np.ndarray, shape (N,)
            Maximum principal stress.
        sigma_2 : np.ndarray, shape (N,)
            Intermediate principal stress.
        sigma_3 : np.ndarray, shape (N,)
            Minimum principal stress.
        tau_max : np.ndarray, shape (N,)
            Absolute maximum shear stress = (σ₁ − σ₃) / 2.
        """
        n = s.shape[0]
        sxx, syy, szz = s[:, 0], s[:, 1], s[:, 2]
        txy, tyz, tzx = s[:, 3], s[:, 4], s[:, 5]

        # Invariants
        I1 = sxx + syy + szz
        I2 = sxx * syy + syy * szz + szz * sxx - txy**2 - tyz**2 - tzx**2
        I3 = sxx * syy * szz + 2 * txy * tyz * tzx - sxx * tyz**2 - syy * tzx**2 - szz * txy**2

        p = I1**2 / 3.0 - I2
        q = 2.0 * I1**3 / 27.0 - I1 * I2 / 3.0 + I3

        # Prevent numeric issues near zero discriminant
        p = np.maximum(p, 0.0)
        sqp = np.sqrt(p / 3.0)
        denom = 2.0 * (p / 3.0) * sqp
        denom = np.where(denom == 0, 1.0, denom)
        cos_arg = np.clip(-q / denom, -1.0, 1.0)
        phi = np.arccos(cos_arg) / 3.0

        mean = I1 / 3.0
        s1 = mean + 2.0 * sqp * np.cos(phi)
        s2 = mean + 2.0 * sqp * np.cos(phi - 2.0 * np.pi / 3.0)
        s3 = mean + 2.0 * sqp * np.cos(phi - 4.0 * np.pi / 3.0)

        # Sort so that s1 >= s2 >= s3
        stacked = np.stack([s1, s2, s3], axis=-1)
        stacked.sort(axis=-1)
        sigma_3 = stacked[:, 0]
        sigma_2 = stacked[:, 1]
        sigma_1 = stacked[:, 2]

        tau_max = (sigma_1 - sigma_3) / 2.0
        return sigma_1, sigma_2, sigma_3, tau_max

    # ------------------------------------------------------------------
    def _build_stress_result(self, sigma: np.ndarray, is_3d: bool) -> StressResult:
        """Assemble a ``StressResult`` from a raw Voigt stress array.

        Given an ``(N, 6)`` array of Voigt stresses (nodal or elemental),
        this factory method computes all derived quantities (Von Mises,
        principal stresses, maximum shear, principal angle) and packages
        them into a ``StressResult`` dataclass.

        The computation path depends on *is_3d*:

        * **is_3d = False** (shells) — 2-D Mohr’s circle for σ₁, σ₂,
          θ_p.  Only in-plane components are populated.
        * **is_3d = True** (solids)  — full 3-D cubic eigenvalue for
          σ₁, σ₂, σ₃.  All six Voigt components are populated.

        Parameters
        ----------
        sigma : np.ndarray, shape (N, 6)
            Voigt stress array [σ_xx, σ_yy, σ_zz, τ_xy, τ_yz, τ_zx].
        is_3d : bool
            If *True*, populate out-of-plane fields and use 3-D
            eigenvalue solver; otherwise use 2-D Mohr’s circle.

        Returns
        -------
        StressResult
        """
        vm = self._von_mises_3d(sigma)

        if is_3d:
            s1, s2, s3, tau = self._principal_3d(sigma)
            angle = np.zeros(sigma.shape[0])
        else:
            s1, s2, tau, angle = self._principal_2d(sigma[:, 0], sigma[:, 1], sigma[:, 3])
            s3 = None

        return StressResult(
            sigma_xx=sigma[:, 0],
            sigma_yy=sigma[:, 1],
            sigma_xy=sigma[:, 3],
            von_mises=vm,
            sigma_1=s1,
            sigma_2=s2,
            tau_max=tau,
            principal_angle=angle,
            sigma_zz=sigma[:, 2] if is_3d else None,
            tau_yz=sigma[:, 4] if is_3d else None,
            tau_zx=sigma[:, 5] if is_3d else None,
            sigma_3=s3,
        )

    def _build_strain_result(self, eps: np.ndarray, is_3d: bool) -> StrainResult:
        """Assemble a ``StrainResult`` from a raw Voigt strain array.

        Computes principal strains and maximum shear strain via in-plane
        Mohr’s circle on the (ε_xx, ε_yy, γ_xy/2) components.  For 3-D
        elements the out-of-plane components (ε_zz, γ_yz, γ_zx) are
        included in the result but the 3-D principal strain computation
        is not yet implemented (``epsilon_3`` is set to *None*).

        Parameters
        ----------
        eps : np.ndarray, shape (N, 6)
            Voigt strain array [ε_xx, ε_yy, ε_zz, γ_xy, γ_yz, γ_zx].
        is_3d : bool
            If *True*, populate out-of-plane strain fields.

        Returns
        -------
        StrainResult
        """
        exx, eyy, gxy = eps[:, 0], eps[:, 1], eps[:, 3]

        avg = (exx + eyy) / 2
        diff = (exx - eyy) / 2
        R = np.sqrt(diff**2 + (gxy / 2) ** 2)

        e1 = avg + R
        e2 = avg - R
        gmax = 2 * R

        return StrainResult(
            epsilon_xx=exx,
            epsilon_yy=eyy,
            gamma_xy=gxy,
            epsilon_1=e1,
            epsilon_2=e2,
            gamma_max=gmax,
            epsilon_zz=eps[:, 2] if is_3d else None,
            gamma_yz=eps[:, 4] if is_3d else None,
            gamma_zx=eps[:, 5] if is_3d else None,
            epsilon_3=None,  # Could compute 3-D principal strains similarly
        )

    # ------------------------------------------------------------------
    # Element area / volume (for area-weighted smoothing)
    # ------------------------------------------------------------------
    def _compute_element_weight(self, element) -> float:
        """Compute the geometric weight (area or volume) for smoothing.

        When ``smoothing="area_weighted"`` is requested, each element's
        contribution to the nodal average is weighted by its area (shell)
        or volume (solid).  Larger elements exert a proportionally
        higher influence on the averaged nodal value.

        Parameters
        ----------
        element : FemElement
            Shell or solid element.

        Returns
        -------
        float
            Element area (shells, in m²) or volume (solids, in m³),
            or 1.0 if the element family is unrecognised.
        """
        if self._is_shell(element):
            return self._compute_shell_area(element)
        elif self._is_solid(element):
            return self._compute_solid_volume(element)
        return 1.0

    @staticmethod
    def _compute_shell_area(element) -> float:
        """Compute the physical area of a shell element via Gauss quadrature.

        Integrates det(J) over the parametric domain using the same
        quadrature rule employed for stiffness integration.

        Returns 1.0 if quadrature data is unavailable.
        """
        gauss_pts = getattr(element, "_gauss_points", None)
        gauss_wts = getattr(element, "_gauss_weights", None)
        if gauss_pts is None or gauss_wts is None:
            return 1.0
        area = 0.0
        for (r, s), w in zip(gauss_pts, gauss_wts):
            _J, det_J = element.J(r, s)
            area += det_J * w
        return abs(area)

    @staticmethod
    def _compute_solid_volume(element) -> float:
        """Compute the physical volume of a solid element via Gauss quadrature.

        Integrates |det(J)| over the parametric domain using the
        element’s own integration rule.
        """
        points, weights = element.integration_points
        vol = 0.0
        for pt, w in zip(points, weights):
            _, det_J, _ = element._compute_jacobian(*pt)
            vol += abs(det_J) * w
        return vol


def compute_von_mises(
    sigma_xx: np.ndarray,
    sigma_yy: np.ndarray,
    sigma_xy: np.ndarray,
    sigma_zz: Optional[np.ndarray] = None,
    tau_yz: Optional[np.ndarray] = None,
    tau_zx: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute von Mises equivalent stress.

    Supports both plane-stress (3 components) and full 3-D (6 components).

    For plane stress (σ_zz = σ_xz = σ_yz = 0):

        σ_vm = √(σ_xx² + σ_yy² - σ_xx·σ_yy + 3·σ_xy²)

    For general 3-D:

        σ_vm = √(½[(σ_xx-σ_yy)² + (σ_yy-σ_zz)² + (σ_zz-σ_xx)²]
                 + 3[τ_xy² + τ_yz² + τ_zx²])

    Parameters
    ----------
    sigma_xx, sigma_yy, sigma_xy : np.ndarray
        In-plane stress components (required).
    sigma_zz, tau_yz, tau_zx : np.ndarray, optional
        Out-of-plane components.  When *None* the plane-stress formula
        is used (equivalent to setting them to zero).

    Returns
    -------
    np.ndarray
        Von Mises equivalent stress.
    """
    if sigma_zz is None:
        # Plane stress shortcut
        return np.sqrt(sigma_xx**2 + sigma_yy**2 - sigma_xx * sigma_yy + 3 * sigma_xy**2)

    szz = np.asarray(sigma_zz)
    tyz = np.asarray(tau_yz) if tau_yz is not None else np.zeros_like(szz)
    tzx = np.asarray(tau_zx) if tau_zx is not None else np.zeros_like(szz)

    return np.sqrt(
        0.5 * ((sigma_xx - sigma_yy) ** 2 + (sigma_yy - szz) ** 2 + (szz - sigma_xx) ** 2)
        + 3.0 * (sigma_xy**2 + tyz**2 + tzx**2)
    )


def compute_principal_stresses(
    sigma_xx: np.ndarray,
    sigma_yy: np.ndarray,
    sigma_xy: np.ndarray,
    sigma_zz: Optional[np.ndarray] = None,
    tau_yz: Optional[np.ndarray] = None,
    tau_zx: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute principal stresses and direction.

    For plane stress (default) returns ``(σ_1, σ_2, θ_p)``.
    For full 3-D returns ``(σ_1, σ_2, σ_3)`` (no angle).

    Parameters
    ----------
    sigma_xx, sigma_yy, sigma_xy : np.ndarray
        In-plane stress components.
    sigma_zz, tau_yz, tau_zx : np.ndarray, optional
        Out-of-plane components.

    Returns
    -------
    sigma_1, sigma_2 : np.ndarray
        Max / min principal stresses.
    theta_or_sigma3 : np.ndarray
        Principal angle (plane stress) **or** third principal stress (3-D).
    """
    if sigma_zz is None:
        sigma_avg = (sigma_xx + sigma_yy) / 2
        sigma_diff = (sigma_xx - sigma_yy) / 2
        R = np.sqrt(sigma_diff**2 + sigma_xy**2)
        return sigma_avg + R, sigma_avg - R, 0.5 * np.arctan2(2 * sigma_xy, sigma_xx - sigma_yy)

    # Full 3-D via StressRecovery helper
    n = len(sigma_xx)
    s = np.column_stack([
        sigma_xx,
        sigma_yy,
        np.asarray(sigma_zz),
        sigma_xy,
        np.asarray(tau_yz) if tau_yz is not None else np.zeros(n),
        np.asarray(tau_zx) if tau_zx is not None else np.zeros(n),
    ])
    s1, s2, s3, _ = StressRecovery._principal_3d(s)
    return s1, s2, s3


def compute_max_shear_stress(
    sigma_xx: np.ndarray,
    sigma_yy: np.ndarray,
    sigma_xy: np.ndarray,
    sigma_zz: Optional[np.ndarray] = None,
    tau_yz: Optional[np.ndarray] = None,
    tau_zx: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute maximum shear stress.

    For plane stress: τ_max = (σ_1 - σ_2) / 2.
    For 3-D: τ_max = (σ_1 - σ_3) / 2.

    Parameters
    ----------
    sigma_xx, sigma_yy, sigma_xy : np.ndarray
        In-plane stress components.
    sigma_zz, tau_yz, tau_zx : np.ndarray, optional
        Out-of-plane components.

    Returns
    -------
    np.ndarray
        Maximum shear stress.
    """
    if sigma_zz is None:
        sigma_diff = (sigma_xx - sigma_yy) / 2
        return np.sqrt(sigma_diff**2 + sigma_xy**2)

    s1, s2, s3 = compute_principal_stresses(sigma_xx, sigma_yy, sigma_xy, sigma_zz, tau_yz, tau_zx)
    return (s1 - s3) / 2
