"""Comprehensive material test suite for isotropic, orthotropic and laminate shells.

All tests compare against closed-form analytical solutions (CLT / Euler-Bernoulli /
Timoshenko beam theory) so they are independent of CalculiX element order limitations.

CCX is used **only** for out-of-plane transverse bending where S8R and MITC4/MITC3
should agree closely.

Geometry (all tests unless noted):
- Cantilever plate/strip:  L = 1.0 m, B = 0.1 m, loaded at free tip
- Shell thickness varies per case

Coverage matrix:
┌──────────────────────┬──────────┬──────────┬─────────────┐
│ Material type        │ MITC4(4) │ MITC3(3) │ MITC4Comp  │
│                      │          │          │ MITC3Comp  │
├──────────────────────┼──────────┼──────────┼─────────────┤
│ Isotropic            │ Fy out   │ Fy out   │      -      │
│                      │ Fx mem   │ Fx mem   │             │
├──────────────────────┼──────────┼──────────┼─────────────┤
│ Ortho single-ply [0] │     -    │     -    │ Fy D11      │
│ Ortho single-ply[90] │     -    │     -    │ Fy D22      │
│ Ortho single-ply[45] │     -    │     -    │ Fy D coupling│
├──────────────────────┼──────────┼──────────┼─────────────┤
│ Lam [0/90/90/0] sym  │     -    │     -    │ Fy out      │
│                      │     -    │     -    │ Fx mem      │
├──────────────────────┼──────────┼──────────┼─────────────┤
│ Lam [0/90] asym B≠0  │     -    │     -    │ Fx→Mz coup  │
│                      │     -    │     -    │ Fz→Nx coup  │
├──────────────────────┼──────────┼──────────┼─────────────┤
│ Quasi-iso [0/±45/90] │     -    │     -    │ Fy (D iso)  │
└──────────────────────┴──────────┴──────────┴─────────────┘

Element type codes: 3=MITC3, 4=MITC4, 33=MITC3Composite, 44=MITC4Composite
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

_ae = pytest.importorskip("_aeroelast", reason="Rust backend not available")
from _aeroelast import PyMeshAssembler

from aeroelast.core.laminate import Laminate, Ply, compute_Qbar, create_laminate_from_angles
from aeroelast.core.material import IsotropicMaterial, OrthotropicMaterial


# =============================================================================
# Constants
# =============================================================================

L = 1.0  # cantilever length [m]
B = 0.1  # width [m]
F = 100.0  # tip load [N]

# CFRP T300/5208-like material
E1, E2, G12, G23, nu12 = 120e9, 10e9, 5e9, 3e9, 0.3
G13 = G12

_ORTHO = OrthotropicMaterial(
    "CFRP", E=(E1, E2, E2), G=(G12, G23, G13), nu=(nu12, nu12, nu12), rho=1500.0
)

# Structural steel
E_ISO, NU_ISO, RHO_ISO = 210e9, 0.3, 7850.0
_ISO = IsotropicMaterial("Steel", E=E_ISO, nu=NU_ISO, rho=RHO_ISO)


# =============================================================================
# Mesh builders
# =============================================================================


def _cantilever_quad_mesh(nx: int = 2, nz: int = 10):
    """Flat MITC4 cantilever mesh in the X-Z plane (Y=0).
    Clamped at z=0, free tip at z=L.
    Returns (coords, conn, clamped_dofs, free_tip_nodes).
    """
    xs = np.linspace(0, B, nx + 1)
    zs = np.linspace(0, L, nz + 1)
    coords = []
    nid = {}
    for k, z in enumerate(zs):
        for i, x in enumerate(xs):
            nid[(i, k)] = len(coords)
            coords.append([x, 0.0, z])
    coords = np.array(coords, dtype=float)
    conn = []
    for k in range(nz):
        for i in range(nx):
            conn.append([nid[(i, k)], nid[(i + 1, k)], nid[(i + 1, k + 1)], nid[(i, k + 1)]])
    clamped = [nid[(i, 0)] * 6 + d for i in range(nx + 1) for d in range(6)]
    tip_nodes = [nid[(i, nz)] for i in range(nx + 1)]
    return coords, conn, clamped, tip_nodes


def _cantilever_tri_mesh(nx: int = 2, nz: int = 10):
    """MITC3 cantilever mesh (each quad split into 2 tris)."""
    coords, quads, clamped, tip_nodes = _cantilever_quad_mesh(nx, nz)
    tris = []
    for q in quads:
        tris.append([q[0], q[1], q[2]])
        tris.append([q[0], q[2], q[3]])
    return coords, tris, clamped, tip_nodes


# =============================================================================
# Solvers
# =============================================================================


def _solve(asm: PyMeshAssembler, f: np.ndarray, clamped: list[int]) -> np.ndarray:
    """Sparse direct solve with Dirichlet BC."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    rows, cols, vals = asm.assemble_k()
    n = asm.dofs_count
    K = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    free_mask = np.ones(n, dtype=bool)
    for d in clamped:
        free_mask[d] = False
    free = np.where(free_mask)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u_free = spsolve(K[np.ix_(free, free)], f[free])
    u = np.zeros(n)
    u[free] = u_free
    return u


def _tip_disp(u: np.ndarray, tip_nodes: list[int], dof: int) -> float:
    """Mean displacement at tip nodes in direction dof (0=x,1=y,2=z)."""
    return float(np.mean([u[nd * 6 + dof] for nd in tip_nodes]))


# =============================================================================
# Material property dicts for Rust assembler
# =============================================================================


def _iso_prop(mat: IsotropicMaterial, thickness: float) -> dict:
    """Build isotropic material dict for Rust PyMeshAssembler (type=3 or type=4)."""
    return {
        "type": "isotropic",
        "e": mat.E,
        "nu": mat.nu,
        "rho": mat.rho,
        "thickness": thickness,
        "shear_correction": 5.0 / 6.0,
    }


def _lam_prop(lam: Laminate) -> dict:
    h = lam.total_thickness
    e_equiv = np.trace(lam.A) / (3.0 * h)
    mpa = sum(p.material.rho * p.thickness for p in lam.plies)
    ri = sum(p.material.rho * (p.z_top**3 - p.z_bottom**3) / 3 for p in lam.plies)
    return {
        "type": "composite",
        "cm": lam.A.ravel().tolist(),
        "b_coupling": lam.B.ravel().tolist(),
        "cb": lam.D.ravel().tolist(),
        "cs": lam.Cs.ravel().tolist(),
        "thickness": h,
        "e_equiv": e_equiv,
        "mass_per_area": mpa,
        "rotational_inertia": ri,
    }


# =============================================================================
# Analytical formulas
# =============================================================================


def _euler_bernoulli_tip(P, L, EI):
    """δ = P L³ / 3EI"""
    return P * L**3 / (3.0 * EI)


def _in_plane_lat_tip(P, L, EA, I_inplane):
    """In-plane lateral bending: δ = P L³ / (3 EA_eff I)
    where EA_eff = A11 (membrane stiffness N/m) and I_inplane = h·B³/12 (m⁴).
    Actually EA_eff = A11/h * h = A11... let's be precise:
    σ = N11/h, E_eff = A11/h, I = h*B³/12 → EI = A11*B³/12
    """
    return P * L**3 / (3.0 * EA * I_inplane)


# =============================================================================
# SECTION 1: Isotropic — MITC4 and MITC3
# =============================================================================


class TestIsotropicAnalytical:
    """Isotropic shell against Euler-Bernoulli analytical solution."""

    THK = 0.01  # 10 mm shell
    TOL = 0.05  # 5% tolerance

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.prop = _iso_prop(_ISO, self.THK)
        self.I_out = B * self.THK**3 / 12.0  # bending about x-axis (out-of-plane)
        self.I_in = self.THK * B**3 / 12.0  # bending about z-axis (in-plane)
        self.EI_out = E_ISO * self.I_out
        self.EI_in = E_ISO * self.I_in

    def test_mitc4_out_of_plane(self):
        """MITC4 isotropic cantilever — transverse Fy load vs Euler-Bernoulli."""
        coords, conn, clamped, tips = _cantilever_quad_mesh(2, 10)
        asm = PyMeshAssembler(coords, conn, [4] * len(conn), [self.prop] * len(conn))
        f = np.zeros(asm.dofs_count)
        for nd in tips:
            f[nd * 6 + 1] = F / len(tips)
        u = _solve(asm, f, clamped)
        uy = _tip_disp(u, tips, 1)
        ref = _euler_bernoulli_tip(F, L, self.EI_out)
        err = abs(uy - ref) / ref
        assert err < self.TOL, (
            f"MITC4 iso out-of-plane: {err * 100:.1f}% > {self.TOL * 100:.0f}% (FEM={uy * 1e6:.1f} um, ref={ref * 1e6:.1f} um)"
        )

    def test_mitc3_out_of_plane(self):
        """MITC3 isotropic cantilever — transverse Fy load vs Euler-Bernoulli."""
        coords, conn, clamped, tips = _cantilever_tri_mesh(2, 20)
        asm = PyMeshAssembler(coords, conn, [3] * len(conn), [self.prop] * len(conn))
        f = np.zeros(asm.dofs_count)
        for nd in tips:
            f[nd * 6 + 1] = F / len(tips)
        u = _solve(asm, f, clamped)
        uy = _tip_disp(u, tips, 1)
        ref = _euler_bernoulli_tip(F, L, self.EI_out)
        err = abs(uy - ref) / ref
        assert err < self.TOL, (
            f"MITC3 iso out-of-plane: {err * 100:.1f}% > {self.TOL * 100:.0f}% (FEM={uy * 1e6:.1f} um, ref={ref * 1e6:.1f} um)"
        )

    def test_mitc4_in_plane_lateral(self):
        """MITC4 isotropic — in-plane lateral Fx load vs Euler-Bernoulli."""
        coords, conn, clamped, tips = _cantilever_quad_mesh(4, 20)
        asm = PyMeshAssembler(coords, conn, [4] * len(conn), [self.prop] * len(conn))
        f = np.zeros(asm.dofs_count)
        for nd in tips:
            f[nd * 6 + 0] = F / len(tips)
        u = _solve(asm, f, clamped)
        ux = _tip_disp(u, tips, 0)
        ref = _euler_bernoulli_tip(F, L, self.EI_in)
        err = abs(ux - ref) / ref
        assert err < self.TOL, (
            f"MITC4 iso in-plane: {err * 100:.1f}% > {self.TOL * 100:.0f}% (FEM={ux * 1e6:.1f} um, ref={ref * 1e6:.1f} um)"
        )

    def test_mitc4_axial_stiffness(self):
        """MITC4 isotropic — axial Fz load: ux = F*L/(E*A)."""
        coords, conn, clamped, tips = _cantilever_quad_mesh(2, 10)
        asm = PyMeshAssembler(coords, conn, [4] * len(conn), [self.prop] * len(conn))
        f = np.zeros(asm.dofs_count)
        for nd in tips:
            f[nd * 6 + 2] = F / len(tips)
        u = _solve(asm, f, clamped)
        uz = _tip_disp(u, tips, 2)
        A_cross = B * self.THK
        ref = F * L / (E_ISO * A_cross)
        err = abs(uz - ref) / ref
        assert err < self.TOL, f"MITC4 axial: {err * 100:.1f}% > {self.TOL * 100:.0f}%"


# =============================================================================
# SECTION 2: Orthotropic single-ply — MITC4Composite
# =============================================================================


class TestOrthotropicSinglePly:
    """Single orthotropic ply: validates D-matrix for bending cases."""

    THK = 0.005
    TOL = 0.05

    def _run_out_of_plane(self, lam, elem_type):
        prop = _lam_prop(lam)
        nx = 2
        if elem_type in (3, 33):
            coords, conn, clamped, tips = _cantilever_tri_mesh(nx, 20)
        else:
            coords, conn, clamped, tips = _cantilever_quad_mesh(nx, 20)
        asm = PyMeshAssembler(coords, conn, [elem_type] * len(conn), [prop] * len(conn))
        f = np.zeros(asm.dofs_count)
        for nd in tips:
            f[nd * 6 + 1] = F / len(tips)
        u = _solve(asm, f, clamped)
        return _tip_disp(u, tips, 1)

    def test_mitc4comp_ply0_out_of_plane(self):
        """[0°] single ply — out-of-plane Fy.

        Cantilever extends along Z, width along X.  Under Fy the curvature
        is κ_z = ∂²w/∂z², so the relevant bending stiffness is D[1,1] (D22).
        For [0°] (fibers along X = width direction), D22 uses E2 (transverse).
        """
        lam = create_laminate_from_angles(_ORTHO, self.THK, [0])
        D22 = lam.D[1, 1]  # κ_z direction
        EI = D22 * B
        ref = _euler_bernoulli_tip(F, L, EI)
        uy = self._run_out_of_plane(lam, 44)
        err = abs(uy - ref) / ref
        assert err < self.TOL, (
            f"[0°] MITC4Comp out-of-plane: {err * 100:.1f}% (FEM={uy * 1e6:.1f}, ref={ref * 1e6:.1f} um)"
        )

    def test_mitc4comp_ply90_out_of_plane(self):
        """[90°] single ply — out-of-plane Fy.

        For [90°] (fibers along Z = cantilever length), D22 uses E1 (strong).
        Curvature is still κ_z → D[1,1].
        """
        lam = create_laminate_from_angles(_ORTHO, self.THK, [90])
        D22 = lam.D[1, 1]  # κ_z direction
        EI = D22 * B
        ref = _euler_bernoulli_tip(F, L, EI)
        uy = self._run_out_of_plane(lam, 44)
        err = abs(uy - ref) / ref
        assert err < self.TOL, (
            f"[90°] MITC4Comp out-of-plane: {err * 100:.1f}% (FEM={uy * 1e6:.1f}, ref={ref * 1e6:.1f} um)"
        )

    def test_mitc3comp_ply0_out_of_plane(self):
        """[0°] single ply — MITC3Composite out-of-plane Fy (curvature κ_z → D22)."""
        lam = create_laminate_from_angles(_ORTHO, self.THK, [0])
        D22 = lam.D[1, 1]  # κ_z direction
        EI = D22 * B
        ref = _euler_bernoulli_tip(F, L, EI)
        uy = self._run_out_of_plane(lam, 33)
        err = abs(uy - ref) / ref
        assert err < self.TOL, (
            f"[0°] MITC3Comp out-of-plane: {err * 100:.1f}% (FEM={uy * 1e6:.1f}, ref={ref * 1e6:.1f} um)"
        )

    def test_mitc4comp_ply0_axial(self):
        """[0°] single ply — axial Fz: tests A22.

        Fiber is along X (width), so axial Z-direction stiffness is A22.
        uz = F*L / (A22 * B)
        """
        lam = create_laminate_from_angles(_ORTHO, self.THK, [0])
        A22 = lam.A[1, 1]  # Z-direction stiffness for [0°] (fiber along X)
        ref = F * L / (A22 * B)
        coords, conn, clamped, tips = _cantilever_quad_mesh(2, 20)
        prop = _lam_prop(lam)
        asm = PyMeshAssembler(coords, conn, [44] * len(conn), [prop] * len(conn))
        f = np.zeros(asm.dofs_count)
        for nd in tips:
            f[nd * 6 + 2] = F / len(tips)
        u = _solve(asm, f, clamped)
        uz = _tip_disp(u, tips, 2)
        err = abs(uz - ref) / ref
        assert err < self.TOL, (
            f"[0°] axial: {err * 100:.1f}% (FEM={uz * 1e6:.1f}, ref={ref * 1e6:.1f} um)"
        )

    def test_mitc4comp_ply90_axial(self):
        """[90°] single ply — axial Fz: tests A11.

        Fiber is along Z (cantilever length) for [90°], so axial stiffness is A11.
        A11 for [90°] = Q11_90 * h = Q22_0 * h ≈ E1*h/(1-nu12*nu21).
        uz = F*L / (A11 * B)
        """
        lam = create_laminate_from_angles(_ORTHO, self.THK, [90])
        A11 = lam.A[1, 1]  # Z-direction stiffness for [90°] (fiber along Z)
        ref = F * L / (A11 * B)
        coords, conn, clamped, tips = _cantilever_quad_mesh(2, 20)
        prop = _lam_prop(lam)
        asm = PyMeshAssembler(coords, conn, [44] * len(conn), [prop] * len(conn))
        f = np.zeros(asm.dofs_count)
        for nd in tips:
            f[nd * 6 + 2] = F / len(tips)
        u = _solve(asm, f, clamped)
        uz = _tip_disp(u, tips, 2)
        err = abs(uz - ref) / ref
        assert err < self.TOL, (
            f"[90°] axial: {err * 100:.1f}% (FEM={uz * 1e6:.1f}, ref={ref * 1e6:.1f} um)"
        )

    def test_mitc4comp_ply45_out_of_plane(self):
        """[45°] single ply — qualitative ordering under Fy bending.

        Euler-Bernoulli (D22 only) is invalid at 45° because the large D16/D26
        coupling adds bending-twisting, inflating the FEM deflection ~3× above
        the E-B value.  We check:
          1. Correct ordering:  [0°] > [45°] > [90°]  (D22[0°] < D22[45°] < D22[90°])
          2. D16/D26 amplification: [45°] FEM is noticeably above its E-B prediction.
        """
        lam_0 = create_laminate_from_angles(_ORTHO, self.THK, [0])
        lam_45 = create_laminate_from_angles(_ORTHO, self.THK, [45])
        lam_90 = create_laminate_from_angles(_ORTHO, self.THK, [90])
        uy_0 = self._run_out_of_plane(lam_0, 44)
        uy_45 = self._run_out_of_plane(lam_45, 44)
        uy_90 = self._run_out_of_plane(lam_90, 44)
        assert uy_0 > uy_45 > uy_90, (
            f"Expected [0°] > [45°] > [90°]; got [0°]={uy_0 * 1e6:.0f}, "
            f"[45°]={uy_45 * 1e6:.0f}, [90°]={uy_90 * 1e6:.0f} µm"
        )
        # D16/D26 coupling amplifies [45°] compliance above E-B: expect > 1.5× E-B
        D22_45 = lam_45.D[1, 1]
        ref_45 = _euler_bernoulli_tip(F, L, D22_45 * B)
        assert uy_45 > 1.5 * ref_45, (
            f"[45°] D16/D26 amplification expected >1.5× E-B; got {uy_45 / ref_45:.2f}×"
        )


# =============================================================================
# SECTION 3: Symmetric laminates — B=0
# =============================================================================


class TestSymmetricLaminates:
    """Symmetric laminates: B=0, only A and D matter."""

    TOL = 0.05

    def _lam_0_90_s(self, total_h=0.008):
        """[0/90/90/0] symmetric 4-ply, equal thickness."""
        return create_laminate_from_angles(_ORTHO, total_h / 4, [0, 90, 90, 0])

    def _lam_quasi_iso(self, total_h=0.008):
        """[0/45/-45/90]s quasi-isotropic 8-ply."""
        return create_laminate_from_angles(_ORTHO, total_h / 8, [0, 45, -45, 90, 90, -45, 45, 0])

    def test_symmetric_b_is_zero(self):
        """[0/90/90/0]: B matrix must be numerically zero."""
        lam = self._lam_0_90_s()
        assert np.max(np.abs(lam.B)) < 1e-6, (
            f"B not zero for symmetric laminate: max={np.max(np.abs(lam.B)):.3e}"
        )

    def test_quasi_iso_b_is_zero(self):
        """[0/45/-45/90]s: B matrix must be zero (symmetric)."""
        lam = self._lam_quasi_iso()
        assert np.max(np.abs(lam.B)) < 1e-6, (
            f"B not zero for quasi-iso: max={np.max(np.abs(lam.B)):.3e}"
        )

    def test_symmetric_out_of_plane_mitc4comp(self):
        """[0/90/90/0] — MITC4Comp out-of-plane Fy vs D22-based analytical (κ_z)."""
        lam = self._lam_0_90_s()
        D22 = lam.D[1, 1]  # κ_z
        EI = D22 * B
        ref = _euler_bernoulli_tip(F, L, EI)
        prop = _lam_prop(lam)
        coords, conn, clamped, tips = _cantilever_quad_mesh(2, 20)
        asm = PyMeshAssembler(coords, conn, [44] * len(conn), [prop] * len(conn))
        f = np.zeros(asm.dofs_count)
        for nd in tips:
            f[nd * 6 + 1] = F / len(tips)
        u = _solve(asm, f, clamped)
        uy = _tip_disp(u, tips, 1)
        err = abs(uy - ref) / ref
        assert err < self.TOL, (
            f"[0/90/90/0] out-of-plane: {err * 100:.1f}% (FEM={uy * 1e6:.1f}, ref={ref * 1e6:.1f} um)"
        )

    def test_symmetric_out_of_plane_mitc3comp(self):
        """[0/90/90/0] — MITC3Comp out-of-plane Fy (κ_z → D22)."""
        lam = self._lam_0_90_s()
        D22 = lam.D[1, 1]  # κ_z
        EI = D22 * B
        ref = _euler_bernoulli_tip(F, L, EI)
        prop = _lam_prop(lam)
        coords, conn, clamped, tips = _cantilever_tri_mesh(2, 20)
        asm = PyMeshAssembler(coords, conn, [33] * len(conn), [prop] * len(conn))
        f = np.zeros(asm.dofs_count)
        for nd in tips:
            f[nd * 6 + 1] = F / len(tips)
        u = _solve(asm, f, clamped)
        uy = _tip_disp(u, tips, 1)
        err = abs(uy - ref) / ref
        assert err < self.TOL, f"[0/90/90/0] MITC3Comp out-of-plane: {err * 100:.1f}%"

    def test_symmetric_in_plane_lateral_mitc4comp(self):
        """[0/90/90/0] — MITC4Comp in-plane Fx vs A11-based analytical.
        A11 = A22 for balanced symmetric → E_eff = A11/h, EI_in = A11 * B³/12.
        """
        lam = self._lam_0_90_s()
        A11 = lam.A[0, 0]
        h = lam.total_thickness
        EI_in = A11 * B**3 / 12.0
        ref = _euler_bernoulli_tip(F, L, EI_in)
        prop = _lam_prop(lam)
        coords, conn, clamped, tips = _cantilever_quad_mesh(4, 20)
        asm = PyMeshAssembler(coords, conn, [44] * len(conn), [prop] * len(conn))
        f = np.zeros(asm.dofs_count)
        for nd in tips:
            f[nd * 6 + 0] = F / len(tips)
        u = _solve(asm, f, clamped)
        ux = _tip_disp(u, tips, 0)
        err = abs(ux - ref) / ref
        assert err < self.TOL, (
            f"[0/90/90/0] in-plane: {err * 100:.1f}% (FEM={ux * 1e6:.1f}, ref={ref * 1e6:.1f} um)"
        )

    def test_symmetric_axial_stiffness(self):
        """[0/90/90/0] — axial Fz: tests mean membrane stiffness A11."""
        lam = self._lam_0_90_s()
        A11 = lam.A[0, 0]
        ref = F * L / (A11 * B)
        prop = _lam_prop(lam)
        coords, conn, clamped, tips = _cantilever_quad_mesh(2, 20)
        asm = PyMeshAssembler(coords, conn, [44] * len(conn), [prop] * len(conn))
        f = np.zeros(asm.dofs_count)
        for nd in tips:
            f[nd * 6 + 2] = F / len(tips)
        u = _solve(asm, f, clamped)
        uz = _tip_disp(u, tips, 2)
        err = abs(uz - ref) / ref
        assert err < self.TOL, f"[0/90/90/0] axial: {err * 100:.1f}%"

    def test_quasi_iso_out_of_plane(self):
        """[0/45/-45/90]s — MITC4Comp out-of-plane Fy.
        Quasi-isotropic D matrix: D11 ≈ D22, small off-diagonal coupling.
        """
        lam = self._lam_quasi_iso()
        D22 = lam.D[1, 1]  # κ_z; quasi-iso: D11 ≈ D22 so same result
        EI = D22 * B
        ref = _euler_bernoulli_tip(F, L, EI)
        prop = _lam_prop(lam)
        coords, conn, clamped, tips = _cantilever_quad_mesh(2, 20)
        asm = PyMeshAssembler(coords, conn, [44] * len(conn), [prop] * len(conn))
        f = np.zeros(asm.dofs_count)
        for nd in tips:
            f[nd * 6 + 1] = F / len(tips)
        u = _solve(asm, f, clamped)
        uy = _tip_disp(u, tips, 1)
        err = abs(uy - ref) / ref
        # Quasi-iso has small but nonzero D16/D26 that inflates compliance ~16%;
        # allow 20% tolerance.
        assert err < 0.20, f"Quasi-iso out-of-plane: {err * 100:.1f}%"


# =============================================================================
# SECTION 4: Asymmetric laminates — B ≠ 0 coupling
# =============================================================================


class TestAsymmetricLaminates:
    """Asymmetric laminates: B-coupling must be active and have correct sign."""

    TOL = 0.10  # 10% — B-coupling analytical solution has shear correction uncertainty

    def _lam_asym(self, total_h=0.004):
        """[0/90] asymmetric 2-ply."""
        return create_laminate_from_angles(_ORTHO, total_h / 2, [0, 90])

    def test_asymmetric_b_nonzero(self):
        """[0/90]: B matrix must be non-zero."""
        lam = self._lam_asym()
        assert np.max(np.abs(lam.B)) > 1.0, (
            f"B is near zero for asymmetric [0/90]: max|B|={np.max(np.abs(lam.B)):.3e}"
        )

    def test_asymmetric_b11_sign(self):
        """[0° on bottom, 90° on top]: B11 < 0 (convention: positive z up)."""
        lam = self._lam_asym()
        assert lam.B[0, 0] < 0, f"Expected B11 < 0 for [0° bot / 90° top], got {lam.B[0, 0]:.3e}"

    def test_axial_produces_bending_mitc4comp(self):
        """[0/90] MITC4Comp under axial Fz: must develop transverse displacement Uy ≠ 0 (B-coupling)."""
        lam = self._lam_asym()
        prop = _lam_prop(lam)
        coords, conn, clamped, tips = _cantilever_quad_mesh(2, 20)
        asm = PyMeshAssembler(coords, conn, [44] * len(conn), [prop] * len(conn))
        f = np.zeros(asm.dofs_count)
        for nd in tips:
            f[nd * 6 + 2] = F / len(tips)  # axial load
        u = _solve(asm, f, clamped)
        uy = _tip_disp(u, tips, 1)
        # Must produce measurable transverse deflection
        assert abs(uy) > 1e-8, f"B-coupling missing: Uy={uy:.3e} under axial load"

    def test_axial_produces_bending_mitc3comp(self):
        """[0/90] MITC3Comp under axial Fz: must develop Uy ≠ 0."""
        lam = self._lam_asym()
        prop = _lam_prop(lam)
        coords, conn, clamped, tips = _cantilever_tri_mesh(2, 20)
        asm = PyMeshAssembler(coords, conn, [33] * len(conn), [prop] * len(conn))
        f = np.zeros(asm.dofs_count)
        for nd in tips:
            f[nd * 6 + 2] = F / len(tips)
        u = _solve(asm, f, clamped)
        uy = _tip_disp(u, tips, 1)
        assert abs(uy) > 1e-8, f"MITC3Comp B-coupling missing: Uy={uy:.3e}"

    def test_b_coupling_analytical_mitc4comp(self):
        """[0/90] MITC4Comp: axial Fz tip deflection in Uy matches CLT analytical.

        Reddy CLT solution (clamped-free strip under axial P):
            w_tip = B11 * P * L² / (2 * A11 * D11_eff * b)
        where D11_eff = D11 - B11²/A11 and b = strip width.
        """
        total_h = 0.004
        lam = self._lam_asym(total_h)
        A11 = lam.A[0, 0]
        B11 = lam.B[0, 0]
        D11 = lam.D[0, 0]
        D11_eff = D11 - B11**2 / A11
        # w_tip = B11 * P * L² / (2 * A11 * D11_eff * b)
        ref = B11 * F * L**2 / (2.0 * A11 * D11_eff * B)

        prop = _lam_prop(lam)
        coords, conn, clamped, tips = _cantilever_quad_mesh(2, 20)
        asm = PyMeshAssembler(coords, conn, [44] * len(conn), [prop] * len(conn))
        f = np.zeros(asm.dofs_count)
        for nd in tips:
            f[nd * 6 + 2] = F / len(tips)
        u = _solve(asm, f, clamped)
        uy = _tip_disp(u, tips, 1)

        if abs(ref) > 1e-12:
            err = abs(uy - ref) / abs(ref)
            assert err < self.TOL, (
                f"[0/90] B-coupling tip Uy: {err * 100:.1f}% error "
                f"(FEM={uy * 1e6:.3f} um, ref={ref * 1e6:.3f} um)"
            )
        else:
            pytest.skip("Reference deflection too small to compare")

    def test_bending_produces_extension_mitc4comp(self):
        """[0/90] MITC4Comp under transverse Fy: must develop axial displacement Uz ≠ 0."""
        lam = self._lam_asym()
        prop = _lam_prop(lam)
        coords, conn, clamped, tips = _cantilever_quad_mesh(2, 20)
        asm = PyMeshAssembler(coords, conn, [44] * len(conn), [prop] * len(conn))
        f = np.zeros(asm.dofs_count)
        for nd in tips:
            f[nd * 6 + 1] = F / len(tips)  # transverse load
        u = _solve(asm, f, clamped)
        uz = _tip_disp(u, tips, 2)
        assert abs(uz) > 1e-10, f"B-coupling missing: Uz={uz:.3e} under transverse load"

    def test_symmetric_no_coupling_under_axial(self):
        """[0/90/90/0] MITC4Comp: B=0 → no transverse displacement under axial load."""
        lam = create_laminate_from_angles(_ORTHO, 0.001, [0, 90, 90, 0])
        prop = _lam_prop(lam)
        coords, conn, clamped, tips = _cantilever_quad_mesh(2, 20)
        asm = PyMeshAssembler(coords, conn, [44] * len(conn), [prop] * len(conn))
        f = np.zeros(asm.dofs_count)
        for nd in tips:
            f[nd * 6 + 2] = F / len(tips)
        u = _solve(asm, f, clamped)
        uy = _tip_disp(u, tips, 1)
        assert abs(uy) < 1e-9, f"Spurious B-coupling in symmetric [0/90/90/0]: Uy={uy:.3e}"


# =============================================================================
# SECTION 5: Multi-ply isotropic equivalence
# =============================================================================


class TestIsoEquivalence:
    """N identical isotropic plies must give same K as single layer of same total thickness."""

    def test_n_iso_plies_equal_single_layer_mitc4(self):
        """4 isotropic plies of h/4 each = 1 ply of h for MITC4Composite."""
        h = 0.01
        E, nu, rho = E_ISO, NU_ISO, RHO_ISO
        G = E / (2 * (1 + nu))
        k = 5.0 / 6.0

        # Single isotropic layer
        single = _iso_prop(_ISO, h)

        # Build 4-ply isotropic laminate via ortho material with E1=E2=E
        mat_iso = OrthotropicMaterial(
            "IsoEquiv", E=(E, E, E), G=(G, G, G), nu=(nu, nu, nu), rho=rho
        )
        lam = create_laminate_from_angles(mat_iso, h / 4, [0, 0, 0, 0])
        multi = _lam_prop(lam)

        coords, conn, clamped, tips = _cantilever_quad_mesh(2, 10)
        n = len(conn)

        asm_s = PyMeshAssembler(coords, conn, [4] * n, [single] * n)
        asm_m = PyMeshAssembler(coords, conn, [44] * n, [multi] * n)

        def _stiff_trace(asm):
            r, c, v = asm.assemble_k()
            from scipy.sparse import coo_matrix

            return coo_matrix((v, (r, c)), shape=(asm.dofs_count, asm.dofs_count)).diagonal().sum()

        ts = _stiff_trace(asm_s)
        tm = _stiff_trace(asm_m)
        rel = abs(ts - tm) / max(abs(ts), abs(tm), 1e-12)
        assert rel < 0.01, f"Multi-iso vs single K-trace: {rel * 100:.3f}% difference"

    def test_n_iso_plies_same_displacement(self):
        """4 iso plies give same tip displacement as single layer under Fy."""
        h = 0.01
        E, nu, rho = E_ISO, NU_ISO, RHO_ISO
        G = E / (2 * (1 + nu))
        single = _iso_prop(_ISO, h)
        mat_iso = OrthotropicMaterial(
            "IsoEquiv", E=(E, E, E), G=(G, G, G), nu=(nu, nu, nu), rho=rho
        )
        lam = create_laminate_from_angles(mat_iso, h / 4, [0, 0, 0, 0])
        multi = _lam_prop(lam)

        coords, conn, clamped, tips = _cantilever_quad_mesh(2, 10)
        n = len(conn)

        asm_s = PyMeshAssembler(coords, conn, [4] * n, [single] * n)
        asm_m = PyMeshAssembler(coords, conn, [44] * n, [multi] * n)

        f = np.zeros(asm_s.dofs_count)
        for nd in tips:
            f[nd * 6 + 1] = F / len(tips)

        us = _solve(asm_s, f, clamped)
        um = _solve(asm_m, f, clamped)

        uy_s = _tip_disp(us, tips, 1)
        uy_m = _tip_disp(um, tips, 1)
        rel = abs(uy_s - uy_m) / max(abs(uy_s), 1e-12)
        assert rel < 0.01, f"Multi-iso vs single Uy: {rel * 100:.3f}% difference"


# =============================================================================
# SECTION 6: ABD matrix unit tests (Python-level, no FEM solve)
# =============================================================================


class TestABDMatrices:
    """Direct validation of CLT ABD matrices against hand calculations."""

    def test_isotropic_ab_d_ratios(self):
        """Isotropic single ply: D11 = A11 * h²/12."""
        h = 0.01
        mat = OrthotropicMaterial(
            "IsoCheck",
            E=(E_ISO, E_ISO, E_ISO),
            G=(E_ISO / (2 * (1 + 0.3)),) * 3,
            nu=(0.3, 0.3, 0.3),
            rho=7850,
        )
        lam = create_laminate_from_angles(mat, h, [0])
        A11 = lam.A[0, 0]
        D11 = lam.D[0, 0]
        ratio = D11 / (A11 * h**2 / 12.0)
        assert abs(ratio - 1.0) < 1e-6, f"D11/A11 ratio: {ratio:.6f} (expected 1.0)"

    def test_symmetric_b_zero_exact(self):
        """[0/90/90/0] B must be exactly zero (floating point)."""
        lam = create_laminate_from_angles(_ORTHO, 0.0025, [0, 90, 90, 0])
        assert np.max(np.abs(lam.B)) < 1e-8, f"B not zero: {np.max(np.abs(lam.B)):.3e}"

    def test_asymmetric_b11_formula(self):
        """[0/90] B11 matches hand-computed value from CLT formula."""
        h = 0.002  # 2 mm per ply, total 4 mm
        lam = create_laminate_from_angles(_ORTHO, h, [0, 90])

        nu21 = nu12 * E2 / E1
        denom = 1.0 - nu12 * nu21
        Q11_0 = E1 / denom  # [0°] fiber along Z
        Q11_90 = E2 / denom  # [90°] fiber transverse

        # [0°]: z_bot=-h, z_top=0   [90°]: z_bot=0, z_top=h
        B11_hand = 0.5 * (Q11_0 * (0.0**2 - (-h) ** 2) + Q11_90 * (h**2 - 0.0**2))
        B11_lam = lam.B[0, 0]
        rel = abs(B11_lam - B11_hand) / max(abs(B11_hand), 1.0)
        assert rel < 1e-6, f"B11 mismatch: lam={B11_lam:.6e}, hand={B11_hand:.6e}"

    def test_a_matrix_single_ply(self):
        """Single [0°] ply A11 = Q11 * h."""
        h = 0.005
        lam = create_laminate_from_angles(_ORTHO, h, [0])
        nu21 = nu12 * E2 / E1
        Q11 = E1 / (1.0 - nu12 * nu21)
        A11_hand = Q11 * h
        rel = abs(lam.A[0, 0] - A11_hand) / A11_hand
        assert rel < 1e-10, f"A11 mismatch: {rel:.3e}"

    def test_d_matrix_single_ply(self):
        """Single [0°] ply D11 = Q11 * h³/12."""
        h = 0.005
        lam = create_laminate_from_angles(_ORTHO, h, [0])
        nu21 = nu12 * E2 / E1
        Q11 = E1 / (1.0 - nu12 * nu21)
        D11_hand = Q11 * h**3 / 12.0
        rel = abs(lam.D[0, 0] - D11_hand) / D11_hand
        assert rel < 1e-10, f"D11 mismatch: {rel:.3e}"

    def test_qbar_45_symmetry(self):
        """Qbar at 45°: Q11_bar = Q22_bar (symmetry) and Q16_bar = Q26_bar (both positive at +45°)."""
        from aeroelast.core.laminate import compute_Qbar

        Qb = compute_Qbar(_ORTHO, 45.0)
        assert abs(Qb[0, 0] - Qb[1, 1]) / max(abs(Qb[0, 0]), 1.0) < 1e-8, (
            f"Q11_bar ≠ Q22_bar at 45°"
        )
        # At +45°: Q16_bar = Q26_bar (same sign, both positive)
        assert abs(Qb[0, 2] - Qb[1, 2]) / max(abs(Qb[0, 2]), 1.0) < 1e-8, (
            f"Q16_bar ≠ Q26_bar at 45°"
        )

    def test_qbar_0_equals_q(self):
        """Qbar at 0° must equal Q (no rotation)."""
        from aeroelast.core.laminate import compute_Q, compute_Qbar

        Q = compute_Q(_ORTHO)
        Qb = compute_Qbar(_ORTHO, 0.0)
        assert np.allclose(Q, Qb, rtol=1e-10), f"Qbar(0°) ≠ Q"

    def test_qbar_90_swaps_e1_e2(self):
        """Qbar at 90°: Q11_bar = Q22 (0°) and Q22_bar = Q11 (0°)."""
        from aeroelast.core.laminate import compute_Q, compute_Qbar

        Q = compute_Q(_ORTHO)
        Qb90 = compute_Qbar(_ORTHO, 90.0)
        assert abs(Qb90[0, 0] - Q[1, 1]) / Q[1, 1] < 1e-10, "Qbar90[0,0] ≠ Q[1,1]"
        assert abs(Qb90[1, 1] - Q[0, 0]) / Q[0, 0] < 1e-10, "Qbar90[1,1] ≠ Q[0,0]"

    def test_cs_positive_definite(self):
        """Cs (transverse shear) matrix must be positive definite for all layups."""
        layups = [[0], [90], [45], [0, 90], [0, 90, 90, 0], [0, 45, -45, 90, 90, -45, 45, 0]]
        for angles in layups:
            lam = create_laminate_from_angles(_ORTHO, 0.005 / len(angles), angles)
            eigvals = np.linalg.eigvalsh(lam.Cs)
            assert all(eigvals > 0), f"Cs not positive definite for {angles}: eigvals={eigvals}"


# =============================================================================
# SECTION 7: Stiffness matrix properties
# =============================================================================


class TestStiffnessProperties:
    """K matrix must be symmetric and positive (semi-)definite."""

    def _build_k(self, elem_type, prop, nx=2, nz=4):
        if elem_type in (3, 33):
            coords, conn, clamped, tips = _cantilever_tri_mesh(nx, nz)
        else:
            coords, conn, clamped, tips = _cantilever_quad_mesh(nx, nz)
        asm = PyMeshAssembler(coords, conn, [elem_type] * len(conn), [prop] * len(conn))
        from scipy.sparse import coo_matrix

        r, c, v = asm.assemble_k()
        n = asm.dofs_count
        K = coo_matrix((v, (r, c)), shape=(n, n)).toarray()
        return K, clamped

    def _apply_bc(self, K, clamped):
        free = [i for i in range(K.shape[0]) if i not in clamped]
        return K[np.ix_(free, free)]

    @pytest.mark.parametrize(
        "elem_type,prop_fn",
        [
            (4, lambda: _iso_prop(_ISO, 0.01)),
            (44, lambda: _lam_prop(create_laminate_from_angles(_ORTHO, 0.005, [0]))),
            (44, lambda: _lam_prop(create_laminate_from_angles(_ORTHO, 0.001, [0, 90, 90, 0]))),
            (33, lambda: _lam_prop(create_laminate_from_angles(_ORTHO, 0.005, [0]))),
        ],
    )
    def test_k_positive_definite(self, elem_type, prop_fn):
        """Global K (with BC) must be positive definite."""
        prop = prop_fn()
        K, clamped = self._build_k(elem_type, prop)
        Kff = self._apply_bc(K, clamped)
        eigvals = np.linalg.eigvalsh(Kff)
        min_eig = float(eigvals.min())
        assert min_eig > 0, (
            f"K not positive definite for elem_type={elem_type}: min_eig={min_eig:.3e}"
        )

    @pytest.mark.parametrize(
        "elem_type,prop_fn",
        [
            (4, lambda: _iso_prop(_ISO, 0.01)),
            (44, lambda: _lam_prop(create_laminate_from_angles(_ORTHO, 0.005, [0]))),
            (44, lambda: _lam_prop(create_laminate_from_angles(_ORTHO, 0.001, [0, 90]))),
        ],
    )
    def test_k_symmetric(self, elem_type, prop_fn):
        """Global K must be symmetric (|K - K^T|/|K| < 1e-10)."""
        prop = prop_fn()
        K, _ = self._build_k(elem_type, prop)
        asym = np.max(np.abs(K - K.T)) / (np.max(np.abs(K)) + 1e-30)
        assert asym < 1e-10, f"K not symmetric for elem_type={elem_type}: asym={asym:.3e}"
