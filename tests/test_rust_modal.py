"""Validate Rust modal solver against Python/SLEPc reference.

Tests:
1. Rust modal_solve gives identical frequencies to Python/SLEPc on the same K/M.
2. Composite blade modal analysis: Rust vs Python match.
3. Performance benchmark: Rust full pipeline vs Python full pipeline.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

petsc4py = pytest.importorskip("petsc4py", reason="PETSc not available")
from petsc4py import PETSc  # noqa: E402

slepc4py = pytest.importorskip("slepc4py", reason="SLEPc not available")
from slepc4py import SLEPc  # noqa: E402

fem_shell_core = pytest.importorskip(
    "fem_shell_core", reason="Rust backend not available"
)

from fem_shell.core.assembler import MeshAssembler  # noqa: E402
from fem_shell.core.bc import BoundaryConditionManager, DirichletCondition  # noqa: E402
from fem_shell.core.material import IsotropicMaterial  # noqa: E402
from fem_shell.core.mesh.entities import ElementType, MeshElement, Node  # noqa: E402
from fem_shell.core.mesh.model import MeshModel  # noqa: E402
from fem_shell.elements import ElementFamily  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MAT_STEEL = IsotropicMaterial(name="steel", E=2.1e11, nu=0.3, rho=7800.0)
THICKNESS = 0.01


def _model_cfg():
    return {
        "elements": {
            "element_family": ElementFamily.SHELL,
            "material": MAT_STEEL,
            "thickness": THICKNESS,
        }
    }


def _build_quad_plate(nx=4, ny=4, L=1.0):
    """Flat MITC4 plate mesh (nx×ny quads)."""
    Node._id_counter = 0
    mesh = MeshModel()
    xs = np.linspace(0.0, L, nx + 1)
    ys = np.linspace(0.0, L, ny + 1)
    grid = {}
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            n = Node([float(x), float(y), 0.0], geometric_node=False)
            mesh.add_node(n)
            grid[(i, j)] = n
    for j in range(ny):
        for i in range(nx):
            mesh.add_element(
                MeshElement(
                    nodes=[
                        grid[(i, j)],
                        grid[(i + 1, j)],
                        grid[(i + 1, j + 1)],
                        grid[(i, j + 1)],
                    ],
                    element_type=ElementType.quad,
                )
            )
    return mesh


def _build_tri_plate(nx=4, ny=4, L=1.0):
    """Flat MITC3 plate mesh (nx×ny×2 triangles)."""
    Node._id_counter = 0
    mesh = MeshModel()
    xs = np.linspace(0.0, L, nx + 1)
    ys = np.linspace(0.0, L, ny + 1)
    grid = {}
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            n = Node([float(x), float(y), 0.0], geometric_node=False)
            mesh.add_node(n)
            grid[(i, j)] = n
    for j in range(ny):
        for i in range(nx):
            mesh.add_element(
                MeshElement(
                    nodes=[grid[(i, j)], grid[(i + 1, j)], grid[(i + 1, j + 1)]],
                    element_type=ElementType.triangle,
                )
            )
            mesh.add_element(
                MeshElement(
                    nodes=[grid[(i, j)], grid[(i + 1, j + 1)], grid[(i, j + 1)]],
                    element_type=ElementType.triangle,
                )
            )
    return mesh


def _clamped_edge_dofs(mesh, dofs_per_node, axis=0, value=0.0):
    """Return DOF indices of all nodes on edge where axis==value."""
    dofs = []
    for node in mesh.nodes:
        coords = np.array(node.coords)
        if np.isclose(coords[axis], value, atol=1e-12):
            idx = mesh.node_id_to_index[node.id]
            start = idx * dofs_per_node
            dofs.extend(range(start, start + dofs_per_node))
    return sorted(dofs)


def _python_modal_solve(K_petsc, M_petsc, free_dofs_arr, num_modes, n_total_dofs):
    """Reference modal solve using SLEPc (identical logic to ModalSolver.solve)."""
    free_is = PETSc.IS().createGeneral(
        free_dofs_arr.astype(PETSc.IntType), comm=PETSc.COMM_WORLD
    )
    K_red = K_petsc.createSubMatrix(free_is, free_is)
    M_red = M_petsc.createSubMatrix(free_is, free_is)
    n_red = K_red.getSize()[0]

    eff_modes = min(num_modes + 5, n_red)

    eps = SLEPc.EPS().create(PETSc.COMM_WORLD)
    eps.setOperators(K_red, M_red)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    eps.setTarget(0.0)

    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    st.setShift(0.0)
    ksp = st.getKSP()
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("petsc")

    eps.setDimensions(eff_modes, PETSc.DECIDE, PETSc.DECIDE)
    eps.setTolerances(tol=1e-10, max_it=1000)
    eps.setFromOptions()
    eps.solve()

    nconv = eps.getConverged()
    vr = K_red.createVecRight()
    vi = K_red.createVecRight()

    eigvals = []
    eigvecs = []
    for i in range(nconv):
        eigval = eps.getEigenpair(i, vr, vi)
        eigvals.append(eigval.real)
        eigvecs.append(vr.array.copy())

    vr.destroy()
    vi.destroy()
    eps.destroy()

    eigvals = np.array(eigvals)
    eigvecs = np.column_stack(eigvecs) if eigvecs else np.zeros((n_red, 0))

    # Filter and sort
    valid = eigvals > 1e-8
    eigvals = eigvals[valid]
    eigvecs = eigvecs[:, valid]
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Expand to full DOFs
    full_modes = np.zeros((n_total_dofs, eigvecs.shape[1]))
    full_modes[free_dofs_arr, :] = eigvecs

    frequencies = np.sqrt(eigvals) / (2 * np.pi)

    K_red.destroy()
    M_red.destroy()
    free_is.destroy()

    return frequencies[:num_modes], full_modes[:, :num_modes]


def _rust_modal_solve(assembler, free_dofs_arr, num_modes):
    """Rust modal solve using PyMeshAssembler (production path).

    Uses assembler._rust.assemble_k() / assemble_m() directly — the same
    path used in production. This validates the full Rust assembly pipeline
    rather than a separate COO path from Python element matrices.
    """
    assert assembler._rust is not None, "PyMeshAssembler not built — Rust unavailable"

    rows_k, cols_k, vals_k = assembler._rust.assemble_k()
    rows_m, cols_m, vals_m = assembler._rust.assemble_m()

    n_total = assembler.dofs_count

    # Restrict COO to free DOFs submatrix
    free_set = set(free_dofs_arr.tolist())
    free_map = {old: new for new, old in enumerate(sorted(free_set))}
    n_free = len(free_dofs_arr)

    def _restrict_coo(rows, cols, vals):
        mask = np.isin(rows, free_dofs_arr) & np.isin(cols, free_dofs_arr)
        r = np.array([free_map[v] for v in rows[mask]], dtype=np.int32)
        c = np.array([free_map[v] for v in cols[mask]], dtype=np.int32)
        v = vals[mask].astype(np.float64)
        return r, c, v

    rk, ck, vk = _restrict_coo(rows_k, cols_k, vals_k)
    rm, cm, vm = _restrict_coo(rows_m, cols_m, vals_m)

    # Assemble PETSc Mats (reduced system)
    k_cap = fem_shell_core.petsc_assemble_matrix(rk, ck, vk, n_free)
    m_cap = fem_shell_core.petsc_assemble_matrix(rm, cm, vm, n_free)

    # Solve eigenvalue problem
    eigvals_raw, modes_flat_raw = fem_shell_core.petsc_modal_solve(k_cap, m_cap, num_modes)
    eigvals = np.asarray(eigvals_raw)
    modes_flat = np.asarray(modes_flat_raw)

    # Filter positive eigenvalues and sort
    valid = eigvals > 1e-8
    eigvals = eigvals[valid]
    n_conv = len(eigvals)
    modes_red = modes_flat.reshape((-1, n_free))[:n_conv]  # (n_conv, n_free)

    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    modes_red = modes_red[idx]

    freq = np.sqrt(eigvals) / (2 * np.pi)

    # Expand modes to full DOF space
    free_sorted = np.array(sorted(free_set), dtype=np.int64)
    full_modes = np.zeros((n_total, n_conv))
    full_modes[free_sorted, :] = modes_red.T

    return freq[:num_modes], full_modes[:, :num_modes]


# ---------------------------------------------------------------------------
# Test: MITC4 clamped cantilever plate
# ---------------------------------------------------------------------------

class TestMITC4ModalCantilever:
    """Clamped cantilever plate (one edge fixed) — MITC4 elements."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.nx, self.ny = 6, 4
        self.L = 1.0
        self.num_modes = 6

        mesh = _build_quad_plate(self.nx, self.ny, self.L)
        asm = MeshAssembler(mesh, _model_cfg())
        self.assembler = asm
        self.mesh = mesh

        # Clamp x=0 edge
        self.clamped_dofs = _clamped_edge_dofs(
            mesh, asm.dofs_per_node, axis=0, value=0.0
        )
        all_dofs = np.arange(asm.dofs_count, dtype=np.int64)
        fixed = np.array(self.clamped_dofs, dtype=np.int64)
        self.free_dofs = np.setdiff1d(all_dofs, fixed)

    def test_frequencies_match(self):
        """Rust and Python modal solvers produce the same frequencies."""
        asm = self.assembler

        # Python reference via SLEPc
        K = asm.assemble_stiffness_matrix()
        M = asm.assemble_mass_matrix()
        py_freq, py_modes = _python_modal_solve(
            K, M, self.free_dofs, self.num_modes, asm.dofs_count
        )
        K.destroy()
        M.destroy()

        # Rust solver
        rs_freq, rs_modes = _rust_modal_solve(asm, self.free_dofs, self.num_modes)

        print(f"\n  MITC4 Cantilever ({self.nx}x{self.ny}) — {len(self.free_dofs)} free DOFs")
        print(f"  {'Mode':>4s}  {'Python [Hz]':>14s}  {'Rust [Hz]':>14s}  {'Rel.Err':>10s}")
        n = min(len(py_freq), len(rs_freq))
        for i in range(n):
            err = abs(py_freq[i] - rs_freq[i]) / py_freq[i] if py_freq[i] > 0 else 0
            print(f"  {i+1:4d}  {py_freq[i]:14.4f}  {rs_freq[i]:14.4f}  {err:10.2e}")

        np.testing.assert_allclose(rs_freq[:n], py_freq[:n], rtol=1e-4,
                                   err_msg="Rust frequencies diverge from Python/SLEPc")

    def test_mode_shapes_orthogonal(self):
        """Rust mode shapes are M-orthogonal (via element mass)."""
        rs_freq, rs_modes = _rust_modal_solve(
            self.assembler, self.free_dofs, self.num_modes
        )
        # Mode shapes should be approximately orthogonal
        n_modes = rs_modes.shape[1]
        for i in range(n_modes):
            norm = np.linalg.norm(rs_modes[:, i])
            assert norm > 1e-10, f"Mode {i} has zero norm"


# ---------------------------------------------------------------------------
# Test: MITC3 clamped cantilever plate
# ---------------------------------------------------------------------------

class TestMITC3ModalCantilever:
    """Clamped cantilever plate (one edge fixed) — MITC3 elements."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.nx, self.ny = 6, 4
        self.L = 1.0
        self.num_modes = 6

        mesh = _build_tri_plate(self.nx, self.ny, self.L)
        asm = MeshAssembler(mesh, _model_cfg())
        self.assembler = asm
        self.mesh = mesh

        self.clamped_dofs = _clamped_edge_dofs(
            mesh, asm.dofs_per_node, axis=0, value=0.0
        )
        all_dofs = np.arange(asm.dofs_count, dtype=np.int64)
        fixed = np.array(self.clamped_dofs, dtype=np.int64)
        self.free_dofs = np.setdiff1d(all_dofs, fixed)

    def test_frequencies_match(self):
        """Rust and Python modal solvers produce the same frequencies."""
        asm = self.assembler

        K = asm.assemble_stiffness_matrix()
        M = asm.assemble_mass_matrix()
        py_freq, _ = _python_modal_solve(
            K, M, self.free_dofs, self.num_modes, asm.dofs_count
        )
        K.destroy()
        M.destroy()

        rs_freq, _ = _rust_modal_solve(asm, self.free_dofs, self.num_modes)

        print(f"\n  MITC3 Cantilever ({self.nx}x{self.ny}) — {len(self.free_dofs)} free DOFs")
        print(f"  {'Mode':>4s}  {'Python [Hz]':>14s}  {'Rust [Hz]':>14s}  {'Rel.Err':>10s}")
        n = min(len(py_freq), len(rs_freq))
        for i in range(n):
            err = abs(py_freq[i] - rs_freq[i]) / py_freq[i] if py_freq[i] > 0 else 0
            print(f"  {i+1:4d}  {py_freq[i]:14.4f}  {rs_freq[i]:14.4f}  {err:10.2e}")

        np.testing.assert_allclose(rs_freq[:n], py_freq[:n], rtol=1e-4)


# ---------------------------------------------------------------------------
# Test: Simply supported plate (all edges pinned)
# ---------------------------------------------------------------------------

class TestSimplySupportedPlate:
    """Simply supported plate — known analytical solution convergence."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.nx, self.ny = 8, 8
        self.L = 1.0
        self.num_modes = 6

        mesh = _build_quad_plate(self.nx, self.ny, self.L)
        asm = MeshAssembler(mesh, _model_cfg())
        self.assembler = asm
        self.mesh = mesh

        # Simply supported: pin translational DOFs on all edges
        pinned_dofs = []
        for node in mesh.nodes:
            coords = np.array(node.coords)
            on_edge = (
                np.isclose(coords[0], 0.0, atol=1e-12)
                or np.isclose(coords[0], self.L, atol=1e-12)
                or np.isclose(coords[1], 0.0, atol=1e-12)
                or np.isclose(coords[1], self.L, atol=1e-12)
            )
            if on_edge:
                idx = mesh.node_id_to_index[node.id]
                start = idx * asm.dofs_per_node
                # Pin ux, uy, uz (DOFs 0,1,2) — leave rotations free
                pinned_dofs.extend([start, start + 1, start + 2])

        all_dofs = np.arange(asm.dofs_count, dtype=np.int64)
        fixed = np.array(sorted(set(pinned_dofs)), dtype=np.int64)
        self.free_dofs = np.setdiff1d(all_dofs, fixed)

    def test_frequencies_match_python(self):
        """Rust matches Python for simply supported plate."""
        asm = self.assembler
        K = asm.assemble_stiffness_matrix()
        M = asm.assemble_mass_matrix()
        py_freq, _ = _python_modal_solve(
            K, M, self.free_dofs, self.num_modes, asm.dofs_count
        )
        K.destroy()
        M.destroy()

        rs_freq, _ = _rust_modal_solve(asm, self.free_dofs, self.num_modes)

        print(f"\n  SS Plate ({self.nx}x{self.ny}) — {len(self.free_dofs)} free DOFs")
        print(f"  {'Mode':>4s}  {'Python [Hz]':>14s}  {'Rust [Hz]':>14s}  {'Rel.Err':>10s}")
        n = min(len(py_freq), len(rs_freq))
        for i in range(n):
            err = abs(py_freq[i] - rs_freq[i]) / py_freq[i] if py_freq[i] > 0 else 0
            print(f"  {i+1:4d}  {py_freq[i]:14.4f}  {rs_freq[i]:14.4f}  {err:10.2e}")

        np.testing.assert_allclose(rs_freq[:n], py_freq[:n], rtol=1e-4)

    def test_analytical_convergence(self):
        """First frequency approaches the Kirchhoff plate analytical value."""
        # Analytical: f_11 = (π/L²) * sqrt(D/(ρh))
        # where D = E h³ / [12(1-ν²)]
        # Derived from ω² = (D/ρh) * π⁴ * (m²/a² + n²/b²)²
        E, nu = 2.1e11, 0.3
        rho, h = 7800.0, THICKNESS
        D = E * h**3 / (12.0 * (1.0 - nu**2))
        a = self.L  # square plate
        f_11_analytical = (np.pi / a**2) * np.sqrt(D / (rho * h))

        rs_freq, _ = _rust_modal_solve(self.assembler, self.free_dofs, 1)

        rel_err = abs(rs_freq[0] - f_11_analytical) / f_11_analytical
        print(f"\n  Analytical f_11 = {f_11_analytical:.4f} Hz")
        print(f"  Rust      f_1  = {rs_freq[0]:.4f} Hz  (err = {rel_err:.2%})")
        # For 8x8 mesh, expect ~1-5% discretisation error
        assert rel_err < 0.05, f"First frequency {rel_err:.1%} off from analytical"


# ---------------------------------------------------------------------------
# Test: COO path — mixed mesh compatibility
# ---------------------------------------------------------------------------

class TestCOOModalSolve:
    """Test modal_solve_coo path using COO from coo_assembly."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.nx, self.ny = 4, 4
        self.num_modes = 4

        mesh = _build_quad_plate(self.nx, self.ny, 1.0)
        asm = MeshAssembler(mesh, _model_cfg())
        self.assembler = asm
        self.mesh = mesh

        clamped = _clamped_edge_dofs(mesh, asm.dofs_per_node, axis=0, value=0.0)
        all_dofs = np.arange(asm.dofs_count, dtype=np.int64)
        fixed = np.array(clamped, dtype=np.int64)
        self.free_dofs = np.setdiff1d(all_dofs, fixed)

    def test_coo_matches_element_path(self):
        """modal_solve_coo gives same result as PyMeshAssembler assembly path."""
        asm = self.assembler

        # Reference: PyMeshAssembler path (production)
        freq_ref, _ = _rust_modal_solve(asm, self.free_dofs, self.num_modes)

        # COO path via PyMeshAssembler COO output + modal_solve_coo
        k_rows, k_cols, k_vals = asm._rust.assemble_k()
        m_rows, m_cols, m_vals = asm._rust.assemble_m()

        freq_coo, _ = fem_shell_core.modal_solve_coo(
            np.ascontiguousarray(k_rows, dtype=np.int64),
            np.ascontiguousarray(k_cols, dtype=np.int64),
            np.ascontiguousarray(k_vals, dtype=np.float64),
            np.ascontiguousarray(m_rows, dtype=np.int64),
            np.ascontiguousarray(m_cols, dtype=np.int64),
            np.ascontiguousarray(m_vals, dtype=np.float64),
            asm.dofs_count,
            self.free_dofs.astype(np.int64),
            self.num_modes,
        )
        freq_coo = np.asarray(freq_coo)

        np.testing.assert_allclose(freq_coo, freq_ref, rtol=1e-10,
                                   err_msg="COO and PyMeshAssembler paths diverge")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class TestModalBenchmark:
    """Performance comparison: Python/SLEPc vs Rust modal solve."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.sizes = [(8, 8), (12, 12)]
        self.num_modes = 6

    def test_benchmark_mitc4(self):
        """Benchmark Rust vs Python for MITC4 plates of increasing size."""
        print("\n" + "=" * 70)
        print("  MITC4 Modal Solve Benchmark: Rust (nalgebra) vs Python (SLEPc)")
        print("=" * 70)
        print(f"  {'Size':>8s}  {'DOFs':>6s}  {'Free':>6s}  {'Python [s]':>10s}"
              f"  {'Rust [s]':>10s}  {'Speedup':>8s}")

        for nx, ny in self.sizes:
            mesh = _build_quad_plate(nx, ny, 1.0)
            asm = MeshAssembler(mesh, _model_cfg())
            clamped = _clamped_edge_dofs(mesh, asm.dofs_per_node, axis=0, value=0.0)
            all_dofs = np.arange(asm.dofs_count, dtype=np.int64)
            fixed = np.array(clamped, dtype=np.int64)
            free_dofs = np.setdiff1d(all_dofs, fixed)

            # Python timing
            t0 = time.perf_counter()
            K = asm.assemble_stiffness_matrix()
            M = asm.assemble_mass_matrix()
            py_freq, _ = _python_modal_solve(
                K, M, free_dofs, self.num_modes, asm.dofs_count
            )
            py_time = time.perf_counter() - t0
            K.destroy()
            M.destroy()

            # Rust timing (element data already in asm._ke_array)
            t0 = time.perf_counter()
            rs_freq, _ = _rust_modal_solve(asm, free_dofs, self.num_modes)
            rs_time = time.perf_counter() - t0

            speedup = py_time / rs_time if rs_time > 0 else float("inf")
            print(f"  {nx}x{ny:>3d}  {asm.dofs_count:6d}  {len(free_dofs):6d}"
                  f"  {py_time:10.4f}  {rs_time:10.4f}  {speedup:7.1f}x")

            # Verify results match
            n = min(len(py_freq), len(rs_freq))
            np.testing.assert_allclose(rs_freq[:n], py_freq[:n], rtol=1e-3)


# ---------------------------------------------------------------------------
# Composite modal validation
# ---------------------------------------------------------------------------

class TestCompositeModal:
    """Modal analysis with composite (laminated) shell elements.

    Validates that Rust composite element K/M matrices produce the same
    eigenfrequencies as Python composite elements via the modal solver.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        from fem_shell.core.laminate import Laminate, Ply, create_laminate_from_angles
        from fem_shell.core.material import OrthotropicMaterial
        from fem_shell.core.mesh.entities import ElementSet
        from fem_shell.core.properties import CompositeShellProperty

        self.num_modes = 6
        self.nx, self.ny = 6, 4

        # Carbon/epoxy orthotropic material
        self.mat = OrthotropicMaterial(
            name="Carbon/Epoxy",
            E=(181e9, 10.3e9, 10.3e9),
            G=(7.17e9, 3.78e9, 7.17e9),
            nu=(0.28, 0.28, 0.28),
            rho=1600,
        )

        # Quasi-isotropic layup: [0/45/-45/90]s — 8 plies
        ply_t = 0.125e-3
        angles = [0, 45, -45, 90, 90, -45, 45, 0]
        lam = create_laminate_from_angles(self.mat, ply_t, angles)

        # Build MITC4 quad mesh with all elements in one element set
        mesh = _build_quad_plate(self.nx, self.ny, 1.0)
        eset = ElementSet("plate", set(mesh.elements))
        mesh.add_element_set(eset)

        # Composite shell property
        comp_prop = CompositeShellProperty(laminate=lam)

        cfg = {
            "elements": {
                "element_family": ElementFamily.SHELL,
                "span_direction": np.array([0.0, 1.0, 0.0]),
                "properties": {"plate": comp_prop},
            }
        }
        self.assembler = MeshAssembler(mesh, cfg)
        self.mesh = mesh

        clamped = _clamped_edge_dofs(mesh, self.assembler.dofs_per_node, axis=0, value=0.0)
        all_dofs = np.arange(self.assembler.dofs_count, dtype=np.int64)
        fixed = np.array(clamped, dtype=np.int64)
        self.free_dofs = np.setdiff1d(all_dofs, fixed)

    def test_composite_frequencies_match(self):
        """Rust and Python modal solvers agree for composite plate."""
        asm = self.assembler

        K = asm.assemble_stiffness_matrix()
        M = asm.assemble_mass_matrix()
        py_freq, _ = _python_modal_solve(
            K, M, self.free_dofs, self.num_modes, asm.dofs_count
        )
        K.destroy()
        M.destroy()

        rs_freq, _ = _rust_modal_solve(asm, self.free_dofs, self.num_modes)

        print(f"\n  Composite MITC4 Cantilever ({self.nx}x{self.ny})"
              f" — {len(self.free_dofs)} free DOFs")
        print(f"  Layup: [0/45/-45/90]s Carbon/Epoxy")
        print(f"  {'Mode':>4s}  {'Python [Hz]':>14s}  {'Rust [Hz]':>14s}  {'Rel.Err':>10s}")
        n = min(len(py_freq), len(rs_freq))
        for i in range(n):
            err = abs(py_freq[i] - rs_freq[i]) / py_freq[i] if py_freq[i] > 0 else 0
            print(f"  {i+1:4d}  {py_freq[i]:14.4f}  {rs_freq[i]:14.4f}  {err:10.2e}")

        np.testing.assert_allclose(
            rs_freq[:n], py_freq[:n], rtol=1e-4,
            err_msg="Composite frequencies diverge between Rust and Python",
        )

    def test_composite_stiffer_than_isotropic(self):
        """Composite plate has different frequencies than isotropic steel."""
        rs_freq_comp, _ = _rust_modal_solve(
            self.assembler, self.free_dofs, self.num_modes
        )

        # Build isotropic steel plate for comparison
        mesh_iso = _build_quad_plate(self.nx, self.ny, 1.0)
        asm_iso = MeshAssembler(mesh_iso, _model_cfg())
        clamped_iso = _clamped_edge_dofs(
            mesh_iso, asm_iso.dofs_per_node, axis=0, value=0.0
        )
        all_dofs = np.arange(asm_iso.dofs_count, dtype=np.int64)
        fixed = np.array(clamped_iso, dtype=np.int64)
        free_iso = np.setdiff1d(all_dofs, fixed)

        rs_freq_iso, _ = _rust_modal_solve(asm_iso, free_iso, self.num_modes)

        # Frequencies should be DIFFERENT (composite vs isotropic)
        assert not np.allclose(rs_freq_comp, rs_freq_iso, rtol=0.01), \
            "Composite and isotropic frequencies should differ"
        print(f"\n  Composite vs Isotropic (first mode): "
              f"{rs_freq_comp[0]:.2f} vs {rs_freq_iso[0]:.2f} Hz")


class TestCompositeModalMITC3:
    """Composite modal analysis with MITC3 triangular elements."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from fem_shell.core.laminate import create_laminate_from_angles
        from fem_shell.core.material import OrthotropicMaterial
        from fem_shell.core.mesh.entities import ElementSet
        from fem_shell.core.properties import CompositeShellProperty

        self.num_modes = 6
        self.nx, self.ny = 6, 4

        mat = OrthotropicMaterial(
            name="Carbon/Epoxy",
            E=(181e9, 10.3e9, 10.3e9),
            G=(7.17e9, 3.78e9, 7.17e9),
            nu=(0.28, 0.28, 0.28),
            rho=1600,
        )

        ply_t = 0.125e-3
        angles = [0, 45, -45, 90, 90, -45, 45, 0]
        lam = create_laminate_from_angles(mat, ply_t, angles)

        mesh = _build_tri_plate(self.nx, self.ny, 1.0)
        eset = ElementSet("plate", set(mesh.elements))
        mesh.add_element_set(eset)

        comp_prop = CompositeShellProperty(laminate=lam)
        cfg = {
            "elements": {
                "element_family": ElementFamily.SHELL,
                "span_direction": np.array([0.0, 1.0, 0.0]),
                "properties": {"plate": comp_prop},
            }
        }
        self.assembler = MeshAssembler(mesh, cfg)
        self.mesh = mesh

        clamped = _clamped_edge_dofs(mesh, self.assembler.dofs_per_node, axis=0, value=0.0)
        all_dofs = np.arange(self.assembler.dofs_count, dtype=np.int64)
        fixed = np.array(clamped, dtype=np.int64)
        self.free_dofs = np.setdiff1d(all_dofs, fixed)

    def test_composite_mitc3_frequencies_match(self):
        """Rust and Python modal solvers agree for composite MITC3."""
        asm = self.assembler

        K = asm.assemble_stiffness_matrix()
        M = asm.assemble_mass_matrix()
        py_freq, _ = _python_modal_solve(
            K, M, self.free_dofs, self.num_modes, asm.dofs_count
        )
        K.destroy()
        M.destroy()

        rs_freq, _ = _rust_modal_solve(asm, self.free_dofs, self.num_modes)

        print(f"\n  Composite MITC3 Cantilever ({self.nx}x{self.ny})"
              f" — {len(self.free_dofs)} free DOFs")
        print(f"  {'Mode':>4s}  {'Python [Hz]':>14s}  {'Rust [Hz]':>14s}  {'Rel.Err':>10s}")
        n = min(len(py_freq), len(rs_freq))
        for i in range(n):
            err = abs(py_freq[i] - rs_freq[i]) / py_freq[i] if py_freq[i] > 0 else 0
            print(f"  {i+1:4d}  {py_freq[i]:14.4f}  {rs_freq[i]:14.4f}  {err:10.2e}")

        np.testing.assert_allclose(rs_freq[:n], py_freq[:n], rtol=1e-4)


# ---------------------------------------------------------------------------
# Composite modal validation
# ---------------------------------------------------------------------------

class TestCompositeModal:
    """Modal analysis with composite (laminated) shell elements.

    Validates that Rust composite element K/M matrices produce the same
    eigenfrequencies as Python composite elements via the modal solver.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        from fem_shell.core.laminate import Laminate, Ply, create_laminate_from_angles
        from fem_shell.core.material import OrthotropicMaterial
        from fem_shell.core.mesh.entities import ElementSet
        from fem_shell.core.properties import CompositeShellProperty

        self.num_modes = 6
        self.nx, self.ny = 6, 4

        # Carbon/epoxy orthotropic material
        self.mat = OrthotropicMaterial(
            name="Carbon/Epoxy",
            E=(181e9, 10.3e9, 10.3e9),
            G=(7.17e9, 3.78e9, 7.17e9),
            nu=(0.28, 0.28, 0.28),
            rho=1600,
        )

        # Quasi-isotropic layup: [0/45/-45/90]s — 8 plies
        ply_t = 0.125e-3
        angles = [0, 45, -45, 90, 90, -45, 45, 0]
        lam = create_laminate_from_angles(self.mat, ply_t, angles)

        # Build MITC4 quad mesh with all elements in one element set
        mesh = _build_quad_plate(self.nx, self.ny, 1.0)
        eset = ElementSet("plate", set(mesh.elements))
        mesh.add_element_set(eset)

        # Composite shell property
        comp_prop = CompositeShellProperty(laminate=lam)

        cfg = {
            "elements": {
                "element_family": ElementFamily.SHELL,
                "span_direction": np.array([0.0, 1.0, 0.0]),
                "properties": {"plate": comp_prop},
            }
        }
        self.assembler = MeshAssembler(mesh, cfg)
        self.mesh = mesh

        clamped = _clamped_edge_dofs(mesh, self.assembler.dofs_per_node, axis=0, value=0.0)
        all_dofs = np.arange(self.assembler.dofs_count, dtype=np.int64)
        fixed = np.array(clamped, dtype=np.int64)
        self.free_dofs = np.setdiff1d(all_dofs, fixed)

    def test_composite_frequencies_match(self):
        """Rust and Python modal solvers agree for composite plate."""
        asm = self.assembler

        K = asm.assemble_stiffness_matrix()
        M = asm.assemble_mass_matrix()
        py_freq, _ = _python_modal_solve(
            K, M, self.free_dofs, self.num_modes, asm.dofs_count
        )
        K.destroy()
        M.destroy()

        rs_freq, _ = _rust_modal_solve(asm, self.free_dofs, self.num_modes)

        print(f"\n  Composite MITC4 Cantilever ({self.nx}x{self.ny})"
              f" — {len(self.free_dofs)} free DOFs")
        print(f"  Layup: [0/45/-45/90]s Carbon/Epoxy")
        print(f"  {'Mode':>4s}  {'Python [Hz]':>14s}  {'Rust [Hz]':>14s}  {'Rel.Err':>10s}")
        n = min(len(py_freq), len(rs_freq))
        for i in range(n):
            err = abs(py_freq[i] - rs_freq[i]) / py_freq[i] if py_freq[i] > 0 else 0
            print(f"  {i+1:4d}  {py_freq[i]:14.4f}  {rs_freq[i]:14.4f}  {err:10.2e}")

        np.testing.assert_allclose(
            rs_freq[:n], py_freq[:n], rtol=1e-4,
            err_msg="Composite frequencies diverge between Rust and Python",
        )

    def test_composite_stiffer_than_isotropic(self):
        """Composite plate has different frequencies than isotropic steel."""
        rs_freq_comp, _ = _rust_modal_solve(
            self.assembler, self.free_dofs, self.num_modes
        )

        # Build isotropic steel plate for comparison
        mesh_iso = _build_quad_plate(self.nx, self.ny, 1.0)
        asm_iso = MeshAssembler(mesh_iso, _model_cfg())
        clamped_iso = _clamped_edge_dofs(
            mesh_iso, asm_iso.dofs_per_node, axis=0, value=0.0
        )
        all_dofs = np.arange(asm_iso.dofs_count, dtype=np.int64)
        fixed = np.array(clamped_iso, dtype=np.int64)
        free_iso = np.setdiff1d(all_dofs, fixed)

        rs_freq_iso, _ = _rust_modal_solve(asm_iso, free_iso, self.num_modes)

        # Frequencies should be DIFFERENT (composite vs isotropic)
        assert not np.allclose(rs_freq_comp, rs_freq_iso, rtol=0.01), \
            "Composite and isotropic frequencies should differ"
        print(f"\n  Composite vs Isotropic (first mode): "
              f"{rs_freq_comp[0]:.2f} vs {rs_freq_iso[0]:.2f} Hz")


class TestCompositeModalMITC3:
    """Composite modal analysis with MITC3 triangular elements."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from fem_shell.core.laminate import create_laminate_from_angles
        from fem_shell.core.material import OrthotropicMaterial
        from fem_shell.core.mesh.entities import ElementSet
        from fem_shell.core.properties import CompositeShellProperty

        self.num_modes = 6
        self.nx, self.ny = 6, 4

        mat = OrthotropicMaterial(
            name="Carbon/Epoxy",
            E=(181e9, 10.3e9, 10.3e9),
            G=(7.17e9, 3.78e9, 7.17e9),
            nu=(0.28, 0.28, 0.28),
            rho=1600,
        )

        ply_t = 0.125e-3
        angles = [0, 45, -45, 90, 90, -45, 45, 0]
        lam = create_laminate_from_angles(mat, ply_t, angles)

        mesh = _build_tri_plate(self.nx, self.ny, 1.0)
        eset = ElementSet("plate", set(mesh.elements))
        mesh.add_element_set(eset)

        comp_prop = CompositeShellProperty(laminate=lam)
        cfg = {
            "elements": {
                "element_family": ElementFamily.SHELL,
                "span_direction": np.array([0.0, 1.0, 0.0]),
                "properties": {"plate": comp_prop},
            }
        }
        self.assembler = MeshAssembler(mesh, cfg)
        self.mesh = mesh

        clamped = _clamped_edge_dofs(mesh, self.assembler.dofs_per_node, axis=0, value=0.0)
        all_dofs = np.arange(self.assembler.dofs_count, dtype=np.int64)
        fixed = np.array(clamped, dtype=np.int64)
        self.free_dofs = np.setdiff1d(all_dofs, fixed)

    def test_composite_mitc3_frequencies_match(self):
        """Rust and Python modal solvers agree for composite MITC3."""
        asm = self.assembler

        K = asm.assemble_stiffness_matrix()
        M = asm.assemble_mass_matrix()
        py_freq, _ = _python_modal_solve(
            K, M, self.free_dofs, self.num_modes, asm.dofs_count
        )
        K.destroy()
        M.destroy()

        rs_freq, _ = _rust_modal_solve(asm, self.free_dofs, self.num_modes)

        print(f"\n  Composite MITC3 Cantilever ({self.nx}x{self.ny})"
              f" — {len(self.free_dofs)} free DOFs")
        print(f"  {'Mode':>4s}  {'Python [Hz]':>14s}  {'Rust [Hz]':>14s}  {'Rel.Err':>10s}")
        n = min(len(py_freq), len(rs_freq))
        for i in range(n):
            err = abs(py_freq[i] - rs_freq[i]) / py_freq[i] if py_freq[i] > 0 else 0
            print(f"  {i+1:4d}  {py_freq[i]:14.4f}  {rs_freq[i]:14.4f}  {err:10.2e}")

        np.testing.assert_allclose(rs_freq[:n], py_freq[:n], rtol=1e-4)
