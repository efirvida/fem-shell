from pathlib import Path
from typing import Dict, Optional, Tuple
from xml.etree.ElementTree import Element, SubElement, ElementTree

import meshio
import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc

from fem_shell.core.bc import BoundaryConditionManager
from fem_shell.core.mesh import MeshModel
from fem_shell.solvers.solver import Solver


class ModalSolver(Solver):
    """Eigenvalue solver for natural frequencies and mode shapes.

    Uses SLEPc shift-and-invert spectral transformation to solve the
    generalized eigenvalue problem  K φ = ω² M φ  for the lowest modes.
    """

    def __init__(self, mesh: MeshModel, fem_model_properties: dict):
        self.num_modes = fem_model_properties["solver"].get("num_modes", 2)
        super().__init__(mesh, fem_model_properties)
        self.comm = PETSc.COMM_WORLD
        self.participation_factors: Optional[np.ndarray] = None

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        K = self.domain.assemble_stiffness_matrix()
        M = self.domain.assemble_mass_matrix()

        F = K.createVecRight()
        F.set(0.0)

        self.bc_applier = BoundaryConditionManager(
            K, F, M, self.domain.dofs_per_node
        )
        self.bc_applier.apply_dirichlet(self.dirichlet_conditions)
        K_red, F_red, M_red = self.bc_applier.reduced_system

        n_red = K_red.getSize()[0]
        eff_num_modes = min(self.num_modes + 5, n_red)

        # Configure SLEPc eigensolver
        eps = SLEPc.EPS().create(self.comm)
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

        eps.setDimensions(eff_num_modes, PETSc.DECIDE, PETSc.DECIDE)
        eps.setTolerances(tol=1e-10, max_it=1000)
        eps.setFromOptions()

        try:
            eps.solve()
        except Exception as exc:
            self._cleanup(eps, K_red, M_red, F_red, F)
            raise RuntimeError(
                f"SLEPc eigenvalue solve failed: {exc}"
            ) from exc

        nconv = eps.getConverged()
        if nconv < self.num_modes:
            self._cleanup(eps, K_red, M_red, F_red, F)
            raise RuntimeError(
                f"Convergence insufficient: {nconv}/{self.num_modes} modes"
            )

        eigvals, eigvecs = self._extract_eigenpairs(eps, nconv, K_red)

        # Compute participation factors BEFORE cleanup destroys M_red
        self.participation_factors = self._compute_participation_factors(
            eigvals, eigvecs, M_red
        )

        self._cleanup(eps, K_red, M_red, F_red, F)
        return self._postprocess_results(eigvals, eigvecs)

    def _extract_eigenpairs(
        self, eps: SLEPc.EPS, nconv: int, K_red: PETSc.Mat
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_red = K_red.getSize()[0]
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

        return np.array(eigvals), np.column_stack(eigvecs)

    def _postprocess_results(
        self, eigvals: np.ndarray, eigvecs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        valid = eigvals > 1e-8
        eigvals = eigvals[valid]
        eigvecs = eigvecs[:, valid]

        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        full_modes = np.zeros((self.domain.dofs_count, eigvecs.shape[1]))
        full_modes[self.bc_applier.free_dofs, :] = eigvecs

        frequencies = np.sqrt(eigvals) / (2 * np.pi)

        return frequencies[: self.num_modes], full_modes[:, : self.num_modes]

    @staticmethod
    def _cleanup(eps, *mats_and_vecs):
        eps.destroy()
        for obj in mats_and_vecs:
            if obj is not None:
                obj.destroy()

    def _compute_participation_factors(
        self,
        eigvals: np.ndarray,
        eigvecs: np.ndarray,
        M_red: PETSc.Mat,
    ) -> np.ndarray:
        """Compute effective mass participation factors in reduced space.

        For each mode *i* and rigid-body direction *d*:

            Γ_i,d  = φᵢᵀ M rₐ
            m_eff  = Γ² / (φᵢᵀ M φᵢ)
            α_i,d  = m_eff / m_total

        For shell elements (6 DOFs/node), computes factors for all 6
        rigid-body modes: 3 translational (Ux, Uy, Uz) and 3 rotational
        (Rx, Ry, Rz).  The rotational rigid-body vectors include the
        translational coupling terms (e.g. rotation about Z:
        ux = −y, uy = x, θz = 1).

        Returns array of shape ``(n_modes, n_rb_dirs)`` with
        percentage values (0–100).
        """
        valid = eigvals > 1e-8
        vecs = eigvecs[:, valid]
        idx = np.argsort(eigvals[valid])
        vecs = vecs[:, idx]

        dpn = self.domain.dofs_per_node
        free_dofs = np.array(sorted(self.bc_applier.free_dofs))

        # Translational rigid-body vectors (unit in each direction)
        n_trans = min(dpn, 3)
        r_vectors = []
        for d in range(n_trans):
            r_full = np.zeros(self.domain.dofs_count)
            r_full[d::dpn] = 1.0
            r_vectors.append(r_full[free_dofs])

        # Rotational rigid-body vectors (rigid rotation about each axis)
        if dpn >= 6:
            coords = self.mesh_obj.coords_array
            for d in range(3):
                r_full = np.zeros(self.domain.dofs_count)
                if d == 0:  # Rotation about X: uy=z, uz=-y, θx=1
                    r_full[1::dpn] = coords[:, 2]
                    r_full[2::dpn] = -coords[:, 1]
                    r_full[3::dpn] = 1.0
                elif d == 1:  # Rotation about Y: ux=-z, uz=x, θy=1
                    r_full[0::dpn] = -coords[:, 2]
                    r_full[2::dpn] = coords[:, 0]
                    r_full[4::dpn] = 1.0
                else:  # Rotation about Z: ux=-y, uy=x, θz=1
                    r_full[0::dpn] = -coords[:, 1]
                    r_full[1::dpn] = coords[:, 0]
                    r_full[5::dpn] = 1.0
                r_vectors.append(r_full[free_dofs])

        n_rb = len(r_vectors)

        # Total mass/inertia per rigid-body direction via rᵀ M r
        m_total = np.zeros(n_rb)
        r_petsc = M_red.createVecRight()
        Mr_petsc = M_red.createVecRight()
        for d in range(n_rb):
            r_petsc.array[:] = r_vectors[d]
            M_red.mult(r_petsc, Mr_petsc)
            m_total[d] = r_petsc.dot(Mr_petsc)

        n_modes = min(vecs.shape[1], self.num_modes)
        factors = np.zeros((n_modes, n_rb))

        phi_petsc = M_red.createVecRight()
        Mphi_petsc = M_red.createVecRight()

        for i in range(n_modes):
            phi = vecs[:, i]
            phi_petsc.array[:] = phi
            M_red.mult(phi_petsc, Mphi_petsc)
            gen_mass = phi_petsc.dot(Mphi_petsc)

            if gen_mass < 1e-30:
                continue

            for d in range(n_rb):
                # Γ_i,d = φᵢᵀ M r_d  =  (M φᵢ)ᵀ r_d
                gamma = np.dot(Mphi_petsc.array, r_vectors[d])
                m_eff = gamma**2 / gen_mass
                if m_total[d] > 1e-30:
                    factors[i, d] = m_eff / m_total[d] * 100.0

        r_petsc.destroy()
        Mr_petsc.destroy()
        phi_petsc.destroy()
        Mphi_petsc.destroy()

        return factors

    def write_modal_results(
        self,
        output_dir: str,
        frequencies: np.ndarray,
        mode_shapes: np.ndarray,
    ) -> None:
        """Write mode shapes as VTU files with a PVD collection.

        Creates one VTU per mode with displacement fields and a PVD
        index file for sequential browsing in ParaView.

        Parameters
        ----------
        output_dir : str
            Directory for output files.  Created if it does not exist.
        frequencies : np.ndarray
            Natural frequencies in Hz, shape ``(n_modes,)``.
        mode_shapes : np.ndarray
            Mode shape vectors, shape ``(n_dofs, n_modes)``.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        vector_form = self.vector_form
        vector_components = [c for v in vector_form.values() for c in v]
        n_components = len(vector_components)
        dpn = self.domain.dofs_per_node

        points = self.mesh_obj.coords_array

        # Build meshio cells list
        cells = []
        for element in self.mesh_obj.element_map.values():
            if element:
                cells.append((element.element_type.name, [element.node_ids]))

        vtu_names = []
        for mode_idx in range(len(frequencies)):
            U = mode_shapes[:, mode_idx].reshape(-1, dpn)

            # Normalize so max translational displacement = 1.0
            trans = U[:, :min(dpn, 3)]
            mag = np.linalg.norm(trans, axis=1)
            max_mag = mag.max()
            if max_mag > 1e-30:
                U = U / max_mag

            point_data: Dict[str, np.ndarray] = {}

            for i, comp in enumerate(vector_components):
                if i < dpn:
                    point_data[comp] = U[:, i]

            for vec_name, components in vector_form.items():
                arr = np.column_stack(
                    [point_data[c] for c in components if c in point_data]
                )
                if arr.shape[1] == 2:
                    arr = np.column_stack([arr, np.zeros(arr.shape[0])])
                point_data[vec_name] = arr

            # Displacement magnitude scalar
            trans_norm = U[:, :min(dpn, 3)]
            point_data["UMAG"] = np.linalg.norm(trans_norm, axis=1)

            fname = f"mode_{mode_idx + 1:02d}.vtu"
            vtu_names.append(fname)
            mesh_obj = meshio.Mesh(points, cells=cells, point_data=point_data)
            mesh_obj.write(str(out / fname))

        # Write PVD collection file
        self._write_pvd(out / "modal_results.pvd", vtu_names, frequencies)

    @staticmethod
    def _write_pvd(
        pvd_path: Path,
        vtu_names: list,
        frequencies: np.ndarray,
    ) -> None:
        """Write a PVD (ParaView Data) collection file."""
        root = Element("VTKFile")
        root.set("type", "Collection")
        root.set("version", "0.1")
        collection = SubElement(root, "Collection")
        for i, (name, freq) in enumerate(zip(vtu_names, frequencies)):
            ds = SubElement(collection, "DataSet")
            ds.set("timestep", f"{freq:.6f}")
            ds.set("group", "")
            ds.set("part", "0")
            ds.set("file", name)

        tree = ElementTree(root)
        with open(pvd_path, "wb") as fh:
            tree.write(fh, xml_declaration=True)
