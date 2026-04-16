"""Standalone BEM solver — computes and projects aerodynamic loads.

Reads a blade definition (WindIO YAML or NuMAD Excel), generates the
shell mesh, runs a single BEM evaluation at given wind conditions, and
projects the resulting forces onto the mesh nodes.  Results are exported
as CSV (sectional + global) and VTU (nodal forces on the shell mesh).
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from fem_shell.core.config import BEMConfig
from fem_shell.core.mesh.model import MeshModel
from fem_shell.models.blade.aerodynamics import BladeAero, load_blade_aero
from fem_shell.solvers.bem.engine import BEMResult, BEMSolver
from fem_shell.solvers.bem.force_projection import ForceProjector


class BEMStandaloneSolver:
    """One-shot BEM solver with force projection onto a shell mesh.

    Intended usage::

        solver = BEMStandaloneSolver(mesh, model_config)
        solver.solve()

    Parameters
    ----------
    mesh : MeshModel
        Already-generated blade shell mesh.
    model_config : dict
        Configuration dictionary built by :class:`FSIRunner`.
    """

    def __init__(self, mesh: MeshModel, model_config: Dict[str, Any]):
        self.mesh = mesh
        self._cfg: Dict[str, Any] = model_config
        self._bem_cfg: Dict[str, Any] = model_config.get("bem", {})
        self._blade_file: Optional[str] = model_config.get("blade_file")

        # Results populated by solve()
        self.blade_aero: Optional[BladeAero] = None
        self.bem_result: Optional[BEMResult] = None
        self.nodal_forces: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def solve(self) -> Dict[str, Any]:
        """Run the full BEM → projection pipeline.

        Returns
        -------
        dict
            Summary with keys ``"thrust"``, ``"torque"``, ``"power"``,
            ``"force_error"``.
        """
        # 1. Load aero data
        self.blade_aero = self._load_aero()

        # 2. BEM solve
        bem_solver = BEMSolver(
            self.blade_aero,
            rho=self._bem_cfg.get("air_density", 1.225),
            mu=self._bem_cfg.get("dynamic_viscosity", 1.81206e-5),
            precone=self._bem_cfg.get("precone", 0.0),
            tilt=self._bem_cfg.get("tilt", 0.0),
            hub_height=self._bem_cfg.get("hub_height", 150.0),
            shear_exp=self._bem_cfg.get("shear_exp", 0.2),
        )

        v_inf = self._bem_cfg.get("wind_speed", 45.0)
        omega = self._bem_cfg.get("omega", 0.0)
        pitch = self._bem_cfg.get("pitch", 0.0)

        self.bem_result = bem_solver.compute(v_inf, omega, pitch)

        # 3. Force projection
        span_dir = self._cfg.get("elements", {}).get("span_direction", [0.0, 0.0, 1.0])
        normal_dir = self._bem_cfg.get("normal_direction", [1.0, 0.0, 0.0])
        tangential_dir = self._bem_cfg.get("tangential_direction", [0.0, 1.0, 0.0])

        projector = ForceProjector(
            self.mesh,
            self.blade_aero,
            span_direction=span_dir,
            normal_direction=normal_dir,
            tangential_direction=tangential_dir,
        )
        self.nodal_forces = projector.project(self.bem_result)
        verification = projector.verify(self.bem_result, self.nodal_forces)

        # 4. Export results
        output_folder = self._cfg.get("solver", {}).get("output_folder", "results")
        self._export(output_folder, verification)

        return {
            "thrust": self.bem_result.thrust,
            "torque": self.bem_result.torque,
            "power": self.bem_result.power,
            "force_error": verification["force_error"],
        }

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _load_aero(self) -> BladeAero:
        """Load blade aerodynamic data from the blade definition file."""
        blade_file = self._blade_file
        if blade_file is None:
            raise RuntimeError(
                "No blade file specified. Ensure the mesh generator config "
                "contains a yaml_file or excel_file parameter."
            )

        return load_blade_aero(
            blade_file,
            default_re=self._bem_cfg.get("default_re", 1e7),
            neuralfoil_model=self._bem_cfg.get("neuralfoil_model", "large"),
            hub_radius=self._bem_cfg.get("hub_radius", 0.0),
            n_blades=self._bem_cfg.get("n_blades", 3),
        )

    def _export(self, output_folder: str, verification: dict) -> None:
        """Write CSV and VTU output files."""
        out = Path(output_folder)
        out.mkdir(parents=True, exist_ok=True)

        result = self.bem_result
        if result is None:
            return

        # --- Sectional loads CSV ---
        header = "r[m],Np[N/m],Tp[N/m],alpha[deg],cl,cd,a,ap"
        data = np.column_stack([
            result.r,
            result.Np,
            result.Tp,
            result.alpha,
            result.cl,
            result.cd,
            result.a,
            result.ap,
        ])
        np.savetxt(out / "bem_sectional_loads.csv", data, delimiter=",", header=header, comments="")

        # --- Global loads CSV ---
        with open(out / "bem_global_loads.csv", "w") as f:
            f.write("quantity,value,unit\n")
            f.write(f"thrust,{result.thrust:.6f},N\n")
            f.write(f"torque,{result.torque:.6f},Nm\n")
            f.write(f"power,{result.power:.6f},W\n")
            f.write(f"force_error,{verification['force_error']:.6e},N\n")

        # --- VTU with nodal forces ---
        if self.nodal_forces is not None:
            try:
                import meshio

                coords = self.mesh.coords_array
                # Build cell connectivity for meshio
                cells = []
                for elem in self.mesh.elements:
                    n_nds = len(elem.connectivity)
                    if n_nds == 3:
                        cell_type = "triangle"
                    elif n_nds == 4:
                        cell_type = "quad"
                    elif n_nds == 6:
                        cell_type = "triangle6"
                    elif n_nds == 8:
                        cell_type = "quad8"
                    else:
                        continue
                    cells.append((cell_type, [elem.connectivity]))

                # Group by cell type
                cell_blocks: dict[str, list] = {}
                for ctype, conn in cells:
                    cell_blocks.setdefault(ctype, []).append(conn[0])
                meshio_cells = [(k, np.array(v)) for k, v in cell_blocks.items()]

                m = meshio.Mesh(
                    points=coords,
                    cells=meshio_cells,
                    point_data={"Force": self.nodal_forces},
                )
                m.write(str(out / "bem_nodal_forces.vtu"))
            except ImportError:
                pass  # meshio not available; skip VTU export
