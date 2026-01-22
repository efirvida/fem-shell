import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from concurrent import futures
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Union

import nevergrad as ng
import numpy as np

from fem_shell.models.blade.model import Blade

HEADER = """
/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2412                                 |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}
\\*---------------------------------------------------------------------------*/
"""


class KeyPoint:
    _id_counter = 0

    def __init__(self, coords: Union[Iterable[float], np.ndarray]):
        coords_arr = np.array(coords, dtype=float)
        if coords_arr.size < 3:
            coords_arr = np.concatenate((coords_arr, np.zeros(3 - coords_arr.size)))
        self.coords = coords_arr
        self.x = coords_arr[0]
        self.y = coords_arr[1]
        self.z = coords_arr[2]
        self.id = KeyPoint._id_counter
        KeyPoint._id_counter += 1

    @classmethod
    def reset(cls):
        cls._id_counter = 0

    def __repr__(self):
        return f"<KeyPoint id={self.id} coords={self.coords.tolist()}>"


class BlockMesh:
    def __init__(self, blade_definition: Blade | None = None) -> None:
        self.definition = blade_definition

    def generate(self, definition):
        """Generate blade mesh structure from blade definition.

        Args:
            blade_definition: List of dictionaries containing blade section definitions

        Returns:
            Dictionary containing points, blocks, splines and boundaries
        """
        KeyPoint.reset()

        # Constants
        NY = 10
        NY_factor = 2
        NX_BL = 10
        NX_OUTTER = NY
        BL_GRADING = (NX_BL, 1, 1)
        OUTTER_GRADING = (1, 1, 1)

        TOP_BL_GRADING = BL_GRADING
        TOP_OUTTER_GRADING = OUTTER_GRADING

        blade_definition = definition["sections"]
        chords = definition["chords"]
        min_chord = chords.min()
        max_chord = chords.max()

        # Ajuste lineal inverso (más celdas cuando la cuerda es pequeña)
        spans = [z["splines"]["af"][0, 2] for z in blade_definition]
        blade_length = spans[-1]
        # 0,3125
        base_nz = int(np.ceil(np.diff(spans).mean())) * 2
        base_nz = int(np.ceil(np.diff(spans).mean()) / 0.3125)

        # Initialize data structures
        points_map = {}
        points = []
        blocks = []
        splines = []
        boundaries = []

        # Helper functions
        def add_point(kp):
            points.append(kp)

        def add_spline(from_point, to_point, spline_coords):
            splines.append(
                {
                    "from_point": from_point,
                    "to_point": to_point,
                    "coords": spline_coords,
                }
            )

        def add_block(point_ids, blocks_x, blocks_y, blocks_z, grading):
            blocks.append(
                {
                    "point_ids": point_ids,
                    "blocks_x": blocks_x,
                    "blocks_y": blocks_y,
                    "blocks_z": blocks_z,
                    "grading": grading,
                }
            )

        # # Add closing section
        # n_fit_sections = 10  # usar las últimas 3 secciones para extrapolar la curvatura

        # # Nuevo valor de Z (como antes)
        # z_last = blade_definition[-1]["splines"]["af"][0, 2]
        # dz = np.diff(spans).mean()
        # z_new = z_last + dz

        # # Nueva sección basada en la última
        # closing_section = deepcopy(blade_definition[-1])

        # # Paso 1: calcular ancho promedio de los últimos 4 'out'
        # last_out_splines = [
        #     blade_definition[i]["splines"]["out"] for i in range(-int(n_fit_sections // 2), 0)
        # ]
        # widths = []
        # for out in last_out_splines:
        #     center = out.mean(axis=0)
        #     radius = np.linalg.norm(out[:, :2] - center[:2], axis=1)
        #     widths.append(np.mean(radius))
        # avg_width = np.mean(widths)

        # # Paso 2: ancho del último perfil
        # last_out = blade_definition[-1]["splines"]["out"]
        # center_last = last_out.mean(axis=0)
        # radius_last = np.linalg.norm(last_out[:, :2] - center_last[:2], axis=1)
        # width_last = np.mean(radius_last)

        # # Paso 3: calcular factor de escala total
        # scale_factor = avg_width / width_last

        # # Paso 4: extrapolar cada spline ("af", "bl", "out")
        # for spline_name, spline in closing_section["splines"].items():
        #     num_points = spline.shape[0]
        #     new_spline = np.zeros_like(spline)

        #     for pt_idx in range(num_points):
        #         x_vals, y_vals, z_vals = [], [], []

        #         for i in range(-n_fit_sections, 0):  # ej: -10 a -1
        #             pt = blade_definition[i]["splines"][spline_name][pt_idx]
        #             x_vals.append(pt[0])
        #             y_vals.append(pt[1])
        #             z_vals.append(pt[2])

        #         px = np.polyfit(z_vals, x_vals, deg=2)
        #         py = np.polyfit(z_vals, y_vals, deg=2)

        #         new_spline[pt_idx, 0] = np.polyval(px, z_new)
        #         new_spline[pt_idx, 1] = np.polyval(py, z_new)
        #         new_spline[pt_idx, 2] = z_new

        #     closing_section["splines"][spline_name] = new_spline

        # # Paso 5: añadir la nueva sección extrapolada
        # blade_definition.append(closing_section)

        # # Paso 6: aplicar escalado progresivo a los últimos 3 perfiles 'out' + extrapolado
        # n_scaled_sections = 2  # últimos 3 + extrapolado

        # # Aplicar el escalado
        # for i in range(-n_scaled_sections, 0):
        #     spline = blade_definition[i]["splines"]["out"]
        #     center = spline.mean(axis=0)
        #     vectors_xy = spline[:, :2] - center[:2]
        #     scaled_xy = vectors_xy * scale_factor
        #     spline[:, :2] = center[:2] + scaled_xy

        # self.definition = deepcopy(blade_definition)

        n_sections = len(blade_definition)

        # Process each section
        for sec_id, sec in enumerate(blade_definition):
            self._process_section(sec, sec_id, points_map, add_point, add_spline)

        # Create blocks between sections
        for sec in range(n_sections - 1):
            chord = chords[sec]
            # nz = int(base_nz * (max_chord / chord) ** 1)
            nz = base_nz

            self._create_blocks_between_sections(
                sec,
                blade_definition,
                points_map,
                boundaries,
                True,
                add_block,
                NY,
                nz,
                NX_BL,
                NX_OUTTER,
                BL_GRADING,
                OUTTER_GRADING,
                NY_factor,
            )

        # Create top blocks
        # self._create_blocks_between_sections(
        #     n_sections - 2,
        #     blade_definition,
        #     points_map,
        #     boundaries,
        #     False,
        #     add_block,
        #     NY,
        #     nz,
        #     NX_BL,
        #     NX_OUTTER,
        #     TOP_BL_GRADING,
        #     TOP_OUTTER_GRADING,
        #     NY_factor,
        # )
        # self._create_top_blocks(
        #     n_sections,
        #     blade_definition,
        #     points_map,
        #     boundaries,
        #     add_block,
        #     NY,
        #     nz,
        #     NX_BL,
        #     (1, 1, 1),
        # )

        return {"points": points, "blocks": blocks, "splines": splines, "boundaries": boundaries}

    def _process_section(self, section, section_id, points_map, add_point, add_spline):
        """Process a single blade section."""
        # Add closing spline for blade
        for group_name in ("af", "bl", "out"):
            self._process_spline_group(
                section, group_name, section_id, points_map, add_point, add_spline
            )
            if group_name != "af":
                kp = section["keypoints"][group_name]
                coords = section["splines"][group_name]
                add_spline(
                    points_map[f"{group_name}-{kp[-1]}-sec-{section_id}"],
                    points_map[f"{group_name}-{kp[0]}-sec-{section_id}"],
                    coords[kp[-1] :],
                )

    def _process_spline_group(
        self, section, group_name, section_id, points_map, add_point, add_spline
    ):
        """Process a group of splines (af, bl, or out)."""
        keypoints = section["keypoints"][group_name]
        coords = section["splines"][group_name]

        # Add keypoints
        for point_id in keypoints:
            kp = KeyPoint(coords=coords[point_id])
            points_map[f"{group_name}-{point_id}-sec-{section_id}"] = kp.id
            add_point(kp)

        # Add splines between keypoints
        group_splines = split_splines(coords, keypoints)
        for kp_idx in range(len(keypoints) - 1):
            from_kp = points_map[f"{group_name}-{keypoints[kp_idx]}-sec-{section_id}"]
            to_kp = points_map[f"{group_name}-{keypoints[kp_idx + 1]}-sec-{section_id}"]
            add_spline(from_kp, to_kp, group_splines[kp_idx])

    def _create_blocks_between_sections(
        self,
        sec,
        blade_definition,
        points_map,
        boundaries,
        inclute_TE_in_blade_patch,
        add_block,
        NY,
        nz,
        NX_BL,
        NX_OUTTER,
        BL_GRADING,
        OUTTER_GRADING,
        NY_factor=2,
    ):
        """Create blocks between two adjacent sections."""
        current_sec = blade_definition[sec]
        next_sec = blade_definition[sec + 1]

        af_kp_current = current_sec["keypoints"]["af"]
        bl_kp_current = current_sec["keypoints"]["bl"]
        out_kp_current = current_sec["keypoints"]["out"]

        af_kp_next = next_sec["keypoints"]["af"]
        bl_kp_next = next_sec["keypoints"]["bl"]
        out_kp_next = next_sec["keypoints"]["out"]

        n_rings = len(af_kp_next) - 1

        for i in range(n_rings):
            # Create blade block
            Bl_blk = self._create_block_points(
                "af", "bl", i, sec, points_map, af_kp_current, bl_kp_current, af_kp_next, bl_kp_next
            )

            # Create outer block
            OUT_blk = self._create_block_points(
                "bl",
                "out",
                i,
                sec,
                points_map,
                bl_kp_current,
                out_kp_current,
                bl_kp_next,
                out_kp_next,
            )

            ny = NY * NY_factor if i in (3, 5) else NY

            if sec != len(blade_definition):
                boundaries.append(
                    [
                        points_map[f"af-{af_kp_current[i]}-sec-{sec}"],
                        points_map[f"af-{af_kp_current[i + 1]}-sec-{sec}"],
                        points_map[f"af-{af_kp_next[i]}-sec-{sec + 1}"],
                        points_map[f"af-{af_kp_next[i + 1]}-sec-{sec + 1}"],
                    ]
                )

            add_block(Bl_blk, NX_BL, ny, nz, BL_GRADING)
            add_block(OUT_blk, NX_OUTTER, ny, nz, OUTTER_GRADING)

        # Create special blocks for the first/last keypoints
        self._create_special_blocks(
            sec,
            points_map,
            add_block,
            af_kp_current,
            bl_kp_current,
            out_kp_current,
            af_kp_next,
            bl_kp_next,
            out_kp_next,
            boundaries,
            inclute_TE_in_blade_patch,
            NX_BL,
            NX_OUTTER,
            NY,
            nz,
            BL_GRADING,
            OUTTER_GRADING,
        )

    def _create_block_points(
        self, from_group, to_group, i, sec, points_map, current_kp1, current_kp2, next_kp1, next_kp2
    ):
        """Create point IDs for a block between two groups."""
        return [
            points_map[f"{from_group}-{current_kp1[i]}-sec-{sec}"],
            points_map[f"{to_group}-{current_kp2[i]}-sec-{sec}"],
            points_map[f"{to_group}-{current_kp2[i + 1]}-sec-{sec}"],
            points_map[f"{from_group}-{current_kp1[i + 1]}-sec-{sec}"],
            points_map[f"{from_group}-{next_kp1[i]}-sec-{sec + 1}"],
            points_map[f"{to_group}-{next_kp2[i]}-sec-{sec + 1}"],
            points_map[f"{to_group}-{next_kp2[i + 1]}-sec-{sec + 1}"],
            points_map[f"{from_group}-{next_kp1[i + 1]}-sec-{sec + 1}"],
        ]

    def _create_special_blocks(
        self,
        sec,
        points_map,
        add_block,
        af_kp_current,
        bl_kp_current,
        out_kp_current,
        af_kp_next,
        bl_kp_next,
        out_kp_next,
        boundaries,
        close_TE,
        NX_BL,
        NX_OUTTER,
        NY,
        nz,
        BL_GRADING,
        OUTTER_GRADING,
    ):
        """Create special blocks for the first/last keypoints."""
        Bl_blk_0 = [
            points_map[f"af-{af_kp_current[-1]}-sec-{sec}"],
            points_map[f"bl-{bl_kp_current[-1]}-sec-{sec}"],
            points_map[f"bl-{bl_kp_current[0]}-sec-{sec}"],
            points_map[f"af-{af_kp_current[0]}-sec-{sec}"],
            points_map[f"af-{af_kp_next[-1]}-sec-{sec + 1}"],
            points_map[f"bl-{bl_kp_next[-1]}-sec-{sec + 1}"],
            points_map[f"bl-{bl_kp_next[0]}-sec-{sec + 1}"],
            points_map[f"af-{af_kp_next[0]}-sec-{sec + 1}"],
        ]

        OUT_blk_0 = [
            points_map[f"bl-{bl_kp_current[-1]}-sec-{sec}"],
            points_map[f"out-{out_kp_current[-1]}-sec-{sec}"],
            points_map[f"out-{out_kp_current[0]}-sec-{sec}"],
            points_map[f"bl-{bl_kp_current[0]}-sec-{sec}"],
            points_map[f"bl-{bl_kp_next[-1]}-sec-{sec + 1}"],
            points_map[f"out-{out_kp_next[-1]}-sec-{sec + 1}"],
            points_map[f"out-{out_kp_next[0]}-sec-{sec + 1}"],
            points_map[f"bl-{bl_kp_next[0]}-sec-{sec + 1}"],
        ]

        add_block(Bl_blk_0, NX_BL, 10, nz, BL_GRADING)
        add_block(OUT_blk_0, NX_OUTTER, 10, nz, OUTTER_GRADING)

        if close_TE:
            boundaries.append(
                [
                    points_map[f"af-{af_kp_current[-1]}-sec-{sec}"],
                    points_map[f"af-{af_kp_current[0]}-sec-{sec}"],
                    points_map[f"af-{af_kp_next[-1]}-sec-{sec + 1}"],
                    points_map[f"af-{af_kp_next[0]}-sec-{sec + 1}"],
                ]
            )

    def _create_top_blocks(
        self,
        n_sections,
        blade_definition,
        points_map,
        boundaries,
        add_block,
        NY,
        nz,
        NX_BL,
        grading,
    ):
        """Create blocks at the top of the blade."""
        af_kp_prev = blade_definition[n_sections - 2]["keypoints"]["af"]
        af_kp_last = blade_definition[n_sections - 1]["keypoints"]["af"]

        # Define top blocks
        blk_top = (
            self._create_top_block_points(
                0, 1, 8, 9, af_kp_prev, af_kp_last, points_map, n_sections
            ),
            self._create_top_block_points(
                1, 2, 7, 8, af_kp_prev, af_kp_last, points_map, n_sections
            ),
            self._create_top_block_points(
                2, 3, 6, 7, af_kp_prev, af_kp_last, points_map, n_sections
            ),
            self._create_top_block_points(
                3, 4, 5, 6, af_kp_prev, af_kp_last, points_map, n_sections
            ),
        )

        # Add boundaries
        boundaries.extend(
            [
                [points_map[f"af-{af_kp_prev[i]}-sec-{n_sections - 2}"] for i in [0, 1, 8, 9]],
                [points_map[f"af-{af_kp_prev[i]}-sec-{n_sections - 2}"] for i in [1, 2, 7, 8]],
                [points_map[f"af-{af_kp_prev[i]}-sec-{n_sections - 2}"] for i in [2, 3, 6, 7]],
                [points_map[f"af-{af_kp_prev[i]}-sec-{n_sections - 2}"] for i in [3, 4, 5, 6]],
            ]
        )

        # Add top blocks
        for blk in blk_top[:-1]:
            add_block(blk, NY, NY, nz, grading)
        add_block(blk_top[-1], NX_BL, NY, nz, grading)

    def _create_top_block_points(
        self, i1, i2, i3, i4, af_kp_prev, af_kp_last, points_map, n_sections
    ):
        """Create point IDs for a top block."""
        return [
            points_map[f"af-{af_kp_prev[i1]}-sec-{n_sections - 2}"],
            points_map[f"af-{af_kp_prev[i2]}-sec-{n_sections - 2}"],
            points_map[f"af-{af_kp_prev[i3]}-sec-{n_sections - 2}"],
            points_map[f"af-{af_kp_prev[i4]}-sec-{n_sections - 2}"],
            points_map[f"af-{af_kp_last[i1]}-sec-{n_sections - 1}"],
            points_map[f"af-{af_kp_last[i2]}-sec-{n_sections - 1}"],
            points_map[f"af-{af_kp_last[i3]}-sec-{n_sections - 1}"],
            points_map[f"af-{af_kp_last[i4]}-sec-{n_sections - 1}"],
        ]

    def write_blockmesh(self, of_case_path: str, definition=None):
        self._of_case_path = of_case_path
        self._dict_path = os.path.join(self._of_case_path, "system", "blockMeshDict")
        scale = 1

        if not definition:
            definition = self.generate(self.definition)
        else:
            definition = self.generate(definition)

        points = definition["points"]
        blocks = definition["blocks"]
        splines = definition["splines"]
        boundaries = definition["boundaries"]

        with open(self._dict_path, "w") as f:
            f.write(HEADER)
            f.write(f"scale {scale};\n\n")
            f.write("vertices \n(\n")
            for pt in points:
                f.write(f"    ({pt.x} {pt.y} {pt.z}) // point id {pt.id}\n")
            f.write(");\n\n")

            f.write("edges \n(\n")
            for spl in splines:
                coords = " ".join([f"({p[0]} {p[1]} {p[2]})" for p in spl["coords"]])
                f.write(f"\n    polyLine {spl['from_point']} {spl['to_point']} ( {coords} )")
            f.write("\n);\n\n")

            f.write("blocks \n(\n")
            for blk in blocks:
                f.write(
                    f"    hex ({' '.join(str(i) for i in blk['point_ids'])}) ({blk['blocks_x']} {blk['blocks_y']} {blk['blocks_z']}) simpleGrading ({' '.join(str(i) for i in blk['grading'])})\n"
                )
            f.write(");\n\n")

            f.write("boundary \n(\n")
            f.write("    bladePatch \n")
            f.write("    {\n")
            f.write("        type patch;\n")
            f.write("        faces\n")
            f.write("        (\n")
            for face in boundaries:
                f.write(f"            ( {' '.join(str(f) for f in face)} )\n")
            f.write("        );\n")
            f.write("    }\n")
            f.write(");\n")

    def optimize(self, work_dir: str):
        original_definition = deepcopy(self.definition)
        test_case = os.path.join(os.path.dirname(__file__), "test_case")

        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)

        bl_kp, out_kp = self._extract_keypoints()
        num_points = bl_kp.size + out_kp.size
        points_per_section = bl_kp.shape[1]
        parametrization = (
            ng.p.Array(shape=(num_points, points_per_section))
            .set_bounds(-10, 10)
            .set_mutation(sigma=2.0)  # Paso de mutación adaptativo
            .set_integer_casting()  # Mantener casting a enteros
        )

        def evaluate(x: np.ndarray) -> float:
            return self._evaluate_candidate(
                x=x.copy(),
                work_dir=work_dir,
                test_case=test_case,
                original_definition=original_definition,
            )

        optimizer = ng.optimizers.NGOpt(
            parametrization=parametrization,
            budget=500_000,
            num_workers=45,
        )
        optimizer.register_callback("tell", self._log_progress)

        with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
            recommendation = optimizer.minimize(
                evaluate,
                executor=executor,
                batch_mode=False,
                verbosity=1,
            )

        np.savetxt(
            os.path.join(os.getcwd(), "results.csv"), recommendation.value, delimiter=",", fmt="%d"
        )

    def _extract_keypoints(self):
        bl_kp = [sec["keypoints"]["bl"][1:] for sec in self.definition]
        out_kp = [sec["keypoints"]["out"][1:] for sec in self.definition]
        return np.array(bl_kp, dtype=np.int64), np.array(out_kp, dtype=np.int64)

    def _evaluate_candidate(
        self,
        x: np.ndarray,
        work_dir: str,
        test_case: str,
        original_definition: List[Dict[str, Any]],
    ) -> float:
        eval_id = f"{os.getpid()}_{threading.get_ident()}_{uuid.uuid4().hex[:8]}"
        eval_dir = os.path.join(work_dir, f"eval_{eval_id}")
        os.makedirs(eval_dir, exist_ok=True)

        try:
            step_definition = self._update_geometry(x, original_definition)
            shutil.copytree(test_case, eval_dir, dirs_exist_ok=True)
            self.write_blockmesh(eval_dir, step_definition)
        except (IndexError, ValueError):
            KeyPoint.reset()
            shutil.rmtree(eval_dir, ignore_errors=True)
            return float("inf")

        blockmesh_path = os.path.join(eval_dir, "system", "blockMeshDict")
        if not wait_for_file(blockmesh_path, timeout=5, interval=0.1):
            shutil.rmtree(eval_dir, ignore_errors=True)
            return float("inf")

        try:
            subprocess.run(
                ["./run.sh"],
                cwd=eval_dir,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            mesh_quality = self._get_mesh_metrics(eval_dir)
            objective = self._calculate_objective(mesh_quality)
        except Exception:
            shutil.rmtree(eval_dir, ignore_errors=True)
            return float("inf")
        finally:
            shutil.rmtree(eval_dir, ignore_errors=True)

        return objective

    def _update_geometry(
        self, variables: np.ndarray, definition: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        updated_definition = deepcopy(definition)
        half = variables.shape[0] // 2
        for i, sec in enumerate(updated_definition):
            sec["keypoints"]["bl"][1:] += variables[:half][i]
            sec["keypoints"]["out"][1:] += variables[half:][i]
        return updated_definition

    def _calculate_objective(self, mesh: Dict[str, float]) -> float:
        # Penalización base para geometrías inválidas
        base_penalty = 1e6 if not mesh["meshOK"] else 0

        # Limitar valores extremos para evitar explosión de penalizaciones
        max_nonorth = min(mesh["maxNonOrth"], 200)  # Capamos a 200° máximo para cálculo
        avg_nonorth = min(mesh["avgNonOrth"], 100)
        max_skew = min(mesh["maxSkew"], 10)
        avg_skew = min(mesh["avgSkew"], 5)
        max_aspect = min(mesh["maxAspectRatio"], 50)
        min_volume = max(mesh["minVolume"], 1e-20)  # Evitar división por cero

        # Cálculo de métricas con límites controlados
        nonOrth_penalty = (max_nonorth / 65) ** 3 + (avg_nonorth / 40) ** 2
        skew_penalty = (max_skew / 3.5) ** 3 + (avg_skew / 2) ** 2
        aspect_penalty = (max_aspect / 10) ** 2
        volume_penalty = 1 / min_volume
        error_penalty = 1000 * (mesh["nErrorDeterminant"] + mesh["nErrorFaceWeight"])
        smoothness_penalty = 10 * (1 - mesh["smoothness"]) ** 2
        complexity_penalty = np.log(mesh["nCells"] / 1e5 + 1)

        # Factores multiplicativos para condiciones críticas
        critical_factor = 1.0
        if (
            max_nonorth > 80
            or max_skew > 4
            or mesh["nErrorDeterminant"] > 0
            or mesh["minVolume"] < 1e-15
        ):
            critical_factor += np.exp(0.5 * (max_nonorth / 80 + max_skew / 4))

        # Pesos dinámicos basados en calidad base
        weights = {
            "nonOrth": 2.0 * critical_factor,
            "skew": 1.5 * critical_factor,
            "aspect": 0.8,
            "errors": 5.0,
            "volume": 3.0,
            "smoothness": 1.2,
            "complexity": 0.5,
        }

        # Cálculo final del objetivo con límite superior
        objective = (
            weights["nonOrth"] * nonOrth_penalty
            + weights["skew"] * skew_penalty
            + weights["aspect"] * aspect_penalty
            + weights["errors"] * error_penalty
            + weights["volume"] * volume_penalty
            + weights["smoothness"] * smoothness_penalty
            + weights["complexity"] * complexity_penalty
            + base_penalty
        )

        # Limitar el máximo valor para estabilidad numérica
        return min(objective, 1e9)

    def _get_mesh_metrics(self, case_path: str) -> Dict[str, float]:
        metrics = {
            "maxNonOrth": 180.0,
            "avgNonOrth": 90.0,
            "maxSkew": 4.0,
            "avgSkew": 2.0,
            "maxAspectRatio": 100.0,
            "minVolume": 1e-18,
            "nErrorDeterminant": 10,
            "nErrorFaceWeight": 10,
            "nCells": 1,
            "smoothness": 0.0,
            "meshOK": False,
        }

        try:
            json_path = os.path.join(case_path, "checkMesh.json")
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    data = json.load(f)
                    for k in metrics:
                        if k in data:
                            metrics[k] = float(data[k])

            log_path = os.path.join(case_path, "log.checkMesh")
            if os.path.exists(log_path):
                with open(log_path, "r") as f:
                    log = f.read()
                    metrics["meshOK"] = "Mesh OK" in log
                    if "nCells" not in data:
                        for line in log.split("\n"):
                            if "cells:" in line:
                                metrics["nCells"] = int(line.strip().split()[-1])
                                break

            boundary_path = os.path.join(case_path, "constant/polyMesh/boundary")
            if os.path.exists(boundary_path):
                with open(boundary_path, "r") as f:
                    content = f.read()
                    num_patches = content.count("type patch;")
                    metrics["smoothness"] = 1.0 / (1.0 + num_patches)

            if not metrics["meshOK"]:
                metrics["nErrorDeterminant"] += 5
                metrics["nErrorFaceWeight"] += 5

        except Exception:
            pass

        for key in metrics:
            if key != "meshOK":
                metrics[key] = float(metrics.get(key, 0.0))

        return metrics

    def _log_progress(self, optimizer: Any, x: Any, value: float):
        if optimizer.num_ask % 10 == 0:
            with open("optimization_log.csv", "a") as f:
                f.write(f"{optimizer.num_ask},{value}\n")


def split_splines(spline: np.ndarray, keypoints: list | np.ndarray) -> list:
    """
    Split a spline into subsplines based on keypoint indices.

    Parameters
    ----------
    spline : np.ndarray
        Array of N points defining the spline (shape NxM)
    keypoints : List[int]
        Indices where the spline should be divided

    Returns
    -------
    List[np.ndarray]
        List of subspline arrays

    Raises
    ------
    ValueError
        If keypoints are not sorted or contain duplicates

    Example
    -------
    >>> spline = np.array([[0, 0], [1, 1], ..., [9, 9]])
    >>> keypoints = [0, 3, 6]
    >>> subsplines = split_splines(spline, keypoints)
    >>> len(subsplines)
    3
    """
    keypoints = sorted(keypoints)
    if not np.all(np.diff(keypoints) > 0):
        raise ValueError("Keypoints must be sorted and unique")

    return [spline[start : end + 1] for start, end in zip(keypoints[:-1], keypoints[1:])]


def wait_for_file(path: str, timeout: float, interval: float = 0.1) -> bool:
    """Espera hasta que el archivo exista y sea accesible"""
    start = time.time()
    while (time.time() - start) < timeout:
        if os.path.isfile(path) and os.access(path, os.R_OK):
            return True
        time.sleep(interval)
    return False
