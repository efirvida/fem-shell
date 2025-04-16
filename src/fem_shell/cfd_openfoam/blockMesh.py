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
| \\      /  F ield         |                                                 |
|  \\    /   O peration     |                                                 |
|   \\  /    A nd           |                                                 |
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

    def generate(self, blade_definition):
        KeyPoint.reset()
        self.definition = blade_definition
        points_map = {}
        points = []
        blocks = []
        splines = []
        boundaries = []

        def add_point(kp):
            points.append(kp)

        def add_spline(from_point, to_point, spline_coords):
            splines.append({
                "from_point": from_point,
                "to_point": to_point,
                "coords": spline_coords,
            })

        def add_block(from_point, blocks_x, blocks_y, blocks_z):
            blocks.append({
                "point_ids": from_point,
                "blocks_x": blocks_x,
                "blocks_y": blocks_y,
                "blocks_z": blocks_z,
            })

        for sec_id, sec in enumerate(blade_definition):
            af_kp = sec["keypoints"]["af"]
            af_coords = sec["splines"]["af"]
            for point_id in af_kp:
                kp = KeyPoint(coords=af_coords[point_id])
                points_map[f"af-{point_id}-sec-{sec_id}"] = kp.id
                add_point(kp)

            af_splines = split_splines(af_coords, af_kp)
            for kp_idx in range(len(af_kp) - 1):
                from_kp = points_map[f"af-{af_kp[kp_idx]}-sec-{sec_id}"]
                to_kp = points_map[f"af-{af_kp[kp_idx + 1]}-sec-{sec_id}"]
                coords = af_splines[kp_idx]
                add_spline(from_kp, to_kp, coords)

            bl_kp = sec["keypoints"]["bl"]
            bl_coords = sec["splines"]["bl"]
            for point_id in bl_kp:
                kp = KeyPoint(coords=bl_coords[point_id])
                points_map[f"bl-{point_id}-sec-{sec_id}"] = kp.id
                add_point(kp)

            bl_splines = split_splines(bl_coords, bl_kp)
            for kp_idx in range(len(bl_kp) - 1):
                from_kp = points_map[f"bl-{bl_kp[kp_idx]}-sec-{sec_id}"]
                to_kp = points_map[f"bl-{bl_kp[kp_idx + 1]}-sec-{sec_id}"]
                coords = bl_splines[kp_idx]
                add_spline(from_kp, to_kp, coords)

            out_kp = sec["keypoints"]["out"]
            out_coords = sec["splines"]["out"]
            for point_id in out_kp:
                kp = KeyPoint(coords=out_coords[point_id])
                points_map[f"out-{point_id}-sec-{sec_id}"] = kp.id
                add_point(kp)
            add_spline(
                points_map[f"bl-{bl_kp[-1]}-sec-{sec_id}"],
                points_map[f"bl-{bl_kp[0]}-sec-{sec_id}"],
                bl_coords[bl_kp[-1] :],
            )

            out_splines = split_splines(out_coords, out_kp)
            for kp_idx in range(len(out_kp) - 1):
                from_kp = points_map[f"out-{out_kp[kp_idx]}-sec-{sec_id}"]
                to_kp = points_map[f"out-{out_kp[kp_idx + 1]}-sec-{sec_id}"]
                coords = out_splines[kp_idx]
                add_spline(from_kp, to_kp, coords)
            add_spline(
                points_map[f"out-{out_kp[-1]}-sec-{sec_id}"],
                points_map[f"out-{out_kp[0]}-sec-{sec_id}"],
                out_coords[out_kp[-1] :],
            )

        NX, NY, NZ = 10, 10, 10

        n_sections = len(blade_definition)
        for sec in range(n_sections - 1):
            af_kp_current = blade_definition[sec]["keypoints"]["af"]
            bl_kp_current = blade_definition[sec]["keypoints"]["bl"]
            out_kp_current = blade_definition[sec]["keypoints"]["out"]

            af_kp_next = blade_definition[sec + 1]["keypoints"]["af"]
            bl_kp_next = blade_definition[sec + 1]["keypoints"]["bl"]
            out_kp_next = blade_definition[sec + 1]["keypoints"]["out"]

            n_rings = len(af_kp_next) - 1
            for i in range(n_rings):
                Bl_blk = [
                    points_map[f"af-{af_kp_current[i]}-sec-{sec}"],
                    points_map[f"bl-{bl_kp_current[i]}-sec-{sec}"],
                    points_map[f"bl-{bl_kp_current[i + 1]}-sec-{sec}"],
                    points_map[f"af-{af_kp_current[i + 1]}-sec-{sec}"],
                    points_map[f"af-{af_kp_next[i]}-sec-{sec + 1}"],
                    points_map[f"bl-{bl_kp_next[i]}-sec-{sec + 1}"],
                    points_map[f"bl-{bl_kp_next[i + 1]}-sec-{sec + 1}"],
                    points_map[f"af-{af_kp_next[i + 1]}-sec-{sec + 1}"],
                ]

                OUT_blk = [
                    points_map[f"bl-{bl_kp_current[i]}-sec-{sec}"],
                    points_map[f"out-{out_kp_current[i]}-sec-{sec}"],
                    points_map[f"out-{out_kp_current[i + 1]}-sec-{sec}"],
                    points_map[f"bl-{bl_kp_current[i + 1]}-sec-{sec}"],
                    points_map[f"bl-{bl_kp_next[i]}-sec-{sec + 1}"],
                    points_map[f"out-{out_kp_next[i]}-sec-{sec + 1}"],
                    points_map[f"out-{out_kp_next[i + 1]}-sec-{sec + 1}"],
                    points_map[f"bl-{bl_kp_next[i + 1]}-sec-{sec + 1}"],
                ]

                mid = n_rings // 2
                if (i >= mid - 1 and i <= mid) or i == 0 or i == n_rings - 1:
                    ny = NY * 5
                else:
                    ny = NY

                add_block(Bl_blk, NX, ny, NZ)
                add_block(OUT_blk, NX, ny, NZ)
                boundaries.append([
                    points_map[f"af-{af_kp_current[i]}-sec-{sec}"],
                    points_map[f"af-{af_kp_current[i + 1]}-sec-{sec}"],
                    points_map[f"af-{af_kp_next[i]}-sec-{sec + 1}"],
                    points_map[f"af-{af_kp_next[i + 1]}-sec-{sec + 1}"],
                ])

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
            add_block(Bl_blk_0, NX, 5, NZ)
            add_block(OUT_blk_0, NX, 5, NZ)

        return {"points": points, "blocks": blocks, "splines": splines, "boundaries": boundaries}

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
                    f"    hex ({' '.join(str(i) for i in blk['point_ids'])}) ({blk['blocks_x']} {blk['blocks_y']} {blk['blocks_z']}) simpleGrading ({15} 1 1 )\n"
                )
            f.write(");\n\n")

            f.write("boundary \n(\n")
            f.write("    blade \n")
            f.write("    {\n")
            f.write("        type wall;\n")
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
            .set_integer_casting()
        )

        def evaluate(x: np.ndarray) -> float:
            return self._evaluate_candidate(
                x=x.copy(),
                work_dir=work_dir,
                test_case=test_case,
                original_definition=original_definition,
            )

        optimizer = ng.optimizers.PSO(
            parametrization=parametrization,
            budget=2000,
            num_workers=2,
        )
        optimizer.register_callback("tell", self._log_progress)

        with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
            recommendation = optimizer.minimize(
                evaluate,
                executor=executor,
                batch_mode=False,
                verbosity=4,
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
        weights = {
            "nonOrth": 2.0,
            "skew": 1.5,
            "aspect": 0.8,
            "errors": 5.0,
            "volume": 3.0,
            "smoothness": 1.2,
            "complexity": 0.5,
        }

        nonOrth_penalty = (mesh["maxNonOrth"] / 65) ** 3 + (mesh["avgNonOrth"] / 40) ** 2
        skew_penalty = (mesh["maxSkew"] / 3.5) ** 3 + (mesh["avgSkew"] / 2) ** 2
        aspect_penalty = (mesh["maxAspectRatio"] / 10) ** 2
        volume_penalty = 1 / (mesh["minVolume"] + 1e-16)
        error_penalty = 1000 * (mesh["nErrorDeterminant"] + mesh["nErrorFaceWeight"])
        smoothness_penalty = 10 * (1 - mesh["smoothness"]) ** 2
        complexity_penalty = np.log(mesh["nCells"] / 1e5 + 1)

        objective = (
            weights["nonOrth"] * nonOrth_penalty
            + weights["skew"] * skew_penalty
            + weights["aspect"] * aspect_penalty
            + weights["errors"] * error_penalty
            + weights["volume"] * volume_penalty
            + weights["smoothness"] * smoothness_penalty
            + weights["complexity"] * complexity_penalty
        )

        if (
            mesh["maxNonOrth"] > 80
            or mesh["maxSkew"] > 4
            or mesh["nErrorDeterminant"] > 0
            or mesh["minVolume"] < 1e-15
        ):
            objective *= np.exp(5 * (mesh["maxNonOrth"] / 80 + mesh["maxSkew"] / 4))

        return objective

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
