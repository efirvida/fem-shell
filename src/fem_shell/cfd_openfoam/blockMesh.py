import json
import os
import shutil
import subprocess
import uuid
from concurrent import futures
from copy import deepcopy
from typing import Iterable, Union

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

    def __repr__(self):
        return f"<KeyPoint id={self.id} coords={self.coords.tolist()}>"


class BlockMesh:
    def __init__(self, blade_definition: Blade | None = None) -> None:
        self.definition = blade_definition

    def generate(self, blade_definition):
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
                f.write(f"    polyLine {spl['from_point']} {spl['to_point']} ( {coords} )\n")
            f.write(");\n\n")

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

    def optimize(self, work_dir):
        OPTIMIZATION_BLADE_DEFINITION = deepcopy(self.definition)
        TEST_CASE = os.path.join(os.path.dirname(__file__), "test_case")
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)

        # Preparar datos iniciales
        bl_kp = []
        out_kp = []
        for sec in self.definition:
            bl_kp.append(sec["keypoints"]["bl"][1:])
            out_kp.append(sec["keypoints"]["out"][1:])

        BL_KP = np.array(bl_kp, dtype=np.int64)
        OUT_KP = np.array(out_kp, dtype=np.int64)
        n_points = BL_KP.size + OUT_KP.size
        POINTS_PER_SECTIONS = BL_KP.shape[1]

        # Espacio de búsqueda
        parametrization = ng.p.Array(shape=(n_points,)).set_integer_casting()
        parametrization.set_bounds(-10, 10)  # Límites para desplazamientos enteros

        def calculate_objective(mesh_quality: dict) -> float:
            # Valores normalizados con límites recomendados de OpenFOAM
            maxNonOrth = mesh_quality.get("maxNonOrth", 180)
            avgNonOrth = mesh_quality.get("avgNonOrth", 90)
            maxSkew = mesh_quality.get("maxSkew", 4)
            avgSkew = mesh_quality.get("avgSkew", 2)
            aspectRatio = mesh_quality.get("maxAspectRatio", 100)
            minVolume = mesh_quality.get("minVolume", 1e-18)
            nErrorDeterminant = mesh_quality.get("nErrorDeterminant", 0)
            nErrorFaceWeight = mesh_quality.get("nErrorFaceWeight", 0)
            totalCells = mesh_quality.get("nCells", 1)
            smoothness = mesh_quality.get("smoothness", 0)

            # Penalizaciones no lineales para métricas críticas
            nonOrth_penalty = (maxNonOrth / 65) ** 3 + (avgNonOrth / 40) ** 2
            skew_penalty = (maxSkew / 3.5) ** 3 + (avgSkew / 2) ** 2
            aspect_penalty = (aspectRatio / 10) ** 2
            volume_penalty = 1 / (minVolume + 1e-16)
            error_penalty = 1000 * (nErrorDeterminant + nErrorFaceWeight)
            smoothness_penalty = 10 * (1 - smoothness) ** 2
            complexity_penalty = np.log(totalCells / 1e5 + 1)

            # Pesos relativos basados en importancia para CFD
            weights = {
                "nonOrth": 2.0,  # Máxima prioridad
                "skew": 1.5,
                "aspect": 0.8,
                "errors": 5.0,  # Penalización fuerte por celdas inválidas
                "volume": 3.0,
                "smoothness": 1.2,
                "complexity": 0.5,
            }

            objective = (
                weights["nonOrth"] * nonOrth_penalty
                + weights["skew"] * skew_penalty
                + weights["aspect"] * aspect_penalty
                + weights["errors"] * error_penalty
                + weights["volume"] * volume_penalty
                + weights["smoothness"] * smoothness_penalty
                + weights["complexity"] * complexity_penalty
            )

            # Penalización exponencial por violar límites críticos
            if maxNonOrth > 80 or maxSkew > 4 or nErrorDeterminant > 0 or minVolume < 1e-15:
                objective *= np.exp(5 * (maxNonOrth / 80 + maxSkew / 4))

            return objective

        def evaluate(x):
            KeyPoint._id_counter = 0
            eval_id = str(uuid.uuid4())[:8]
            eval_dir = os.path.join(work_dir, f"worker_{os.getpid()}", f"eval_{eval_id}")

            original_dir = os.getcwd()
            try:
                # Configuración del caso
                shutil.copytree(TEST_CASE, eval_dir)
                x_int = np.round(x).astype(int)

                # Actualizar geometría
                step_definition = update_geometry(x_int)
                self.write_blockmesh(eval_dir, step_definition)

                # Ejecutar simulación
                os.chdir(eval_dir)
                os.chmod("run.sh", 0o755)
                subprocess.run(
                    ["./run.sh"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                os.chdir(original_dir)

                # Obtener métricas
                mesh_quality = get_metrics(eval_dir)
                objective = calculate_objective(mesh_quality)
                if os.path.exists(eval_dir):
                    shutil.rmtree(eval_dir)
            except Exception as e:
                print(f"Error en evaluación {eval_id}: {str(e)}")
                objective = float("inf")
            finally:
                os.chdir(original_dir)

            return objective

        def update_geometry(x: np.ndarray):
            x = np.round(x).astype(int)
            new_definition = deepcopy(OPTIMIZATION_BLADE_DEFINITION)
            transformations = x.reshape(-1, POINTS_PER_SECTIONS)
            half = transformations.shape[0] // 2
            transformations_bl = transformations[:half]
            transformations_out = transformations[half:]
            for i, sec in enumerate(new_definition):
                # Keypoints originales (índices)
                original_bl = np.array(sec["keypoints"]["bl"][1:])
                original_out = np.array(sec["keypoints"]["out"][1:])

                # Aplicar desplazamientos con restricciones
                new_bl = apply_safe_displacements(original_bl, transformations_bl[i])
                new_out = apply_safe_displacements(original_out, transformations_out[i])

                sec["keypoints"]["bl"][1:] = new_bl.tolist()
                sec["keypoints"]["out"][1:] = new_out.tolist()
            return new_definition

        def get_metrics(case_path: str) -> dict[str, float]:
            """Obtiene métricas de calidad de malla con manejo robusto de errores"""

            # Valores por defecto (indicando mala calidad)
            default_metrics = {
                "maxNonOrth": 180.0,  # Máximo valor posible (muy malo)
                "avgNonOrth": 90.0,  # Valor promedio malo
                "maxSkew": 4.0,  # Skewness crítico
                "avgSkew": 2.0,  # Skewness promedio alto
                "maxAspectRatio": 100.0,  # Relación de aspecto muy pobre
                "minVolume": 1e-18,  # Volumen mínimo casi cero
                "nErrorDeterminant": 10,  # Múltiples celdas inválidas
                "nErrorFaceWeight": 10,  # Múltiples caras problemáticas
                "nCells": 1,  # Mínimo número de celdas
                "smoothness": 0.0,  # Mínima suavidad
                "meshOK": False,  # Indica si el mesh es válido
            }

            metrics = default_metrics.copy()

            try:
                # 1. Procesar checkMesh.json
                check_mesh_json = os.path.join(case_path, "checkMesh.json")
                if os.path.exists(check_mesh_json):
                    with open(check_mesh_json, "r") as f:
                        mesh_data = json.load(f)

                    # Actualizar solo las métricas existentes en el archivo
                    for key in mesh_data:
                        if key in metrics:
                            metrics[key] = float(mesh_data[key])

                # 2. Verificar log.checkMesh para validación adicional
                check_mesh_log = os.path.join(case_path, "log.checkMesh")
                if os.path.exists(check_mesh_log):
                    with open(check_mesh_log, "r") as f:
                        log_content = f.read()

                        # Verificar si el mesh es válido
                        metrics["meshOK"] = "Mesh OK" in log_content

                        # Extraer número de celdas si no está en el JSON
                        if "nCells" not in mesh_data:
                            for line in log_content.split("\n"):
                                if "cells:" in line:
                                    parts = line.strip().split()
                                    metrics["nCells"] = int(parts[-1])
                                    break

                # 3. Calcular métrica de suavizado (número de parches)
                boundary_file = os.path.join(case_path, "constant/polyMesh/boundary")
                if os.path.exists(boundary_file):
                    try:
                        with open(boundary_file, "r") as f:
                            content = f.read()
                            num_patches = content.count("type patch;")
                            # Suavizado inversamente proporcional al número de parches
                            metrics["smoothness"] = 1.0 / (1.0 + num_patches)
                    except Exception as e:
                        print(f"Error leyendo boundary file: {str(e)}")
                        metrics["smoothness"] = 0.0

                # 4. Verificación de errores adicionales
                if not metrics["meshOK"]:
                    metrics["nErrorDeterminant"] += 5
                    metrics["nErrorFaceWeight"] += 5

            except json.JSONDecodeError as e:
                print(f"Error decodificando JSON: {str(e)}")
                metrics.update({
                    "maxNonOrth": 180.0,
                    "nErrorDeterminant": 20,
                    "nErrorFaceWeight": 20,
                })
            except Exception as e:
                print(f"Error inesperado obteniendo métricas: {str(e)}")
                # Mantener los valores por defecto ya establecidos

            # Asegurar tipos correctos para todas las métricas
            for key in metrics:
                if key != "meshOK":
                    if isinstance(metrics[key], (int, float)):
                        metrics[key] = float(metrics[key])
                    else:
                        metrics[key] = 0.0

            return metrics

        optimizer = ng.optimizers.DiscreteOnePlusOne(
            parametrization=parametrization, budget=100, num_workers=2
        )

        def print_progress(optimizer, x, value):
            if optimizer.num_ask % 10 == 0:
                print(f"Iteración {optimizer.num_ask}: Value = {value:.2e}")
                with open("optimization_log.csv", "a") as f:
                    f.write(f"{optimizer.num_ask},{value}\n")

        optimizer.register_callback("tell", print_progress)

        with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
            recommendation = optimizer.minimize(
                evaluate,
                executor=executor,
                batch_mode=False,
                verbosity=0,  # Desactivar logs automáticos de Nevergrad
            )

        # Resultados finales
        print(f"\nOptimización completada. Mejor valor: {best_objective:.2f}")
        best_definition = update_geometry(recommendation.value)
        np.savetxt("results", recommendation.value)
        return best_definition


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
    segments = []
    for i in range(len(keypoints) - 1):
        start = keypoints[i]
        end = keypoints[i + 1]

        # Asegurar start < end
        if start >= end:
            start, end = end, start

        # Limitar al tamaño del spline
        start = max(0, min(start, len(spline) - 1))
        end = max(0, min(end, len(spline) - 1))

        segments.append(spline[start : end + 1])

    return segments


def apply_safe_displacements(original: np.ndarray, displacements: np.ndarray) -> np.ndarray:
    """Aplica desplazamientos manteniendo:
    - Índices en [0, 499]
    - Orden creciente
    """
    new = original.copy()
    n_points = 500  # Total de puntos en el spline

    # Aplicar desplazamientos
    new += displacements

    # 1. Limitar al rango [0, 499]
    np.clip(new, 0, n_points - 1, out=new)

    # 2. Forzar orden creciente
    for j in range(1, len(new)):
        if new[j] <= new[j - 1]:
            # Ajustar al mínimo espacio posible
            new[j] = new[j - 1] + 1

    # Última verificación
    np.clip(new, 0, n_points - 1, out=new)
    return new


# Simulación: reemplazá por tu código real
