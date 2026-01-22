from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import shapely as shp
from scipy.interpolate import CubicSpline

from fem_shell.core.material import Material, OrthotropicMaterial
from fem_shell.core.mesh import MeshModel
from fem_shell.core.mesh.generators import BladeMesh, RotorMesh
from fem_shell.core.viewer import BladeGeometryVisualizer
from fem_shell.elements import ElementFamily


class Blade:
    """
    Wind turbine blade model with mesh generation and visualization capabilities.

    This class wraps the BladeMesh generator and provides additional
    functionality for blade visualization and geometry inspection.

    Parameters
    ----------
    blade_yaml : str
        Path to the blade YAML definition file
    element_size : float, optional
        Target element size for meshing (default: 0.1)
    n_samples : int, optional
        Number of samples for airfoil discretization (default: 300)
    """

    def __init__(self, blade_yaml: str, element_size: float = 0.1, n_samples: int = 300) -> None:
        self.yaml_file = blade_yaml
        self.element_size = element_size
        self.n_samples = n_samples

        self._mesh_generator: Optional[BladeMesh] = None
        self.mesh: Optional[MeshModel] = None
        self._numad_mesh: Dict = {}
        self._numad_blade = None

    def generate_mesh(self, renumber: Optional[str] = None):
        """
        Generate the blade mesh from the YAML definition.

        Parameters
        ----------
        renumber : str, optional
            Renumbering algorithm to apply after mesh generation.
            - None (default): No renumbering
            - "simple": Direct index assignment
            - "rcm": Reverse Cuthill-McKee for bandwidth reduction
        """
        self._mesh_generator = BladeMesh(
            yaml_file=self.yaml_file,
            element_size=self.element_size,
            n_samples=self.n_samples,
        )
        self.mesh = self._mesh_generator.generate(renumber=renumber, verbose=True)
        self._numad_mesh = self._mesh_generator.numad_mesh_data
        self._numad_blade = self._mesh_generator.numad_blade

    def view(self) -> None:
        """Visualize the blade mesh."""
        if self.mesh is None:
            raise RuntimeError("Mesh has not been generated yet. Call generate_mesh() first.")
        self.mesh.view()

    def show_plots(self) -> None:
        """Display blade geometry plots."""
        if self._numad_blade is None:
            raise RuntimeError("Mesh has not been generated yet. Call generate_mesh() first.")
        visualizer = BladeGeometryVisualizer(self._numad_blade)
        visualizer.plot_airfoil_type_distribution()
        visualizer.plot_chord_distribution()
        visualizer.plot_pitch_axis_position()
        visualizer.plot_thickness_to_chord_ratio()
        visualizer.plot_twist_distribution()
        plt.show()

    def write_mesh(self, filename: str, **kwargs) -> None:
        """Write the blade mesh to a file."""
        if self.mesh is None:
            raise RuntimeError("Mesh has not been generated yet. Call generate_mesh() first.")
        self.mesh.write_mesh(filename, **kwargs)

    def generate_blocks_sections(self):
        """Genera la estructura completa de datos para perfiles aerodinámicos y cilindros,
        estableciendo el Leading Edge (LE) como el punto medio entre los puntos centrales
        del spline.
        """
        if self._numad_blade is None or self._numad_mesh is None:
            raise RuntimeError("Mesh has not been generated yet. Call generate_mesh() first.")

        BL_CHORD_FRACTION = 0.2
        OUTTER_CHORD_FRACTION = 0.7

        n_points = 500
        n_spline_divisions = 5
        tolerance = 1e-6

        keypoints_in_section = self._numad_blade.keypoints.key_points.T
        airfoil_list = []

        # Identificar splines que coinciden con keypoints
        spline_indices = []
        for i in range(len(self._numad_mesh["splineZi"])):
            spline_z = self._numad_mesh["splineZi"][i][0]
            if any(abs(spline_z - sec[2][0]) < tolerance for sec in keypoints_in_section):
                spline_indices.append(i)

        def find_offset_points(profile_points, offset_x, offset_y):
            """
            Para cada punto de profile_points devuelve una tupla
            (closest_offset_point_index, x_intersect, y_intersect).
            Si no hay intersección, se devuelve (-1, np.nan, np.nan).
            """
            offset_curve = shp.LineString(np.column_stack((offset_x, offset_y)))
            offset_points = []
            offset_points_array = np.column_stack((offset_x, offset_y))

            for i in range(1, len(profile_points) - 1):
                px, py = profile_points[i]

                # Tangente local y normal
                prev_point = profile_points[i - 1]
                next_point = profile_points[i + 1]
                dx = next_point[0] - prev_point[0]
                dy = next_point[1] - prev_point[1]
                normal = np.array([-dy, dx])
                normal /= np.linalg.norm(normal)

                # Línea normal extendida
                line_length = 1000
                normal_line = shp.LineString(
                    [
                        (px - normal[0] * line_length, py - normal[1] * line_length),
                        (px + normal[0] * line_length, py + normal[1] * line_length),
                    ]
                )

                # Intersección con la curva offset
                intersection = normal_line.intersection(offset_curve)
                if intersection.is_empty:
                    continue

                # Si hay múltiples intersecciones, seleccionamos la más cercana al perfil base
                if isinstance(intersection, shp.MultiPoint):
                    points = np.array([[pt.x, pt.y] for pt in intersection.geoms])
                elif isinstance(intersection, shp.Point):
                    points = np.array([[intersection.x, intersection.y]])
                else:
                    continue  # Si es otra cosa, la ignoramos

                # Calcular distancia de cada punto de intersección al punto de perfil original
                distances = np.linalg.norm(points - np.array([px, py]), axis=1)
                closest_idx = np.argmin(distances)
                closest_point = points[closest_idx]

                # Encontrar el punto más cercano en la curva offset al punto de intersección seleccionado
                # Esto es para obtener el índice del punto en el array original
                offset_points_array = np.column_stack((offset_x, offset_y))
                dists_to_offset = np.linalg.norm(offset_points_array - closest_point, axis=1)
                closest_offset_point_index = np.argmin(dists_to_offset)

                # Guardar resultado
                offset_points.append(
                    (
                        closest_offset_point_index,
                        *closest_point,  # Usamos el punto de intersección, no el punto del array
                    )
                )

            return np.array(offset_points, dtype=float)

        def reparameterize_spline(x, y, n_points=500):
            """
            Reparametriza un spline usando interpolación cúbica a lo largo de la longitud de arco.
            Maneja casos donde hay puntos duplicados o no estrictamente crecientes.
            """
            # Convertir a arrays numpy
            x = np.asarray(x).flatten()
            y = np.asarray(y).flatten()

            # Eliminar puntos duplicados consecutivos
            mask = np.ones(len(x), dtype=bool)
            for i in range(1, len(x)):
                if x[i] == x[i - 1] and y[i] == y[i - 1]:
                    mask[i] = False
            x = x[mask]
            y = y[mask]

            # Si después de eliminar duplicados quedan muy pocos puntos
            if len(x) < 4:
                return x.tolist()[1:-1], y.tolist()[1:-1]

            # Calcular la longitud de arco acumulada
            dx = np.diff(x)
            dy = np.diff(y)
            ds = np.sqrt(dx**2 + dy**2)
            s = np.insert(np.cumsum(ds), 0, 0)  # Longitud de arco acumulada

            # Crear splines cúbicos para x e y en función de s
            try:
                cs_x = CubicSpline(s, x)
                cs_y = CubicSpline(s, y)
                s_new = np.linspace(0, s[-1], n_points)
                new_x = cs_x(s_new)
                new_y = cs_y(s_new)
            except:
                # Fallback a interpolación lineal si CubicSpline falla
                s_new = np.linspace(0, s[-1], n_points)
                new_x = np.interp(s_new, s, x)
                new_y = np.interp(s_new, s, y)

            return new_x, new_y

        def generate_equidistant_points(start_point, end_point, spline_points, n_points):
            """
            Genera n_points puntos equidistantes a lo largo del spline entre start_point y end_point.

            Parámetros:
            - start_point: Punto inicial (x, y)
            - end_point: Punto final (x, y)
            - spline_points: Tupla con listas de coordenadas x e y del spline (x_spline, y_spline)
            - n_points: Número total de puntos a generar (incluyendo start y end)

            Retorna:
            - Lista de tuplas con:
            * puntos equidistantes [(x1, y1), (x2, y2), ...]
            * índices de los puntos más cercanos en el spline original [idx1, idx2, ...]
            """
            x_spline, y_spline = spline_points

            # Encontrar los índices de los puntos más cercanos en el spline
            def find_closest_idx(point, x, y):
                distances = np.sqrt((np.array(x) - point[0]) ** 2 + (np.array(y) - point[1]) ** 2)
                return np.argmin(distances)

            start_idx = find_closest_idx(start_point, x_spline, y_spline)
            end_idx = find_closest_idx(end_point, x_spline, y_spline)

            # Extraer el segmento del spline entre los puntos
            if start_idx < end_idx:
                segment_x = x_spline[start_idx : end_idx + 1]
                segment_y = y_spline[start_idx : end_idx + 1]
                original_indices = list(range(start_idx, end_idx + 1))
            else:
                segment_x = x_spline[end_idx : start_idx + 1][::-1]  # Invertir para mantener orden
                segment_y = y_spline[end_idx : start_idx + 1][::-1]
                original_indices = list(
                    range(start_idx, end_idx - 1, -1)
                )  # Índices en orden inverso

            # Crear LineString del segmento
            segment_line = shp.LineString(list(zip(segment_x, segment_y)))
            total_length = segment_line.length

            # Calcular distancias para los puntos equidistantes
            distances = np.linspace(0, total_length, n_points)
            equidistant_points = [segment_line.interpolate(d) for d in distances]

            # Para cada punto interpolado, encontrar el índice más cercano en el segmento original
            closest_indices = []
            for point in equidistant_points:
                # Calcular distancias a todos los puntos del segmento
                dists = np.sqrt(
                    (np.array(segment_x) - point.x) ** 2 + (np.array(segment_y) - point.y) ** 2
                )
                closest_segment_idx = np.argmin(dists)
                # Mapear al índice original del spline
                closest_original_idx = original_indices[closest_segment_idx]
                closest_indices.append(closest_original_idx)

            # Retornar tanto los puntos como sus índices correspondientes
            points = np.array([(point.x, point.y) for point in equidistant_points])
            return np.array(
                [
                    [closest_indices[i], points[i][0], points[i][1]]
                    for i in range(len(closest_indices))
                ]
            )

        def rotate_spline_arrays(x_coords, y_coords, new_start_index):
            """
            Rota los arrays de coordenadas del spline para que el punto en new_start_index sea el primero.

            Parámetros:
            - x_coords: Array NumPy con coordenadas x [x1, x2, ..., xn]
            - y_coords: Array NumPy con coordenadas y [y1, y2, ..., yn]
            - new_start_index: Índice del punto que debe convertirse en el primero (basado en 0)

            Retorna:
            - Tupla con los arrays rotados (x_rotated, y_rotated)
            """
            if not 0 <= new_start_index < len(x_coords):
                raise ValueError("Índice fuera de rango")
            if len(x_coords) != len(y_coords):
                raise ValueError("Los arrays x e y deben tener la misma longitud")

            x_rotated = np.concatenate([x_coords[new_start_index:], x_coords[:new_start_index]])
            y_rotated = np.concatenate([y_coords[new_start_index:], y_coords[:new_start_index]])
            # return numpy array
            return np.array([x_rotated, y_rotated]).T

        for section_id, spline_id in enumerate(spline_indices):
            # Datos del spline actual
            chord_length = self._numad_blade.geometry.ichord[section_id]
            airfoil_spline_x = self._numad_mesh["splineXi"][spline_id][2:-2].copy()
            airfoil_spline_y = self._numad_mesh["splineYi"][spline_id][2:-2].copy()
            z_spline = self._numad_mesh["splineZi"][spline_id][2:-2].copy()
            current_z = z_spline[0]

            # AIRFOIL SECTION
            coords = list(zip(airfoil_spline_x, airfoil_spline_y))
            airfoil_polygon = shp.Polygon(coords)

            # Encontrar LE y TE con el nuevo método
            le_airfoil_idx = airfoil_spline_x.size // 2
            le_airfoil = (airfoil_spline_x[le_airfoil_idx], airfoil_spline_y[le_airfoil_idx])

            x_airfoil_lower = airfoil_spline_x[:le_airfoil_idx]
            y_airfoil_lower = airfoil_spline_y[:le_airfoil_idx]
            x_airfoil_lower, y_airfoil_lower = reparameterize_spline(
                x_airfoil_lower, y_airfoil_lower, n_points // 2
            )

            x_airfoil_upper = airfoil_spline_x[le_airfoil_idx:]
            y_airfoil_upper = airfoil_spline_y[le_airfoil_idx:]
            x_airfoil_upper, y_airfoil_upper = reparameterize_spline(
                x_airfoil_upper, y_airfoil_upper, n_points // 2
            )

            airfoil_spline_x = np.concatenate([x_airfoil_lower, x_airfoil_upper])
            airfoil_spline_y = np.concatenate([y_airfoil_lower, y_airfoil_upper])

            te_airfoil_0 = (airfoil_spline_x[0], airfoil_spline_y[0])
            te_airfoil_1 = (airfoil_spline_x[-1], airfoil_spline_y[-1])

            le_airfoil_idx = airfoil_spline_x.size // 2
            le_offset_points = 15
            upper_airfoil_points = generate_equidistant_points(
                te_airfoil_1,
                le_airfoil,
                (airfoil_spline_x, airfoil_spline_y),
                n_spline_divisions,
            )
            upper_airfoil_points_idx = upper_airfoil_points[:, 0]

            # Generar puntos equidistantes para el perfil inferior (4 puntos: TE1 + 2 intermedios + LE)
            lower_airfoil_points = generate_equidistant_points(
                te_airfoil_0,
                le_airfoil,
                (airfoil_spline_x, airfoil_spline_y),
                n_spline_divisions,
            )

            lower_airfoil_points_idx = lower_airfoil_points[:, 0]

            airfoil_keypoints = sorted(
                {
                    int(p)
                    for p in [
                        *lower_airfoil_points_idx,
                        le_airfoil_idx - le_offset_points,
                        le_airfoil_idx + le_offset_points,
                        *upper_airfoil_points_idx,
                    ]
                    if p != le_airfoil_idx
                }
            )

            airfoil_spline_z = np.full_like(airfoil_spline_x, current_z)
            airfoil_spline = np.column_stack([airfoil_spline_x, airfoil_spline_y, airfoil_spline_z])

            # ==============================
            # BOUNDARY LAYER SECTION
            # ==============================
            # Crear buffer para obtener el spline de offset
            buffer_distance = chord_length * BL_CHORD_FRACTION
            bl_polygon = airfoil_polygon.buffer(buffer_distance)
            # Extraer contorno exterior y reparametrizar a 500 puntos
            bl_spline_x, bl_spline_y = bl_polygon.exterior.xy
            bl_spline_x, bl_spline_y = reparameterize_spline(
                bl_spline_x[::-1], bl_spline_y[::-1], n_points
            )
            bl_spline_x, bl_spline_y = bl_spline_x[2:-2], bl_spline_y[2:-2]

            le_bl_idx = bl_spline_y.size // 2

            upper_bl_points = find_offset_points(
                upper_airfoil_points[:, 1:], bl_spline_x, bl_spline_y
            )
            upper_bl_points_idx = upper_bl_points[:, 0]
            lower_bl_points = find_offset_points(
                lower_airfoil_points[:, 1:], bl_spline_x, bl_spline_y
            )
            lower_bl_points_idx = lower_bl_points[:, 0]

            # bl_end_point_idx = bl_spline_x.shape[0] - 3
            # bl_start_point_idx = 3

            bl_end_point_idx = bl_spline_x.size - (
                (bl_spline_x.size - upper_bl_points_idx[0].astype(int)) // 4
            )
            bl_start_point_idx = lower_bl_points_idx[0].astype(int) // 4
            bl_keypoints = sorted(
                [
                    int(pt)
                    for pt in [
                        bl_start_point_idx,
                        *lower_bl_points_idx,
                        le_bl_idx - int(le_offset_points * 1.2),
                        le_bl_idx + le_offset_points,
                        *upper_bl_points_idx,
                        bl_end_point_idx,
                    ]
                ]
            )
            bl_spline = rotate_spline_arrays(bl_spline_x, bl_spline_y, bl_keypoints[0])
            bl_keypoints = np.array(bl_keypoints) - bl_keypoints[0]

            bl_spline_z = np.full((bl_spline.shape[0], 1), current_z)
            bl_spline = np.hstack([bl_spline, bl_spline_z])

            # ==============================
            # OUTERDOMAIN SECTION
            # ==============================
            outter_buffer_distance = airfoil_polygon.buffer(chord_length * OUTTER_CHORD_FRACTION)
            outter_spline_x, outter_spline_y = outter_buffer_distance.exterior.xy
            outter_spline_x, outter_spline_y = reparameterize_spline(
                outter_spline_x[::-1], outter_spline_y[::-1], n_points
            )
            upper_outter_points = find_offset_points(
                upper_airfoil_points[:, 1:], outter_spline_x, outter_spline_y
            )
            upper_outter_points_idx = upper_outter_points[:, 0]
            lower_outter_points = find_offset_points(
                lower_airfoil_points[:, 1:], outter_spline_x, outter_spline_y
            )
            lower_outter_points_idx = lower_outter_points[:, 0]

            le_outter_idx = bl_spline_y.size // 2

            outter_end_point_idx = (
                outter_spline_x.size + upper_outter_points_idx[0].astype(int)
            ) // 2
            outter_start_point_idx = lower_outter_points_idx[0].astype(int) // 2

            outter_keypoints = sorted(
                [
                    int(pt)
                    for pt in [
                        outter_start_point_idx,
                        *lower_outter_points_idx,
                        le_outter_idx - int(le_offset_points * 1.4),
                        le_outter_idx + le_offset_points,
                        *upper_outter_points_idx,
                        outter_end_point_idx,
                    ]
                ]
            )
            outter_spline = rotate_spline_arrays(
                outter_spline_x, outter_spline_y, outter_keypoints[0]
            )
            outter_keypoints = np.array(outter_keypoints) - outter_keypoints[0]

            outter_spline_z = np.full((outter_spline.shape[0], 1), current_z)
            outter_spline = np.hstack([outter_spline, outter_spline_z])

            airfoil_list.append(
                {
                    "keypoints": {
                        "af": airfoil_keypoints,
                        "bl": bl_keypoints,
                        "out": outter_keypoints,
                    },
                    "splines": {
                        "af": airfoil_spline,
                        "bl": bl_spline,
                        "out": outter_spline,
                    },
                }
            )

        # airfoil_list[-1]["splines"]["out"] = airfoil_list[-2]["splines"]["out"]
        # airfoil_list[-1]["keypoints"]["out"] = airfoil_list[-2]["keypoints"]["out"]

        self.blocks_definition = {
            "chords": self._numad_blade.geometry.ichord,
            "sections": airfoil_list,
        }

        # Graficar resultados
        # plt.figure(figsize=(8, 6))

        # PLOT AIRFOIL
        # plt.plot(airfoil_spline[:, 0], airfoil_spline[:, 1], "x", label="AIRFOIL")
        # for i, pt in enumerate(airfoil_keypoints):
        #     plt.scatter(airfoil_spline[pt][0], airfoil_spline[pt][1])
        #     plt.text(airfoil_spline[pt][0], airfoil_spline[pt][1], f"{i}", size=20)

        # PLOT BL
        # plt.plot(bl_spline[:, 0], bl_spline[:, 1], "b-", label="BL")
        # for i, pt in enumerate(bl_spline):
        #     if i in bl_keypoints:
        #         plt.scatter(pt[0], pt[1])
        #         plt.text(pt[0], bl_spline[i][1], f"{i}")

        # PLOT OUTTER
        # plt.plot(outter_spline[:, 0], outter_spline[:, 1], "b-", label="OUTTER DOMAIN")
        # for i, pt in enumerate(outter_spline):
        #     if i in outter_keypoints:
        #         plt.scatter(pt[0], pt[1])
        #         plt.text(pt[0], outter_spline[i][1], f"{i}")

        # plt.gca().set_aspect("equal")
        # plt.show()


class Rotor:
    """
    Wind turbine rotor model with mesh generation capabilities.

    This class wraps the RotorMesh generator and provides additional
    functionality for rotor visualization.

    Parameters
    ----------
    blade_yaml : str
        Path to the blade YAML definition file
    n_blades : int
        Number of blades in the rotor
    hub_radius : float, optional
        Radial distance from rotation axis to blade root
    element_size : float, optional
        Target element size for meshing (default: 0.1)
    n_samples : int, optional
        Number of samples for airfoil discretization (default: 300)
    """

    def __init__(
        self,
        blade_yaml: str,
        n_blades: int,
        hub_radius: Optional[float] = None,
        element_size: float = 0.1,
        n_samples: int = 300,
    ) -> None:
        self.blade_yaml = blade_yaml
        self.n_blades = n_blades
        self.hub_radius = hub_radius
        self.element_size = element_size
        self.n_samples = n_samples

        self._mesh_generator: Optional[RotorMesh] = None
        self.mesh: Optional[MeshModel] = None

    def generate_mesh(self, renumber: Optional[str] = None):
        """
        Generate the rotor mesh.

        Parameters
        ----------
        renumber : str, optional
            Renumbering algorithm to apply after mesh generation.
        """
        self._mesh_generator = RotorMesh(
            yaml_file=self.blade_yaml,
            n_blades=self.n_blades,
            hub_radius=self.hub_radius,
            element_size=self.element_size,
            n_samples=self.n_samples,
        )
        self.mesh = self._mesh_generator.generate(renumber=renumber, verbose=True)

    def view(self) -> None:
        """Visualize the rotor mesh."""
        if self.mesh is None:
            raise RuntimeError("Mesh has not been generated yet. Call generate_mesh() first.")
        self.mesh.view()

    def write_mesh(self, filename: str, **kwargs) -> None:
        """Write the rotor mesh to a file."""
        if self.mesh is None:
            raise RuntimeError("Mesh has not been generated yet. Call generate_mesh() first.")
        self.mesh.write_mesh(filename, **kwargs)


def material_factory(
    material_data: Dict[str, Union[Dict[str, float], Dict[str, Tuple[float, float, float]]]],
) -> Union[Material, OrthotropicMaterial]:
    """
    Factory.s function to create a material object (either Isotropic or Orthotropic)
    based on .sthe provided material data.

    Parameters
    ----------
    material_data : dict
        A dictionary containing material properties. It should include the keys:
        - 'name' : str
            The name of the material.
        - 'density' : float
            The density of the material (kg/m^3).
        - 'elastic' : dict
            A dictionary containing the elastic properties:
            - 'E' : float or list of floats
                Young's Modulus (single value for isotropic, list for orthotropic).
            - 'nu' : float or list of floats
                Poisson's ratio (single value for isotropic, list for orthotropic).
            - 'G' : list of floats
                Shear Modulus (required for orthotropic materials only).

    Returns
    -------
    Union[IsotropicMaterial, OrthotropicMaterial]
        - An instance of `IsotropicMaterial` if the elastic properties are provided as single values or
        if the orthotropic material has equal values for E, G, and nu.
        - An instance of `OrthotropicMaterial` if the elastic properties are provided as lists and they are not identical.
    """
    material_name = material_data["name"]
    density = material_data["density"]
    elastic_props = material_data["elastic"]

    # Check if elastic properties are provided as lists for E, G, and nu (orthotropic material)
    if isinstance(elastic_props["E"], Iterable):
        # Check if all components are identical (i.e., isotropic material)
        if (
            all(val == elastic_props["E"][0] for val in elastic_props["E"])
            and all(val == elastic_props["nu"][0] for val in elastic_props["nu"])
            and all(val == elastic_props["G"][0] for val in elastic_props["G"])
        ):
            # It's an isotropic material, even if it was defined as orthotropic
            E = elastic_props["E"][0]
            nu = elastic_props["nu"][0]
            return Material(name=material_name, E=E, nu=nu, rho=density)
        else:
            # It's a true orthotropic material
            E = tuple(elastic_props["E"])
            G = tuple(elastic_props["G"])
            nu = tuple(elastic_props["nu"])
            return OrthotropicMaterial(name=material_name, E=E, G=G, nu=nu, rho=density)
    else:
        # It's an isotropic material
        E = elastic_props["E"]
        nu = elastic_props["nu"]
        return Material(name=material_name, E=E, nu=nu, rho=density)


@dataclass
class LayerDef:
    material: str
    thickens: float
    angle: float


@dataclass
class BladeSection:
    set_name: str
    x_dir: np.ndarray
    xy_dir: np.ndarray
    layers: List[LayerDef]


class BladeModel:
    def __init__(self, blade_yaml: str, element_size: float = 0.1, n_samples: int = 300) -> None:
        self.blade_mesh_obj = Blade(blade_yaml, element_size, n_samples)
        print(self.get_element_material(1))

    def generage_mesh(self):
        self._materials_db = {}
        self._sections_db = {}
        self.blade_mesh_obj.generate_mesh()
        self.mesh = self.blade_mesh_obj.mesh
        self._numad_mesh = self.blade_mesh_obj._numad_mesh

    def view(self):
        self.blade_mesh_obj.view()

    def write_mesh(self, filename, **kwargs):
        self.blade_mesh_obj.mesh.write_mesh(filename, **kwargs)

    @property
    def materials(self) -> Dict[str, Union[Material, OrthotropicMaterial]]:
        if self._materials_db:
            return self._materials_db
        else:
            for material in self._numad_mesh["materials"]:
                material = material_factory(material)
                self._materials_db[material.name] = material
        return self._materials_db

    @property
    def sections(self) -> Dict[str, BladeSection]:
        if self._sections_db:
            return self._sections_db
        else:
            for section in self._numad_mesh["sections"]:
                layers = [LayerDef(*l) for l in section["layup"]]
                section = BladeSection(
                    set_name=section["elementSet"],
                    x_dir=section["xDir"],
                    xy_dir=section["xyDir"],
                    layers=layers,
                )
                self._sections_db[section.set_name] = section
            return self._sections_db

    def get_element_material(self, element_id: int) -> BladeSection:
        element = self.mesh.get_element_by_id(element_id)
        element_sets = self.mesh.get_element_associated_set(element)

        # use the first asociate element set
        valid_set = element_sets[0]
        return self.sections[valid_set.name]

    def to_model_dict(self):
        model = {}
        for element in self.mesh.elements:
            section = self.get_element_material(element_id=element.id)
            model[element.id] = {
                "element_family": ElementFamily.SHELL,
                "layers": section.layers,
                "x_dir": section.x_dir,
                "xy_dir": section.xy_dir,
            }

            return model
