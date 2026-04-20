import io
from typing import Optional

import matplotlib.pyplot as plt
import polars as pl


class FSIDataVisualizer:
    def __init__(self, file_path: str):
        """
        Inicializa el visualizador con los datos del archivo.

        Args:
            file_path (str): Ruta al archivo de datos FSI
        """
        # Leer datos con Polars
        with open(file_path, "r") as f:
            content = f.read()
        csv_content = (
            content.replace("   ", ",").replace("  ", ",").replace(" ", ",").replace("\n,", "\n")
        )[1:]

        self.df = pl.read_csv(io.StringIO(csv_content), separator=",", has_header=True)

        # Detectar dimensionalidad (2D o 3D)
        self.dimensions = self._detect_dimensions()
        print(f"Datos cargados: {self.df.shape[0]} puntos temporales en {self.dimensions}D")

    def _detect_dimensions(self) -> int:
        """Detecta automáticamente si los datos son 2D o 3D."""
        coord_cols = [col for col in self.df.columns if "Coordinate" in col]
        return len(coord_cols)

    def plot_displacement(self, component: Optional[int] = None, save_path: Optional[str] = None):
        """
        Grafica los desplazamientos a lo largo del tiempo.

        Args:
            component (int, optional): Componente específico a graficar (0, 1, 2). Si None, grafica todas.
            save_path (str, optional): Ruta para guardar la figura. Si None, muestra la figura.
        """
        plt.figure(figsize=(10, 6))

        if component is not None:
            # Graficar componente específica
            plt.plot(
                self.df["Time"],
                self.df[f"Displacement{component}"],
                label=f"Componente {component}",
                linewidth=2,
            )
        else:
            # Graficar todas las componentes
            for dim in range(self.dimensions):
                plt.plot(
                    self.df["Time"],
                    self.df[f"Displacement{dim}"],
                    label=f"Dirección {dim}",
                    linewidth=2,
                )

        plt.title("Evolución temporal del desplazamiento")
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Desplazamiento [m]")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figura guardada en: {save_path}")
        else:
            plt.show()

    def plot_force(self, component: Optional[int] = None, save_path: Optional[str] = None):
        """
        Grafica las fuerzas a lo largo del tiempo.

        Args:
            component (int, optional): Componente específica a graficar (0, 1, 2). Si None, grafica todas.
            save_path (str, optional): Ruta para guardar la figura. Si None, muestra la figura.
        """
        plt.figure(figsize=(10, 6))

        if component is not None:
            # Graficar componente específica
            plt.plot(
                self.df["Time"],
                self.df[f"Force{component}"],
                label=f"Componente {component}",
                linewidth=2,
            )
        else:
            # Graficar todas las componentes
            for dim in range(self.dimensions):
                plt.plot(
                    self.df["Time"], self.df[f"Force{dim}"], label=f"Dirección {dim}", linewidth=2
                )

        plt.title("Evolución temporal de la fuerza")
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Fuerza [N]")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figura guardada en: {save_path}")
        else:
            plt.show()

    def plot_trajectory(self, save_path: Optional[str] = None):
        """
        Grafica la trayectoria en 2D/3D.

        Args:
            save_path (str, optional): Ruta para guardar la figura. Si None, muestra la figura.
        """
        fig = plt.figure(figsize=(10, 8))

        if self.dimensions == 2:
            ax = fig.add_subplot(111)
            ax.plot(
                self.df["Coordinate0"] + self.df["Displacement0"],
                self.df["Coordinate1"] + self.df["Displacement1"],
                color="red",
                linewidth=2,
                label="Trayectoria",
            )
            ax.scatter(
                self.df["Coordinate0"][0] + self.df["Displacement0"][0],
                self.df["Coordinate1"][0] + self.df["Displacement1"][0],
                color="green",
                s=100,
                label="Inicio",
            )
            ax.scatter(
                self.df["Coordinate0"][-1] + self.df["Displacement0"][-1],
                self.df["Coordinate1"][-1] + self.df["Displacement1"][-1],
                color="blue",
                s=100,
                label="Fin",
            )
            ax.set_xlabel("Coordenada X [m]")
            ax.set_ylabel("Coordenada Y [m]")

        elif self.dimensions == 3:
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(
                self.df["Coordinate0"] + self.df["Displacement0"],
                self.df["Coordinate1"] + self.df["Displacement1"],
                self.df["Coordinate2"] + self.df["Displacement2"],
                color="red",
                linewidth=2,
                label="Trayectoria",
            )
            ax.scatter(
                self.df["Coordinate0"][0] + self.df["Displacement0"][0],
                self.df["Coordinate1"][0] + self.df["Displacement1"][0],
                self.df["Coordinate2"][0] + self.df["Displacement2"][0],
                color="green",
                s=100,
                label="Inicio",
            )
            ax.scatter(
                self.df["Coordinate0"][-1] + self.df["Displacement0"][-1],
                self.df["Coordinate1"][-1] + self.df["Displacement1"][-1],
                self.df["Coordinate2"][-1] + self.df["Displacement2"][-1],
                color="blue",
                s=100,
                label="Fin",
            )
            ax.set_xlabel("Coordenada X [m]")
            ax.set_ylabel("Coordenada Y [m]")
            ax.set_zlabel("Coordenada Z [m]")

        plt.title("Trayectoria del punto de interés")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figura guardada en: {save_path}")
        else:
            plt.show()
