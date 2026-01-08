"""
Checkpoint Manager for Dynamic Solvers.

Provides disk-based checkpointing and restart capabilities for dynamic FEM solvers,
following OpenFOAM conventions for write intervals and time-based folder structure.

This module is designed to be solver-agnostic and works with numpy arrays,
allowing any solver (LinearDynamic, FSI, etc.) to use it via composition.

Escritura asíncrona: Los checkpoints se escriben en un thread separado para no
bloquear el bucle principal de la simulación, mejorando significativamente el
performance cuando la escritura es frecuente.
"""

import logging
import os
import re
import threading
from dataclasses import dataclass
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple

import meshio
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CheckpointInfo:
    """Information about a checkpoint on disk."""

    time: float
    time_step: int
    path: str


class AsyncCheckpointWriter:
    """
    Worker thread que maneja la escritura asíncrona de checkpoints.

    Permite que el solver continúe ejecutándose sin esperar a que se complete
    la escritura a disco. Los datos se encolan y se escriben en segundo plano.
    El archivo PVD se actualiza de forma incremental después de cada escritura
    para garantizar que esté al día incluso si la simulación se cancela.

    Parameters
    ----------
    mesh_obj : MeshModel
        Objeto de malla para exportar a VTU.
    vector_form : Dict[str, List[str]]
        Mapeo de nombres de campos vectoriales a componentes.
    dofs_per_node : int
        Grados de libertad por nodo.
    output_folder : str
        Carpeta base para los archivos de salida (para PVD).
    time_precision : int
        Decimales para nombres de carpetas de tiempo.
    save_deformed_mesh : bool
        Si True, guarda la malla deformada en formato HDF5.
    deformed_mesh_scale : float
        Factor de escala para los desplazamientos al guardar la malla deformada.
    """

    def __init__(
        self,
        mesh_obj: Any,
        vector_form: Dict[str, List[str]],
        dofs_per_node: int,
        output_folder: str,
        time_precision: int = 6,
        save_deformed_mesh: bool = True,
        deformed_mesh_scale: float = 1.0,
    ):
        self.mesh_obj = mesh_obj
        self.vector_form = vector_form
        self.dofs_per_node = dofs_per_node
        self.output_folder = output_folder
        self.time_precision = time_precision
        self.save_deformed_mesh = save_deformed_mesh
        self.deformed_mesh_scale = deformed_mesh_scale

        self._queue: Queue = Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=False)
        self._thread.start()
        self._written_times: List[Tuple[float, str]] = []
        self._lock = threading.Lock()

    def _worker(self) -> None:
        """
        Worker thread que procesa tareas de escritura de la cola.

        Se ejecuta continuamente hasta que se señale el evento de stop.
        """
        while not self._stop_event.is_set():
            try:
                # Timeout para permitir que el thread revise _stop_event regularmente
                task = self._queue.get(timeout=0.1)
                if task is None:  # Señal de parada
                    break

                self._execute_write_task(task)
                self._queue.task_done()

            except Exception:
                # Queue.Empty exception - timeout, continuar esperando
                continue
            except Exception as e:
                logger.error(f"Error in async checkpoint writer: {e}", exc_info=True)

    def _execute_write_task(self, task: Dict[str, Any]) -> None:
        """
        Ejecuta una tarea de escritura de checkpoint.

        Parameters
        ----------
        task : Dict[str, Any]
            Diccionario con información del checkpoint a escribir.
        """
        try:
            t = task["t"]
            time_step = task["time_step"]
            dt = task["dt"]
            state = task["state"]
            u_full = task["u_full"]
            v_full = task.get("v_full")
            a_full = task.get("a_full")
            checkpoint_dir = task["checkpoint_dir"]

            # Guardar estado para restart (NPZ)
            npz_path = os.path.join(checkpoint_dir, "state.npz")
            np.savez_compressed(
                npz_path,
                t=t,
                time_step=time_step,
                dt=dt,
                **state,
            )

            # Escribir VTU para visualización (con malla deformada)
            vtu_path = os.path.join(checkpoint_dir, "fields.vtu")
            self._write_vtu(vtu_path, u_full, v_full, a_full)

            # Guardar malla deformada en HDF5
            if self.save_deformed_mesh:
                mesh_path = os.path.join(checkpoint_dir, "deformed_mesh.h5")
                try:
                    self._write_deformed_mesh(mesh_path, u_full)
                except Exception as e:
                    logger.error(f"Failed to write deformed mesh: {e}", exc_info=True)
                    print(f"  ❌ ERROR writing deformed mesh: {e}", flush=True)

            # Registrar para PVD
            with self._lock:
                self._written_times.append((
                    t,
                    os.path.join(f"{t:.{self.time_precision}g}", "fields.vtu"),
                ))
                # Update PVD incrementally so it's always up-to-date even if simulation is cancelled
                self._update_pvd_incremental()

            logger.info(
                "Checkpoint written asynchronously: t=%.6f s (step %d) → %s",
                t,
                time_step,
                checkpoint_dir,
            )

        except Exception as e:
            logger.error(f"Failed to write checkpoint: {e}", exc_info=True)
            # Print to stdout for visibility even without logging configured
            print(f"  ❌ ERROR writing checkpoint: {e}", flush=True)

    def enqueue_write(
        self,
        t: float,
        time_step: int,
        dt: float,
        state: Dict[str, np.ndarray],
        u_full: np.ndarray,
        checkpoint_dir: str,
        v_full: Optional[np.ndarray] = None,
        a_full: Optional[np.ndarray] = None,
    ) -> None:
        """
        Encola una tarea de escritura de checkpoint.

        No bloquea - devuelve inmediatamente. La escritura ocurre en el thread worker.

        Parameters
        ----------
        t : float
            Tiempo actual.
        time_step : int
            Número del paso de tiempo.
        dt : float
            Tamaño del paso de tiempo.
        state : Dict[str, np.ndarray]
            Vectores de estado para restart.
        u_full : np.ndarray
            Vector de desplazamiento completo.
        checkpoint_dir : str
            Directorio donde escribir el checkpoint.
        v_full : np.ndarray, optional
            Vector de velocidad completo.
        a_full : np.ndarray, optional
            Vector de aceleración completo.
        """
        # Copiar arrays para evitar que se modifiquen durante la escritura
        task = {
            "t": float(t),
            "time_step": int(time_step),
            "dt": float(dt),
            "state": {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in state.items()},
            "u_full": u_full.copy(),
            "v_full": v_full.copy() if v_full is not None else None,
            "a_full": a_full.copy() if a_full is not None else None,
            "checkpoint_dir": checkpoint_dir,
        }
        self._queue.put(task)

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Espera a que se completen todas las escrituras pendientes.

        Bloquea hasta que la cola esté vacía.

        Parameters
        ----------
        timeout : float, optional
            Timeout en segundos. None para esperar indefinidamente.

        Returns
        -------
        bool
            True si todas las tareas se completaron, False si timeout.
        """
        try:
            self._queue.join()  # Espera a que task_done() se llame para todas las tareas
            return True
        except Exception:
            return False

    def shutdown(self, timeout: Optional[float] = 10.0) -> None:
        """
        Apaga el worker thread.

        Espera a que se completen las tareas pendientes y luego detiene el thread.

        Parameters
        ----------
        timeout : float, optional
            Timeout en segundos para esperar a que se completen las tareas.
        """
        logger.debug("Shutting down AsyncCheckpointWriter...")

        # Esperar a que se completen las tareas en cola
        try:
            self._queue.join()
        except Exception as e:
            logger.warning(f"Error waiting for queue completion: {e}")

        # Señalizar al worker que se detenga
        self._stop_event.set()

        # Esperar a que el thread termine
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("AsyncCheckpointWriter thread did not terminate within timeout")
        else:
            logger.debug("AsyncCheckpointWriter thread already terminated")

    def _update_pvd_incremental(self) -> None:
        """
        Update PVD file incrementally after each checkpoint write.

        Must be called while holding self._lock.
        """
        pvd_path = os.path.join(self.output_folder, "results.pvd")
        try:
            with open(pvd_path, "w") as f:
                f.write('<?xml version="1.0"?>\n')
                f.write('<VTKFile type="Collection" version="1.0">\n')
                f.write("  <Collection>\n")

                for t, vtu_rel_path in sorted(self._written_times, key=lambda x: x[0]):
                    f.write(f'    <DataSet timestep="{t}" part="0" file="{vtu_rel_path}"/>\n')

                f.write("  </Collection>\n")
                f.write("</VTKFile>\n")
        except Exception as e:
            logger.warning(f"Failed to update PVD file: {e}")

    def _write_vtu(
        self,
        vtu_path: str,
        u_full: np.ndarray,
        v_full: Optional[np.ndarray] = None,
        a_full: Optional[np.ndarray] = None,
    ) -> None:
        """Write VTU file with solution fields using deformed coordinates."""
        # Get vector components from vector_form
        vector_components = [comp for vec in self.vector_form.values() for comp in vec]
        n_components = len(vector_components)

        # Reshape to (n_nodes, n_dofs_per_node) and take relevant components
        U = u_full.reshape(-1, self.dofs_per_node)[:, :n_components]

        point_data: Dict[str, np.ndarray] = {}

        # Store individual components as scalar fields
        for i, comp in enumerate(vector_components):
            point_data[comp] = U[:, i]

        # Create vector fields and magnitudes
        for vec_name, components in self.vector_form.items():
            vec_data = np.column_stack([point_data[c] for c in components])

            # Ensure 3D for VTK compatibility
            if vec_data.shape[1] == 2:
                vec_data = np.hstack([vec_data, np.zeros((vec_data.shape[0], 1))])

            point_data[vec_name] = vec_data
            point_data[f"{vec_name}_magnitude"] = np.linalg.norm(vec_data, axis=1)

        # Add velocity if provided
        if v_full is not None:
            V = v_full.reshape(-1, self.dofs_per_node)[:, :n_components]
            if V.shape[1] == 2:
                V = np.hstack([V, np.zeros((V.shape[0], 1))])
            elif V.shape[1] < 3:
                V = np.hstack([V, np.zeros((V.shape[0], 3 - V.shape[1]))])
            point_data["Velocity"] = V[:, :3]
            point_data["Velocity_magnitude"] = np.linalg.norm(V[:, :3], axis=1)

        # Add acceleration if provided
        if a_full is not None:
            A = a_full.reshape(-1, self.dofs_per_node)[:, :n_components]
            if A.shape[1] == 2:
                A = np.hstack([A, np.zeros((A.shape[0], 1))])
            elif A.shape[1] < 3:
                A = np.hstack([A, np.zeros((A.shape[0], 3 - A.shape[1]))])
            point_data["Acceleration"] = A[:, :3]
            point_data["Acceleration_magnitude"] = np.linalg.norm(A[:, :3], axis=1)

        # Build cell connectivity
        cells_dict: Dict[str, List] = {}
        for element in self.mesh_obj.element_map.values():
            if element:
                cell_type = element.element_type.name
                if cell_type not in cells_dict:
                    cells_dict[cell_type] = []
                cells_dict[cell_type].append(element.node_ids)

        cells = [(ctype, np.array(conns)) for ctype, conns in cells_dict.items()]

        # Compute deformed coordinates for VTU visualization
        original_coords = self.mesh_obj.coords_array
        # Extract translational DOFs (first 3 components typically)
        n_nodes = original_coords.shape[0]
        uvw = U[:, :3] if U.shape[1] >= 3 else np.hstack([U, np.zeros((n_nodes, 3 - U.shape[1]))])
        deformed_coords = original_coords + uvw * self.deformed_mesh_scale

        # Create and write mesh with deformed coordinates
        mesh = meshio.Mesh(
            deformed_coords,
            cells,
            point_data=point_data,
        )
        mesh.write(vtu_path, file_format="vtu")

    def _write_deformed_mesh(self, mesh_path: str, u_full: np.ndarray) -> None:
        """
        Write deformed mesh to HDF5 format.

        Parameters
        ----------
        mesh_path : str
            Path to output HDF5 file.
        u_full : np.ndarray
            Full displacement vector.
        """
        # Create a deformed copy of the mesh without modifying the original
        deformed_mesh = self.mesh_obj.get_deformed_copy(
            u_full,
            scale=self.deformed_mesh_scale,
            dofs_per_node=self.dofs_per_node,
        )
        deformed_mesh.save(mesh_path, format="hdf5")

    def get_written_times(self) -> List[Tuple[float, str]]:
        """Obtiene la lista de tiempos escritos."""
        with self._lock:
            return list(self._written_times)

    def load_existing_times(self, times: List[Tuple[float, str]]) -> None:
        """
        Load existing checkpoint times from disk into the writer.

        This is used when resuming a simulation to include previously
        written checkpoints in the PVD file.

        Parameters
        ----------
        times : List[Tuple[float, str]]
            List of (time, vtu_relative_path) tuples from existing checkpoints.
        """
        with self._lock:
            # Merge existing times with any already written
            existing_set = {t for t, _ in self._written_times}
            for t, path in times:
                if t not in existing_set:
                    self._written_times.append((t, path))
            # Sort by time
            self._written_times.sort(key=lambda x: x[0])


class CheckpointManager:
    """
    Manages disk-based checkpointing for dynamic solvers.

    Writes converged timestep data to disk at specified intervals, enabling:
    - Simulation restart from last written checkpoint
    - VTU output for visualization/animation in ParaView
    - Memory optimization by not accumulating all timesteps in RAM

    Follows OpenFOAM conventions:
    - Folders named by physical time (e.g., "0.001000/")
    - Only converged timesteps are written
    - PVD file for time series visualization

    Utiliza escritura asíncrona: los checkpoints se escriben en un thread separado
    para no bloquear el bucle principal de la simulación.

    Parameters
    ----------
    output_folder : str
        Base directory for output files.
    write_interval : float
        Write checkpoint every N seconds of physical time. 0 = disabled.
    mesh_obj : MeshModel
        Mesh object for VTU export (coordinates and connectivity).
    vector_form : Dict[str, List[str]]
        Mapping of vector field names to component names.
        Example: {"Displacement": ["Ux", "Uy", "Uz"]}
    dofs_per_node : int
        Degrees of freedom per node (e.g., 6 for shells).
    time_precision : int
        Decimal places for time folder names (default: 6).
    async_write : bool
        Enable asynchronous (non-blocking) checkpoint writing. Default: True.
    save_deformed_mesh : bool
        If True, saves the deformed mesh in HDF5 format. Default: True.
    deformed_mesh_scale : float
        Scale factor for displacements when saving deformed mesh. Default: 1.0.
    """

    def __init__(
        self,
        output_folder: str,
        write_interval: float,
        mesh_obj: Any,
        vector_form: Dict[str, List[str]],
        dofs_per_node: int,
        time_precision: int = 6,
        async_write: bool = True,
        save_deformed_mesh: bool = True,
        deformed_mesh_scale: float = 1.0,
        write_initial_state: bool = True,
    ):
        self.output_folder = output_folder
        self.write_interval = write_interval
        self.mesh_obj = mesh_obj
        self.vector_form = vector_form
        self.dofs_per_node = dofs_per_node
        self.time_precision = time_precision
        self.async_write = async_write
        self.save_deformed_mesh = save_deformed_mesh
        self.deformed_mesh_scale = deformed_mesh_scale
        self.write_initial_state = write_initial_state

        # Track written checkpoints for PVD
        self._written_times: List[Tuple[float, str]] = []

        # Initialize async writer if enabled
        self._async_writer: Optional[AsyncCheckpointWriter] = None
        if self.async_write:
            self._async_writer = AsyncCheckpointWriter(
                mesh_obj=mesh_obj,
                vector_form=vector_form,
                dofs_per_node=dofs_per_node,
                output_folder=output_folder,
                time_precision=time_precision,
                save_deformed_mesh=save_deformed_mesh,
                deformed_mesh_scale=deformed_mesh_scale,
            )

        # Ensure output folder exists
        if self.write_interval > 0 or self.write_initial_state:
            os.makedirs(self.output_folder, exist_ok=True)

    def should_write(self, t: float) -> bool:
        """
        Check if checkpoint should be written at this time.

        Uses the same logic as OpenFOAM's writeControl adjustableRunTime:
        writes at multiples of write_interval.

        Parameters
        ----------
        t : float
            Current physical time.

        Returns
        -------
        bool
            True if checkpoint should be written.
        """
        if self.write_initial_state and t == 0:
            return True

        if self.write_interval <= 0:
            return False

        # Skip t=0 (initial condition) unless explicitly requested above
        if t <= 0:
            return False

        # Check if t is a multiple of write_interval (with tolerance for floating point)
        # Use a relative tolerance that scales with the time value to handle accumulated errors
        # This ensures we write at 0.1, 0.2, 0.3, ... not 0.001, 0.101, 0.201, ...
        n_intervals = t / self.write_interval
        n_intervals_rounded = round(n_intervals)
        
        # Use relative tolerance that grows with time magnitude
        # For t=27.0, write_interval=0.1: tolerance = max(1e-6, 27.0 * 1e-9) = 2.7e-8
        tolerance = max(1e-6, abs(t) * 1e-9)
        
        is_multiple = abs(n_intervals - n_intervals_rounded) < tolerance

        return is_multiple

    def write(
        self,
        t: float,
        time_step: int,
        dt: float,
        state: Dict[str, np.ndarray],
        u_full: np.ndarray,
        v_full: Optional[np.ndarray] = None,
        a_full: Optional[np.ndarray] = None,
    ) -> str:
        """
        Write checkpoint to disk.

        Si async_write=True, esta operación no bloquea y devuelve inmediatamente.
        La escritura ocurre en un thread separado en segundo plano.

        Parameters
        ----------
        t : float
            Current physical time.
        time_step : int
            Current timestep number.
        dt : float
            Time step size.
        state : Dict[str, np.ndarray]
            State arrays to save for restart. Keys should include:
            - "u_red": reduced displacement vector
            - "v_red": reduced velocity vector
            - "a_red": reduced acceleration vector
            Any additional keys will be saved as well.
        u_full : np.ndarray
            Full (expanded) displacement vector for VTU export.
        v_full : np.ndarray, optional
            Full velocity vector for VTU export.
        a_full : np.ndarray, optional
            Full acceleration vector for VTU export.

        Returns
        -------
        str
            Path to the checkpoint folder.
        """
        # Create time folder - format like OpenFOAM with timeFormat general
        # Use 'g' format to remove trailing zeros (0.001 instead of 0.001000)
        time_str = f"{t:.{self.time_precision}g}"
        checkpoint_dir = os.path.join(self.output_folder, time_str)
        os.makedirs(checkpoint_dir, exist_ok=True)

        if self.async_write and self._async_writer is not None:
            # Enqueue for asynchronous writing - non-blocking
            self._async_writer.enqueue_write(
                t=t,
                time_step=time_step,
                dt=dt,
                state=state,
                u_full=u_full,
                checkpoint_dir=checkpoint_dir,
                v_full=v_full,
                a_full=a_full,
            )
        else:
            # Synchronous writing (original behavior)
            # Save state for restart (NPZ format)
            npz_path = os.path.join(checkpoint_dir, "state.npz")
            np.savez_compressed(
                npz_path,
                t=t,
                time_step=time_step,
                dt=dt,
                **state,
            )

            # Write VTU for visualization (with deformed coordinates)
            vtu_path = os.path.join(checkpoint_dir, "fields.vtu")
            self._write_vtu(vtu_path, u_full, v_full, a_full)

            # Save deformed mesh in HDF5
            if self.save_deformed_mesh:
                mesh_path = os.path.join(checkpoint_dir, "deformed_mesh.h5")
                self._write_deformed_mesh(mesh_path, u_full)

            # Track for PVD
            self._written_times.append((t, os.path.join(time_str, "fields.vtu")))

            # Update PVD file (only for sync mode)
            self._update_pvd()

        logger.info("Checkpoint enqueued: t=%.6f s (step %d) → %s", t, time_step, checkpoint_dir)
        if not self.async_write:
            print(f"  📁 Checkpoint saved: {checkpoint_dir}", flush=True)
        else:
            print(f"  📁 Checkpoint queued for async write: {checkpoint_dir}", flush=True)

        return checkpoint_dir

    def find_latest(self) -> Optional[CheckpointInfo]:
        """
        Find the latest checkpoint in the output folder.

        Scans for time-named folders and returns info about the most recent one.

        Returns
        -------
        CheckpointInfo or None
            Information about the latest checkpoint, or None if none found.
        """
        if not os.path.exists(self.output_folder):
            return None

        # Pattern for time folders (e.g., "0", "0.001000", "1.234567")
        time_pattern = re.compile(r"^(\d+(?:\.\d+)?)$")

        latest_time = -1.0
        latest_info = None

        for entry in os.listdir(self.output_folder):
            entry_path = os.path.join(self.output_folder, entry)
            if not os.path.isdir(entry_path):
                continue

            match = time_pattern.match(entry)
            if not match:
                continue

            # Check if state.npz exists
            npz_path = os.path.join(entry_path, "state.npz")
            if not os.path.exists(npz_path):
                continue

            folder_time = float(match.group(1))
            if folder_time > latest_time:
                latest_time = folder_time

                # Load minimal info to get time_step
                try:
                    with np.load(npz_path) as data:
                        time_step = int(data["time_step"])
                    latest_info = CheckpointInfo(
                        time=folder_time,
                        time_step=time_step,
                        path=entry_path,
                    )
                except Exception as e:
                    logger.warning("Failed to read checkpoint %s: %s", npz_path, e)
                    continue

        return latest_info

    def find_first(self) -> Optional[CheckpointInfo]:
        """
        Find the earliest checkpoint in the output folder.

        Returns
        -------
        CheckpointInfo or None
            Information about the earliest checkpoint, or None if none found.
        """
        if not os.path.exists(self.output_folder):
            return None

        time_pattern = re.compile(r"^(\d+(?:\.\d+)?)$")

        first_time = float("inf")
        first_info = None

        for entry in os.listdir(self.output_folder):
            entry_path = os.path.join(self.output_folder, entry)
            if not os.path.isdir(entry_path):
                continue

            match = time_pattern.match(entry)
            if not match:
                continue

            npz_path = os.path.join(entry_path, "state.npz")
            if not os.path.exists(npz_path):
                continue

            folder_time = float(match.group(1))
            if folder_time < first_time:
                try:
                    with np.load(npz_path) as data:
                        time_step = int(data["time_step"])
                    first_time = folder_time
                    first_info = CheckpointInfo(
                        time=folder_time,
                        time_step=time_step,
                        path=entry_path,
                    )
                except Exception as e:
                    logger.warning("Failed to read checkpoint %s: %s", npz_path, e)
                    continue

        return first_info

    def load(self, path: str) -> Dict[str, Any]:
        """
        Load checkpoint data from disk.

        Parameters
        ----------
        path : str
            Path to checkpoint folder.

        Returns
        -------
        dict
            Dictionary containing:
            - "t": physical time
            - "time_step": timestep number
            - "dt": time step size
            - "u_red": reduced displacement
            - "v_red": reduced velocity
            - "a_red": reduced acceleration
            Plus any additional arrays stored in the checkpoint.
        """
        npz_path = os.path.join(path, "state.npz")

        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Checkpoint not found: {npz_path}")

        with np.load(npz_path) as data:
            result = {key: data[key] for key in data.files}

        # Convert scalar arrays to Python types
        for key in ["t", "time_step", "dt"]:
            if key in result and result[key].ndim == 0:
                result[key] = result[key].item()

        logger.info(
            "Checkpoint loaded: t=%.6f s (step %d) from %s",
            result["t"],
            result["time_step"],
            path,
        )

        return result

    def _write_vtu(
        self,
        vtu_path: str,
        u_full: np.ndarray,
        v_full: Optional[np.ndarray] = None,
        a_full: Optional[np.ndarray] = None,
    ) -> None:
        """Write VTU file with solution fields (synchronous mode)."""
        # Get vector components from vector_form
        vector_components = [comp for vec in self.vector_form.values() for comp in vec]
        n_components = len(vector_components)

        # Reshape to (n_nodes, n_dofs_per_node) and take relevant components
        U = u_full.reshape(-1, self.dofs_per_node)[:, :n_components]

        point_data: Dict[str, np.ndarray] = {}

        # Store individual components as scalar fields
        for i, comp in enumerate(vector_components):
            point_data[comp] = U[:, i]

        # Create vector fields and magnitudes
        for vec_name, components in self.vector_form.items():
            vec_data = np.column_stack([point_data[c] for c in components])

            # Ensure 3D for VTK compatibility
            if vec_data.shape[1] == 2:
                vec_data = np.hstack([vec_data, np.zeros((vec_data.shape[0], 1))])

            point_data[vec_name] = vec_data
            point_data[f"{vec_name}_magnitude"] = np.linalg.norm(vec_data, axis=1)

        # Add velocity if provided
        if v_full is not None:
            V = v_full.reshape(-1, self.dofs_per_node)[:, :n_components]
            if V.shape[1] == 2:
                V = np.hstack([V, np.zeros((V.shape[0], 1))])
            elif V.shape[1] < 3:
                V = np.hstack([V, np.zeros((V.shape[0], 3 - V.shape[1]))])
            point_data["Velocity"] = V[:, :3]
            point_data["Velocity_magnitude"] = np.linalg.norm(V[:, :3], axis=1)

        # Add acceleration if provided
        if a_full is not None:
            A = a_full.reshape(-1, self.dofs_per_node)[:, :n_components]
            if A.shape[1] == 2:
                A = np.hstack([A, np.zeros((A.shape[0], 1))])
            elif A.shape[1] < 3:
                A = np.hstack([A, np.zeros((A.shape[0], 3 - A.shape[1]))])
            point_data["Acceleration"] = A[:, :3]
            point_data["Acceleration_magnitude"] = np.linalg.norm(A[:, :3], axis=1)

        # Build cell connectivity
        cells_dict: Dict[str, List] = {}
        for element in self.mesh_obj.element_map.values():
            if element:
                cell_type = element.element_type.name
                if cell_type not in cells_dict:
                    cells_dict[cell_type] = []
                cells_dict[cell_type].append(element.node_ids)

        cells = [(ctype, np.array(conns)) for ctype, conns in cells_dict.items()]

        # Compute deformed coordinates for VTU visualization
        original_coords = self.mesh_obj.coords_array
        n_nodes = original_coords.shape[0]
        uvw = U[:, :3] if U.shape[1] >= 3 else np.hstack([U, np.zeros((n_nodes, 3 - U.shape[1]))])
        deformed_coords = original_coords + uvw * self.deformed_mesh_scale

        # Create and write mesh with deformed coordinates
        mesh = meshio.Mesh(
            deformed_coords,
            cells,
            point_data=point_data,
        )
        mesh.write(vtu_path, file_format="vtu")

    def _write_deformed_mesh(self, mesh_path: str, u_full: np.ndarray) -> None:
        """
        Write deformed mesh to HDF5 format (synchronous mode).

        Parameters
        ----------
        mesh_path : str
            Path to output HDF5 file.
        u_full : np.ndarray
            Full displacement vector.
        """
        deformed_mesh = self.mesh_obj.get_deformed_copy(
            u_full,
            scale=self.deformed_mesh_scale,
            dofs_per_node=self.dofs_per_node,
        )
        deformed_mesh.save(mesh_path, format="hdf5")

    def _update_pvd(self) -> None:
        """Update PVD file with all written timesteps."""
        # Si usamos escritura asíncrona, reconstruir desde disco
        if self.async_write and self._async_writer is not None:
            self._written_times = self._async_writer.get_written_times()

        pvd_path = os.path.join(self.output_folder, "results.pvd")

        with open(pvd_path, "w") as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="Collection" version="1.0">\n')
            f.write("  <Collection>\n")

            for t, vtu_rel_path in sorted(self._written_times, key=lambda x: x[0]):
                f.write(f'    <DataSet timestep="{t}" part="0" file="{vtu_rel_path}"/>\n')

            f.write("  </Collection>\n")
            f.write("</VTKFile>\n")

    def rebuild_pvd_from_disk(self) -> None:
        """
        Rebuild PVD file by scanning existing checkpoint folders.

        Useful after restart to include all previously written timesteps.
        Also updates the async writer's internal list if using async mode.
        """
        if not os.path.exists(self.output_folder):
            return

        # Pattern matches both integer (0, 1, 2) and decimal (0.1, 0.001) time folders
        time_pattern = re.compile(r"^(\d+(?:\.\d+)?)$")
        self._written_times = []

        for entry in os.listdir(self.output_folder):
            entry_path = os.path.join(self.output_folder, entry)
            if not os.path.isdir(entry_path):
                continue

            match = time_pattern.match(entry)
            if not match:
                continue

            vtu_path = os.path.join(entry_path, "fields.vtu")
            if os.path.exists(vtu_path):
                t = float(match.group(1))
                self._written_times.append((t, os.path.join(entry, "fields.vtu")))

        if self._written_times:
            # Update async writer's list so new writes merge correctly
            if self._async_writer is not None:
                self._async_writer.load_existing_times(self._written_times)
            self._update_pvd()
            logger.info("PVD rebuilt with %d timesteps", len(self._written_times))

    def finalize(self, timeout: Optional[float] = None) -> None:
        """
        Finaliza el gestor de checkpoints, esperando a que se completen todas las escrituras pendientes.

        Debe llamarse al final de la simulación para asegurar que todos los checkpoints
        se escriban completamente a disco y que el archivo PVD se actualice correctamente.

        Parameters
        ----------
        timeout : float, optional
            Timeout en segundos para esperar a que se completen las escrituras.
            Si es None, espera indefinidamente.
        """
        if self._async_writer is not None:
            logger.info("Waiting for asynchronous checkpoint writes to complete...")
            self._async_writer.wait_for_completion(timeout=timeout)

            # Actualizar PVD con todos los tiempos escritos
            self._written_times = self._async_writer.get_written_times()
            self._update_pvd()

            self._async_writer.shutdown(timeout=timeout)
            logger.info("Asynchronous checkpoint writing completed")

    @property
    def written_count(self) -> int:
        """Number of checkpoints written."""
        return len(self._written_times)

    @property
    def written_times(self) -> List[float]:
        """List of times that have been written to disk."""
        return [t for t, _ in self._written_times]
