"""
Mesh I/O subpackage.

This package provides functions for reading and writing meshes in various formats.
"""

from fem_shell.core.mesh.io.readers import load_hdf5, load_mesh, load_meshio, load_pickle
from fem_shell.core.mesh.io.writers import (
    write_ccx_mesh,
    write_gmsh_mesh,
    write_hdf5,
    write_mesh,
    write_meshio,
    write_pickle,
    write_plot3d,
)

__all__ = [
    # Writers
    "write_mesh",
    "write_meshio",
    "write_plot3d",
    "write_ccx_mesh",
    "write_gmsh_mesh",
    "write_hdf5",
    "write_pickle",
    # Readers
    "load_mesh",
    "load_meshio",
    "load_hdf5",
    "load_pickle",
]
