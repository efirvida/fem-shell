"""
Mesh module for fem_shell.

This module re-exports all mesh-related classes and functions from the mesh package
for backward compatibility. New code should import directly from the mesh subpackage.

Example
-------
Legacy import (still works):
>>> from fem_shell.core.mesh import MeshModel, Node

Preferred import:
>>> from fem_shell.core.mesh import MeshModel, Node
>>> # or for specific submodules:
>>> from fem_shell.core.mesh.generators import SquareShapeMesh
>>> from fem_shell.core.mesh.io import write_mesh
"""

# Re-export everything from the mesh package for backward compatibility
from fem_shell.core.mesh import (  # Entities; Model; Generators; I/O
    ELEMENT_NODES_MAP,
    BladeMesh,
    BoxSurfaceMesh,
    ElementSet,
    ElementType,
    MeshElement,
    MeshModel,
    MultiFlapMesh,
    Node,
    NodeSet,
    RotorMesh,
    SquareShapeMesh,
    load_hdf5,
    load_mesh,
    load_meshio,
    load_pickle,
    write_ccx_mesh,
    write_gmsh_mesh,
    write_hdf5,
    write_mesh,
    write_meshio,
    write_pickle,
    write_plot3d,
    selectors,
)

__all__ = [
    # Entities
    "Node",
    "MeshElement",
    "NodeSet",
    "ElementSet",
    "ElementType",
    "ELEMENT_NODES_MAP",
    # Model
    "MeshModel",
    # Generators
    "SquareShapeMesh",
    "BoxSurfaceMesh",
    "MultiFlapMesh",
    "BladeMesh",
    "RotorMesh",
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
    "selectors",
]


if __name__ == "__main__":
    # Example usage
    unit_mesh = BoxSurfaceMesh.create_box((0, 0, 5), (1, 1, 10), 5, 5, 50)
    unit_mesh.rotate_mesh((1, 0, 0), 20)
    unit_mesh.view()
