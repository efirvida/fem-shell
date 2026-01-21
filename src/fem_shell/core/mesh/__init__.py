"""
Mesh package for fem_shell.

This package provides comprehensive mesh handling capabilities including:
- Mesh entities (Node, MeshElement, NodeSet, ElementSet)
- Mesh model (MeshModel)
- Mesh generators (SquareShapeMesh, BoxSurfaceMesh, MultiFlapMesh, BladeMesh, RotorMesh)
- I/O functions for various file formats

Usage
-----
>>> from fem_shell.core.mesh import MeshModel, Node, MeshElement
>>> from fem_shell.core.mesh import SquareShapeMesh, BladeMesh, RotorMesh

Creating a simple rectangular mesh:

>>> mesh = SquareShapeMesh.create_rectangle(width=1.0, height=2.0, nx=10, ny=20)
>>> mesh.view()

Creating a wind turbine blade mesh:

>>> blade_mesh = BladeMesh("blade_definition.yaml", element_size=0.05)
>>> mesh = blade_mesh.generate(renumber="rcm")

Creating a wind turbine rotor mesh:

>>> rotor_mesh = RotorMesh("blade_definition.yaml", n_blades=3, hub_radius=1.5)
>>> mesh = rotor_mesh.generate(renumber="rcm")

Loading and saving meshes:

>>> mesh.save("my_mesh.h5")
>>> loaded_mesh = MeshModel.load("my_mesh.h5")
"""

# Core entities
from fem_shell.core.mesh.entities import (
    ELEMENT_NODES_MAP,
    ElementSet,
    ElementType,
    MeshElement,
    Node,
    NodeSet,
)

# Mesh generators
from fem_shell.core.mesh.generators import (
    BladeMesh,
    BoxSurfaceMesh,
    CylindricalSurfaceMesh,
    HyperbolicParaboloidMesh,
    MultiFlapMesh,
    RaaschHookMesh,
    RotorMesh,
    SphericalSurfaceMesh,
    SquareShapeMesh,
)

# I/O functions
from fem_shell.core.mesh.io import (
    load_hdf5,
    load_mesh,
    load_pickle,
    write_ccx_mesh,
    write_gmsh_mesh,
    write_hdf5,
    write_mesh,
    write_meshio,
    write_pickle,
    write_plot3d,
)

# Main mesh model
from fem_shell.core.mesh.model import MeshModel

# Selection utilities
from fem_shell.core.mesh import selectors

# Utility functions
from fem_shell.core.mesh.utils import (
    detect_open_boundaries,
    close_open_boundaries,
    volumetric_remesh,
    boolean_union_meshes,
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
    "CylindricalSurfaceMesh",
    "HyperbolicParaboloidMesh",
    "RaaschHookMesh",
    "SphericalSurfaceMesh",
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
    "load_hdf5",
    "load_pickle",
    # Utilities
    "selectors",
    "detect_open_boundaries",
    "close_open_boundaries",
    "volumetric_remesh",
    "boolean_union_meshes",
]
