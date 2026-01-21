"""
Beam Analysis Example: Flexion, Torsion, Tension, and Compression
==================================================================

This example demonstrates the behavior of 3D solid elements under
different loading conditions using the LinearStaticSolver.

Loading cases:
1. Tension (axial load along beam axis)
2. Compression (axial load opposite to beam axis)
3. Bending/Flexion (transverse load at tip)
4. Torsion (moment about beam axis)

Element types compared:
- HEXA8 (8-node linear hexahedron) - structured mesh
- TETRA4 (4-node linear tetrahedron) - structured mesh
- HEXA20 (20-node quadratic hexahedron) - structured mesh
- TETRA10 (10-node quadratic tetrahedron) - structured mesh
- PYRAMID (mixed tet+pyramid mesh via Gmsh)

Results include comparison with analytical solutions:
- Tension/Compression: δ = PL/(AE)
- Bending (cantilever): δ = PL³/(3EI)
- Torsion (rectangular): θ = TL/(GJ), J ≈ 0.1406·a⁴ for square section
  Note: Saint-Venant formula assumes free warping. With clamped BC, actual
  displacement is ~30% lower due to warping restraint. This is expected behavior.

Author: FEM-Shell Project
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add src to path if running from examples folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from fem_shell.core.mesh.generators import BoxVolumeMesh, MixedElementBeamMesh
from fem_shell.core.mesh.entities import Node, MeshElement, ElementType
from fem_shell.core.material import IsotropicMaterial
from fem_shell.core.bc import DirichletCondition
from fem_shell.elements import ElementFamily
from fem_shell.solvers import LinearStaticSolver

try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False


# =============================================================================
# Beam Parameters
# =============================================================================

# Geometry
LENGTH = 10.0  # Beam length (X direction)
WIDTH = 1.0    # Cross-section width (Y direction)
HEIGHT = 1.0   # Cross-section height (Z direction)
RADIUS = 0.5   # Radius for circular cross-section beam

# Mesh density (refined for better bending/torsion accuracy)
NX = 40  # Elements along length (was 16)
NY = 6   # Elements across width (was 3)
NZ = 6   # Elements across height (was 3)

# Material properties (Steel)
E = 210e9      # Young's modulus [Pa]
NU = 0.3       # Poisson's ratio
RHO = 7850.0   # Density [kg/m³]

# =============================================================================
# Element types by cross-section
# =============================================================================

# Rectangular cross-section beam element types
RECTANGULAR_ELEMENT_TYPES = [
    'hex',        # 8-node hexahedron (structured)
    'tet',        # 4-node tetrahedron (structured)
    'pyramid',    # Linear pyramid mesh (Gmsh)
    'hex_q',      # 20-node quadratic hexahedron (structured)
    'tet_q',      # 10-node quadratic tetrahedron (structured)
    'pyramid_q',  # Quadratic pyramid mesh (Gmsh)
    'mixed',      # Mixed elements: hex, tet, pyramid, wedge (Gmsh)
]

# Circular cross-section beam element types
CIRCULAR_ELEMENT_TYPES = RECTANGULAR_ELEMENT_TYPES


# =============================================================================
# Mesh Generation Functions
# =============================================================================

def create_beam_mesh(mesh_type='hex', cross_section='rectangular', refinement=1):
    """
    Create a 3D beam mesh.
    
    Parameters
    ----------
    mesh_type : str
        For rectangular section:
            'hex', 'tet' - structured meshes (linear)
            'hex_q', 'tet_q' - structured meshes (quadratic/second order)
            'pyramid' - mesh with pyramid transition elements (linear)
            'pyramid_q' - mesh with quadratic pyramid elements
            'mixed' - unstructured mixed element mesh (hex, tet, pyramid, wedge)
        For circular section:
            'tet' - tetrahedral mesh (Gmsh)
            'tet_q' - quadratic tetrahedral mesh (Gmsh)
    cross_section : str
        'rectangular' or 'circular'
    refinement : int
        Mesh refinement factor (1=base, 2=2x finer, etc.)
        
    Returns
    -------
    mesh : MeshModel
    """
    # Reset node IDs for consistent results
    Node._id_counter = 0
    MeshElement._id_counter = 0
    
    # Apply refinement factor
    nx = NX * refinement
    ny = NY * refinement
    nz = NZ * refinement
    
    if cross_section == 'circular':
        # Circular cross-section always uses Gmsh
        return _create_circular_beam_mesh(quadratic=(mesh_type == 'tet_q'), refinement=refinement)
    
    # Rectangular cross-section
    if mesh_type in ['hex', 'tet']:
        return _create_structured_mesh(mesh_type, quadratic=False, nx=nx, ny=ny, nz=nz)
    elif mesh_type in ['hex_q', 'tet_q']:
        base_type = mesh_type.replace('_q', '')
        # Use coarser mesh for quadratic elements (more nodes per element)
        return _create_structured_mesh(base_type, quadratic=True, nx=20*refinement, ny=3*refinement, nz=3*refinement)
    elif mesh_type == 'pyramid':
        return _create_pyramid_mesh(quadratic=False, refinement=refinement)
    elif mesh_type == 'pyramid_q':
        return _create_pyramid_mesh(quadratic=True, refinement=refinement)
    elif mesh_type == 'mixed':
        return _create_mixed_mesh(refinement=refinement)
    else:
        raise ValueError(f"Unknown mesh type: {mesh_type}")


def _create_structured_mesh(element_type, quadratic=False, nx=None, ny=None, nz=None):
    """Create structured mesh with BoxVolumeMesh."""
    center = (LENGTH / 2, WIDTH / 2, HEIGHT / 2)
    dims = (LENGTH, WIDTH, HEIGHT)
    
    # Use global defaults if not specified
    if nx is None:
        nx = NX
    if ny is None:
        ny = NY
    if nz is None:
        nz = NZ
    
    mesh = BoxVolumeMesh(
        center=center,
        dims=dims,
        nx=nx,
        ny=ny,
        nz=nz,
        element_type=element_type,
        quadratic=quadratic,
    )
    
    return mesh.generate()


def _create_pyramid_mesh(quadratic=False, refinement=1):
    """Create a beam mesh using pyramid elements via Gmsh.
    
    Uses Gmsh to generate a mesh that includes pyramid elements by
    creating a hex mesh and then converting some elements to pyramids.
    
    Parameters
    ----------
    quadratic : bool
        If True, generate quadratic (second-order) elements
    refinement : int
        Mesh refinement factor (1=base, 2=2x finer, etc.)
    """
    if not GMSH_AVAILABLE:
        raise ImportError("Gmsh is required for pyramid meshes")
    
    from fem_shell.core.mesh.model import MeshModel
    
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("beam_pyramid")
        
        # Create box geometry
        box = gmsh.model.occ.addBox(0, 0, 0, LENGTH, WIDTH, HEIGHT)
        gmsh.model.occ.synchronize()
        
        # Set mesh size (coarser for quadratic elements, refined by factor)
        scale = 2.0 if quadratic else 1.0
        nx, ny, nz = NX * refinement, NY * refinement, NZ * refinement
        mesh_size = min(LENGTH/nx, WIDTH/ny, HEIGHT/nz) * scale
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.8)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size * 1.2)
        
        # 2D: Create quad mesh first
        gmsh.option.setNumber("Mesh.RecombineAll", 1)  # Recombine into quads
        gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal-Delaunay for quads
        
        # 3D: Hybrid mesh with pyramids for hex-tet transition
        # This creates hex-dominant mesh with pyramids at transitions
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
        gmsh.option.setNumber("Mesh.Recombine3DAll", 0)  # Don't recombine to all hex
        gmsh.option.setNumber("Mesh.Recombine3DConformity", 2)  # pyramids+trihedra
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 0)  # No subdivision
        
        # Set element order
        if quadratic:
            gmsh.option.setNumber("Mesh.ElementOrder", 2)
            # Use incomplete elements (13-node pyramids, not 14-node)
            gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
        
        gmsh.model.mesh.generate(3)
        
        # Extract mesh
        return _extract_gmsh_mesh(MeshModel)
        
    finally:
        gmsh.finalize()


def _create_circular_beam_mesh(quadratic=False, refinement=1):
    """Create a circular cross-section beam mesh using Gmsh.
    
    Creates a cylinder along the X-axis with tetrahedral elements.
    
    Parameters
    ----------
    quadratic : bool
        If True, create quadratic (10-node) tetrahedra
    refinement : int
        Mesh refinement factor (1=base, 2=2x finer, etc.)
    """
    if not GMSH_AVAILABLE:
        raise ImportError("Gmsh is required for circular beam meshes")
    
    from fem_shell.core.mesh.model import MeshModel
    
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("beam_circular")
        
        # Create cylinder along X-axis
        # Gmsh cylinder: center of first face, axis direction, radius, length
        cylinder = gmsh.model.occ.addCylinder(0, 0, 0, LENGTH, 0, 0, RADIUS)
        gmsh.model.occ.synchronize()
        
        # Set mesh size (coarser for quadratic elements, refined by factor)
        nx, ny = NX * refinement, NY * refinement
        if quadratic:
            mesh_size = min(LENGTH/(nx//2), 2*RADIUS/(ny//2))
        else:
            mesh_size = min(LENGTH/nx, 2*RADIUS/ny)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.8)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size * 1.2)
        
        # Use tetrahedral mesh
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        
        # Set element order
        if quadratic:
            gmsh.option.setNumber("Mesh.ElementOrder", 2)
        else:
            gmsh.option.setNumber("Mesh.ElementOrder", 1)
        
        gmsh.model.mesh.generate(3)
        
        # Extract mesh
        return _extract_gmsh_mesh(MeshModel)
        
    finally:
        gmsh.finalize()


def _create_mixed_mesh(refinement=1):
    """Create an unstructured mixed element mesh using Gmsh.
    
    Creates a mesh with hexahedra, tetrahedra, pyramids, and wedges.
    
    Parameters
    ----------
    refinement : int
        Mesh refinement factor (1=base, 2=2x finer, etc.)
    """
    if not GMSH_AVAILABLE:
        raise ImportError("Gmsh is required for mixed meshes")
    
    from fem_shell.core.mesh.model import MeshModel
    
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("beam_mixed")
        
        # Create box geometry
        box = gmsh.model.occ.addBox(0, 0, 0, LENGTH, WIDTH, HEIGHT)
        gmsh.model.occ.synchronize()
        
        # Set mesh size with some variation (reduced for better accuracy)
        nx, ny, nz = NX * refinement, NY * refinement, NZ * refinement
        mesh_size = min(LENGTH/nx, WIDTH/ny, HEIGHT/nz) * 0.8
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.5)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size * 1.5)
        
        # Add some randomization for unstructured feel
        gmsh.option.setNumber("Mesh.RandomFactor", 1e-3)
        
        # Use algorithm that creates quad-dominant 2D mesh
        gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal-Delaunay for quads
        
        # Partial recombination: use recombination with threshold
        gmsh.option.setNumber("Mesh.RecombineAll", 1)  # Try to recombine
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 0)  # Simple
        
        # 3D: Hybrid mesh algorithm with pyramids for hex-tet transition
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
        gmsh.option.setNumber("Mesh.Recombine3DAll", 0)  # Not all to hex
        gmsh.option.setNumber("Mesh.Recombine3DConformity", 2)  # pyramids+trihedra
        
        gmsh.model.mesh.generate(3)
        
        # Extract mesh
        return _extract_gmsh_mesh(MeshModel)
        
    finally:
        gmsh.finalize()


def _extract_gmsh_mesh(MeshModelClass):
    """Extract mesh from Gmsh into MeshModel, supporting all element types."""
    mesh_model = MeshModelClass()
    
    # Get nodes
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = coords.reshape(-1, 3)
    
    node_map = {}
    for i, tag in enumerate(node_tags):
        node = Node(coords[i], geometric_node=True)
        mesh_model.add_node(node)
        node_map[tag] = node
    
    # Gmsh element type to our ElementType mapping
    GMSH_TO_ELEMENT_TYPE = {
        4: ElementType.tetra,      # 4-node tetrahedron
        5: ElementType.hexahedron, # 8-node hexahedron
        6: ElementType.wedge,      # 6-node prism
        7: ElementType.pyramid,    # 5-node pyramid
        11: ElementType.tetra10,   # 10-node tetrahedron
        12: ElementType.hexahedron20,  # 20-node hexahedron
        13: ElementType.wedge15,   # 15-node prism
        19: ElementType.pyramid13, # 13-node pyramid (Gmsh type 19, not 14!)
    }
    
    # Get 3D elements
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=3)
    
    for etype, etags, enodes in zip(elem_types, elem_tags, elem_node_tags):
        elem_name, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(etype)
        nodes_per_elem = num_nodes
        
        our_elem_type = GMSH_TO_ELEMENT_TYPE.get(etype)
        if our_elem_type is None:
            print(f"  Warning: Skipping unknown Gmsh element type {etype} ({elem_name})")
            continue
            
        for i, tag in enumerate(etags):
            elem_nodes = [node_map[n] for n in enodes[i*nodes_per_elem:(i+1)*nodes_per_elem]]
            element = MeshElement(elem_nodes, our_elem_type)
            mesh_model.add_element(element)
    
    return mesh_model


def create_model_properties(material):
    """
    Create the fem_model_properties dictionary for LinearStaticSolver.
    
    Parameters
    ----------
    material : IsotropicMaterial
        Material properties
        
    Returns
    -------
    dict : Model properties configuration
    """
    return {
        "solver": {
            "output_folder": "output",
        },
        "elements": {
            "material": material,
            "element_family": ElementFamily.SOLID,
            # HEXA8 now uses full integration by default (reduced_integration=False)
            # which avoids hourglass modes without requiring hourglass control
        }
    }


def get_fixed_face_dofs(solver):
    """
    Get DOFs for the fixed face (x=0) of the cantilever beam.
    
    Parameters
    ----------
    solver : LinearStaticSolver
        The solver instance
        
    Returns
    -------
    set : Set of DOFs to fix
    """
    # Find nodes at x_min manually (node sets from Gmsh may be incorrect)
    mesh = solver.mesh_obj
    nodes = list(mesh.nodes)
    x_coords = [n.coords[0] for n in nodes]
    x_min = min(x_coords)
    tol = 1e-6
    
    dofs = set()
    dofs_per_node = solver.domain.dofs_per_node
    node_id_to_index = mesh.node_id_to_index
    
    for node in nodes:
        if abs(node.coords[0] - x_min) < tol:
            idx = node_id_to_index[node.id]
            for d in range(dofs_per_node):
                dofs.add(idx * dofs_per_node + d)
    
    return dofs


def get_tip_node_ids(solver):
    """
    Get node IDs at the beam tip (x=L).
    
    Parameters
    ----------
    solver : LinearStaticSolver
        The solver instance
        
    Returns
    -------
    list : List of node IDs at the tip
    """
    # Find nodes at x_max manually (node sets from Gmsh may be incorrect)
    mesh = solver.mesh_obj
    nodes = list(mesh.nodes)
    x_coords = [n.coords[0] for n in nodes]
    x_max = max(x_coords)
    tol = 1e-6
    
    tip_nodes = []
    for node in nodes:
        if abs(node.coords[0] - x_max) < tol:
            tip_nodes.append(node.id)
    
    return tip_nodes


def get_tip_dofs(solver, direction='x'):
    """
    Get DOFs at the beam tip for a specific direction.
    
    Parameters
    ----------
    solver : LinearStaticSolver
    direction : str
        'x', 'y', or 'z'
        
    Returns
    -------
    list : List of DOFs
    """
    tip_node_ids = get_tip_node_ids(solver)
    dofs_per_node = solver.domain.dofs_per_node
    node_id_to_index = solver.mesh_obj.node_id_to_index
    
    dir_map = {'x': 0, 'y': 1, 'z': 2}
    offset = dir_map.get(direction, 0)
    
    dofs = []
    for node_id in tip_node_ids:
        idx = node_id_to_index[node_id]
        dofs.append(idx * dofs_per_node + offset)
    
    return dofs


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_beam(element_type, load_case, material, total_force=1000.0, cross_section='rectangular', refinement=1):
    """
    Analyze beam under specified loading using LinearStaticSolver.
    
    Parameters
    ----------
    element_type : str
        Element/mesh type
    load_case : str
        'tension', 'compression', 'bending_z', 'bending_y', 'torsion'
    material : IsotropicMaterial
    total_force : float
        Total applied force [N]
    cross_section : str
        'rectangular' or 'circular'
    refinement : int
        Mesh refinement factor (1=base, 2=2x finer, etc.)
        
    Returns
    -------
    dict : Results dictionary
    """
    # Create mesh
    mesh = create_beam_mesh(element_type, cross_section=cross_section, refinement=refinement)
    
    # Create model properties
    model_props = create_model_properties(material)
    
    # Create solver
    solver = LinearStaticSolver(mesh, model_props)
    
    # Apply fixed boundary condition at x=0
    fixed_dofs = get_fixed_face_dofs(solver)
    dirichlet_bc = DirichletCondition(dofs=fixed_dofs, value=0.0)
    solver.add_dirichlet_conditions([dirichlet_bc])
    
    # Get tip node info
    tip_node_ids = get_tip_node_ids(solver)
    n_tip_nodes = len(tip_node_ids)
    force_per_node = total_force / n_tip_nodes
    
    # Apply load based on case
    if load_case == 'tension':
        # Force in +X direction
        tip_dofs_x = get_tip_dofs(solver, 'x')
        solver.add_force_on_dofs(tip_dofs_x, [force_per_node])
        
    elif load_case == 'compression':
        # Force in -X direction
        tip_dofs_x = get_tip_dofs(solver, 'x')
        solver.add_force_on_dofs(tip_dofs_x, [-force_per_node])
        
    elif load_case == 'bending_z':
        # Force in +Z direction
        tip_dofs_z = get_tip_dofs(solver, 'z')
        solver.add_force_on_dofs(tip_dofs_z, [force_per_node])
        
    elif load_case == 'bending_y':
        # Force in +Y direction
        tip_dofs_y = get_tip_dofs(solver, 'y')
        solver.add_force_on_dofs(tip_dofs_y, [force_per_node])
        
    elif load_case == 'torsion':
        # Apply torsion by couples in Y and Z
        apply_torsion_load(solver, tip_node_ids, total_force)
    
    # Solve
    u = solver.solve()
    
    # Extract results
    u_array = u.getArray()
    
    # Calculate tip displacement (average over tip nodes)
    dofs_per_node = solver.domain.dofs_per_node
    node_id_to_index = mesh.node_id_to_index
    
    tip_displacements = []
    for node_id in tip_node_ids:
        idx = node_id_to_index[node_id]
        ux = u_array[idx * dofs_per_node]
        uy = u_array[idx * dofs_per_node + 1]
        uz = u_array[idx * dofs_per_node + 2]
        tip_displacements.append([ux, uy, uz])
    
    tip_displacements = np.array(tip_displacements)
    tip_avg = np.mean(tip_displacements, axis=0)
    max_displacement = np.max(np.abs(u_array))
    
    return {
        'element_type': element_type,
        'load_case': load_case,
        'mesh': mesh,
        'solver': solver,
        'displacement': u_array.copy(),
        'max_displacement': max_displacement,
        'tip_displacement_avg': tip_avg,
        'n_elements': len(mesh.elements),
        'n_nodes': len(mesh.nodes),
    }


def apply_torsion_load(solver, tip_node_ids, total_moment=1000.0):
    """
    Apply torsion load (moment about X axis at tip).
    
    Creates a couple by applying tangential forces proportional to 
    distance from centroid. The moment is: M = sum(r_i × F_i)
    
    For tangential forces F_i = k * r_i (proportional to distance),
    the total moment is: M = k * sum(r_i^2)
    Therefore: k = M / sum(r_i^2)
    
    Parameters
    ----------
    solver : LinearStaticSolver
    tip_node_ids : list
        Node IDs at the beam tip
    total_moment : float
        Total moment to apply about X axis
    """
    mesh = solver.mesh_obj
    node_id_to_index = mesh.node_id_to_index
    dofs_per_node = solver.domain.dofs_per_node
    
    # Find centroid of tip face
    tip_coords = []
    for node_id in tip_node_ids:
        node = None
        for n in mesh.nodes:
            if n.id == node_id:
                node = n
                break
        if node is not None:
            tip_coords.append(node.coords)
    
    tip_coords = np.array(tip_coords)
    y_center = np.mean(tip_coords[:, 1])
    z_center = np.mean(tip_coords[:, 2])
    
    # Calculate sum of r^2 for all nodes (excluding center nodes)
    sum_r_squared = 0.0
    radii = []
    for i in range(len(tip_coords)):
        dy = tip_coords[i, 1] - y_center
        dz = tip_coords[i, 2] - z_center
        r = np.sqrt(dy**2 + dz**2)
        radii.append(r)
        if r > 1e-10:
            sum_r_squared += r**2
    
    if sum_r_squared < 1e-10:
        print("Warning: All tip nodes at center, cannot apply torsion")
        return
    
    # Calculate scaling factor k = M / sum(r_i^2)
    k = total_moment / sum_r_squared
    
    # Apply tangential forces F_i = k * r_i
    for i, node_id in enumerate(tip_node_ids):
        y, z = tip_coords[i, 1:3]
        
        dy = y - y_center
        dz = z - z_center
        r = radii[i]
        
        if r > 1e-10:
            # Force magnitude proportional to distance
            force_mag = k * r
            
            # Tangent direction: (-dz/r, dy/r) in the y-z plane (counterclockwise)
            fy = -force_mag * dz / r
            fz = force_mag * dy / r
            
            idx = node_id_to_index[node_id]
            dof_y = idx * dofs_per_node + 1
            dof_z = idx * dofs_per_node + 2
            
            solver.add_force_on_dofs([dof_y], [fy])
            solver.add_force_on_dofs([dof_z], [fz])


def compare_element_types(load_case, element_types=None, cross_section='rectangular', refinement=1):
    """
    Compare different element types for a given load case.
    
    Parameters
    ----------
    load_case : str
        Load case to analyze
    element_types : list, optional
        List of element/mesh types to compare.
    cross_section : str
        'rectangular' or 'circular'
    refinement : int
        Mesh refinement factor (1=base, 2=2x finer, etc.)
        
    Returns
    -------
    dict : Results for each element type
    """
    if element_types is None:
        if cross_section == 'rectangular':
            element_types = RECTANGULAR_ELEMENT_TYPES
        else:
            element_types = CIRCULAR_ELEMENT_TYPES
    
    material = IsotropicMaterial(name="Steel", E=E, nu=NU, rho=RHO)
    total_force = 1000.0  # N
    
    results = {}
    for elem_type in element_types:
        try:
            print(f"  Analyzing {elem_type.upper()} elements...")
            results[elem_type] = analyze_beam(elem_type, load_case, material, total_force, 
                                              cross_section=cross_section, refinement=refinement)
        except Exception as e:
            print(f"  WARNING: {elem_type.upper()} failed - {e}")
            continue
    
    return results


# =============================================================================
# Analytical Solutions
# =============================================================================

def analytical_tension_displacement(P, L, A, E):
    """δ = PL/(AE)"""
    return P * L / (A * E)


def analytical_bending_displacement(P, L, E, I):
    """δ = PL³/(3EI) for cantilever with tip load"""
    return P * L**3 / (3 * E * I)


def analytical_torsion_angle(T, L, G, J):
    """θ = TL/(GJ) for rectangular section.
    
    For a square section, J = 0.1406 * a^4 where a is the side length.
    The maximum displacement at corner is θ * r where r = a*sqrt(2)/2.
    """
    return T * L / (G * J)


def analytical_torsion_displacement(T, L, G, a, b):
    """
    Maximum displacement at corner for rectangular section under torsion.
    
    For a square section (a=b), J ≈ 0.1406 * a^4 (torsion constant, not polar moment)
    θ = TL/(GJ)
    Max displacement = θ * r_corner where r_corner = sqrt((a/2)^2 + (b/2)^2)
    
    Parameters
    ----------
    T : float
        Applied torque
    L : float
        Beam length
    G : float
        Shear modulus
    a, b : float
        Cross-section dimensions
    """
    # Torsion constant for rectangular section (not polar moment of inertia)
    # For a/b = 1: J ≈ 0.1406 * a^4
    # General formula: J = k * a * b^3 where k depends on a/b ratio
    ratio = max(a, b) / min(a, b)
    if ratio == 1:
        k = 0.1406
    else:
        # Approximate formula for k
        k = 1/3 * (1 - 0.630 / ratio + 0.052 / ratio**5)
    
    J = k * max(a, b) * min(a, b)**3
    
    theta = T * L / (G * J)
    r_corner = np.sqrt((a/2)**2 + (b/2)**2)
    
    return theta * r_corner


def analytical_circular_torsion_displacement(T, L, G, R):
    """
    Maximum displacement at outer radius for circular section under torsion.
    
    For circular section, J = π·R⁴/2 (polar moment of inertia)
    θ = TL/(GJ)
    Max displacement = θ * R
    
    Parameters
    ----------
    T : float
        Applied torque
    L : float
        Beam length
    G : float
        Shear modulus
    R : float
        Radius of circular section
    """
    J = np.pi * R**4 / 2  # Polar moment of inertia for circle
    theta = T * L / (G * J)
    return theta * R


def analytical_circular_bending(P, L, E, R):
    """
    Bending displacement for circular cross-section cantilever.
    
    I = π·R⁴/4 (second moment of area for circle)
    δ = PL³/(3EI)
    """
    I = np.pi * R**4 / 4
    return P * L**3 / (3 * E * I)


# =============================================================================
# Visualization and Output
# =============================================================================

def plot_comparison_results(all_results, save_path=None):
    """
    Create comparison plots for different load cases and element types.
    Includes analytical reference values as horizontal lines.
    """
    # Calculate analytical solutions
    A = WIDTH * HEIGHT
    I = WIDTH * HEIGHT**3 / 12
    G = E / (2 * (1 + NU))
    P = 1000.0  # Applied force (same as in compare_element_types)
    
    analytical = {
        'tension': analytical_tension_displacement(P, LENGTH, A, E) * 1000,  # mm
        'compression': analytical_tension_displacement(P, LENGTH, A, E) * 1000,
        'bending_z': analytical_bending_displacement(P, LENGTH, E, I) * 1000,
        'bending_y': analytical_bending_displacement(P, LENGTH, E, I) * 1000,
        'torsion': analytical_torsion_displacement(P, LENGTH, G, WIDTH, HEIGHT) * 1000,
    }
    
    load_cases = list(all_results.keys())
    n_cases = len(load_cases)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, n_cases, figsize=(7*n_cases, 6), squeeze=False)
    
    # Color palette for many element types
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for idx, (load_case, comparison) in enumerate(all_results.items()):
        ax = axes[0, idx]
        
        elem_types = list(comparison.keys())
        max_disps = [comparison[et]['max_displacement'] * 1000 for et in elem_types]  # mm
        n_elems = [comparison[et]['n_elements'] for et in elem_types]
        
        # Create labels with element counts
        labels = [f"{et}\n({n})" for et, n in zip(elem_types, n_elems)]
        
        x_pos = np.arange(len(elem_types))
        bars = ax.bar(x_pos, max_disps, color=colors[:len(elem_types)])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_xlabel('Element Type (# elements)')
        ax.set_ylabel('Max Displacement [mm]')
        ax.set_title(f'{load_case.upper()}')
        
        # Add value labels on bars
        for bar, val in zip(bars, max_disps):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.5f}', ha='center', va='bottom', fontsize=7, rotation=90)
        
        # Add analytical reference line if available
        if analytical.get(load_case) is not None:
            ax.axhline(y=analytical[load_case], color='red', linestyle='--', 
                      linewidth=2, label=f'Analytical: {analytical[load_case]:.5f}')
            ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()


def print_results_table(all_results, cross_section='rectangular'):
    """
    Print formatted results table with comparison to analytical values.
    
    Parameters
    ----------
    all_results : dict
        Dictionary with load case results.
    cross_section : str
        'rectangular' or 'circular'
    """
    G = E / (2 * (1 + NU))
    P = 1000.0  # Applied force
    
    if cross_section == 'rectangular':
        # Rectangular cross-section properties
        A = WIDTH * HEIGHT
        I_y = WIDTH * HEIGHT**3 / 12
        I_z = HEIGHT * WIDTH**3 / 12
        
        analytical_refs = {
            'tension': analytical_tension_displacement(P, LENGTH, A, E) * 1000,
            'compression': analytical_tension_displacement(P, LENGTH, A, E) * 1000,
            'bending_z': analytical_bending_displacement(P, LENGTH, E, I_y) * 1000,
            'bending_y': analytical_bending_displacement(P, LENGTH, E, I_z) * 1000,
            'torsion': analytical_torsion_displacement(P, LENGTH, G, WIDTH, HEIGHT) * 1000,
        }
        section_desc = f"W={WIDTH}m, H={HEIGHT}m"
    else:
        # Circular cross-section properties
        A = np.pi * RADIUS**2
        
        analytical_refs = {
            'tension': analytical_tension_displacement(P, LENGTH, A, E) * 1000,
            'compression': analytical_tension_displacement(P, LENGTH, A, E) * 1000,
            'bending_z': analytical_circular_bending(P, LENGTH, E, RADIUS) * 1000,
            'bending_y': analytical_circular_bending(P, LENGTH, E, RADIUS) * 1000,
            'torsion': analytical_circular_torsion_displacement(P, LENGTH, G, RADIUS) * 1000,
        }
        section_desc = f"R={RADIUS}m"
    
    print("-"*115)
    print(f"{'Load Case':<12} {'Element':<14} {'#Elem':>6} {'#Nodes':>6} {'Max Disp [mm]':>14} "
          f"{'Analytical':>12} {'Error %':>10} {'Tip Disp [mm]':>15}")
    print("-"*115)
    
    for load_case, comparison in all_results.items():
        for elem_type, results in comparison.items():
            ref = analytical_refs.get(load_case, None)
            max_d = results['max_displacement'] * 1000
            tip = results['tip_displacement_avg'] * 1000
            n_elem = results['n_elements']
            n_nodes = results['n_nodes']
            
            # Calculate relevant tip displacement for comparison
            if load_case in ['tension', 'compression']:
                tip_disp = abs(tip[0])
            elif load_case == 'bending_z':
                tip_disp = abs(tip[2])
            elif load_case == 'bending_y':
                tip_disp = abs(tip[1])
            elif load_case == 'torsion':
                tip_disp = max_d  # For torsion, max displacement is relevant
            else:
                tip_disp = max_d
            
            # Calculate error
            if ref and ref > 0:
                error = ((tip_disp - ref) / ref) * 100
                error_str = f"{error:+.1f}%"
            else:
                error_str = "N/A"
            
            ref_str = f"{ref:.6f}" if ref else "N/A"
            
            print(f"{load_case:<12} {elem_type.upper():<14} {n_elem:>6} {n_nodes:>6} {max_d:>14.6f} "
                  f"{ref_str:>12} {error_str:>10} {tip_disp:>15.6f}")
        print()
    
    print("-"*115)
    print(f"\nAnalytical Reference Values ({cross_section.upper()} section, {section_desc}):")
    print(f"  P=1000N, L={LENGTH}m, E={E/1e9:.0f}GPa, ν={NU}")
    print(f"  Tension/Compression: δ = PL/(AE) = {analytical_refs['tension']:.6f} mm")
    print(f"  Bending (tip load):  δ = PL³/(3EI) = {analytical_refs['bending_z']:.6f} mm")
    
    if cross_section == 'rectangular':
        print(f"  Torsion (corner):    δ = θ·r = {analytical_refs['torsion']:.6f} mm  (Saint-Venant, free warping)")
        print()
        print("NOTE on Torsion: The ~30% difference for rectangular sections is expected.")
        print("      Saint-Venant formula assumes free warping. With fixed BC at x=0 (warping")
        print("      restrained), actual displacement is lower.")
    else:
        print(f"  Torsion (edge):      δ = θ·R = {analytical_refs['torsion']:.6f} mm")
        print()
        print("NOTE: Circular sections show better torsion agreement due to zero warping.")
    print("="*115)


def compute_tet_volume(p0, p1, p2, p3):
    """
    Compute signed volume of tetrahedron.
    Positive volume means correct orientation (outward normals).
    """
    v1 = np.array(p1) - np.array(p0)
    v2 = np.array(p2) - np.array(p0)
    v3 = np.array(p3) - np.array(p0)
    return np.dot(v3, np.cross(v1, v2)) / 6.0


def fix_element_orientation(elem_nodes, node_coords, elem_type):
    """
    Fix element node ordering to ensure positive volume (correct face orientation).
    
    For tetrahedra: swap nodes 0 and 1 if volume is negative.
    """
    if elem_type == 10:  # VTK_TETRA
        p0 = node_coords[elem_nodes[0]]
        p1 = node_coords[elem_nodes[1]]
        p2 = node_coords[elem_nodes[2]]
        p3 = node_coords[elem_nodes[3]]
        
        vol = compute_tet_volume(p0, p1, p2, p3)
        
        if vol < 0:
            # Swap nodes 0 and 1 to fix orientation
            return [elem_nodes[1], elem_nodes[0], elem_nodes[2], elem_nodes[3]]
    
    return elem_nodes


def export_vtk(results, filename):
    """
    Export deformed mesh to VTK format for visualization in ParaView.
    
    Parameters
    ----------
    results : dict
        Analysis results
    filename : str
        Output VTK filename
    """
    mesh = results['mesh']
    u = results['displacement']
    solver = results['solver']
    scale = 100  # Displacement magnification for visualization
    
    # Get nodes list sorted by ID
    nodes_list = list(mesh.nodes)
    nodes_sorted = sorted(nodes_list, key=lambda n: n.id)
    
    dofs_per_node = solver.domain.dofs_per_node
    node_id_to_index = mesh.node_id_to_index
    
    # Get deformed node coordinates
    nodes_deformed = []
    for node in nodes_sorted:
        x, y, z = node.coords
        idx = node_id_to_index[node.id]
        ux = u[idx * dofs_per_node] * scale
        uy = u[idx * dofs_per_node + 1] * scale
        uz = u[idx * dofs_per_node + 2] * scale
        nodes_deformed.append([x + ux, y + uy, z + uz])
    
    nodes_deformed = np.array(nodes_deformed)
    
    # VTK element types and node ordering mappings (Gmsh -> VTK)
    # Linear elements have same ordering, quadratic need reordering
    VTK_TYPES = {
        'tetra': 10,
        'hexahedron': 12,
        'wedge': 13,
        'pyramid': 14,
        'tetra10': 24,
        'hexahedron20': 25,
        'wedge15': 26,
        'pyramid13': 27,
    }
    
    # Gmsh to VTK node ordering for quadratic elements
    # Reference: https://gmsh.info/doc/texinfo/gmsh.html#Node-ordering
    #            https://vtk.org/doc/nightly/html/ (element classes)
    GMSH_TO_VTK_ORDER = {
        # TETRA10: Gmsh edges {0-1,1-2,0-2,0-3,2-3,1-3} -> VTK {0-1,1-2,0-2,0-3,1-3,2-3}
        'tetra10': [0, 1, 2, 3, 4, 5, 6, 7, 9, 8],
        
        # HEXA20: Gmsh -> VTK edge midpoint mapping
        # Gmsh: 8-19 on edges {0-1,0-3,0-4,1-2,1-5,2-3,2-6,3-7,4-5,4-7,5-6,6-7}
        # VTK:  8-19 on edges {0-1,1-2,2-3,3-0,4-5,5-6,6-7,7-4,0-4,1-5,2-6,3-7}
        'hexahedron20': [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 16, 9, 17, 10, 18, 19, 12, 15, 13, 14],
        
        # WEDGE15: Gmsh -> VTK 
        # Gmsh: 6-14 on edges {0-1,0-2,0-3,1-2,1-4,2-5,3-4,3-5,4-5}
        # VTK:  6-14 on edges {0-1,1-2,2-0,3-4,4-5,5-3,0-3,1-4,2-5}
        'wedge15': [0, 1, 2, 3, 4, 5, 6, 8, 12, 7, 13, 14, 9, 11, 10],
        
        # PYRAMID13: Gmsh -> VTK
        # Gmsh: 5-12 on edges {0-1,0-3,0-4,1-2,1-4,2-3,2-4,3-4}
        # VTK:  5-12 on edges {0-1,1-2,2-3,3-0,0-4,1-4,2-4,3-4}
        # vtk_nodes[i] = gmsh_nodes[order[i]]
        'pyramid13': [0, 1, 2, 3, 4, 5, 8, 10, 6, 7, 9, 11, 12],
    }
    
    # Build element connectivity
    elements = []
    elem_types = []
    
    for mesh_elem in mesh.elements:
        elem_type_name = mesh_elem.element_type.name.lower()
        
        # Get VTK type
        vtk_type = VTK_TYPES.get(elem_type_name)
        if vtk_type is None:
            print(f"Warning: Unknown element type {elem_type_name}, skipping")
            continue
        
        # Convert node IDs to indices
        elem_node_indices = [node_id_to_index[n.id] for n in mesh_elem.nodes]
        
        # Apply Gmsh -> VTK node reordering if needed
        if elem_type_name in GMSH_TO_VTK_ORDER:
            order = GMSH_TO_VTK_ORDER[elem_type_name]
            elem_node_indices = [elem_node_indices[i] for i in order]
        
        # Fix element orientation if needed (for linear tetra)
        elem_node_indices = fix_element_orientation(elem_node_indices, nodes_deformed, vtk_type)
        
        elements.append(elem_node_indices)
        elem_types.append(vtk_type)
    
    # Write VTK file
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"Beam Analysis: {results['load_case']} - {results['element_type']}\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        
        # Points
        f.write(f"POINTS {len(nodes_deformed)} float\n")
        for node in nodes_deformed:
            f.write(f"{node[0]:.10f} {node[1]:.10f} {node[2]:.10f}\n")
        
        # Cells
        total_size = sum(len(e) + 1 for e in elements)
        f.write(f"\nCELLS {len(elements)} {total_size}\n")
        for elem in elements:
            f.write(f"{len(elem)} " + " ".join(map(str, elem)) + "\n")
        
        # Cell types
        f.write(f"\nCELL_TYPES {len(elements)}\n")
        for vtype in elem_types:
            f.write(f"{vtype}\n")
        
        # Point data - displacement magnitude
        f.write(f"\nPOINT_DATA {len(nodes_deformed)}\n")
        f.write("SCALARS displacement_magnitude float 1\n")
        f.write("LOOKUP_TABLE default\n")
        
        for node in nodes_sorted:
            idx = node_id_to_index[node.id]
            ux = u[idx * dofs_per_node]
            uy = u[idx * dofs_per_node + 1]
            uz = u[idx * dofs_per_node + 2]
            mag = np.sqrt(ux**2 + uy**2 + uz**2)
            f.write(f"{mag:.10e}\n")
        
        # Displacement vectors
        f.write("\nVECTORS displacement float\n")
        for node in nodes_sorted:
            idx = node_id_to_index[node.id]
            ux = u[idx * dofs_per_node]
            uy = u[idx * dofs_per_node + 1]
            uz = u[idx * dofs_per_node + 2]
            f.write(f"{ux:.10e} {uy:.10e} {uz:.10e}\n")
    
    print(f"VTK file saved: {filename}")


# =============================================================================
# Main Program
# =============================================================================

def run_analysis(cross_section='rectangular', load_cases=None, skip_plots=False, skip_vtk=False, refinement=1):
    """
    Run beam analysis for a specific cross-section type.
    
    Parameters
    ----------
    cross_section : str
        'rectangular' or 'circular'
    load_cases : list, optional
        List of load cases to analyze. Default: all load cases.
    skip_plots : bool
        If True, skip generating plots.
    skip_vtk : bool
        If True, skip exporting VTK files.
    refinement : int
        Mesh refinement factor (1=base, 2=2x finer, etc.)
        
    Returns
    -------
    dict
        Results dictionary.
    """
    # Get element types for this cross-section
    if cross_section == 'rectangular':
        element_types = RECTANGULAR_ELEMENT_TYPES
        section_desc = f"L={LENGTH}m, W={WIDTH}m, H={HEIGHT}m"
    else:
        element_types = CIRCULAR_ELEMENT_TYPES
        section_desc = f"L={LENGTH}m, R={RADIUS}m"
    
    print("\n" + "="*100)
    print(f"   {cross_section.upper()} SECTION BEAM ANALYSIS")
    print(f"   Geometry: {section_desc}")
    if refinement > 1:
        print(f"   Mesh refinement: {refinement}x")
    print("="*100 + "\n")
    
    print(f"Material: E={E/1e9:.0f} GPa, ν={NU}")
    print(f"Element types: {', '.join(element_types)}")
    if refinement > 1:
        print(f"Mesh density: {NX*refinement}×{NY*refinement}×{NZ*refinement} (refined {refinement}x)")
    print()
    
    # Define load cases
    all_load_cases = ['tension', 'compression', 'bending_z', 'bending_y', 'torsion']
    if load_cases is None:
        load_cases = all_load_cases
    
    print("Load cases: " + ", ".join(load_cases))
    print()
    
    # Analyze all cases
    all_results = {}
    
    for load_case in load_cases:
        print(f"\n{'='*60}")
        print(f"Analyzing load case: {load_case.upper()}")
        print("="*60)
        all_results[load_case] = compare_element_types(
            load_case, element_types=element_types, cross_section=cross_section, refinement=refinement
        )
    
    # Print summary table
    print("\n" + "="*100)
    print(f"{cross_section.upper()} SECTION RESULTS SUMMARY")
    print("="*100 + "\n")
    print_results_table(all_results, cross_section=cross_section)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Export VTK files for visualization
    if not skip_vtk:
        print(f"\nExporting VTK files for {cross_section} section...")
        for load_case, comparison in all_results.items():
            for elem_type, results in comparison.items():
                vtk_file = os.path.join(output_dir, f"beam_{cross_section}_{elem_type}_{load_case}.vtk")
                export_vtk(results, vtk_file)
    
    return all_results


def main(sections=None, load_cases=None, skip_plots=False, skip_vtk=False, refinement=1):
    """Run the complete beam analysis example.
    
    Parameters
    ----------
    sections : list, optional
        List of cross-sections to analyze. Default: ['rectangular', 'circular']
    load_cases : list, optional
        List of load cases to analyze. Default: all load cases.
    skip_plots : bool
        If True, skip generating plots.
    skip_vtk : bool
        If True, skip exporting VTK files.
    refinement : int
        Mesh refinement factor (1=base, 2=2x finer, etc.)
    """
    
    print("\n" + "="*100)
    print("   BEAM ANALYSIS: Flexion, Torsion, Tension, Compression")
    print("   Comparing 3D Solid Element Types for Different Cross-Sections")
    print("   Using LinearStaticSolver with PETSc")
    print("="*100 + "\n")
    
    # Determine which cross-sections to analyze
    available_sections = ['rectangular', 'circular']
    if sections is None:
        sections = available_sections
    
    all_section_results = {}
    
    for section in sections:
        results = run_analysis(
            cross_section=section,
            load_cases=load_cases,
            skip_plots=True,  # Generate combined plots at the end
            skip_vtk=skip_vtk,
            refinement=refinement
        )
        all_section_results[section] = results
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots for each section
    if not skip_plots:
        print("\nGenerating comparison plots...")
        for section, results in all_section_results.items():
            plot_file = os.path.join(output_dir, f"beam_{section}_comparison.png")
            plot_comparison_results(results, save_path=plot_file)
            print(f"  Plot saved: {plot_file}")
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE!")
    print("="*100)
    print(f"Output files saved in: {output_dir}")
    
    return all_section_results


if __name__ == "__main__":
    import argparse
    
    all_sections = ['rectangular', 'circular']
    all_load_cases = ['tension', 'compression', 'bending_z', 'bending_y', 'torsion']
    
    parser = argparse.ArgumentParser(description='Beam analysis with various 3D solid elements')
    parser.add_argument('--sections', '-s', nargs='+', choices=all_sections,
                        default=None, help=f'Cross-sections to analyze (default: all)')
    parser.add_argument('--load-cases', '-l', nargs='+', choices=all_load_cases,
                        default=None, help=f'Load cases to analyze (default: all)')
    parser.add_argument('--skip-plots', action='store_true', help='Skip generating plots')
    parser.add_argument('--skip-vtk', action='store_true', help='Skip VTK export')
    parser.add_argument('--refinement', '-r', type=int, default=1,
                        help='Mesh refinement factor (1=base, 2=2x finer, etc.)')
    
    args = parser.parse_args()
    
    results = main(
        sections=args.sections,
        load_cases=args.load_cases,
        skip_plots=args.skip_plots,
        skip_vtk=args.skip_vtk,
        refinement=args.refinement
    )
