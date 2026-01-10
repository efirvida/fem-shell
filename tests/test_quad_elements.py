"""Test suite for QUAD family finite elements (QUAD4, QUAD8, QUAD9)."""

import numpy as np
import pytest

from fem_shell.core.material import IsotropicMaterial
from fem_shell.elements.QUAD import QUAD4, QUAD8, QUAD9


def check_stiffness_matrix(K, expected_size, tol=1e-6):
    """
    Verify stiffness matrix properties.

    Parameters
    ----------
    K : np.ndarray
        Stiffness matrix to check.
    expected_size : int
        Expected matrix size (n_dofs × n_dofs).
    tol : float, optional
        Numerical tolerance for comparisons.

    Raises
    ------
    AssertionError
        If matrix doesn't meet required properties.
    """
    # Check matrix dimensions
    assert K.shape == (expected_size, expected_size), (
        f"Expected size {expected_size}x{expected_size}, got {K.shape}"
    )

    # Verify symmetry
    assert np.allclose(K, K.T, atol=tol), (
        f"Matrix not symmetric. Max asymmetry: {np.max(np.abs(K - K.T))}"
    )

    # Check eigenvalues (rigid body modes and positive definiteness)
    eigvals = np.linalg.eigvalsh(K)
    max_eigval = np.max(np.abs(eigvals))
    # Use relative tolerance for detecting zero eigenvalues
    zero_threshold = max_eigval * 1e-10 if max_eigval > 0 else 1e-10
    zero_eigvals = np.sum(np.abs(eigvals) < zero_threshold)  # Count rigid body modes
    assert zero_eigvals == 3, f"Expected 3 rigid modes, found {zero_eigvals}. Eigenvalues: {eigvals[:5]}"

    # Verify positive semi-definiteness (excluding rigid modes)
    non_zero_eigvals = eigvals[zero_eigvals:]
    assert np.all(non_zero_eigvals > -zero_threshold), (
        f"Negative eigenvalues found: {non_zero_eigvals[non_zero_eigvals < 0]}"
    )


@pytest.fixture
def sample_material():
    """Create an isotropic material sample for testing."""
    return IsotropicMaterial(name="steel", E=210e9, nu=0.3, rho=7800)


@pytest.fixture
def quad4_nodes():
    """
    Return nodes for a standard QUAD4 element in natural coordinates.

    Node ordering:
        3 --------- 2
        |           |
        |  (0,0)    |
        |           |
        0 --------- 1
    """
    return np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])


@pytest.fixture
def quad8_nodes():
    """
    Return nodes for a standard QUAD8 element in natural coordinates.

    Node ordering:
        3 --- 6 --- 2
        |           |
        7  (0,0)    5
        |           |
        0 --- 4 --- 1
    """
    return np.array([
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1],  # Corner nodes
        [0, -1],
        [1, 0],
        [0, 1],
        [-1, 0],  # Mid-side nodes
    ])


@pytest.fixture
def quad9_nodes():
    """
    Return nodes for a standard QUAD9 element in natural coordinates.

    Node ordering:
        3 --- 6 --- 2
        |    8      |
        7  (0,0)    5
        |           |
        0 --- 4 --- 1
    """
    return np.array([
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1],  # Corner nodes
        [0, -1],
        [1, 0],
        [0, 1],
        [-1, 0],  # Mid-side nodes
        [0, 0],  # Center node
    ])


# Element test configurations
ELEMENT_CONFIGS = {
    "QUAD4": {
        "class": QUAD4,
        "nodes": "quad4_nodes",
        "connectivity": (0, 1, 2, 3),
        "dofs": 8,
        "shape_funcs": {
            (-1, -1): [1, 0, 0, 0],
            (1, -1): [0, 1, 0, 0],
            (1, 1): [0, 0, 1, 0],
            (-1, 1): [0, 0, 0, 1],
            (0, 0): [0.25, 0.25, 0.25, 0.25],
        },
        "has_derivatives": True,
    },
    "QUAD8": {
        "class": QUAD8,
        "nodes": "quad8_nodes",
        "connectivity": (0, 1, 2, 3, 4, 5, 6, 7),
        "dofs": 16,
        "shape_funcs": {
            (-1, -1): [1, 0, 0, 0, 0, 0, 0, 0],
            (1, -1): [0, 1, 0, 0, 0, 0, 0, 0],
            (0, -1): [0, 0, 0, 0, 1, 0, 0, 0],
            (1, 0): [0, 0, 0, 0, 0, 1, 0, 0],
        },
        "has_derivatives": False,
    },
    "QUAD9": {
        "class": QUAD9,
        "nodes": "quad9_nodes",
        "connectivity": (0, 1, 2, 3, 4, 5, 6, 7, 8),
        "dofs": 18,
        "shape_funcs": {
            (-1, -1): [1, 0, 0, 0, 0, 0, 0, 0, 0],
            (0, -1): [0, 0, 0, 0, 1, 0, 0, 0, 0],
            (0, 0): [0, 0, 0, 0, 0, 0, 0, 0, 1],
        },
        "has_derivatives": False,
        "n_int_points": 9,
    },
}


@pytest.fixture(params=list(ELEMENT_CONFIGS.keys()))
def element_setup(request, sample_material, quad4_nodes, quad8_nodes, quad9_nodes):
    """
    Fixture providing element test data for parametrized tests.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Pytest request object providing parameterization.
    """
    config = ELEMENT_CONFIGS[request.param]
    nodes = {
        "quad4_nodes": quad4_nodes,
        "quad8_nodes": quad8_nodes,
        "quad9_nodes": quad9_nodes,
    }[config["nodes"]]

    element = config["class"](nodes, config["connectivity"], sample_material)

    return {
        "element": element,
        "name": request.param,
        "expected_dofs": config["dofs"],
        "shape_funcs": config["shape_funcs"],
        "has_derivatives": config.get("has_derivatives", False),
        "n_int_points": config.get("n_int_points", None),
    }


def test_shape_functions(element_setup):
    """Verify shape functions at specific natural coordinates."""
    elem = element_setup["element"]
    for (xi, eta), expected in element_setup["shape_funcs"].items():
        computed = elem.shape_functions(xi, eta)
        assert np.allclose(computed, expected), (
            f"{element_setup['name']} shape functions failed at ({xi}, {eta})\n"
            f"Expected: {expected}\nGot: {computed}"
        )


def test_jacobian(element_setup):
    """Test Jacobian computation at element center."""
    elem = element_setup["element"]
    J, detJ, invJ = elem._compute_jacobian(0, 0)

    # For standard elements, Jacobian should be identity
    assert np.allclose(J, np.eye(2)), f"Jacobian not identity for {element_setup['name']}:\n{J}"
    assert np.isclose(detJ, 1.0), "Jacobian determinant should be 1"
    assert np.allclose(invJ, np.eye(2)), "Inverse Jacobian should be identity"


def test_stiffness_matrix(element_setup):
    """Verify stiffness matrix properties."""
    elem = element_setup["element"]
    check_stiffness_matrix(elem.K, element_setup["expected_dofs"])


def test_mass_matrix(element_setup):
    """Check mass matrix symmetry and positive definiteness."""
    elem = element_setup["element"]
    if hasattr(elem, "M"):
        M = elem.M
        assert np.allclose(M, M.T), "Mass matrix must be symmetric"
        eigvals = np.linalg.eigvalsh(M)
        assert np.all(eigvals > -1e-10), "Mass matrix must be positive semi-definite"


def test_body_load(element_setup):
    """Verify body load vector properties."""
    elem = element_setup["element"]
    load_vector = elem.body_load(np.array([1, 2]))

    assert load_vector.shape == (element_setup["expected_dofs"],), "Load vector size mismatch"

    # For QUAD4, check load distribution symmetry
    if element_setup["name"] == "QUAD4":
        assert np.isclose(load_vector[0], load_vector[2]), "X-load not symmetric"
        assert np.isclose(load_vector[1], load_vector[3]), "Y-load not symmetric"


def test_shape_derivatives(element_setup):
    """Test shape function derivatives if implemented."""
    if element_setup["has_derivatives"]:
        elem = element_setup["element"]
        dN_dxi, dN_deta = elem.shape_function_derivatives(0, 0)

        # Expected values for QUAD4 at center
        expected_dxi = np.array([-0.25, 0.25, 0.25, -0.25])
        expected_deta = np.array([-0.25, -0.25, 0.25, 0.25])

        assert np.allclose(dN_dxi, expected_dxi), "dN/dxi mismatch"
        assert np.allclose(dN_deta, expected_deta), "dN/deta mismatch"


def test_integration_points(element_setup):
    """Verify integration scheme properties."""
    if element_setup["n_int_points"] is not None:
        elem = element_setup["element"]
        points, weights = elem.integration_points

        assert len(points) == element_setup["n_int_points"], "Wrong number of points"
        assert len(weights) == element_setup["n_int_points"], "Wrong number of weights"

        # Total weight should equal element area in natural coordinates (4)
        assert np.isclose(sum(weights), 4.0), "Weights should sum to element area"


@pytest.mark.parametrize("element_type", ["QUAD9"])
def test_distorted_jacobian(element_type, sample_material, quad9_nodes):
    """Test Jacobian remains positive for distorted elements."""
    if element_type == "QUAD9":
        # Create a deliberately distorted element
        distorted_nodes = np.array([
            [0, 0],
            [2, 0.1],
            [2.1, 2],
            [0, 1.9],  # Corners
            [1, 0],
            [2, 1],
            [1, 2],
            [0, 1],  # Midsides
            [1, 1],  # Center
        ])

        elem = QUAD9(distorted_nodes, range(9), sample_material)

        # Verify Jacobian at all integration points
        for xi, eta in elem.integration_points[0]:
            _, detJ, _ = elem._compute_jacobian(xi, eta)
            assert detJ > 0, f"Negative Jacobian at ({xi}, {eta})"


@pytest.mark.parametrize("element_type", ["QUAD4", "QUAD8", "QUAD9"])
def test_B_matrix(element_type, sample_material, quad4_nodes, quad8_nodes, quad9_nodes):
    """Verify strain-displacement matrix computation."""
    # Setup element based on type
    if element_type == "QUAD4":
        elem = QUAD4(quad4_nodes, (0, 1, 2, 3), sample_material)
        expected_dx = np.array([-0.25, 0.25, 0.25, -0.25])
        expected_dy = np.array([-0.25, -0.25, 0.25, 0.25])
    elif element_type == "QUAD8":
        elem = QUAD8(quad8_nodes, range(8), sample_material)
        expected_dx = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, -0.5])
        expected_dy = np.array([0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.5, 0.0])
    elif element_type == "QUAD9":
        elem = QUAD9(quad9_nodes, range(9), sample_material)
        expected_dx = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, -0.5, 0.0])
        expected_dy = np.array([0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.5, 0.0, 0.0])

    B = elem.compute_B_matrix(0, 0)

    # Check derivatives in B matrix
    assert np.allclose(B[0, 0::2], expected_dx), f"{element_type} dN/dx mismatch"
    assert np.allclose(B[1, 1::2], expected_dy), f"{element_type} dN/dy mismatch"


def test_rigid_body_modes(element_setup):
    """Verify stiffness matrix has correct rigid body modes."""
    elem = element_setup["element"]
    K = elem.K
    coords = elem.node_coords

    # Define rigid body modes (translation X, Y and rotation)
    translation_x = np.array([1, 0] * len(coords))
    translation_y = np.array([0, 1] * len(coords))
    rotation = np.array([(-y, x) for x, y in coords]).flatten()

    # Use relative tolerance based on matrix norm
    K_norm = np.linalg.norm(K, ord='fro')
    rel_tol = 1e-10  # Relative tolerance

    # Check each mode produces zero strain energy
    for i, mode in enumerate([translation_x, translation_y, rotation]):
        residual = K @ mode
        max_error = np.max(np.abs(residual))
        # Use relative error check
        rel_error = max_error / K_norm if K_norm > 0 else max_error
        assert rel_error < rel_tol, (
            f"Rigid mode {i} failed. Relative error: {rel_error}\n"
            f"Max absolute error: {max_error}, K_norm: {K_norm}"
        )


def test_constitutive_matrix(sample_material):
    """Verify material matrix properties."""
    # Using QUAD4 as representative for material matrix
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    elem = QUAD4(nodes, (0, 1, 2, 3), sample_material)

    C = elem.C
    assert np.allclose(C, C.T), "Material matrix must be symmetric"
    assert np.all(np.linalg.eigvalsh(C) > 0), "Material matrix must be positive definite"
