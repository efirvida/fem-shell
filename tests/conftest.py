"""Shared pytest fixtures for the fem-shell test suite."""

import os

import pytest

# ---------------------------------------------------------------------------
# Blade YAML fixture
# ---------------------------------------------------------------------------
# Candidate paths for the IEA-15-240-RWT turbine definition YAML.
# The file is large so it lives outside the repo in the simulations tree,
# but a copy may also exist inside examples/.
_BLADE_YAML_CANDIDATES = [
    # Inside repo (preferred when present)
    os.path.join(
        os.path.dirname(__file__), "..", "examples", "reference_turbines", "yamls",
        "IEA-15-240-RWT.yaml",
    ),
    # Sibling simulations directory (dev environment)
    os.path.join(
        os.path.dirname(__file__), "..", "..", "simulations", "blade", "solid",
        "IEA-15-240-RWT.yaml",
    ),
    os.path.join(
        os.path.dirname(__file__), "..", "..", "simulations", "IEA-15-240-RWT", "fsi",
        "NoTower", "solid", "IEA-15-240-RWT.yaml",
    ),
]


def _find_blade_yaml():
    for p in _BLADE_YAML_CANDIDATES:
        resolved = os.path.abspath(p)
        if os.path.isfile(resolved):
            return resolved
    return None


@pytest.fixture(scope="session")
def iea_blade_yaml():
    """Return the path to IEA-15-240-RWT.yaml or skip the test."""
    path = _find_blade_yaml()
    if path is None:
        pytest.skip(
            "IEA-15-240-RWT.yaml not found. "
            "Place a copy at examples/reference_turbines/yamls/ or alongside "
            "the simulations/blade/solid/ tree."
        )
    return path


@pytest.fixture(scope="session")
def iea_numad_blade(iea_blade_yaml):
    """Return a fully initialised numadBlade for the IEA-15-240-RWT."""
    import numpy as np

    from aeroelast.models.blade.numad import Blade as numadBlade

    # Same setup sequence used by BladeMesh.generate()
    blade = numadBlade()
    blade.read_yaml(iea_blade_yaml)
    for stat in blade.definition.stations:
        stat.airfoil.resample(n_samples=150)
    blade.update_blade()
    n_stations = blade.geometry.coordinates.shape[2]
    blade.expand_blade_geometry_te(0.001 * np.ones(n_stations))
    return blade
