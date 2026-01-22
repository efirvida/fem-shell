"""
FSI Simulation Configuration Module.

This module provides a YAML-based configuration system for FSI simulations,
allowing users to define complete simulations without writing Python code.

Example YAML configuration:
    mesh:
      source: "file"
      file:
        path: "mesh.h5"

    material:
      type: "isotropic"
      E: 4.0e6
      nu: 0.3
      rho: 3000.0

    solver:
      type: "LinearDynamicFSI"
      total_time: 5.0
      time_step: 0.001
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

# =============================================================================
# preCICE Configuration Parser
# =============================================================================


@dataclass
class PreciceTimeConfig:
    """Time configuration extracted from preCICE config XML."""

    max_time: Optional[float] = None
    time_window_size: Optional[float] = None


@dataclass
class PreciceWatchpoint:
    """Watch-point configuration from preCICE config XML."""

    name: str
    mesh: str
    coordinate: str


@dataclass
class PreciceConfigInfo:
    """Information extracted from preCICE configuration XML.

    This class parses a preCICE XML configuration file and extracts:
    - Time parameters (max-time, time-window-size)
    - Watch-points for a specific participant
    """

    time: PreciceTimeConfig
    watchpoints: List[PreciceWatchpoint]
    participant: Optional[str] = None

    @classmethod
    def from_xml(
        cls, xml_path: Union[str, Path], participant: str = "Solid"
    ) -> "PreciceConfigInfo":
        """Parse preCICE configuration XML and extract relevant information.

        Parameters
        ----------
        xml_path : str or Path
            Path to the preCICE configuration XML file.
        participant : str
            Name of the participant to extract watch-points for.

        Returns
        -------
        PreciceConfigInfo
            Extracted configuration information.
        """
        xml_path = Path(xml_path)
        if not xml_path.exists():
            raise FileNotFoundError(f"preCICE config file not found: {xml_path}")

        # preCICE XML uses custom namespace prefixes (data:, mapping:, m2n:, etc.)
        # without declaring them, which causes standard XML parsers to fail.
        # We preprocess the file to remove namespace prefixes before parsing.
        with open(xml_path, "r") as f:
            xml_content = f.read()

        # Remove namespace prefixes from tags (e.g., data:vector -> data-vector)
        # Also handles nested prefixes like basis-function:compact-polynomial-c6
        import re

        # Replace namespace colons in tags with hyphens
        # Pattern matches tag names that contain colons
        xml_content = re.sub(r"<(/?)([\w-]+):([\w-]+)", r"<\1\2-\3", xml_content)
        # Run again for nested namespace prefixes (e.g., basis-function:compact-polynomial-c6)
        xml_content = re.sub(r"<(/?)([\w-]+):([\w-]+)", r"<\1\2-\3", xml_content)

        root = ET.fromstring(xml_content)

        # Extract time configuration from coupling-scheme
        # After preprocessing, coupling-scheme:parallel-implicit becomes coupling-scheme-parallel-implicit
        time_config = PreciceTimeConfig()
        for coupling_scheme in root.iter():
            if coupling_scheme.tag.startswith("coupling-scheme-"):
                # Find max-time
                max_time_elem = coupling_scheme.find("max-time")
                if max_time_elem is not None:
                    time_config.max_time = float(max_time_elem.get("value", 0))

                # Find time-window-size
                tw_elem = coupling_scheme.find("time-window-size")
                if tw_elem is not None:
                    time_config.time_window_size = float(tw_elem.get("value", 0))
                break

        # Extract watch-points for the specified participant
        watchpoints = []
        for part_elem in root.findall("participant"):
            if part_elem.get("name") == participant:
                for wp in part_elem.findall("watch-point"):
                    watchpoints.append(
                        PreciceWatchpoint(
                            name=wp.get("name", ""),
                            mesh=wp.get("mesh", ""),
                            coordinate=wp.get("coordinate", ""),
                        )
                    )
                break

        return cls(time=time_config, watchpoints=watchpoints, participant=participant)

    def get_watchpoint_files(self) -> List[str]:
        """Generate list of watchpoint output file names.

        Returns
        -------
        List[str]
            List of expected watchpoint log file names.
        """
        return [f"precice-{self.participant}-watchpoint-{wp.name}.log" for wp in self.watchpoints]


class MeshSource(str, Enum):
    """Source type for mesh data."""

    FILE = "file"
    GENERATOR = "generator"


class MaterialType(str, Enum):
    """Type of material model."""

    ISOTROPIC = "isotropic"
    ORTHOTROPIC = "orthotropic"


class ElementFamily(str, Enum):
    """Element family type."""

    PLANE = "PLANE"
    SHELL = "SHELL"


class SolverType(str, Enum):
    """Type of solver to use."""

    LINEAR_STATIC = "LinearStatic"
    LINEAR_DYNAMIC = "LinearDynamic"
    LINEAR_DYNAMIC_FSI = "LinearDynamicFSI"


class MeshGeneratorType(str, Enum):
    """Available mesh generators."""

    SQUARE = "SquareShapeMesh"
    BOX = "BoxSurfaceMesh"
    MULTIFLAP = "MultiFlapMesh"
    ROTOR = "RotorMesh"


# =============================================================================
# Configuration Data Classes
# =============================================================================


@dataclass
class MeshFileConfig:
    """Configuration for loading mesh from file."""

    path: str
    format: str = "auto"

    def __post_init__(self):
        if self.format not in ("auto", "hdf5", "pickle"):
            raise ValueError(f"Invalid mesh format: {self.format}")


@dataclass
class SquareMeshParams:
    """Parameters for SquareShapeMesh generator."""

    width: float
    height: float
    nx: int
    ny: int
    quadratic: bool = False
    triangular: bool = False


@dataclass
class BoxMeshParams:
    """Parameters for BoxSurfaceMesh generator."""

    center: tuple
    dims: tuple
    nx: int
    ny: int
    nz: int
    quadratic: bool = False
    triangular: bool = False


@dataclass
class MultiFlapMeshParams:
    """Parameters for MultiFlapMesh generator."""

    n_flaps: int
    flap_width: float
    flap_height: float
    x_spacing: float
    base_height: float = 0.05
    nx_flap: int = 4
    ny_flap: int = 20
    nx_base_segment: int = 10
    ny_base: int = 2
    quadratic: bool = False


@dataclass
class RotorMeshParams:
    """Parameters for RotorMesh generator."""

    yaml_file: str  # Path to blade YAML definition
    n_blades: int = 3
    hub_radius: Optional[float] = None  # If None, uses blade definition
    element_size: float = 0.5
    n_samples: int = 300


@dataclass
class MeshGeneratorConfig:
    """Configuration for mesh generation."""

    type: str
    params: Dict[str, Any]

    def get_params_class(self):
        """Get the appropriate params dataclass for the generator type."""
        mapping = {
            MeshGeneratorType.SQUARE.value: SquareMeshParams,
            MeshGeneratorType.BOX.value: BoxMeshParams,
            MeshGeneratorType.MULTIFLAP.value: MultiFlapMeshParams,
            MeshGeneratorType.ROTOR.value: RotorMeshParams,
        }
        return mapping.get(self.type)

    def get_typed_params(self):
        """Get typed parameters for the mesh generator."""
        params_class = self.get_params_class()
        if params_class is None:
            raise ValueError(f"Unknown mesh generator type: {self.type}")
        return params_class(**self.params)


@dataclass
class MeshConfig:
    """Complete mesh configuration."""

    source: str
    file: Optional[MeshFileConfig] = None
    generator: Optional[MeshGeneratorConfig] = None
    # Output mesh file (format inferred from extension)
    # Supported formats: .vtk, .vtu, .msh, .inp (CalculiX), .h5/.hdf5, .obj, .stl
    output_file: Optional[str] = None

    def __post_init__(self):
        if self.source == MeshSource.FILE.value:
            if self.file is None:
                raise ValueError("Mesh source is 'file' but no file config provided")
        elif self.source == MeshSource.GENERATOR.value:
            if self.generator is None:
                raise ValueError("Mesh source is 'generator' but no generator config provided")
        else:
            raise ValueError(f"Invalid mesh source: {self.source}")


@dataclass
class IsotropicMaterialConfig:
    """Configuration for isotropic material."""

    name: str
    E: float
    nu: float
    rho: float

    def __post_init__(self):
        if self.E <= 0:
            raise ValueError(f"Young's modulus must be positive: {self.E}")
        if not -1 < self.nu < 0.5:
            raise ValueError(f"Poisson's ratio must be in (-1, 0.5): {self.nu}")
        if self.rho <= 0:
            raise ValueError(f"Density must be positive: {self.rho}")


@dataclass
class OrthotropicMaterialConfig:
    """Configuration for orthotropic material."""

    name: str
    E: tuple  # (E1, E2, E3)
    G: tuple  # (G12, G23, G31)
    nu: tuple  # (nu12, nu23, nu31)
    rho: float

    def __post_init__(self):
        if len(self.E) != 3:
            raise ValueError("E must have 3 components")
        if len(self.G) != 3:
            raise ValueError("G must have 3 components")
        if len(self.nu) != 3:
            raise ValueError("nu must have 3 components")


@dataclass
class MaterialConfig:
    """Complete material configuration."""

    type: str
    name: str = "Material"
    # Isotropic properties
    E: Optional[Union[float, List[float]]] = None
    nu: Optional[Union[float, List[float]]] = None
    rho: Optional[float] = None
    # Orthotropic additional property
    G: Optional[List[float]] = None

    def __post_init__(self):
        if self.type not in (MaterialType.ISOTROPIC.value, MaterialType.ORTHOTROPIC.value):
            raise ValueError(f"Invalid material type: {self.type}")

    def get_material_config(self):
        """Get the appropriate material configuration object."""
        if self.type == MaterialType.ISOTROPIC.value:
            if self.E is None or self.nu is None or self.rho is None:
                raise ValueError("Isotropic material requires E, nu, and rho")
            return IsotropicMaterialConfig(
                name=self.name,
                E=float(self.E) if not isinstance(self.E, list) else self.E[0],
                nu=float(self.nu) if not isinstance(self.nu, list) else self.nu[0],
                rho=float(self.rho),
            )
        else:
            if self.E is None or self.G is None or self.nu is None or self.rho is None:
                raise ValueError("Orthotropic material requires E, G, nu, and rho")
            return OrthotropicMaterialConfig(
                name=self.name,
                E=tuple(self.E) if isinstance(self.E, list) else (self.E, self.E, self.E),
                G=tuple(self.G),
                nu=tuple(self.nu) if isinstance(self.nu, list) else (self.nu, self.nu, self.nu),
                rho=float(self.rho),
            )


@dataclass
class ElementConfig:
    """Element configuration."""

    family: str
    thickness: Optional[float] = None

    def __post_init__(self):
        if self.family not in (ElementFamily.PLANE.value, ElementFamily.SHELL.value):
            raise ValueError(f"Invalid element family: {self.family}")
        if self.family == ElementFamily.SHELL.value and self.thickness is None:
            raise ValueError("Shell elements require thickness parameter")


@dataclass
class NewmarkConfig:
    """Newmark-β integration parameters."""

    beta: float = 0.25
    gamma: float = 0.5


@dataclass
class DampingConfig:
    """Rayleigh damping parameters."""

    eta_m: float = 1.0e-4
    eta_k: float = 1.0e-4


@dataclass
class SolverConfig:
    """Solver configuration.

    For FSI simulations, total_time and time_step can be omitted if they
    should be read from the preCICE configuration XML file.
    """

    type: str
    total_time: Optional[float] = None  # Can be read from preCICE config
    time_step: Optional[float] = None  # Can be read from preCICE config
    newmark: Optional[NewmarkConfig] = None
    damping: Optional[DampingConfig] = None
    use_critical_dt: bool = False
    safety_factor: float = 0.8

    def __post_init__(self):
        valid_types = [s.value for s in SolverType]
        if self.type not in valid_types:
            raise ValueError(f"Invalid solver type: {self.type}. Valid: {valid_types}")
        # For FSI solvers, time parameters can be read from preCICE config later
        # So we only validate if they are provided
        if self.total_time is not None and self.total_time <= 0:
            raise ValueError(f"total_time must be positive: {self.total_time}")
        if self.time_step is not None and self.time_step <= 0:
            raise ValueError(f"time_step must be positive: {self.time_step}")

    def update_from_precice(self, precice_info: "PreciceConfigInfo") -> None:
        """Update time parameters from preCICE configuration.

        Parameters
        ----------
        precice_info : PreciceConfigInfo
            Parsed preCICE configuration information.
        """
        if self.total_time is None and precice_info.time.max_time is not None:
            self.total_time = precice_info.time.max_time
        if self.time_step is None and precice_info.time.time_window_size is not None:
            self.time_step = precice_info.time.time_window_size

    def validate_complete(self) -> None:
        """Validate that all required parameters are set.

        Raises
        ------
        ValueError
            If required parameters are missing.
        """
        if self.total_time is None:
            raise ValueError(
                "total_time not set. Provide in YAML or ensure preCICE config has max-time."
            )
        if self.time_step is None:
            raise ValueError(
                "time_step not set. Provide in YAML or ensure preCICE config has time-window-size."
            )


@dataclass
class DirichletBCConfig:
    """Dirichlet boundary condition configuration."""

    nodeset: str
    value: float = 0.0
    components: Optional[List[int]] = None  # Optional: specific DOF components


@dataclass
class BodyForceConfig:
    """Body force configuration."""

    value: List[float]


@dataclass
class BoundaryConditionsConfig:
    """Complete boundary conditions configuration."""

    dirichlet: List[DirichletBCConfig] = field(default_factory=list)
    body_forces: List[BodyForceConfig] = field(default_factory=list)


@dataclass
class PreciceInterfaceConfig:
    """preCICE interface configuration."""

    coupling_mesh: str
    write_data: List[str] = field(default_factory=lambda: ["Displacement"])
    read_data: List[str] = field(default_factory=lambda: ["Force"])


@dataclass
class CouplingConfig:
    """FSI coupling (preCICE) configuration.

    All preCICE parameters are configured inline - no external adapter file needed.

    Examples
    --------
    coupling:
      participant: "Solid"
      config_file: "../precice-config.xml"
      coupling_mesh: "Solid-Mesh"
      write_data:
        - "Displacement"
      read_data:
        - "Force"
      boundaries:
        - "left"
        - "top"
        - "right"
    """

    boundaries: List[str]
    participant: str
    config_file: str
    coupling_mesh: str
    write_data: List[str] = field(default_factory=lambda: ["Displacement"])
    read_data: List[str] = field(default_factory=lambda: ["Force"])
    # Force limiting parameters
    force_max_cap: Optional[float] = None
    force_ramp_time: Optional[float] = None


@dataclass
class OutputConfig:
    """Output and checkpoint configuration."""

    folder: str = "results"
    write_interval: float = 0.0
    start_from: str = "startTime"
    start_time: float = 0.0
    save_deformed_mesh: bool = True
    deformed_mesh_scale: float = 1.0
    write_vtk: bool = True
    vtk_file: Optional[str] = None
    write_initial_state: bool = True


@dataclass
class PostprocessConfig:
    """Post-processing configuration.

    Watchpoint files can be specified explicitly or auto-detected from
    the preCICE configuration.
    """

    # Legacy single watchpoint (kept for backward compatibility)
    watchpoint_file: Optional[str] = None
    # Multiple watchpoint files (auto-populated from preCICE config)
    watchpoint_files: Optional[List[str]] = None
    plots: Optional[Dict[str, str]] = None

    def get_all_watchpoint_files(self) -> List[str]:
        """Get list of all watchpoint files.

        Returns
        -------
        List[str]
            All configured watchpoint files.
        """
        files = []
        if self.watchpoint_files:
            files.extend(self.watchpoint_files)
        if self.watchpoint_file and self.watchpoint_file not in files:
            files.append(self.watchpoint_file)
        return files

    def update_from_precice(self, precice_info: "PreciceConfigInfo") -> None:
        """Update watchpoint files from preCICE configuration.

        Parameters
        ----------
        precice_info : PreciceConfigInfo
            Parsed preCICE configuration information.
        """
        auto_files = precice_info.get_watchpoint_files()
        if auto_files:
            if self.watchpoint_files is None:
                self.watchpoint_files = []
            for f in auto_files:
                if f not in self.watchpoint_files:
                    self.watchpoint_files.append(f)


@dataclass
class FSISimulationConfig:
    """Complete FSI simulation configuration."""

    mesh: MeshConfig
    material: MaterialConfig
    elements: ElementConfig
    solver: SolverConfig
    boundary_conditions: BoundaryConditionsConfig
    coupling: Optional[CouplingConfig] = None
    output: Optional[OutputConfig] = None
    postprocess: Optional[PostprocessConfig] = None

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "FSISimulationConfig":
        """Load configuration from YAML file.

        Parameters
        ----------
        yaml_path : str or Path
            Path to the YAML configuration file.

        Returns
        -------
        FSISimulationConfig
            Validated configuration object.

        Raises
        ------
        FileNotFoundError
            If the YAML file does not exist.
        ValueError
            If the configuration is invalid.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data, base_path=yaml_path.parent)

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], base_path: Optional[Path] = None
    ) -> "FSISimulationConfig":
        """Create configuration from dictionary.

        Parameters
        ----------
        data : dict
            Configuration dictionary.
        base_path : Path, optional
            Base path for resolving relative file paths.

        Returns
        -------
        FSISimulationConfig
            Validated configuration object.
        """
        # Parse mesh configuration
        mesh_data = data.get("mesh", {})
        mesh_file = None
        mesh_generator = None

        if mesh_data.get("source") == MeshSource.FILE.value:
            file_data = mesh_data.get("file", {})
            # Resolve relative paths
            if base_path and not Path(file_data.get("path", "")).is_absolute():
                file_data["path"] = str(base_path / file_data.get("path", ""))
            mesh_file = MeshFileConfig(**file_data)
        elif mesh_data.get("source") == MeshSource.GENERATOR.value:
            gen_data = mesh_data.get("generator", {})
            mesh_generator = MeshGeneratorConfig(
                type=gen_data.get("type"),
                params=gen_data.get("params", {}),
            )

        mesh_config = MeshConfig(
            source=mesh_data.get("source"),
            file=mesh_file,
            generator=mesh_generator,
            output_file=mesh_data.get(
                "output_file", mesh_data.get("output_vtk")
            ),  # backward compat
        )

        # Parse material configuration
        mat_data = data.get("material", {})
        material_config = MaterialConfig(
            type=mat_data.get("type", "isotropic"),
            name=mat_data.get("name", "Material"),
            E=mat_data.get("E"),
            nu=mat_data.get("nu"),
            rho=mat_data.get("rho"),
            G=mat_data.get("G"),
        )

        # Parse element configuration
        elem_data = data.get("elements", {})
        element_config = ElementConfig(
            family=elem_data.get("family", "PLANE"),
            thickness=elem_data.get("thickness"),
        )

        # Parse solver configuration
        solver_data = data.get("solver", {})
        newmark_data = solver_data.get("newmark", {})
        damping_data = solver_data.get("damping", {})

        newmark_config = NewmarkConfig(**newmark_data) if newmark_data else None
        damping_config = DampingConfig(**damping_data) if damping_data else None

        solver_config = SolverConfig(
            type=solver_data.get("type", "LinearDynamicFSI"),
            total_time=solver_data.get("total_time"),
            time_step=solver_data.get("time_step"),
            newmark=newmark_config,
            damping=damping_config,
            use_critical_dt=solver_data.get("use_critical_dt", False),
            safety_factor=solver_data.get("safety_factor", 0.8),
        )

        # Parse boundary conditions
        bc_data = data.get("boundary_conditions", {})
        dirichlet_list = []
        for bc in bc_data.get("dirichlet", []):
            dirichlet_list.append(
                DirichletBCConfig(
                    nodeset=bc.get("nodeset"),
                    value=bc.get("value", 0.0),
                    components=bc.get("components"),
                )
            )

        body_forces_list = []
        for bf in bc_data.get("body_forces", []):
            body_forces_list.append(BodyForceConfig(value=bf.get("value")))

        bc_config = BoundaryConditionsConfig(
            dirichlet=dirichlet_list,
            body_forces=body_forces_list,
        )

        # Parse coupling configuration (optional for non-FSI solvers)
        coupling_config = None
        coupling_data = data.get("coupling")
        if coupling_data:
            boundaries = coupling_data.get("boundaries", [])

            # Resolve config_file path
            config_file = coupling_data.get("config_file")
            if base_path and config_file and not Path(config_file).is_absolute():
                config_file = str(base_path / config_file)

            # Support both flat structure and nested 'interface' structure
            # Flat: coupling.coupling_mesh, coupling.write_data, coupling.read_data
            # Nested: coupling.interface.coupling_mesh, coupling.interface.write_data, etc.
            interface_data = coupling_data.get("interface", {})

            # Handle write_data and read_data as lists (check interface first, then flat)
            write_data = interface_data.get("write_data") or coupling_data.get(
                "write_data", ["Displacement"]
            )
            read_data = interface_data.get("read_data") or coupling_data.get("read_data", ["Force"])
            coupling_mesh = interface_data.get("coupling_mesh") or coupling_data.get(
                "coupling_mesh"
            )

            # Ensure they are lists
            if isinstance(write_data, str):
                write_data = [write_data]
            if isinstance(read_data, str):
                read_data = [read_data]

            coupling_config = CouplingConfig(
                boundaries=boundaries,
                participant=coupling_data.get("participant"),
                config_file=config_file,
                coupling_mesh=coupling_mesh,
                write_data=write_data,
                read_data=read_data,
                force_max_cap=coupling_data.get("force_max_cap"),
                force_ramp_time=coupling_data.get("force_ramp_time"),
            )

        # Parse output configuration
        output_config = None
        output_data = data.get("output")
        if output_data:
            output_config = OutputConfig(
                folder=output_data.get("folder", "results"),
                write_interval=output_data.get("write_interval", 0.0),
                start_from=output_data.get("start_from", "startTime"),
                start_time=output_data.get("start_time", 0.0),
                save_deformed_mesh=output_data.get("save_deformed_mesh", True),
                deformed_mesh_scale=output_data.get("deformed_mesh_scale", 1.0),
                write_vtk=output_data.get("write_vtk", True),
                vtk_file=output_data.get("vtk_file"),
                write_initial_state=output_data.get("write_initial_state", True),
            )

        # Parse postprocess configuration
        postprocess_config = None
        postprocess_data = data.get("postprocess")
        if postprocess_data:
            postprocess_config = PostprocessConfig(
                watchpoint_file=postprocess_data.get("watchpoint_file"),
                plots=postprocess_data.get("plots"),
            )

        return cls(
            mesh=mesh_config,
            material=material_config,
            elements=element_config,
            solver=solver_config,
            boundary_conditions=bc_config,
            coupling=coupling_config,
            output=output_config,
            postprocess=postprocess_config,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        result = {
            "mesh": {
                "source": self.mesh.source,
            },
            "material": {
                "type": self.material.type,
                "name": self.material.name,
                "E": self.material.E,
                "nu": self.material.nu,
                "rho": self.material.rho,
            },
            "elements": {
                "family": self.elements.family,
            },
            "solver": {
                "type": self.solver.type,
                "total_time": self.solver.total_time,
                "time_step": self.solver.time_step,
            },
            "boundary_conditions": {
                "dirichlet": [
                    {"nodeset": bc.nodeset, "value": bc.value}
                    for bc in self.boundary_conditions.dirichlet
                ],
            },
        }

        # Add mesh specifics
        if self.mesh.file:
            result["mesh"]["file"] = {
                "path": self.mesh.file.path,
                "format": self.mesh.file.format,
            }
        if self.mesh.generator:
            result["mesh"]["generator"] = {
                "type": self.mesh.generator.type,
                "params": self.mesh.generator.params,
            }

        # Add optional configurations
        if self.elements.thickness:
            result["elements"]["thickness"] = self.elements.thickness

        if self.material.G:
            result["material"]["G"] = self.material.G

        if self.solver.newmark:
            result["solver"]["newmark"] = {
                "beta": self.solver.newmark.beta,
                "gamma": self.solver.newmark.gamma,
            }

        if self.solver.damping:
            result["solver"]["damping"] = {
                "eta_m": self.solver.damping.eta_m,
                "eta_k": self.solver.damping.eta_k,
            }

        if self.coupling:
            result["coupling"] = {
                "participant": self.coupling.participant,
                "config_file": self.coupling.config_file,
                "coupling_mesh": self.coupling.coupling_mesh,
                "write_data": self.coupling.write_data,
                "read_data": self.coupling.read_data,
                "boundaries": self.coupling.boundaries,
            }
            if self.coupling.force_max_cap is not None:
                result["coupling"]["force_max_cap"] = self.coupling.force_max_cap
            if self.coupling.force_ramp_time is not None:
                result["coupling"]["force_ramp_time"] = self.coupling.force_ramp_time

        if self.output:
            result["output"] = {
                "folder": self.output.folder,
                "write_interval": self.output.write_interval,
                "start_from": self.output.start_from,
                "start_time": self.output.start_time,
                "save_deformed_mesh": self.output.save_deformed_mesh,
                "deformed_mesh_scale": self.output.deformed_mesh_scale,
            }

        if self.postprocess:
            result["postprocess"] = {}
            if self.postprocess.watchpoint_file:
                result["postprocess"]["watchpoint_file"] = self.postprocess.watchpoint_file
            if self.postprocess.plots:
                result["postprocess"]["plots"] = self.postprocess.plots

        return result

    def save_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file.

        Parameters
        ----------
        yaml_path : str or Path
            Path to the output YAML file.
        """
        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def validate(self) -> List[str]:
        """Validate the complete configuration.

        Returns
        -------
        list of str
            List of validation warnings (empty if all OK).
        """
        warnings = []

        # Check FSI solver requires coupling config
        if self.solver.type == SolverType.LINEAR_DYNAMIC_FSI.value:
            if not self.coupling:
                warnings.append("FSI solver requires coupling configuration")
            elif not self.coupling.boundaries:
                warnings.append("FSI coupling requires at least one boundary")

        # Check boundary conditions reference valid nodesets
        # (This would need mesh to be loaded to fully validate)

        return warnings

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [
            "FSI Simulation Configuration",
            "=" * 40,
            f"Mesh: {self.mesh.source}",
        ]
        if self.mesh.file:
            lines.append(f"  File: {self.mesh.file.path}")
        if self.mesh.generator:
            lines.append(f"  Generator: {self.mesh.generator.type}")

        lines.extend(
            [
                f"Material: {self.material.type} ({self.material.name})",
                f"  E={self.material.E}, nu={self.material.nu}, rho={self.material.rho}",
                f"Elements: {self.elements.family}",
                f"Solver: {self.solver.type}",
            ]
        )

        # Handle optional time parameters
        total_time = self.solver.total_time or "auto (from preCICE)"
        time_step = self.solver.time_step or "auto (from preCICE)"
        lines.append(f"  Time: 0 → {total_time}s (dt={time_step}s)")
        lines.append(f"Boundary Conditions: {len(self.boundary_conditions.dirichlet)} Dirichlet")

        if self.coupling:
            lines.append(f"Coupling: {len(self.coupling.boundaries)} boundaries")

        return "\n".join(lines)

    def load_precice_config(self) -> Optional[PreciceConfigInfo]:
        """Load and parse the preCICE configuration XML file.

        Returns
        -------
        PreciceConfigInfo or None
            Parsed preCICE configuration, or None if not applicable.
        """
        if not self.coupling:
            return None

        # Get the preCICE config file path
        config_file = self.coupling.config_file
        participant = self.coupling.participant

        if not config_file or not Path(config_file).exists():
            return None

        return PreciceConfigInfo.from_xml(config_file, participant=participant or "Solid")

    def auto_complete_from_precice(self) -> None:
        """Auto-complete configuration from preCICE XML file.

        This method reads the preCICE configuration and fills in:
        - total_time from <max-time>
        - time_step from <time-window-size>
        - watchpoint_files from <watch-point> elements

        Only values that are not already set in the YAML will be updated.
        """
        precice_info = self.load_precice_config()
        if not precice_info:
            return

        # Update solver time parameters
        self.solver.update_from_precice(precice_info)

        # Update postprocess watchpoint files
        if self.postprocess is None:
            self.postprocess = PostprocessConfig()
        self.postprocess.update_from_precice(precice_info)

    def validate_complete(self) -> None:
        """Validate that all required parameters are set after auto-completion.

        Raises
        ------
        ValueError
            If required parameters are missing.
        """
        self.solver.validate_complete()
