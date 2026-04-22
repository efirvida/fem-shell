# MITC4 Covariant Element Specification

## Purpose
This specification defines the migration of the MITC4 shell element from a projected 2D "Flat-Shell" formulation to a full 3D Covariant formulation to eliminate parasitic shear and membrane locking in warped/curved geometries.

## Requirements

### Mathematical Requirements
| ID | Requirement | Strength |
|----|-------------|-----------|
| MATH-01 | The element MUST use the covariant metric tensor $g_{\alpha\beta} = \mathbf{x}_{,\alpha} \cdot \mathbf{x}_{,\beta}$ ($\alpha, \beta \in \{r, s\}$) for all membrane calculations. | MUST |
| MATH-02 | Integration MUST be performed using the surface Jacobian $\sqrt{g} = \sqrt{g_{rr}g_{ss} - g_{rs}^2}$. | MUST |
| MATH-03 | Membrane strains SHALL be computed in the covariant basis and modified using MITC4+ blending coefficients $a_i$ at tying points. | SHALL |
| MATH-04 | Strains MUST be mapped to a point-wise local orthonormal frame $\{\mathbf{e}_1, \mathbf{e}_2\}$ for constitutive law application. | MUST |

### Implementation Requirements
| ID | Requirement | Strength |
|----|-------------|-----------|
| IMPL-01 | `Mitc4Precomputed` MUST remove 2D projection `local_coords` and store full 3D node coordinates. | MUST |
| IMPL-02 | `compute_ke_local` SHALL implement the integration loop: $\sqrt{g} \to$ Covariant Strains $\to$ Blending $\to$ Orthonormal Mapping $\to$ B-Matrix. | SHALL |
| IMPL-03 | The global stiffness transformation MUST maintain consistency with the 3D covariant basis. | MUST |

### Validation & Acceptance Criteria
| ID | Requirement | Strength |
|----|-------------|-----------|
| VAL-01 | "Twisted Beam" benchmark: Relative displacement error MUST be $\le 5\%$ vs Ko et al. (2017). | MUST |
| VAL-02 | "Pinched Cylinder" benchmark: Relative displacement error MUST be $\le 5\%$ vs Ko et al. (2017). | MUST |
| VAL-03 | Cantilever Plate Parity: Relative error MUST be $\le 1\%$ vs CalculiX in `test_ccx_cantilever_plate_parity.py`. | MUST |
| VAL-04 | Locking Test: Convergence of thin plate displacement MUST be stable as $t/L \to 10^{-4}$. | MUST |

### Non-Functional Requirements
| ID | Requirement | Strength |
|----|-------------|-----------|
| NFR-01 | Assembly time for the covariant element SHALL NOT increase by more than 20% compared to the projected implementation. | SHOULD |

## Scenarios

### Scenario: Warped Geometry Accuracy (Twisted Beam)
- GIVEN a twisted beam geometry from Ko et al. (2017)
- WHEN the covariant MITC4 element is used for linear analysis
- THEN the nodal displacements MUST match the benchmark within 5% relative error
- AND no parasitic shear locking SHALL be observed

### Scenario: Thin Plate Limit (Locking Verification)
- GIVEN a cantilever plate with thickness $t$ varying from $L/100$ to $L/10000$
- WHEN the stiffness matrix is assembled and solved
- THEN the displacement results MUST remain stable and not show artificial stiffening (locking)

### Scenario: Parity with CalculiX
- GIVEN the cantilever plate model used in `test_ccx_cantilever_plate_parity.py`
- WHEN solved using the covariant formulation
- THEN the displacement field MUST align with CalculiX results within 1% relative error
