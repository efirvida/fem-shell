"""
Failure Criteria for Composite Materials.

This module implements common failure criteria for fiber-reinforced
composite laminae, including:
- Tsai-Wu criterion (quadratic interaction)
- Hashin criterion (mode-specific failure)
- Maximum stress criterion

References
----------
- Tsai, S.W. and Wu, E.M. (1971). "A General Theory of Strength for
  Anisotropic Materials." J. Composite Materials, 5, 58-80.
- Hashin, Z. (1980). "Failure Criteria for Unidirectional Fiber Composites."
  J. Applied Mechanics, 47, 329-334.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np

from fem_shell.core.laminate import StrengthProperties


class FailureMode(Enum):
    """Enumeration of composite failure modes."""

    NO_FAILURE = "no_failure"
    FIBER_TENSION = "fiber_tension"
    FIBER_COMPRESSION = "fiber_compression"
    MATRIX_TENSION = "matrix_tension"
    MATRIX_COMPRESSION = "matrix_compression"
    SHEAR = "shear"
    COMBINED = "combined"  # For Tsai-Wu


@dataclass
class FailureResult:
    """
    Result of failure criterion evaluation.

    Attributes
    ----------
    failed : bool
        True if failure index >= 1
    failure_index : float
        Value of failure criterion (>= 1 indicates failure)
    mode : FailureMode
        Type of failure (for mode-specific criteria)
    margin_of_safety : float
        Safety margin = 1/sqrt(FI) - 1 (for quadratic criteria)
    reserve_factor : float
        Load factor to failure = 1/sqrt(FI)
    """

    failed: bool
    failure_index: float
    mode: FailureMode
    margin_of_safety: float = 0.0
    reserve_factor: float = 1.0

    def __post_init__(self):
        if self.failure_index > 0:
            self.reserve_factor = 1 / np.sqrt(self.failure_index)
            self.margin_of_safety = self.reserve_factor - 1
        else:
            self.reserve_factor = float("inf")
            self.margin_of_safety = float("inf")


def tsai_wu_failure_index(
    sigma: np.ndarray,
    strength: StrengthProperties,
    f12_coefficient: float = -0.5,
) -> FailureResult:
    """
    Compute Tsai-Wu failure index for a composite lamina.

    The Tsai-Wu criterion is a quadratic polynomial criterion:
    F1·σ1 + F2·σ2 + F11·σ1² + F22·σ2² + F66·τ12² + 2F12·σ1·σ2 = 1

    Parameters
    ----------
    sigma : np.ndarray
        Stress in ply principal coordinates [σ1, σ2, τ12] in Pa
    strength : StrengthProperties
        Material strength properties
    f12_coefficient : float, optional
        Coefficient for F12 term. Default is -0.5 (conservative).
        F12 = f12_coefficient / sqrt(Xt·Xc·Yt·Yc)

    Returns
    -------
    FailureResult
        Failure analysis result with index and mode

    Notes
    -----
    The strength parameters are:
    - F1 = 1/Xt - 1/Xc (linear fiber term)
    - F2 = 1/Yt - 1/Yc (linear matrix term)
    - F11 = 1/(Xt·Xc) (quadratic fiber term)
    - F22 = 1/(Yt·Yc) (quadratic matrix term)
    - F66 = 1/S² (shear term)
    - F12 = interaction term (typically -0.5/sqrt(Xt·Xc·Yt·Yc))

    The failure index FI is the value of the left-hand side.
    Failure occurs when FI >= 1.

    Examples
    --------
    >>> strength = StrengthProperties(Xt=1500e6, Xc=1500e6, Yt=40e6, Yc=246e6, S12=68e6)
    >>> sigma = np.array([500e6, 20e6, 30e6])  # [σ1, σ2, τ12]
    >>> result = tsai_wu_failure_index(sigma, strength)
    >>> print(f"FI = {result.failure_index:.3f}, Failed = {result.failed}")
    """
    s1, s2, t12 = sigma

    Xt, Xc = strength.Xt, strength.Xc
    Yt, Yc = strength.Yt, strength.Yc
    S = strength.S12

    # Strength parameters
    F1 = 1 / Xt - 1 / Xc
    F2 = 1 / Yt - 1 / Yc
    F11 = 1 / (Xt * Xc)
    F22 = 1 / (Yt * Yc)
    F66 = 1 / S**2
    F12 = f12_coefficient / np.sqrt(Xt * Xc * Yt * Yc)

    # Failure index
    FI = F1 * s1 + F2 * s2 + F11 * s1**2 + F22 * s2**2 + F66 * t12**2 + 2 * F12 * s1 * s2

    failed = FI >= 1.0

    return FailureResult(
        failed=failed,
        failure_index=FI,
        mode=FailureMode.COMBINED if failed else FailureMode.NO_FAILURE,
    )


def hashin_failure_indices(
    sigma: np.ndarray,
    strength: StrengthProperties,
) -> Dict[FailureMode, FailureResult]:
    """
    Compute Hashin failure indices for all failure modes.

    The Hashin criterion distinguishes between four failure modes:
    1. Fiber tension (σ1 > 0)
    2. Fiber compression (σ1 < 0)
    3. Matrix tension (σ2 > 0)
    4. Matrix compression (σ2 < 0)

    Parameters
    ----------
    sigma : np.ndarray
        Stress in ply principal coordinates [σ1, σ2, τ12] in Pa
    strength : StrengthProperties
        Material strength properties

    Returns
    -------
    Dict[FailureMode, FailureResult]
        Dictionary mapping failure modes to their results

    Notes
    -----
    **Fiber Tension (σ1 ≥ 0):**
    (σ1/Xt)² + (τ12/S)² = 1

    **Fiber Compression (σ1 < 0):**
    |σ1/Xc| = 1

    **Matrix Tension (σ2 ≥ 0):**
    (σ2/Yt)² + (τ12/S)² = 1

    **Matrix Compression (σ2 < 0):**
    (σ2/2ST)² + [(Yc/2ST)² - 1](σ2/Yc) + (τ12/S)² = 1

    where ST is the transverse shear strength.

    Examples
    --------
    >>> strength = StrengthProperties(Xt=1500e6, Xc=1500e6, Yt=40e6, Yc=246e6, S12=68e6)
    >>> sigma = np.array([500e6, -100e6, 30e6])
    >>> results = hashin_failure_indices(sigma, strength)
    >>> for mode, result in results.items():
    ...     print(f"{mode.value}: FI = {result.failure_index:.3f}")
    """
    s1, s2, t12 = sigma

    Xt, Xc = strength.Xt, strength.Xc
    Yt, Yc = strength.Yt, strength.Yc
    S = strength.S12
    ST = strength.S23  # Transverse shear strength

    results = {}

    # Fiber tension (σ1 >= 0)
    if s1 >= 0:
        FI_ft = (s1 / Xt) ** 2 + (t12 / S) ** 2
        results[FailureMode.FIBER_TENSION] = FailureResult(
            failed=FI_ft >= 1.0,
            failure_index=FI_ft,
            mode=FailureMode.FIBER_TENSION,
        )
    else:
        # Fiber compression (σ1 < 0)
        FI_fc = (abs(s1) / Xc) ** 2
        results[FailureMode.FIBER_COMPRESSION] = FailureResult(
            failed=FI_fc >= 1.0,
            failure_index=FI_fc,
            mode=FailureMode.FIBER_COMPRESSION,
        )

    # Matrix tension (σ2 >= 0)
    if s2 >= 0:
        FI_mt = (s2 / Yt) ** 2 + (t12 / S) ** 2
        results[FailureMode.MATRIX_TENSION] = FailureResult(
            failed=FI_mt >= 1.0,
            failure_index=FI_mt,
            mode=FailureMode.MATRIX_TENSION,
        )
    else:
        # Matrix compression (σ2 < 0)
        FI_mc = (s2 / (2 * ST)) ** 2 + ((Yc / (2 * ST)) ** 2 - 1) * (s2 / Yc) + (t12 / S) ** 2
        results[FailureMode.MATRIX_COMPRESSION] = FailureResult(
            failed=FI_mc >= 1.0,
            failure_index=FI_mc,
            mode=FailureMode.MATRIX_COMPRESSION,
        )

    return results


def max_stress_failure_index(
    sigma: np.ndarray,
    strength: StrengthProperties,
) -> FailureResult:
    """
    Compute maximum stress failure criterion.

    The maximum stress criterion compares each stress component
    to its corresponding strength independently (no interaction).

    Parameters
    ----------
    sigma : np.ndarray
        Stress in ply principal coordinates [σ1, σ2, τ12] in Pa
    strength : StrengthProperties
        Material strength properties

    Returns
    -------
    FailureResult
        Failure result with the maximum ratio as failure index

    Notes
    -----
    Failure ratios:
    - R1t = σ1/Xt (fiber tension)
    - R1c = |σ1|/Xc (fiber compression)
    - R2t = σ2/Yt (matrix tension)
    - R2c = |σ2|/Yc (matrix compression)
    - R12 = |τ12|/S (shear)

    FI = max(R1, R2, R12)
    """
    s1, s2, t12 = sigma

    Xt, Xc = strength.Xt, strength.Xc
    Yt, Yc = strength.Yt, strength.Yc
    S = strength.S12

    # Compute individual ratios
    if s1 >= 0:
        R1 = s1 / Xt
        mode1 = FailureMode.FIBER_TENSION
    else:
        R1 = abs(s1) / Xc
        mode1 = FailureMode.FIBER_COMPRESSION

    if s2 >= 0:
        R2 = s2 / Yt
        mode2 = FailureMode.MATRIX_TENSION
    else:
        R2 = abs(s2) / Yc
        mode2 = FailureMode.MATRIX_COMPRESSION

    R12 = abs(t12) / S

    # Find maximum
    ratios = [R1, R2, R12]
    modes = [mode1, mode2, FailureMode.SHEAR]

    max_idx = np.argmax(ratios)
    FI = ratios[max_idx]
    mode = modes[max_idx]

    return FailureResult(
        failed=FI >= 1.0,
        failure_index=FI,
        mode=mode,  # Always return the critical mode
    )


def evaluate_ply_failure(
    sigma: np.ndarray,
    strength: StrengthProperties,
    criterion: str = "tsai-wu",
) -> FailureResult:
    """
    Evaluate ply failure using specified criterion.

    Parameters
    ----------
    sigma : np.ndarray
        Stress in ply principal coordinates [σ1, σ2, τ12] in Pa
    strength : StrengthProperties
        Material strength properties
    criterion : str
        Failure criterion: "tsai-wu", "hashin", or "max-stress"

    Returns
    -------
    FailureResult
        Failure result for the specified criterion.
        For Hashin, returns the most critical mode.
    """
    criterion = criterion.lower().replace("_", "-")

    if criterion == "tsai-wu":
        return tsai_wu_failure_index(sigma, strength)

    elif criterion == "hashin":
        results = hashin_failure_indices(sigma, strength)
        # Return most critical mode
        max_result = max(results.values(), key=lambda r: r.failure_index)
        return max_result

    elif criterion == "max-stress":
        return max_stress_failure_index(sigma, strength)

    else:
        raise ValueError(
            f"Unknown criterion: {criterion}. Use 'tsai-wu', 'hashin', or 'max-stress'."
        )


def stress_transformation_matrix(theta_deg: float) -> np.ndarray:
    """
    Compute stress transformation matrix from laminate to ply coordinates.

    Parameters
    ----------
    theta_deg : float
        Ply angle in degrees

    Returns
    -------
    np.ndarray
        3x3 transformation matrix T such that σ_12 = T @ σ_xy

    Notes
    -----
    Transforms stress from laminate (x,y) coordinates to ply
    principal (1,2) coordinates:

    T = [c²    s²     2sc  ]
        [s²    c²    -2sc  ]
        [-sc   sc   c²-s² ]

    where c = cos(θ), s = sin(θ)
    """
    theta = np.radians(theta_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    c2, s2 = c**2, s**2

    return np.array([[c2, s2, 2 * s * c], [s2, c2, -2 * s * c], [-s * c, s * c, c2 - s2]])


def strain_transformation_matrix(theta_deg: float) -> np.ndarray:
    """
    Compute strain transformation matrix from laminate to ply coordinates.

    Parameters
    ----------
    theta_deg : float
        Ply angle in degrees

    Returns
    -------
    np.ndarray
        3x3 transformation matrix T_eps such that ε_12 = T_eps @ ε_xy

    Notes
    -----
    For engineering strains [εxx, εyy, γxy]:

    T_eps = [c²    s²     sc   ]
            [s²    c²    -sc   ]
            [-2sc  2sc   c²-s²]
    """
    theta = np.radians(theta_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    c2, s2 = c**2, s**2

    return np.array([[c2, s2, s * c], [s2, c2, -s * c], [-2 * s * c, 2 * s * c, c2 - s2]])
