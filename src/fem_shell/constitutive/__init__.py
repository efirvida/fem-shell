"""
Constitutive models package for fem_shell.

This package contains material constitutive relations and failure criteria
for composite materials analysis.
"""

from fem_shell.constitutive.failure import (
    FailureMode,
    FailureResult,
    tsai_wu_failure_index,
    hashin_failure_indices,
    max_stress_failure_index,
    evaluate_ply_failure,
)

__all__ = [
    "FailureMode",
    "FailureResult",
    "tsai_wu_failure_index",
    "hashin_failure_indices",
    "max_stress_failure_index",
    "evaluate_ply_failure",
]
