"""
Conservative force clipping for FSI coupling.
"""

from typing import Dict, Optional, Tuple

import numpy as np


class ForceClipper:
    """
    Conservative force clipping to prevent pathological spikes in FSI coupling.

    Only clips force magnitudes that exceed a specified threshold. Does NOT smooth
    or average forces—preserves the actual CFD solution in the normal range.

    Strategy: Detect when nodal force magnitude is excessive (e.g., > 10x typical),
    and scale it down to a reasonable cap. All other forces pass through unchanged.

    Parameters
    ----------
    force_max_cap : Optional[float]
        Hard cap on per-node force magnitude. If None, no clipping is applied.
        Recommended: estimate from steady-state or pre-simulation (e.g., 500 kN for blades).
    """

    def __init__(self, force_max_cap: Optional[float] = None):
        self.force_max_cap = force_max_cap
        self._clipped_count = 0
        self._total_count = 0

    def apply(self, force_data: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Apply conservative clipping to force data.

        Parameters
        ----------
        force_data : np.ndarray
            Raw force data from preCICE. Shape: (n_nodes, n_dims) or (n_components,).

        Returns
        -------
        clipped_force : np.ndarray
            Force data with excessive magnitudes clipped, same shape as input.
        diagnostics : Dict[str, float]
            Statistics: mean, max, n_clipped (count of clipped nodes).
        """
        if self.force_max_cap is None:
            # No clipping
            force_mags = (
                np.linalg.norm(force_data, axis=1) if force_data.ndim == 2 else np.abs(force_data)
            )
            return force_data.copy(), {
                "mean": float(np.mean(force_mags)),
                "max": float(np.max(force_mags)),
                "n_clipped": 0,
                "cap": None,
            }

        # Ensure 2D for consistent processing
        if force_data.ndim == 1:
            n_dims = force_data.shape[0] if len(force_data.shape) == 1 else 1
            force_2d = force_data.reshape(-1, max(n_dims, 1))
            reshape_1d = True
        else:
            force_2d = force_data
            reshape_1d = False

        force_clipped = force_2d.copy()

        # Compute per-node magnitude
        force_mags = np.linalg.norm(force_clipped, axis=1, keepdims=True)
        force_mags = np.maximum(force_mags, 1e-12)  # Avoid division by zero

        # Identify and clip excessive magnitudes
        clip_mask = force_mags[:, 0] > self.force_max_cap
        n_clipped = np.sum(clip_mask)

        if n_clipped > 0:
            scale_factors = self.force_max_cap / force_mags[clip_mask, 0]
            force_clipped[clip_mask] *= scale_factors[:, np.newaxis]

        self._clipped_count += n_clipped
        self._total_count += force_2d.shape[0]

        # Diagnostics
        force_mags_final = np.linalg.norm(force_clipped, axis=1)
        diagnostics = {
            "mean": float(np.mean(force_mags_final)),
            "max": float(np.max(force_mags_final)),
            "n_clipped": int(n_clipped),
            "cap": self.force_max_cap,
        }

        # Reshape to match input if needed
        if reshape_1d:
            return force_clipped.flatten(), diagnostics
        return force_clipped, diagnostics

    def get_statistics(self) -> Dict[str, float]:
        """Return overall clipping statistics."""
        if self._total_count == 0:
            return {"clipped_fraction": 0.0, "total_nodes_processed": 0}
        return {
            "clipped_fraction": self._clipped_count / self._total_count,
            "total_nodes_processed": self._total_count,
        }
