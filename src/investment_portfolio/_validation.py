"""Shared input-validation utilities."""

from __future__ import annotations

import numpy as np


def validate_finite(*, arr: np.ndarray, name: str = "input") -> None:
    """Raise if *arr* contains NaN or Inf values.

    Args:
        arr: Array to check.
        name: Human-readable name for error messages.

    Raises:
        ValueError: If any element is NaN or +/-Inf.
    """
    if not np.all(np.isfinite(arr)):
        msg = f"{name} contains NaN or Inf"
        raise ValueError(msg)
