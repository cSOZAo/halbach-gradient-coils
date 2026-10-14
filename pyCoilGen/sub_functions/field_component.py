"""Projection onto B0 in the geometry's Cartesian frame.

The low-level engine retains +Z as its default for compatibility. The Halbach
workflow supplies its selected physical B0 explicitly (default +Y).
"""

import numpy as np


def field_direction(input_args=None):
    vector = np.asarray(getattr(input_args, 'b_0_direction', [0, 0, 1]), dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError('b_0_direction must contain three finite components')
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError('b_0_direction must be nonzero')
    return vector / norm


def project_field(field, input_args=None):
    """Project a (3, ...) Cartesian field or sensitivity tensor onto B0."""
    return np.tensordot(field_direction(input_args), field, axes=(0, 0))
