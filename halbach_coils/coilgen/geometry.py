"""
Geometry helpers shared across the gradient / leads / shell steps.

Contents
--------
- Elementary rotation matrices (rotx / roty / rotz).
- Rodrigues rotation for the cylinder mesh rotation.
- ``internal_field_axis``: physical G{x,y,z} -> pyCoilGen internal axis.
- ``rotated_cylinder_axis``: bore axis after the R_y(pi/2) mesh rotation.
- ``build_target_field_file``: spherical target-field .npy for an axis.
- Fusion STL dimension detection and wire STL measurement.

All angles in radians, all lengths in metres unless noted.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import trimesh


# ---------------------------------------------------------------------------
# Elementary rotations
# ---------------------------------------------------------------------------

def rotx(t):
    return np.array([[1, 0, 0],
                     [0, np.cos(t), -np.sin(t)],
                     [0, np.sin(t), np.cos(t)]])


def roty(t):
    return np.array([[np.cos(t), 0, np.sin(t)],
                     [0, 1, 0],
                     [-np.sin(t), 0, np.cos(t)]])


def rotz(t):
    return np.array([[np.cos(t), -np.sin(t), 0],
                     [np.sin(t), np.cos(t), 0],
                     [0, 0, 1]])


def rodrigues_rotation_matrix(axis, angle) -> np.ndarray:
    """Rotation matrix for a rotation of ``angle`` rad about ``axis``."""
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


# ---------------------------------------------------------------------------
# Axis mapping
# ---------------------------------------------------------------------------

_AXIS_MAP = {'x': 'y', 'y': 'z', 'z': 'x'}


def internal_field_axis(gradient_axis: str) -> str:
    """
    pyCoilGen ``field_shape_function`` value for a physical gradient axis.

    The cylinder is rotated (CYL_ROT_AXIS=Y, CYL_ROT_ANGLE=90 deg) so the
    generated fields are permuted: physical Gx/Gy/Gz map to internal y/z/x.
    Wire STLs are named ``..._wire_0_{internal}.stl``.
    """
    return _AXIS_MAP[gradient_axis.lower()]


# Target-grid rotation per physical axis (MATLAB Lin1/Lin2/Lin3 convention).
_ROTATIONS_BY_AXIS = {
    'x': rotx(0),            # identity  -> Lin1 (Gx)
    'y': rotz(np.pi / 2),    # +90 deg Z -> Lin2 (Gy)
    'z': roty(-np.pi / 2),   # -90 deg Y -> Lin3 (Gz)
}
_FNAME_BY_AXIS = {'x': 'lin_1', 'y': 'lin_2', 'z': 'lin_3'}
_FILENAME_BY_AXIS = {
    'x': 'OSI2_GradTarget_Lin1.npy',
    'y': 'OSI2_GradTarget_Lin2.npy',
    'z': 'OSI2_GradTarget_Lin3.npy',
}


def rotated_cylinder_axis(rot_axis=(0, 1, 0), rot_angle=np.pi / 2) -> np.ndarray:
    """Bore axis (unit vector) after applying the cylinder mesh rotation."""
    R = rodrigues_rotation_matrix(rot_axis, rot_angle)
    return R @ np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Target field file
# ---------------------------------------------------------------------------

def build_target_field_file(axis: str,
                            target_rx: float,
                            target_ry: float,
                            target_rz: float,
                            resol_radial: int,
                            resol_angular: int,
                            target_dir: str) -> tuple[str, str]:
    """
    Write the .npy target-field file for ``axis`` (an internal pyCoilGen axis).

    Builds a spherical (r, theta, phi) grid inside the target ellipsoid and
    stores the scalar field ``x1`` (linear in X); rotating the coordinate
    cloud relabels it to the requested axis — same convention as the MATLAB
    script. Returns ``(path, field_name)``.
    """
    d1 = 1 / (resol_radial - 1)
    d2 = np.pi / (resol_angular - 1)
    d3 = 2 * np.pi / (resol_angular - 1)
    ra = np.arange(0, 1 + d1 / 2, d1)
    theta = np.arange(0, np.pi + d2 / 2, d2)
    phi = np.arange(-np.pi, np.pi + d3 / 2, d3)
    R, T, P = np.meshgrid(ra, theta, phi, indexing='ij')   # mimic MATLAB ndgrid
    r, t, p = R.ravel(), T.ravel(), P.ravel()

    x1 = target_rx * r * np.sin(t) * np.cos(p)
    x2 = target_ry * r * np.sin(t) * np.sin(p)
    x3 = target_rz * r * np.cos(t)
    points = np.vstack([x1, x2, x3])

    coords = _ROTATIONS_BY_AXIS[axis] @ points
    fname = _FNAME_BY_AXIS[axis]
    path = os.path.join(target_dir, _FILENAME_BY_AXIS[axis])

    data = {'coords': coords.astype(np.float64),
            fname: x1.astype(np.float64)}
    # pyCoilGen does [loaded] = np.load(..., allow_pickle=True)
    np.save(path, np.array([data], dtype=object), allow_pickle=True)
    return path, fname


# ---------------------------------------------------------------------------
# Fusion shell + wire STL measurement
# ---------------------------------------------------------------------------

def detect_fusion_cylinder_dims(stl_a: str, stl_b: str) -> Dict[str, float]:
    """
    Measure the full Fusion cylinder from both printable halves.

    Fusion STLs are in millimetres with the cylinder axis along +Z.
    Returns axial span and radial range in metres (after mm -> m scaling).
    """
    zs_mm = []
    rs_mm = []
    for path in (stl_a, stl_b):
        tm = trimesh.load(path)
        zs_mm.append(tm.vertices[:, 2])
        rs_mm.append(np.linalg.norm(tm.vertices[:, :2], axis=1))

    z_all = np.concatenate(zs_mm)
    r_all = np.concatenate(rs_mm)
    z_min_mm = float(z_all.min())
    z_max_mm = float(z_all.max())

    return {
        'axial_min_m': z_min_mm * 0.001,
        'axial_max_m': z_max_mm * 0.001,
        'axial_center_m': (z_min_mm + z_max_mm) * 0.0005,
        'axial_length_m': (z_max_mm - z_min_mm) * 0.001,
        'inner_r_m': float(r_all.min()) * 0.001,
        'outer_r_m': float(r_all.max()) * 0.001,
        'z_min_mm': z_min_mm,
        'z_max_mm': z_max_mm,
    }


def measure_wire_dims(stl_path: str,
                      rot_axis=(0, 1, 0),
                      rot_angle=np.pi / 2) -> Dict[str, float]:
    """Axial and radial extent of a wire STL in the pyCoilGen frame [m]."""
    axis = rotated_cylinder_axis(rot_axis, rot_angle)
    tm = trimesh.load(stl_path)
    v = np.asarray(tm.vertices, dtype=np.float64)
    axial_coords = v @ axis
    axial_proj = np.outer(axial_coords, axis)
    radial_vecs = v - axial_proj
    radii = np.linalg.norm(radial_vecs, axis=1)

    return {
        'inner_r': float(radii.min()),
        'outer_r': float(radii.max()),
        'axial_min': float(axial_coords.min()),
        'axial_max': float(axial_coords.max()),
        'axial_center': float((axial_coords.min() + axial_coords.max()) / 2.0),
        'axial_extent': float(axial_coords.max() - axial_coords.min()),
        'path': stl_path,
    }


def format_dims_mm(dims: Dict[str, float], prefix: str = '') -> str:
    return (f"{prefix}axial [{dims['axial_min']*1000:.1f}, "
            f"{dims['axial_max']*1000:.1f}] mm  "
            f"centre {dims['axial_center']*1000:.2f} mm  "
            f"radial [{dims['inner_r']*1000:.2f}, {dims['outer_r']*1000:.2f}] mm")
