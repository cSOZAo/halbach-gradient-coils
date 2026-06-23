"""
Shared parameters and geometry helpers for the coil negative-mold workflow:

    gradiente_belen_santi_main.py  ->  wire STL
    add_coil_leads.py              ->  wire STL with leads
    generate_coil_shell_split.py   ->  printable shell halves

Edit the USER PARAMETERS block below once; import this module from the
individual scripts or run ``run_coil_mold_pipeline.py``.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import trimesh

# =============================================================================
# USER PARAMETERS — single source of truth for the mold workflow
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Gradient design (must match gradiente_belen_santi_main.py) --------------
GRADIENT_AXIS = 'y'
TIKHONOV_FACTOR = 2500
NUM_LEVELS = 26

CYL_HEIGHT = 0.44               # [m]
CYL_RADIUS = 0.152              # [m]
CYL_ROT_AXIS = (0, 1, 0)
CYL_ROT_ANGLE = np.pi / 2

CONDUCTOR_WIDTH = 0.0018
CROSS_SECTION_A_FRAC = 1.0
CROSS_SECTION_B_FRAC = 1.0
CROSS_SECTION_N = 12

# ---- Result paths -----------------------------------------------------------
RESULTS_DIR = os.path.join(
    BASE_DIR, 'resultados', f'resultados_grande_{GRADIENT_AXIS}', 'final_2',
)

def wire_stem() -> str:
    """Base filename (no extension) for the exported wire STL."""
    return f'Gradient_G{GRADIENT_AXIS}_tk{TIKHONOV_FACTOR}_lvl{NUM_LEVELS}_wire_0_z'


def wire_stl_path(with_leads: bool = False) -> str:
    stem = wire_stem()
    if with_leads:
        stem += '_with_leads'
    return os.path.join(RESULTS_DIR, f'{stem}.stl')


def leads_stl_path() -> str:
    """Standalone lead tubes exported by add_coil_leads.py (pass 2 of shell cut)."""
    return os.path.join(RESULTS_DIR, f'{wire_stem()}_leads_only.stl')


# ---- Fusion 360 printable shell halves --------------------------------------
SHELL_STL_DIR = os.path.join(BASE_DIR, 'cilindros_gradientes_grandes')

# Which cylinder pair to carve.  Dimensions are read from the STL files at
# runtime (detect_fusion_cylinder_dims); table below is reference only.
#
# Measured 2026-06-22 from cilindros_gradientes_grandes/ (mm, axis Z):
#   Layer  Halves      Z span        Length  Inner R  Outer R  Wall
#   1      g_1a/g_1b   0 – 446.5     446.5   152.4    160.0    7.6
#   2      g_2a/g_2b   0 – 446.5     446.5   143.8    151.4    7.6
#   3      g_3a/g_3b   0 – 446.5     446.5   135.2    143.0    7.8
#
# Half split (zigzag junction):  g_*a → z 213.25–446.5,  g_*b → z 0–233.25
FUSION_SHELL_REFERENCE_MM = {
    1: {'z_min': 0.0, 'z_max': 446.5, 'length': 446.5, 'inner_r': 152.4, 'outer_r': 160.0},
    2: {'z_min': 0.0, 'z_max': 446.5, 'length': 446.5, 'inner_r': 143.8, 'outer_r': 151.4},
    3: {'z_min': 0.0, 'z_max': 446.5, 'length': 446.5, 'inner_r': 135.2, 'outer_r': 143.0},
}
GRADIENT_LAYER = 2

# ---- Shell subtraction ------------------------------------------------------
# Two-pass direct mesh boolean (no voxel remesh — keeps full-depth grooves):
#   pass 1: coil-only STL  (gradient windings)
#   pass 2: leads-only STL (inlet/outlet paths, exported by add_coil_leads)
ALIGN_WIRE_STL = wire_stl_path(with_leads=False)
LEADS_WIRE_STL = leads_stl_path()
SUBTRACT_WIRE_STL = wire_stl_path(with_leads=True)   # legacy / fallback

# Normal expansion along wire surface normals before boolean subtraction.
# Widens grooves slightly and helps merge overlapping turns in dense areas.
# Too large → grooves wider than wire; too small → thin flash in crossovers.
GROOVE_EXPANSION = 0.00035         # [m] 0.35 mm per side — coil pass
LEAD_GROOVE_EXPANSION = 0.00025    # [m] 0.25 mm per side — leads pass

# Second full-mesh subtract with slightly larger expansion removes flash in
# crossover zones without shaving the whole outer cylinder (which costs depth).
COIL_SECOND_SUBTRACT = True
COIL_SECOND_EXPANSION = 0.00050   # [m] total expansion for pass 1b (> GROOVE_EXPANSION)

# Uniform outer peel — keep small; prefer COIL_SECOND_SUBTRACT above.
OUTER_SKIN_TRIM = 0.0002           # [m] 0.2 mm; set 0 to disable

# Voxel remesh (disabled by default — coarse pitch yields blocky / hollow grooves).
RESOLVE_SELF_INTERSECTIONS = False
VOXEL_PITCH = 0.0004
VOXEL_SLAB_LENGTH = 0.090
VOXEL_SLAB_OVERLAP = 0.012
SMOOTH_ITERATIONS = 8
CIRCULAR_SEGMENTS = 256
OUTPUT_IN_MM = True
OUTPUT_DIR = ''                  # empty -> next to subtract STL

# ---- Lead geometry (add_coil_leads.py) --------------------------------------
LEAD_DIRECTION = np.array([-1.0, 0.0, 0.0])
SECTOR_MIN_Z = 0.10
SECTOR_MAX_ABS_Y = 0.05
CYL_AXIS = None                  # None -> infer from LEAD_DIRECTION
SHELL_RADIUS = None              # None -> infer from mesh

CUT_LOOP_LENGTH = 0.040
GAP_AXIAL_LENGTH = 0.012
WIRE_ISOLATE_HALF = 0.008
TANGENT_RADIUS = 0.006
WIRE_TANGENT_RUN = 0.004
FACE_TOWARD_GAP = 0.003
PEEL_OUT = 0.006
LEAD_LENGTH = 0.02
LEAD_BLEND = 0.030
TIP_FAN = 0.015
LEAD_STEPS = 128
LEAD_0_SPREAD_SIGN = 1
LEAD_1_SPREAD_SIGN = -1
EXIT_DIRECTION = None

CS_BLEND_RINGS = 8
JUNCTION_RIGID_STEPS = 2
JUNCTION_PLANE_RINGS = 4

# ---- Pipeline control (run_coil_mold_pipeline.py) ---------------------------
RUN_GRADIENT = False             # True -> invoke pyCoilGen (slow)
RUN_LEADS = True
RUN_SHELL = True

# =============================================================================
# Geometry helpers
# =============================================================================


def rodrigues_rotation_matrix(axis, angle) -> np.ndarray:
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def rotated_cylinder_axis(rot_axis=CYL_ROT_AXIS, rot_angle=CYL_ROT_ANGLE) -> np.ndarray:
    R = rodrigues_rotation_matrix(rot_axis, rot_angle)
    return R @ np.array([0.0, 0.0, 1.0])


def shell_half_paths(layer: int = GRADIENT_LAYER) -> Tuple[str, str]:
    stl_a = os.path.join(SHELL_STL_DIR, f'g_{layer}a.stl')
    stl_b = os.path.join(SHELL_STL_DIR, f'g_{layer}b.stl')
    return stl_a, stl_b


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
                      rot_axis=CYL_ROT_AXIS,
                      rot_angle=CYL_ROT_ANGLE) -> Dict[str, float]:
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


def derive_align_wire_path(subtract_path: str) -> str:
    """Return the coil-only STL path paired with a *_with_leads* subtract STL."""
    base, ext = os.path.splitext(subtract_path)
    if base.endswith('_with_leads'):
        candidate = base[:-len('_with_leads')] + ext
        if os.path.isfile(candidate):
            return candidate
    return subtract_path


def format_dims_mm(dims: Dict[str, float], prefix: str = '') -> str:
    return (f"{prefix}axial [{dims['axial_min']*1000:.1f}, "
            f"{dims['axial_max']*1000:.1f}] mm  "
            f"centre {dims['axial_center']*1000:.2f} mm  "
            f"radial [{dims['inner_r']*1000:.2f}, {dims['outer_r']*1000:.2f}] mm")
