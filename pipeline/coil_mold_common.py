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
import sys
from typing import Dict, Tuple

import numpy as np
import trimesh

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PIPELINE_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from output_utils import (
    PIPELINE_OUTPUT_BASE,
    design_folder_name,
    gradient_project_stem as _default_gradient_project_stem,
    read_active_stem,
    unique_run_dir,
    write_active_stem,
)

# =============================================================================
# USER PARAMETERS — single source of truth for the mold workflow
# =============================================================================

PYCOILGEN_ROOT = os.path.dirname(PROJECT_ROOT)
BASE_DIR = PROJECT_ROOT

# ---- Gradient design (gradiente_belen_santi_main.py) ------------------------
GRADIENT_AXIS = 'y'
TIKHONOV_FACTOR = 2500
NUM_LEVELS = 26

TARGET_RX = 0.125
TARGET_RY = 0.125
TARGET_RZ = 0.125
RESOL_RADIAL = 8
RESOL_ANGULAR = 28

CYL_HEIGHT = 0.43               # [m]
CYL_RADIUS = 0.150              # [m]
CYL_N_CIRC = 200
CYL_N_LONG = 10
CYL_ROT_AXIS = (0, 1, 0)
CYL_ROT_ANGLE = np.pi / 2

CUT_WIDTH = 0.001
POT_OFFSET_FACTOR = 0.5
MIN_LOOP_SIGNIF = 5
NORMAL_SHIFT = -0.005
NORMAL_SHIFT_SMOOTH = [7, 7, 7]

CONDUCTOR_WIDTH = 0.00225
CROSS_SECTION_A_FRAC = 1.6
CROSS_SECTION_B_FRAC = 0.7
CROSS_SECTION_N = 16

ENABLE_FASTHENRY = True
FASTHENRY_BIN = r'C:\Program Files (x86)\FastFieldSolvers\FastHenry2\FastHenry2.exe'
SPECIFIC_CONDUCTIVITY_CONDUCTOR = 1.8e-8
SMOOTH_FACTOR = 3

# ---- Result paths -----------------------------------------------------------
# Pipeline outputs: resultados/pipeline/Gy_tk2500_lvl26/ (or …(2) on re-run).
# Set by init_pipeline_run() before the first pipeline step.
RESULTS_DIR_ENV = 'COIL_MOLD_RESULTS_DIR'
RESULTS_DIR = ''
PROJECT_STEM = ''


def design_run_stem() -> str:
    return design_folder_name(GRADIENT_AXIS, TIKHONOV_FACTOR, NUM_LEVELS)


def gradient_project_stem() -> str:
    return PROJECT_STEM or _default_gradient_project_stem(
        GRADIENT_AXIS, TIKHONOV_FACTOR, NUM_LEVELS,
    )


def set_project_stem(stem: str) -> None:
    global PROJECT_STEM
    PROJECT_STEM = stem
    if RESULTS_DIR:
        write_active_stem(RESULTS_DIR, stem)


def set_results_dir(path: str) -> str:
    """Pin output to an existing run folder (used by the pipeline orchestrator)."""
    global RESULTS_DIR
    RESULTS_DIR = os.path.abspath(path)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def results_dir_from_env() -> str:
    """Return RESULTS_DIR passed by run_coil_mold_pipeline.py to child scripts."""
    return os.environ.get(RESULTS_DIR_ENV, '').strip()


def sync_project_stem_from_disk() -> str:
    """Reload PROJECT_STEM from .active_project_stem after an external step."""
    if not RESULTS_DIR:
        return PROJECT_STEM
    stem = read_active_stem(RESULTS_DIR, gradient_project_stem())
    if stem:
        set_project_stem(stem)
    return PROJECT_STEM


def get_results_dir() -> str:
    """Return the active pipeline run folder (must already be allocated)."""
    if not RESULTS_DIR:
        raise RuntimeError(
            'RESULTS_DIR is not set. Run run_coil_mold_pipeline.py, or call '
            'ensure_pipeline_output_dir() before using path helpers.',
        )
    return RESULTS_DIR


def ensure_pipeline_output_dir() -> str:
    """
    Resolve RESULTS_DIR for a pipeline step.

    Order: existing global → env var from orchestrator → new run folder.
    """
    if RESULTS_DIR:
        return RESULTS_DIR
    env_dir = results_dir_from_env()
    if env_dir:
        return set_results_dir(env_dir)
    return init_pipeline_run()


def init_pipeline_run() -> str:
    """Allocate one run folder under resultados/pipeline/ (once per process)."""
    global RESULTS_DIR, PROJECT_STEM
    if RESULTS_DIR:
        return RESULTS_DIR
    RESULTS_DIR = unique_run_dir(PIPELINE_OUTPUT_BASE, design_run_stem())
    PROJECT_STEM = ''
    return RESULTS_DIR


def target_fields_dir() -> str:
    return os.path.join(get_results_dir(), 'target_fields')


def wire_stem() -> str:
    """Base filename (no extension) for the exported wire STL."""
    return f'{gradient_project_stem()}_wire_0_z'


def wire_stl_path(with_leads: bool = False) -> str:
    from output_utils import resolve_lead_stl_paths, resolve_wire_stl_path

    wire = resolve_wire_stl_path(
        get_results_dir(), GRADIENT_AXIS, TIKHONOV_FACTOR, NUM_LEVELS,
    )
    if not with_leads:
        return wire
    with_leads_path, _, _ = resolve_lead_stl_paths(wire)
    return with_leads_path


def leads_stl_path() -> str:
    """Standalone lead tubes exported by add_coil_leads.py."""
    return os.path.join(get_results_dir(), f'{wire_stem()}_leads_only.stl')


def coil_open_stl_path() -> str:
    """Open gradient loop (cut gap, no leads) exported by add_coil_leads.py."""
    return os.path.join(get_results_dir(), f'{wire_stem()}_coil_open.stl')


def refresh_stl_paths() -> None:
    """Resolve wire / lead STL paths after gradient or leads steps."""
    from output_utils import resolve_lead_stl_paths, resolve_wire_stl_path

    global ALIGN_WIRE_STL, COIL_OPEN_STL, LEADS_WIRE_STL, SUBTRACT_WIRE_STL
    results = get_results_dir()
    ALIGN_WIRE_STL = resolve_wire_stl_path(
        results, GRADIENT_AXIS, TIKHONOV_FACTOR, NUM_LEVELS,
    )
    SUBTRACT_WIRE_STL, COIL_OPEN_STL, LEADS_WIRE_STL = resolve_lead_stl_paths(
        ALIGN_WIRE_STL,
    )


# ---- Fusion 360 printable shell halves --------------------------------------
SHELL_STL_DIR = os.path.join(PROJECT_ROOT, 'assets', 'cilindros_gradientes_grandes')

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
# Align with closed coil-only STL; subtract using the open cable + leads.
# Pad → subtract → restore (direct mesh boolean, no voxel remesh).
#
# SUBTRACT_MODE:
#   'with_leads'              — one boolean on *_with_leads.stl
#   'with_leads_by_component' — coil_open + leads_only sequentially (stable)
#   'two_pass'                — legacy closed coil + leads_only (extra gap groove)
SUBTRACT_MODE = 'with_leads'
ALIGN_WIRE_STL = ''
COIL_OPEN_STL = ''
LEADS_WIRE_STL = ''
SUBTRACT_WIRE_STL = ''

# Normal expansion along wire surface normals before boolean subtraction.
# Widens grooves slightly and helps merge overlapping turns in dense areas.
# Too large → grooves wider than wire; too small → thin flash in crossovers.
GROOVE_EXPANSION = 0         # [m] 0.35 mm per side — coil pass
LEAD_GROOVE_EXPANSION = 0.00025    # [m] 0.25 mm per side — leads-only 2nd pass

# Pad the shell outward before coil subtract so wire that extends past the
# design outer radius still cuts solid material (single even-odd pass).
# Must cover oval semi-axis (~3.6 mm); wire mesh peaks ~2.4 mm past Fusion outer_r.
_WIRE_SEMI_A = CROSS_SECTION_A_FRAC * CONDUCTOR_WIDTH
SHELL_OUTER_PAD = max(0.0005, 0.65 * _WIRE_SEMI_A)

# Extra boolean on each lead tube after the main with_leads subtract (stable).
LEADS_SECOND_SUBTRACT = True

# Second full-mesh subtract (legacy) — off when using pad+restore below.
COIL_SECOND_SUBTRACT = False
COIL_SECOND_EXPANSION = 0.00050   # [m] only used if COIL_SECOND_SUBTRACT is True

# Extra uniform outer peel after restore-to-design (usually 0 with pad workflow).
OUTER_SKIN_TRIM = 0.0              # [m]; set >0 only for fine flash cleanup

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
RUN_GRADIENT = True             # True -> invoke pyCoilGen (slow)
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
    import re
    base, ext = os.path.splitext(subtract_path)
    coil_only = re.sub(r'_with_leads(?:\(\d+\))?$', '', base) + ext
    if os.path.isfile(coil_only):
        return coil_only
    legacy = base[:-len('_with_leads')] + ext if base.endswith('_with_leads') else base
    if os.path.isfile(legacy):
        return legacy
    return subtract_path


def format_dims_mm(dims: Dict[str, float], prefix: str = '') -> str:
    return (f"{prefix}axial [{dims['axial_min']*1000:.1f}, "
            f"{dims['axial_max']*1000:.1f}] mm  "
            f"centre {dims['axial_center']*1000:.2f} mm  "
            f"radial [{dims['inner_r']*1000:.2f}, {dims['outer_r']*1000:.2f}] mm")
