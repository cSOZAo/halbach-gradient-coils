"""
Main design script — port of script_belen_santi.m, reorganized for
easy tuning and with custom metrics/plots.

Keeps the same underlying logic as the frozen Gx/Gy/Gz copies
(spherical target grid, oval conductor cross-section, cylinder mesh
rotated so its axis is perpendicular to Halbach B0 along +Y), but:
  * All user-tunable parameters live in the block at the top of the
    file (USER PARAMETERS). Everything below is derived from them.
  * Built-in pyCoilGen plots are replaced by three custom metrics:
      1) slope of the produced gradient [mT / (m · A)]
      2) visual comparison target field vs generated field
      3) scatter of generated field + linear fit + RMSE [mT / (m · A)]
"""

import os                                          # filesystem paths
import shutil                                      # executable lookup
import sys                                         # local package path
import logging                                     # pyCoilGen logger
import numpy as np                                 # arrays / math
import matplotlib.pyplot as plt                    # plots
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (enables 3D axes)

# Prefer the pyCoilGen package in this workspace over any installed copy.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from pyCoilGen.pyCoilGen_release import pyCoilGen  # solver entry point


# =============================================================================
# USER PARAMETERS — edit these to change the coil design
# =============================================================================

# ---- Gradient direction --------------------------------------------------
GRADIENT_AXIS   = 'y'           # 'x' | 'y' | 'z' — which linear gradient
                                # (Lin1=x, Lin2=y, Lin3=z in MATLAB terms)

# ---- Target region (ellipsoid inside which the gradient must be linear) -
TARGET_RX       = 0.125          # [m] target ellipsoid semi-axis along X
TARGET_RY       = 0.125          # [m]                             along Y
TARGET_RZ       = 0.125          # [m]                             along Z
RESOL_RADIAL    = 8             # radial samples of the spherical grid
RESOL_ANGULAR   = 28            # angular samples (theta & phi)

# ---- Cylinder mesh (where the wire windings live) -----------------------
CYL_HEIGHT      = 0.43          # [m] cylinder length (along its axis)
CYL_RADIUS      = 0.151             # [m] cylinder radius (first of MATLAB radii)
CYL_N_CIRC      = 200           # circumferential mesh divisions
CYL_N_LONG      = 10            # longitudinal  mesh divisions
CYL_ROT_AXIS    = (0, 1, 0)     # rotation axis applied to the mesh
CYL_ROT_ANGLE   = np.pi / 2     # [rad] rotation angle — aligns axis perp. to B0

# ---- Coil / stream function design -------------------------------------
NUM_LEVELS          = 26         # number of SF levels = number of windings
TIKHONOV_FACTOR     = 2500       # Tikhonov regularization (higher = smoother)
CUT_WIDTH           = 0.001     # [m] interconnection cut width
POT_OFFSET_FACTOR   = 0.5       # offset 0..1 for min/max contours
MIN_LOOP_SIGNIF     = 5         # [%] min field contribution to keep a loop
NORMAL_SHIFT        = -0.005    # [m] separation of go/return paths
NORMAL_SHIFT_SMOOTH = [7, 7, 7] # smoothing for the normal-shift algorithm

# ---- Conductor cross-section (oval groove profile) ---------------------
CONDUCTOR_WIDTH     = 0.0023   # [m] conductor width (scales the oval)
CROSS_SECTION_N     = 12        # points around the oval (closed curve)
CROSS_SECTION_A_FRAC = 2.0      # X semi-axis = A_FRAC * CONDUCTOR_WIDTH
CROSS_SECTION_B_FRAC = 1.0      # Y semi-axis = B_FRAC * CONDUCTOR_WIDTH

# ---- Electrical metrics ------------------------------------------------
# FastHenry models the final wire path with rectangular conductor segments.
# To match the oval's cross-sectional area (pi * A * B) and width, we set the
# rectangular width to 2*B and height to (pi/2)*A.
ENABLE_FASTHENRY = True
FASTHENRY_BIN = r'C:\Program Files (x86)\FastFieldSolvers\FastHenry2\FastHenry2.exe'
FASTHENRY_CONDUCTOR_WIDTH = 2.0 * CROSS_SECTION_B_FRAC * CONDUCTOR_WIDTH   # [m] Segment width (along surface)
FASTHENRY_CONDUCTOR_HEIGHT = (np.pi / 2.0) * CROSS_SECTION_A_FRAC * CONDUCTOR_WIDTH # [m] Segment height (normal)
SPECIFIC_CONDUCTIVITY_CONDUCTOR = 1.8e-8      # [Ohm*m] copper resistivity used by pyCoilGen

# ---- Output ------------------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
RUTA_SALIDA  = os.path.join(SCRIPT_DIR, 'output')
TARGET_DIR   = os.path.join(RUTA_SALIDA, 'target_fields')

# =============================================================================
# END USER PARAMETERS — do not edit below unless you know what you are doing
# =============================================================================

# Map the requested gradient axis to the internal configuration.
# Because the cylinder is rotated (CYL_ROT_AXIS = Y, CYL_ROT_ANGLE = 90 deg),
# the generated fields are permuted relative to the unrotated cylinder:
# - internal 'y' configuration generates a physical X gradient
# - internal 'z' configuration generates a physical Y gradient
# - internal 'x' configuration generates a physical Z gradient
INTERNAL_AXIS = {'x': 'y', 'y': 'z', 'z': 'x'}[GRADIENT_AXIS]

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

os.makedirs(RUTA_SALIDA, exist_ok=True)
os.makedirs(TARGET_DIR,  exist_ok=True)


def resolve_fasthenry_bin(configured_path):
    """Return the configured FastHenry binary, or a PATH match if present."""
    candidates = []
    if configured_path:
        candidates.append(os.path.expandvars(os.path.expanduser(configured_path)))
    for exe_name in ('FastHenry2.exe', 'fasthenry.exe', 'fasthenry'):
        found = shutil.which(exe_name)
        if found:
            candidates.append(found)
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return candidates[0] if candidates else ''


FASTHENRY_BIN_RESOLVED = resolve_fasthenry_bin(FASTHENRY_BIN)
FASTHENRY_AVAILABLE = bool(FASTHENRY_BIN_RESOLVED and os.path.isfile(FASTHENRY_BIN_RESOLVED))


# =============================================================================
# 1. ROTATION MATRICES (used to orient the target grids)
# =============================================================================
# Three axis-aligned rotations. Lin1 uses identity (Gx), Lin2 uses Rz(+90°)
# (Gy), Lin3 uses Ry(-90°) (Gz) — same convention as MATLAB script.

def rotx(t): return np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
def roty(t): return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
def rotz(t): return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])

_rotations_by_axis = {
    'x': rotx(0),            # identity  → Lin1 (Gx)
    'y': rotz(np.pi / 2),    # +90° Z    → Lin2 (Gy)
    'z': roty(-np.pi / 2),   # -90° Y    → Lin3 (Gz)
}
_fname_by_axis   = {'x': 'lin_1', 'y': 'lin_2', 'z': 'lin_3'}
_filename_by_ax  = {'x': 'OSI2_GradTarget_Lin1.npy',
                    'y': 'OSI2_GradTarget_Lin2.npy',
                    'z': 'OSI2_GradTarget_Lin3.npy'}


# =============================================================================
# 2. BUILD THE TARGET FIELD FILE
# =============================================================================
# Spherical (r, theta, phi) grid inside the target ellipsoid; the scalar
# field stored is x1 (linear in X) — rotating the coords relabels it.

def build_target_field_file(axis):
    """Write the .npy target file for the requested axis."""
    d1 = 1 / (RESOL_RADIAL - 1)
    d2 = np.pi / (RESOL_ANGULAR - 1)
    d3 = 2 * np.pi / (RESOL_ANGULAR - 1)
    ra    = np.arange(0, 1 + d1 / 2, d1)
    theta = np.arange(0, np.pi + d2 / 2, d2)
    phi   = np.arange(-np.pi, np.pi + d3 / 2, d3)
    R, T, P = np.meshgrid(ra, theta, phi, indexing='ij')   # mimic MATLAB ndgrid
    r, t, p = R.ravel(), T.ravel(), P.ravel()

    # Cartesian coordinates on the ellipsoidal shells
    x1 = TARGET_RX * r * np.sin(t) * np.cos(p)
    x2 = TARGET_RY * r * np.sin(t) * np.sin(p)
    x3 = TARGET_RZ * r * np.cos(t)
    points = np.vstack([x1, x2, x3])

    # Rotate the coordinate cloud, keep scalar values = x1 (like MATLAB)
    coords = _rotations_by_axis[axis] @ points
    fname  = _fname_by_axis[axis]
    path   = os.path.join(TARGET_DIR, _filename_by_ax[axis])

    data = {'coords': coords.astype(np.float64),
            fname: x1.astype(np.float64)}
    # pyCoilGen does [loaded] = np.load(..., allow_pickle=True)
    np.save(path, np.array([data], dtype=object), allow_pickle=True)
    return path, fname


target_path, target_fname = build_target_field_file(INTERNAL_AXIS)
print(f"[target] built {os.path.basename(target_path)}  (field = {target_fname})")


# =============================================================================
# 3. CONDUCTOR CROSS-SECTION (oval — closed ellipse)
# =============================================================================
# Small closed ellipse sampled uniformly in angle. The array is the groove
# profile used when sweeping the wire into a 3D STL.

_theta = np.linspace(0, 2 * np.pi, CROSS_SECTION_N, endpoint=True)
cross_sectional_points = np.vstack([
    CROSS_SECTION_A_FRAC * CONDUCTOR_WIDTH * np.sin(_theta),
    CROSS_SECTION_B_FRAC * CONDUCTOR_WIDTH * np.cos(_theta),
])


# =============================================================================
# 4. RUN pyCoilGen
# =============================================================================
# Assemble the arg dict and invoke the solver. Every valid key is registered
# in pyCoilGen/sub_functions/parse_input.py or in a mesh/export plugin.

os.chdir(RUTA_SALIDA)   # pyCoilGen resolves target_fields/ relative to CWD

cylinder_mesh_parameter_list = [
    CYL_HEIGHT, CYL_RADIUS,          # height, radius  [m]
    CYL_N_CIRC, CYL_N_LONG,          # divisions: circular, longitudinal
    *CYL_ROT_AXIS,                   # rotation axis (unit vector)
    CYL_ROT_ANGLE,                   # rotation angle [rad]
]

arg_dict = {
    # --- target field ---
    'target_field_definition_file':       os.path.basename(target_path),
    'target_field_definition_field_name': target_fname,
    # --- coil mesh ---
    'coil_mesh_file':               'create cylinder mesh',
    'cylinder_mesh_parameter_list': cylinder_mesh_parameter_list,
    'surface_is_cylinder_flag':     True,
    # --- stream function / winding ---
    'min_loop_significance':        MIN_LOOP_SIGNIF,
    'levels':                       NUM_LEVELS,
    'pot_offset_factor':            POT_OFFSET_FACTOR,
    'interconnection_cut_width':    CUT_WIDTH,
    # --- conductor geometry ---
    'conductor_cross_section_width': FASTHENRY_CONDUCTOR_WIDTH,
    'conductor_cross_section_height': FASTHENRY_CONDUCTOR_HEIGHT,
    'specific_conductivity_conductor': SPECIFIC_CONDUCTIVITY_CONDUCTOR,
    'cross_sectional_points':        cross_sectional_points.tolist(),
    'normal_shift_length':           NORMAL_SHIFT,
    'normal_shift_smooth_factors':   NORMAL_SHIFT_SMOOTH,
    # --- post-processing ---
    'skip_postprocessing':          False,
    'skip_inductance_calculation':  not ENABLE_FASTHENRY,
    'fasthenry_bin':                FASTHENRY_BIN_RESOLVED,
    'smooth_factor':                3,
    'make_cylindrical_pcb':         False,
    'save_stl_flag':                True,
    # CAD_filename controls which STLs are written. The default template
    # ('{project}_{mesh}_{part_index}_{field_function}.stl') makes
    # export_cad_file.py iterate over ['surface', 'wire'], producing both a
    # cylinder-surface STL and a wire STL. Removing the '{mesh}' placeholder
    # keeps the wire export only and skips the surface STL.
    'CAD_filename':                 '{project}_wire_{part_index}_{field_function}.stl',
    # --- optimization ---
    'tikhonov_reg_factor':          TIKHONOV_FACTOR,
    # --- output ---
    'output_directory':             RUTA_SALIDA,
    'field_shape_function':         INTERNAL_AXIS,
    'project_name':                 f'Gradient_G{GRADIENT_AXIS}'
                                    f'_tk{int(TIKHONOV_FACTOR)}_lvl{NUM_LEVELS}',
}

print("\n" + "=" * 70)
print(f"  Running pyCoilGen — G{GRADIENT_AXIS.upper()}  |  Tikhonov={TIKHONOV_FACTOR:g}"
      f"  |  levels={NUM_LEVELS}")
print(f"  (Internal mapped axis : {INTERNAL_AXIS})")
print(f"  FastHenry enabled     : {ENABLE_FASTHENRY}")
print(f"  FastHenry binary      : {FASTHENRY_BIN_RESOLVED or 'not configured'}")
print(f"  FastHenry available   : {FASTHENRY_AVAILABLE}")
if ENABLE_FASTHENRY and not FASTHENRY_AVAILABLE:
    print("  WARNING: FastHenry2 was not found. R/L metrics will be marked as n/a.")
print("=" * 70)
solution = pyCoilGen(log, arg_dict)


# =============================================================================
# 5. METRICS — slope, RMSE, and the data needed for the plots
# =============================================================================
# Extract the realized field at 1 A and regress it against the gradient
# axis. Slope and RMSE are reported in physical engineering units.

_axis_index = {'x': 0, 'y': 1, 'z': 2}[INTERNAL_AXIS]
_axis_label = GRADIENT_AXIS.upper()

coords       = solution.target_field.coords                              # (3, N) [m]
coord_grad   = coords[_axis_index, :]                                    # [m]
# pyCoilGen optimizes Bz internally → index 2 is the relevant component.
layout_field = solution.solution_errors.combined_field_layout_per1Amp[2] # [T/A]
target_field = solution.solution_errors.target_field_1A.b[2]             # [T/A]

# Convert to mT/A for readability.
layout_mT = layout_field * 1000.0
target_mT = target_field * 1000.0

# Linear regression: y = slope * x + intercept  (slope in mT/(m·A)).
slope_mTmA, intercept = np.polyfit(coord_grad, layout_mT, 1)
layout_fit = slope_mTmA * coord_grad + intercept

# RMSE of the residuals, expressed as a gradient-equivalent error
# (mT/A divided by the coord range in m → mT/(m·A)).
residuals         = layout_mT - layout_fit                               # [mT/A]
coord_range       = coord_grad.max() - coord_grad.min()                  # [m]
rmse_mTA          = float(np.sqrt(np.mean(residuals ** 2)))              # [mT/A]
rmse_gradient_mTmA = rmse_mTA / coord_range if coord_range > 0 else 0.0  # [mT/(m·A)]

# Normalize target to the slope of the realized field so both share scale.
# Rationale: pyCoilGen scales the internal target arbitrarily; only its
# *shape* carries meaning. Scaling it to the realized slope makes the
# "objective vs generated" comparison visually interpretable.
target_range = float(target_mT.max() - target_mT.min())
if target_range > 0:
    target_scaled_mT = (target_mT - target_mT.mean()) * (
        (slope_mTmA * coord_range) / target_range
    ) + layout_mT.mean()
else:
    target_scaled_mT = target_mT

# ----- Wire length ------------------------------------------------------
# pyCoilGen already computes the swept wire length inside
# create_sweep_along_surface.py and stores it as `wire_path.v_length`
# (line 123 of that file). We also recompute it independently from the
# 3D wire vertices `wire_path.v` (shape 3xN) as the sum of the Euclidean
# segment lengths — this serves as a sanity check that the value pyCoilGen
# stored matches a straight integration of the polyline.
wire_lengths_stored   = []   # value stored by pyCoilGen on each part [m]
wire_lengths_computed = []   # value re-derived here on each part      [m]
for part in solution.coil_parts:
    wp = getattr(part, 'wire_path', None)
    if wp is None or wp.v is None:
        continue
    # Recompute: sum of |v[:, k+1] - v[:, k]| across all consecutive points.
    seg = np.linalg.norm(np.diff(wp.v, axis=1), axis=0)
    wire_lengths_computed.append(float(np.sum(seg)))
    # Stored value (may not exist if create_sweep_along_surface was skipped).
    wire_lengths_stored.append(float(getattr(wp, 'v_length', np.nan)))

total_wire_length_computed = float(np.sum(wire_lengths_computed)) if wire_lengths_computed else 0.0
total_wire_length_stored   = float(np.nansum(wire_lengths_stored))   if wire_lengths_stored   else 0.0

# ----- Electrical metrics ------------------------------------------------
# `ohmian_resistance` is pyCoilGen's direct DC estimate from path length and
# swept cross-section area. `coil_resistance` and `coil_inductance` are the
# FastHenry results, available only when FastHenry2 is installed and enabled.
def _float_or_nan(value):
    try:
        if value is None:
            return np.nan
        return float(np.asarray(value).squeeze())
    except (TypeError, ValueError):
        return np.nan


def _sum_finite_or_nan(values):
    finite = [v for v in values if np.isfinite(v)]
    return float(np.sum(finite)) if finite else np.nan


electrical_metrics = []
for part_index, part in enumerate(solution.coil_parts):
    fh_resistance = _float_or_nan(getattr(part, 'coil_resistance', np.nan))
    fh_inductance = _float_or_nan(getattr(part, 'coil_inductance', np.nan))
    fh_cross_section = _float_or_nan(getattr(part, 'coil_cross_section', np.nan))
    if not (ENABLE_FASTHENRY and FASTHENRY_AVAILABLE):
        fh_resistance = np.nan
        fh_inductance = np.nan
        fh_cross_section = np.nan

    electrical_metrics.append({
        'part_index': part_index,
        'coil_length_m': _float_or_nan(getattr(part, 'coil_length', np.nan)),
        'ohmian_resistance_ohm': _float_or_nan(getattr(part, 'ohmian_resistance', np.nan)),
        'fasthenry_resistance_ohm': fh_resistance,
        'fasthenry_inductance_H': fh_inductance,
        'fasthenry_cross_section_m2': fh_cross_section,
    })

total_ohmian_resistance = _sum_finite_or_nan(
    [m['ohmian_resistance_ohm'] for m in electrical_metrics]
)
total_fasthenry_resistance = _sum_finite_or_nan(
    [m['fasthenry_resistance_ohm'] for m in electrical_metrics]
)
total_fasthenry_inductance_sum = _sum_finite_or_nan(
    [m['fasthenry_inductance_H'] for m in electrical_metrics]
)

# ----- Other figures of merit reported by pyCoilGen --------------------
# Pulled defensively (some attrs are None when the corresponding stage was
# skipped, e.g. inductance / gradient calculation).
ferr = getattr(solution.solution_errors, 'field_error_vals', None)
def _g(obj, name):
    return getattr(obj, name, None) if obj is not None else None

max_rel_err_layout  = _g(ferr, 'max_rel_error_layout_vs_target')
mean_rel_err_layout = _g(ferr, 'mean_rel_error_layout_vs_target')
max_rel_err_loops   = _g(ferr, 'max_rel_error_unconnected_contours_vs_target')
mean_rel_err_loops  = _g(ferr, 'mean_rel_error_unconnected_contours_vs_target')
opt_current_layout  = getattr(solution.solution_errors, 'opt_current_layout', None)

cgrad = getattr(solution, 'coil_gradient', None)
mean_grad_target = _g(cgrad, 'mean_gradient_in_target_direction')
std_grad_target  = _g(cgrad, 'std_gradient_in_target_direction')

def _fmt(val, fmt='.6g'):
    """Format helper that tolerates None / NaN."""
    if val is None:
        return 'n/a'
    try:
        if isinstance(val, (float, np.floating)) and not np.isfinite(val):
            return 'n/a'
        return format(val, fmt)
    except (TypeError, ValueError):
        return str(val)

print("\n" + "-" * 70)
print("  METRICS")
print("-" * 70)
print(f"  Gradient axis          : {_axis_label}")
print(f"  Slope (realized coil)  : {slope_mTmA:.4f}  mT/(m·A)")
print(f"  RMSE of residuals      : {rmse_mTA:.4f}  mT/A")
print(f"  RMSE / coord range     : {rmse_gradient_mTmA:.4f}  mT/(m·A)")
print(f"  Wire length (stored)   : {total_wire_length_stored:.4f}  m"
      f"   (per part: {[f'{x:.4f}' for x in wire_lengths_stored]})")
print(f"  Wire length (recomputed): {total_wire_length_computed:.4f}  m"
      f"   (per part: {[f'{x:.4f}' for x in wire_lengths_computed]})")
print(f"  Ohmic R estimate       : {_fmt(total_ohmian_resistance)}  ohm")
print(f"  FastHenry R            : {_fmt(total_fasthenry_resistance)}  ohm")
print(f"  FastHenry L (part sum) : {_fmt(total_fasthenry_inductance_sum)}  H")
print("-" * 70)


# =============================================================================
# 5b. SAVE METRICS TXT — one file per run, named after the project
# =============================================================================
# Dump every meaningful number for this run so multiple sweeps can be
# diffed/compared offline without reopening pickles. Includes all user
# parameters at the top (so the file is self-describing).
metrics_path = os.path.join(
    RUTA_SALIDA, f"{arg_dict['project_name']}_metrics.txt"
)

with open(metrics_path, 'w', encoding='utf-8') as fh:
    fh.write("# pyCoilGen run metrics\n")
    fh.write(f"project_name              = {arg_dict['project_name']}\n")
    fh.write(f"gradient_axis             = {GRADIENT_AXIS}\n")
    fh.write(f"internal_axis             = {INTERNAL_AXIS}\n")
    fh.write("\n[USER PARAMETERS]\n")
    fh.write(f"target_rx_m               = {TARGET_RX}\n")
    fh.write(f"target_ry_m               = {TARGET_RY}\n")
    fh.write(f"target_rz_m               = {TARGET_RZ}\n")
    fh.write(f"resol_radial              = {RESOL_RADIAL}\n")
    fh.write(f"resol_angular             = {RESOL_ANGULAR}\n")
    fh.write(f"cyl_height_m              = {CYL_HEIGHT}\n")
    fh.write(f"cyl_radius_m              = {CYL_RADIUS}\n")
    fh.write(f"cyl_n_circ                = {CYL_N_CIRC}\n")
    fh.write(f"cyl_n_long                = {CYL_N_LONG}\n")
    fh.write(f"num_levels                = {NUM_LEVELS}\n")
    fh.write(f"tikhonov_factor           = {TIKHONOV_FACTOR}\n")
    fh.write(f"cut_width_m               = {CUT_WIDTH}\n")
    fh.write(f"pot_offset_factor         = {POT_OFFSET_FACTOR}\n")
    fh.write(f"min_loop_significance_pct = {MIN_LOOP_SIGNIF}\n")
    fh.write(f"normal_shift_m            = {NORMAL_SHIFT}\n")
    fh.write(f"normal_shift_smooth       = {NORMAL_SHIFT_SMOOTH}\n")
    fh.write(f"conductor_width_m         = {CONDUCTOR_WIDTH}\n")
    fh.write(f"cross_section_n           = {CROSS_SECTION_N}\n")
    fh.write(f"enable_fasthenry          = {ENABLE_FASTHENRY}\n")
    fh.write(f"fasthenry_bin             = {FASTHENRY_BIN_RESOLVED}\n")
    fh.write(f"fasthenry_available       = {FASTHENRY_AVAILABLE}\n")
    fh.write(f"fasthenry_conductor_width_m  = {FASTHENRY_CONDUCTOR_WIDTH}\n")
    fh.write(f"fasthenry_conductor_height_m = {FASTHENRY_CONDUCTOR_HEIGHT}\n")
    fh.write(f"specific_conductivity_conductor_ohm_m = {SPECIFIC_CONDUCTIVITY_CONDUCTOR}\n")

    fh.write("\n[REGRESSION ON REALIZED FIELD]\n")
    fh.write(f"slope_mT_per_m_per_A      = {_fmt(slope_mTmA)}\n")
    fh.write(f"intercept_mT_per_A        = {_fmt(intercept)}\n")
    fh.write(f"rmse_residual_mT_per_A    = {_fmt(rmse_mTA)}\n")
    fh.write(f"rmse_per_range_mT_per_m_per_A = {_fmt(rmse_gradient_mTmA)}\n")
    fh.write(f"coord_range_m             = {_fmt(coord_range)}\n")
    fh.write(f"n_target_points           = {coord_grad.size}\n")

    fh.write("\n[pyCoilGen FIELD ERRORS]\n")
    fh.write(f"max_rel_err_layout_vs_target_pct  = {_fmt(max_rel_err_layout)}\n")
    fh.write(f"mean_rel_err_layout_vs_target_pct = {_fmt(mean_rel_err_layout)}\n")
    fh.write(f"max_rel_err_loops_vs_target_pct   = {_fmt(max_rel_err_loops)}\n")
    fh.write(f"mean_rel_err_loops_vs_target_pct  = {_fmt(mean_rel_err_loops)}\n")
    fh.write(f"opt_current_layout_A      = {_fmt(opt_current_layout)}\n")

    fh.write("\n[pyCoilGen GRADIENT (target direction)]\n")
    fh.write(f"mean_gradient_mT_per_m_per_A = {_fmt(mean_grad_target)}\n")
    fh.write(f"std_gradient_mT_per_m_per_A  = {_fmt(std_grad_target)}\n")

    fh.write("\n[WIRE LENGTH]\n")
    fh.write(f"total_wire_length_m_stored      = {_fmt(total_wire_length_stored)}\n")
    fh.write(f"total_wire_length_m_recomputed  = {_fmt(total_wire_length_computed)}\n")
    for i, (s, c) in enumerate(zip(wire_lengths_stored, wire_lengths_computed)):
        fh.write(f"part_{i}_wire_length_m_stored      = {_fmt(s)}\n")
        fh.write(f"part_{i}_wire_length_m_recomputed  = {_fmt(c)}\n")

    fh.write("\n[ELECTRICAL METRICS]\n")
    fh.write(f"total_ohmian_resistance_ohm       = {_fmt(total_ohmian_resistance)}\n")
    fh.write(f"total_fasthenry_resistance_ohm    = {_fmt(total_fasthenry_resistance)}\n")
    fh.write(f"total_fasthenry_inductance_H_sum_of_parts = {_fmt(total_fasthenry_inductance_sum)}\n")
    for item in electrical_metrics:
        idx = item['part_index']
        fh.write(f"part_{idx}_coil_length_m              = {_fmt(item['coil_length_m'])}\n")
        fh.write(f"part_{idx}_ohmian_resistance_ohm      = {_fmt(item['ohmian_resistance_ohm'])}\n")
        fh.write(f"part_{idx}_fasthenry_resistance_ohm   = {_fmt(item['fasthenry_resistance_ohm'])}\n")
        fh.write(f"part_{idx}_fasthenry_inductance_H     = {_fmt(item['fasthenry_inductance_H'])}\n")
        fh.write(f"part_{idx}_fasthenry_cross_section_m2 = {_fmt(item['fasthenry_cross_section_m2'])}\n")

print(f"  Metrics written to     : {metrics_path}")


# =============================================================================
# 6. PLOTS
# =============================================================================
# Only the three plots the user asked for. Coordinate converted to cm for
# readability on the X axis.

coord_cm = coord_grad * 100.0
order    = np.argsort(coord_grad)  # sort for clean line plots


# --- Plot 1: objective vs generated field --------------------------------
fig1, ax1 = plt.subplots(figsize=(10, 6),
                         num=f'Objective vs Generated — G{_axis_label}')
ax1.plot(coord_cm[order], target_scaled_mT[order],
         'r-', linewidth=2.0,
         label='Objective (target field, scaled)')
ax1.scatter(coord_cm, layout_mT,
            s=14, alpha=0.55, color='steelblue',
            label='Generated (layout @ 1 A)')
ax1.set_xlabel(f'{_axis_label} [cm]')
ax1.set_ylabel(r'$B_z$ [mT/A]')
ax1.set_title(f'Objective vs Generated Field — G{_axis_label}')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='best')
fig1.tight_layout()


# --- Plot 2: linearity scatter + regression + RMSE -----------------------
fig2, ax2 = plt.subplots(figsize=(10, 6),
                         num=f'Linearity — G{_axis_label}')
ax2.scatter(coord_cm, layout_mT,
            s=14, alpha=0.55, color='steelblue', label='Generated field')
ax2.plot(coord_cm[order], layout_fit[order],
         'r-', linewidth=2.0,
         label=f'Linear fit: slope = {slope_mTmA:.4f} mT/(m·A)')
ax2.set_xlabel(f'{_axis_label} [cm]')
ax2.set_ylabel(r'$B_z$ [mT/A]')
ax2.set_title(f'Linearity of Generated Gradient — G{_axis_label}\n'
              f'RMSE (residual / range) = {rmse_gradient_mTmA:.4f} mT/(m·A)')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='best')
fig2.tight_layout()


# --- Console summary panel (always shown) --------------------------------
print(f"\n  Slope reported on plot : {slope_mTmA:.4f}  mT/(m·A)")
print(f"  RMSE reported on plot  : {rmse_gradient_mTmA:.4f}  mT/(m·A)")
print(f"  Results directory      : {RUTA_SALIDA}\n")

if plt.get_backend().lower() == 'agg':
    plt.close('all')
else:
    plt.show()
