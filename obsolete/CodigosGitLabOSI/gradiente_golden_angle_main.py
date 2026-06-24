"""
Main design script — port of GoldenAngle_GradientCoil.m, reorganized
for easy tuning and with custom metrics/plots.

Same logic as the frozen Ch1/Ch2/Ch3 copies (golden-angle / magic-angle
rotations, spherical target grid, large cylinder for the A4IM project),
but:
  * All user-tunable parameters live at the top (USER PARAMETERS).
  * Select the channel with CHANNEL = 1 | 2 | 3. Each channel has its
    own gradient direction (a non-axis-aligned unit vector).
  * Built-in pyCoilGen plots are replaced by three custom metrics:
      1) slope of the produced gradient along its direction [mT/(m·A)]
      2) visual comparison target field vs generated field
      3) scatter of generated field + linear fit + RMSE [mT/(m·A)]
"""

import os                                          # filesystem paths
import logging                                     # pyCoilGen logger
import numpy as np                                 # arrays / math
import matplotlib.pyplot as plt                    # plots
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (enables 3D axes)

from pyCoilGen.pyCoilGen_release import pyCoilGen  # solver entry point


# =============================================================================
# USER PARAMETERS — edit these to change the coil design
# =============================================================================

# ---- Channel selection --------------------------------------------------
CHANNEL         = 1            # 1 | 2 | 3 — which golden-angle gradient

# ---- Target region (ellipsoid inside which the gradient must be linear) -
TARGET_RX       = 0.125        # [m] target ellipsoid semi-axis along X
TARGET_RY       = 0.125        # [m]                             along Y
TARGET_RZ       = 0.125        # [m]                             along Z
RESOL_RADIAL    = 4            # radial samples of the spherical grid
RESOL_ANGULAR   = 28           # angular samples (theta & phi)

# ---- Cylinder mesh (where the wire windings live) -----------------------
CYL_HEIGHT      = 0.418        # [m] cylinder length (along its axis)
CYL_RADIUS_CH1  = 0.156        # [m] radius for Channel 1
CYL_RADIUS_CH2  = 0.1575       # [m] radius for Channel 2
CYL_RADIUS_CH3  = 0.159        # [m] radius for Channel 3
CYL_DIVS_CH1    = (50, 50)     # (circumferential, longitudinal) for Ch1
CYL_DIVS_CH2    = (92, 42)     #                                 for Ch2
CYL_DIVS_CH3    = (92, 42)     #                                 for Ch3
CYL_ROT_AXIS    = (0, 1, 0)    # rotation axis applied to the mesh
CYL_ROT_ANGLE   = np.pi / 2    # [rad] rotation angle (perp. to B0 along Y)

# ---- Coil / stream function design -------------------------------------
NUM_LEVELS              = 42       # number of SF levels = number of windings
TIKHONOV_FACTOR         = 3000     # Tikhonov regularization (higher = smoother)
CUT_WIDTH               = 0.04     # [m] interconnection cut width
POT_OFFSET_FACTOR       = 0.5      # offset 0..1 for min/max contours
MIN_LOOP_SIGNIF_CH1     = 2        # [%] min field contribution to keep a loop
MIN_LOOP_SIGNIF_CH2     = 6        # MATLAB uses 6 for Ch2 only
MIN_LOOP_SIGNIF_CH3     = 2
NORMAL_SHIFT            = -0.012   # [m] separation of go/return paths
NORMAL_SHIFT_SMOOTH     = [7, 7, 7]
SMOOTH_FACTOR           = 1        # >1 enables track smoothing (MATLAB used 1)

# ---- Conductor cross-section (oval groove profile) ---------------------
CONDUCTOR_WIDTH         = 0.002 / 2   # [m] (= 0.001 m, MATLAB convention)
CROSS_SECTION_N         = 11          # points around the oval (closed)
CROSS_SECTION_A_FRAC    = 0.7         # X semi-axis = A_FRAC * CONDUCTOR_WIDTH
CROSS_SECTION_B_FRAC    = 1.0         # Y semi-axis = B_FRAC * CONDUCTOR_WIDTH

# ---- Output ------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RUTA_SALIDA = os.path.join(BASE_DIR, f'resultados_golden_angle_main_Ch{CHANNEL}')
TARGET_DIR  = os.path.join(RUTA_SALIDA, 'target_fields')

# =============================================================================
# END USER PARAMETERS — do not edit below unless you know what you are doing
# =============================================================================


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

os.makedirs(RUTA_SALIDA, exist_ok=True)
os.makedirs(TARGET_DIR,  exist_ok=True)


# =============================================================================
# 1. GOLDEN-ANGLE ROTATIONS (one per channel)
# =============================================================================
# roty(magic angle) followed by rotx with ±120° offsets. The three resulting
# unit vectors are the gradient directions for Ch1/Ch2/Ch3.

def rotx(t): return np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
def roty(t): return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])

_magic = -np.arctan(np.sqrt(2))       # magic angle (≈ -54.74°)
_120   = np.deg2rad(120)              # ±120° offset
_270   = np.deg2rad(270)              # common 270° offset (MATLAB 3*pi/180*90)

_rot_by_ch = {
    1: rotx(0         + _270) @ roty(_magic),   # Channel 1
    2: rotx(-_120     + _270) @ roty(_magic),   # Channel 2
    3: rotx( _120     + _270) @ roty(_magic),   # Channel 3
}
rot_matrix       = _rot_by_ch[CHANNEL]
gradient_axis_vec = rot_matrix @ np.array([1.0, 0.0, 0.0])   # unit vector


# =============================================================================
# 2. BUILD THE TARGET FIELD FILE FOR THIS CHANNEL
# =============================================================================
# Spherical (r, theta, phi) grid inside the target ellipsoid; the field
# values stored are x1 (linear in X) — rotating the coords relabels it.

def build_target_field_file(rot_mat, channel_idx):
    """Write the .npy target file for the selected channel."""
    d1 = 1 / (RESOL_RADIAL  - 1)
    d2 = np.pi / (RESOL_ANGULAR - 1)
    d3 = 2 * np.pi / (RESOL_ANGULAR - 1)
    ra    = np.arange(0, 1 + d1/2, d1)
    theta = np.arange(0, np.pi + d2/2, d2)
    phi   = np.arange(-np.pi, np.pi + d3/2, d3)
    R, T, P = np.meshgrid(ra, theta, phi, indexing='ij')   # mimic MATLAB ndgrid
    r, t, p = R.ravel(), T.ravel(), P.ravel()

    # Cartesian coordinates on the ellipsoidal shells
    x1 = TARGET_RX * r * np.sin(t) * np.cos(p)
    x2 = TARGET_RY * r * np.sin(t) * np.sin(p)
    x3 = TARGET_RZ * r * np.cos(t)
    points = np.vstack([x1, x2, x3])

    # Rotate the coordinate cloud, keep scalar values = x1 (MATLAB convention)
    coords = rot_mat @ points
    path   = os.path.join(TARGET_DIR, f'OSI2_GradTarget_GA{channel_idx}.npy')
    data   = {'coords': coords.astype(np.float64),
              'lin_1':  x1.astype(np.float64)}
    np.save(path, np.array([data], dtype=object), allow_pickle=True)
    return path


target_path = build_target_field_file(rot_matrix, CHANNEL)
print(f"[target] built {os.path.basename(target_path)}")


# =============================================================================
# 3. CONDUCTOR CROSS-SECTION (clean ellipse — replaces MATLAB two-lobe bug)
# =============================================================================

_theta = np.linspace(0, 2 * np.pi, CROSS_SECTION_N, endpoint=True)
cross_sectional_points = np.vstack([
    CROSS_SECTION_A_FRAC * CONDUCTOR_WIDTH * np.sin(_theta),
    CROSS_SECTION_B_FRAC * CONDUCTOR_WIDTH * np.cos(_theta),
])


# =============================================================================
# 4. PER-CHANNEL CONFIG TABLE
# =============================================================================
# Per-channel cylinder geometry & loop significance. All other knobs are
# shared and live in the USER PARAMETERS block above.

_per_channel = {
    1: dict(radius=CYL_RADIUS_CH1, divs=CYL_DIVS_CH1, min_signif=MIN_LOOP_SIGNIF_CH1,
            force_cut=None),
    2: dict(radius=CYL_RADIUS_CH2, divs=CYL_DIVS_CH2, min_signif=MIN_LOOP_SIGNIF_CH2,
            force_cut=['low', 'low', 'high', 'high', 'high',
                       'high', 'high', 'high', 'high']),
    3: dict(radius=CYL_RADIUS_CH3, divs=CYL_DIVS_CH3, min_signif=MIN_LOOP_SIGNIF_CH3,
            force_cut=None),
}
cfg = _per_channel[CHANNEL]


# =============================================================================
# 5. RUN pyCoilGen
# =============================================================================
# Assemble arg dict and invoke the solver. CWD is moved so the target
# file is resolved by find_file() under ./target_fields/.

os.chdir(RUTA_SALIDA)

cylinder_mesh_parameter_list = [
    CYL_HEIGHT, cfg['radius'],       # height, radius [m]
    cfg['divs'][0], cfg['divs'][1],  # divs: circumferential, longitudinal
    *CYL_ROT_AXIS,                   # rotation axis
    CYL_ROT_ANGLE,                   # rotation angle [rad]
]

arg_dict = {
    # --- target field ---
    'target_field_definition_file':       os.path.basename(target_path),
    'target_field_definition_field_name': 'lin_1',
    # --- coil mesh ---
    'coil_mesh_file':               'create cylinder mesh',
    'cylinder_mesh_parameter_list': cylinder_mesh_parameter_list,
    'surface_is_cylinder_flag':     True,
    # --- stream function / winding ---
    'min_loop_significance':        cfg['min_signif'],
    'levels':                       NUM_LEVELS,
    'pot_offset_factor':            POT_OFFSET_FACTOR,
    'interconnection_cut_width':    CUT_WIDTH,
    # --- conductor geometry ---
    'conductor_cross_section_width': CONDUCTOR_WIDTH,
    'cross_sectional_points':        cross_sectional_points.tolist(),
    'normal_shift_length':           NORMAL_SHIFT,
    'normal_shift_smooth_factors':   NORMAL_SHIFT_SMOOTH,
    # --- post-processing ---
    'skip_postprocessing':          False,
    'skip_inductance_calculation':  True,
    'smooth_factor':                SMOOTH_FACTOR,
    'make_cylindrical_pcb':         False,
    'save_stl_flag':                True,
    # --- optimization ---
    'tikhonov_reg_factor':          TIKHONOV_FACTOR,
    # --- output ---
    'output_directory':             RUTA_SALIDA,
    'field_shape_function':         'x',
    'project_name':                 f'GoldenAngle_main_Ch{CHANNEL}'
                                    f'_tk{int(TIKHONOV_FACTOR)}_lvl{NUM_LEVELS}',
}
if cfg['force_cut'] is not None:
    arg_dict['force_cut_selection'] = cfg['force_cut']

print("\n" + "=" * 70)
print(f"  Running pyCoilGen — GoldenAngle Ch{CHANNEL}  |  "
      f"Tikhonov={TIKHONOV_FACTOR:g}  |  levels={NUM_LEVELS}")
print(f"  Gradient direction vec = ({gradient_axis_vec[0]:+.3f}, "
      f"{gradient_axis_vec[1]:+.3f}, {gradient_axis_vec[2]:+.3f})")
print("=" * 70)
solution = pyCoilGen(log, arg_dict)


# =============================================================================
# 6. METRICS — slope, RMSE, and the data needed for the plots
# =============================================================================
# The gradient direction is gradient_axis_vec (not a coordinate axis), so we
# project the target coords onto it before regressing the realized Bz.

coords       = solution.target_field.coords                              # (3, N) [m]
# Project each target point onto the channel's gradient direction.
coord_grad   = gradient_axis_vec @ coords                                # (N,) [m]
# pyCoilGen optimizes Bz internally → index 2 is the relevant component.
layout_field = solution.solution_errors.combined_field_layout_per1Amp[2] # [T/A]
target_field = solution.solution_errors.target_field_1A.b[2]             # [T/A]

# Convert to mT/A for readability.
layout_mT = layout_field * 1000.0
target_mT = target_field * 1000.0

# Linear regression: Bz = slope * (coord along gradient direction) + intercept
slope_mTmA, intercept = np.polyfit(coord_grad, layout_mT, 1)
layout_fit = slope_mTmA * coord_grad + intercept

# RMSE of residuals, expressed as a gradient-equivalent error
# (mT/A / coord range in m → mT/(m·A)).
residuals          = layout_mT - layout_fit                              # [mT/A]
coord_range        = coord_grad.max() - coord_grad.min()                 # [m]
rmse_mTA           = float(np.sqrt(np.mean(residuals ** 2)))             # [mT/A]
rmse_gradient_mTmA = rmse_mTA / coord_range if coord_range > 0 else 0.0  # [mT/(m·A)]

# Normalize target to share scale with the realized field (visual comparison).
target_range = float(target_mT.max() - target_mT.min())
if target_range > 0:
    target_scaled_mT = (target_mT - target_mT.mean()) * (
        (slope_mTmA * coord_range) / target_range
    ) + layout_mT.mean()
else:
    target_scaled_mT = target_mT

print("\n" + "-" * 70)
print("  METRICS")
print("-" * 70)
print(f"  Channel                : {CHANNEL}")
print(f"  Gradient direction vec : ({gradient_axis_vec[0]:+.4f}, "
      f"{gradient_axis_vec[1]:+.4f}, {gradient_axis_vec[2]:+.4f})")
print(f"  Slope (realized coil)  : {slope_mTmA:.4f}  mT/(m·A)")
print(f"  RMSE of residuals      : {rmse_mTA:.4f}  mT/A")
print(f"  RMSE / coord range     : {rmse_gradient_mTmA:.4f}  mT/(m·A)")
print("-" * 70)


# =============================================================================
# 7. PLOTS
# =============================================================================
# Only the three plots the user asked for. The "gradient axis" is the
# projection onto gradient_axis_vec; converted to cm for readability.

coord_cm = coord_grad * 100.0
order    = np.argsort(coord_grad)


# --- Plot 1: objective vs generated field --------------------------------
fig1, ax1 = plt.subplots(figsize=(10, 6),
                         num=f'Objective vs Generated — Ch{CHANNEL}')
ax1.plot(coord_cm[order], target_scaled_mT[order],
         'r-', linewidth=2.0,
         label='Objective (target field, scaled)')
ax1.scatter(coord_cm, layout_mT,
            s=14, alpha=0.55, color='steelblue',
            label='Generated (layout @ 1 A)')
ax1.set_xlabel('Position along gradient direction [cm]')
ax1.set_ylabel(r'$B_z$ [mT/A]')
ax1.set_title(f'Objective vs Generated Field — Channel {CHANNEL}\n'
              f'Gradient dir = ({gradient_axis_vec[0]:+.3f}, '
              f'{gradient_axis_vec[1]:+.3f}, {gradient_axis_vec[2]:+.3f})')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='best')
fig1.tight_layout()


# --- Plot 2: linearity scatter + regression + RMSE -----------------------
fig2, ax2 = plt.subplots(figsize=(10, 6),
                         num=f'Linearity — Ch{CHANNEL}')
ax2.scatter(coord_cm, layout_mT,
            s=14, alpha=0.55, color='steelblue', label='Generated field')
ax2.plot(coord_cm[order], layout_fit[order],
         'r-', linewidth=2.0,
         label=f'Linear fit: slope = {slope_mTmA:.4f} mT/(m·A)')
ax2.set_xlabel('Position along gradient direction [cm]')
ax2.set_ylabel(r'$B_z$ [mT/A]')
ax2.set_title(f'Linearity of Generated Gradient — Channel {CHANNEL}\n'
              f'RMSE (residual / range) = {rmse_gradient_mTmA:.4f} mT/(m·A)')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='best')
fig2.tight_layout()


# --- Console summary ----------------------------------------------------
print(f"\n  Slope reported on plot : {slope_mTmA:.4f}  mT/(m·A)")
print(f"  RMSE reported on plot  : {rmse_gradient_mTmA:.4f}  mT/(m·A)")
print(f"  Results directory      : {RUTA_SALIDA}\n")

plt.show()
