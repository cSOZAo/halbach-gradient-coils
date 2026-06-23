"""
Port of GoldenAngle_GradientCoil.m to pyCoilGen — CHANNEL 2 [FROZEN].

Source: CoilGen_MatLab/GoldenAngle_GradientCoil.m (Sebastian Littin,
University Medical Center Freiburg, July 2024). Generates a golden-angle
gradient coil set for a Halbach magnet with B0 along Y (A4IM / OSI project).

This is the FROZEN per-channel reference for Channel 2. Do not modify.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (enables 3D axes)

from pyCoilGen.pyCoilGen_release import pyCoilGen
import pyCoilGen.plotting as pcg_plt


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


# =============================================================================
# 0.  PATHS
# =============================================================================
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
RUTA_SALIDA   = os.path.join(BASE_DIR, 'resultados_golden_angle_Ch2')
TARGET_FIELDS = os.path.join(RUTA_SALIDA, 'target_fields')
os.makedirs(RUTA_SALIDA,   exist_ok=True)
os.makedirs(TARGET_FIELDS, exist_ok=True)


# =============================================================================
# 1.  ROTATION MATRICES — MAGIC-ANGLE / GOLDEN-ANGLE DIRECTIONS
#     MATLAB lines 17-28
# =============================================================================
# Build three rotations as roty(magic_angle) followed by rotx with three
# different angles (0, -120°, +120°) plus a common 270° offset.
# The three resulting unit vectors are NOT axis-aligned; they're the
# golden-angle gradient directions for the A4IM project.

def rotx(t):
    return np.array([[1, 0, 0],
                     [0, np.cos(t), -np.sin(t)],
                     [0, np.sin(t),  np.cos(t)]])

def roty(t):
    return np.array([[ np.cos(t), 0, np.sin(t)],
                     [ 0,         1, 0        ],
                     [-np.sin(t), 0, np.cos(t)]])

def rotz(t):
    return np.array([[np.cos(t), -np.sin(t), 0],
                     [np.sin(t),  np.cos(t), 0],
                     [0,          0,         1]])

rot_ang_1 = -np.arctan(np.sqrt(2))     # magic angle ≈ -54.74°
rot_ang_2 = np.deg2rad(120)            # 120°
rot_ang_3 = np.deg2rad(270)            # 3*pi/180*90 = 270° in MATLAB

rot1 = rotx(0           + rot_ang_3) @ roty(rot_ang_1)
rot2 = rotx(-rot_ang_2  + rot_ang_3) @ roty(rot_ang_1)
rot3 = rotx( rot_ang_2  + rot_ang_3) @ roty(rot_ang_1)

base_x = np.array([1.0, 0.0, 0.0])
vec_1 = rot1 @ base_x   # gradient direction Channel 1
vec_2 = rot2 @ base_x   # gradient direction Channel 2
vec_3 = rot3 @ base_x   # gradient direction Channel 3

# Orthogonality sanity check (MATLAB lines 44-46).
print("[sanity] vec_1 . vec_2 =", float(np.dot(vec_1, vec_2)))
print("[sanity] vec_1 . vec_3 =", float(np.dot(vec_1, vec_3)))
print("[sanity] vec_2 . vec_3 =", float(np.dot(vec_2, vec_3)))


# =============================================================================
# 2.  SPHERICAL TARGET GRID
#     MATLAB lines 48-73
# =============================================================================
# Same construction as belen_santi, but a much LARGER target region.

rx = ry = rz = 0.125       # [m] target ellipsoid semi-axes
resol_radial  = 4
resol_angular = 28

d1 = 1 / (resol_radial - 1)
d2 = np.pi / (resol_angular - 1)
d3 = 2 * np.pi / (resol_angular - 1)
ra_1d    = np.arange(0, 1 + d1/2, d1)
theta_1d = np.arange(0, np.pi + d2/2, d2)
phi_1d   = np.arange(-np.pi, np.pi + d3/2, d3)

ra, theta, phi = np.meshgrid(ra_1d, theta_1d, phi_1d, indexing='ij')
ra_f, theta_f, phi_f = ra.ravel(), theta.ravel(), phi.ravel()

x1 = rx * ra_f * np.sin(theta_f) * np.cos(phi_f)
x2 = ry * ra_f * np.sin(theta_f) * np.sin(phi_f)
x3 = rz * ra_f * np.cos(theta_f)
Points = np.vstack([x1, x2, x3])

target_rot1 = rot1 @ Points
target_rot2 = rot2 @ Points
target_rot3 = rot3 @ Points

# All three CoilGen calls in MATLAB use field name 'lin_1', with values
# = x1 in every target file. So all three .npy files just need
# {coords, lin_1=x1}.
lin_values = x1.copy()


# =============================================================================
# 3.  SAVE TARGET FIELDS AS .npy
#     MATLAB lines 75-88 (.mat → .npy here)
# =============================================================================
def save_target(filename, coords, lin1):
    data = {'coords': coords.astype(np.float64),
            'lin_1': lin1.astype(np.float64)}
    np.save(filename, np.array([data], dtype=object), allow_pickle=True)

save_target(os.path.join(TARGET_FIELDS, 'OSI2_GradTarget_GA1.npy'), target_rot1, lin_values)
save_target(os.path.join(TARGET_FIELDS, 'OSI2_GradTarget_GA2.npy'), target_rot2, lin_values)
save_target(os.path.join(TARGET_FIELDS, 'OSI2_GradTarget_GA3.npy'), target_rot3, lin_values)
print(f"[targets] wrote GA1/GA2/GA3 into {TARGET_FIELDS}")


# =============================================================================
# 4.  CONDUCTOR CROSS-SECTION
#     MATLAB lines 169-171 (Channel 2 offset = -5)
# =============================================================================
# MATLAB shapes [sin(...) , -8 + sin(...)] give two semicircles separated by
# 8*conductor_width — same disjoint-lobe bug as in belen_santi. Replaced
# with a clean closed ellipse (see CLAUDE.md → pyCoilGen API rules).
circular_resolution = 11
conductor_width     = 0.002 / 2   # [m] = 0.001

_theta_cs = np.linspace(0, 2 * np.pi, circular_resolution, endpoint=True)
cross_sectional_points = np.vstack([
    0.7 * conductor_width * np.sin(_theta_cs),   # X semi-axis
    1.0 * conductor_width * np.cos(_theta_cs),   # Y semi-axis
])


# =============================================================================
# 5.  COIL PARAMETERS — CHANNEL 2
#     MATLAB lines 167-205
# =============================================================================
cut_width                   = 0.04
min_loop_signifcance        = 6        # MATLAB Ch2 (different from Ch1/Ch3)
pot_offset_factor           = 0.5
normal_shift_smooth_factors = [7, 7, 7]
normal_shift                = -0.012
tikonov_factor              = 3000
num_levels                  = 42

# MATLAB radii(2) = 0.1575
radius_ch2 = 0.1575

# Cylinder mesh: height 0.418 m, 92×42 divisions, rotated [0 1 0] by π/2
cylinder_mesh_parameter_list = [
    0.418,        # height [m]
    radius_ch2,   # radius [m]
    92, 42,       # circumferential / longitudinal divisions
    0, 1, 0,      # rotation axis (Y)
    np.pi / 2,    # rotation angle
]

# Channel 2 MATLAB uses force_cut_selection — forces specific cut sides
# for the first 9 loops. Keeps the routing consistent across runs.
force_cut_selection = ['low', 'low', 'high', 'high', 'high',
                       'high', 'high', 'high', 'high']


# =============================================================================
# 6.  RUN pyCoilGen
#     MATLAB lines 141-162
# =============================================================================
os.chdir(RUTA_SALIDA)

arg_dict = {
    # --- target field (Channel 2 uses GA2.mat / lin_1) ---
    'target_field_definition_file':       'OSI2_GradTarget_GA2.npy',
    'target_field_definition_field_name': 'lin_1',
    # --- coil mesh ---
    'coil_mesh_file':              'create cylinder mesh',
    'cylinder_mesh_parameter_list': cylinder_mesh_parameter_list,
    'surface_is_cylinder_flag':    True,
    # --- stream function / winding ---
    'min_loop_significance':       min_loop_signifcance,
    'levels':                      num_levels,
    'pot_offset_factor':           pot_offset_factor,
    'interconnection_cut_width':   cut_width,
    'force_cut_selection':         force_cut_selection,
    # --- conductor geometry ---
    'conductor_cross_section_width': conductor_width,
    'cross_sectional_points':        cross_sectional_points.tolist(),
    'normal_shift_length':           normal_shift,
    'normal_shift_smooth_factors':   normal_shift_smooth_factors,
    # --- post-processing ---
    'skip_postprocessing':          False,
    'skip_inductance_calculation':  True,
    # MATLAB had smooth_flag=true, smooth_factor=1. In pyCoilGen smoothing
    # is enabled only when smooth_factor > 1, so MATLAB literal port = no
    # smoothing. Kept at 1 to match MATLAB. Bump in the main script if needed.
    'smooth_factor':                1,
    'make_cylindrical_pcb':         False,
    'save_stl_flag':                True,
    # --- optimization ---
    'tikhonov_reg_factor':          tikonov_factor,
    # --- output ---
    'output_directory':             RUTA_SALIDA,
    'field_shape_function':         'x',
    'project_name':                 f'GoldenAngle_Ch2_tk{tikonov_factor}_lvl{num_levels}',
}

print("\n" + "=" * 70)
print("  Running pyCoilGen — GoldenAngle (Channel 2 / GA2) [FROZEN]")
print(f"  Cylinder: L=0.418 m, R={radius_ch2:.4f} m | Tikhonov={tikonov_factor}, levels={num_levels}")
print("=" * 70)

solution = pyCoilGen(log, arg_dict)


# =============================================================================
# 7.  BUILT-IN PLOTS (MATLAB lines 246-265)
# =============================================================================
coil_name = 'A4IM_GoldenAngle_Ch2'
for plot_fn_name in [
    'plot_2D_contours_with_sf',
    'plot_coil_track_with_resulting_bfield',
    'plot_various_error_metrics',
]:
    fn = getattr(pcg_plt, plot_fn_name, None)
    if fn is None:
        print(f"  [plot] {plot_fn_name} not available")
        continue
    try:
        fn([solution], 0, coil_name)
    except Exception as e:
        print(f"  [plot] {plot_fn_name} failed: {e}")

plt.show()
