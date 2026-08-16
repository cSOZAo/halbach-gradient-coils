"""
Main design script WITH automatic lead wires — extended copy of
`gradiente_belen_santi_main.py`.

Everything in the original script is preserved verbatim. On top of it,
after pyCoilGen finishes, this script:
  * builds a short smooth polyline from each free wire end out of the
    cylinder bore (cubic Bezier blend + straight axial run),
  * sweeps the same oval cross-section along those polylines using a
    parallel-transport frame,
  * merges the lead tubes with the existing swept coil mesh and writes
    a second STL next to the original one.

The working script (`gradiente_belen_santi_main.py`) is NOT modified —
this is a parallel copy so experiments here can't break it.

All lead-related parameters are in the `LEAD PARAMETERS` block below.
Set `ADD_LEADS = False` to reproduce the original script's output
exactly (no extra STL is written).
"""

import os                                          # filesystem paths
import logging                                     # pyCoilGen logger
import numpy as np                                 # arrays / math
import matplotlib.pyplot as plt                    # plots
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (enables 3D axes)

from pyCoilGen.pyCoilGen_release import pyCoilGen  # solver entry point

# Lead-wire post-processing helper (local module)
from lead_wires import (
    rotated_cylinder_axis,
    add_lead_wires,
    plot_leads_3d,
)


# =============================================================================
# USER PARAMETERS — edit these to change the coil design
# =============================================================================

# ---- Gradient direction --------------------------------------------------
GRADIENT_AXIS   = 'x'           # 'x' | 'y' | 'z' — which linear gradient
                                # (Lin1=x, Lin2=y, Lin3=z in MATLAB terms)

# ---- Target region (ellipsoid inside which the gradient must be linear) -
TARGET_RX       = 0.02          # [m] target ellipsoid semi-axis along X
TARGET_RY       = 0.02          # [m]                             along Y
TARGET_RZ       = 0.02          # [m]                             along Z
RESOL_RADIAL    = 4             # radial samples of the spherical grid
RESOL_ANGULAR   = 28            # angular samples (theta & phi)

# ---- Cylinder mesh (where the wire windings live) -----------------------
CYL_HEIGHT      = 0.10          # [m] cylinder length (along its axis)
CYL_RADIUS      = 0.05          # [m] cylinder radius (first of MATLAB radii)
CYL_N_CIRC      = 200           # circumferential mesh divisions
CYL_N_LONG      = 20            # longitudinal  mesh divisions
CYL_ROT_AXIS    = (0, 1, 0)     # rotation axis applied to the mesh
CYL_ROT_ANGLE   = np.pi / 2     # [rad] rotation angle — aligns axis perp. to B0

# ---- Coil / stream function design -------------------------------------
NUM_LEVELS          = 16         # number of SF levels = number of windings
TIKHONOV_FACTOR     = 3e4       # Tikhonov regularization (higher = smoother)
CUT_WIDTH           = 0.001     # [m] interconnection cut width
POT_OFFSET_FACTOR   = 0.5       # offset 0..1 for min/max contours
MIN_LOOP_SIGNIF     = 5         # [%] min field contribution to keep a loop
NORMAL_SHIFT        = -0.003    # [m] separation of go/return paths
NORMAL_SHIFT_SMOOTH = [7, 7, 7] # smoothing for the normal-shift algorithm

# ---- Conductor cross-section (oval groove profile) ---------------------
CONDUCTOR_WIDTH     = 0.002     # [m] conductor width (scales the oval)
CROSS_SECTION_N     = 12        # points around the oval (closed curve)
CROSS_SECTION_A_FRAC = 0.7      # X semi-axis = A_FRAC * CONDUCTOR_WIDTH
CROSS_SECTION_B_FRAC = 0.7      # Y semi-axis = B_FRAC * CONDUCTOR_WIDTH

# ---- Output ------------------------------------------------------------
# BASE_DIR = pruebas/ (parent of obsolete/), so outputs stay relocatable.
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_SALIDA  = os.path.join(BASE_DIR, 'resultados', f'resultados_main_{GRADIENT_AXIS}_2')
TARGET_DIR   = os.path.join(RUTA_SALIDA, 'target_fields')


# =============================================================================
# LEAD PARAMETERS — automatic cable start & end
# =============================================================================
# Set ADD_LEADS = False to reproduce the original script exactly.
# All lengths are in metres.

ADD_LEADS             = True

# Radial gap between the cylinder surface and the lead's axial run.
# 2*CONDUCTOR_WIDTH keeps the lead clearly outside the coil tubes.
LEAD_RADIAL_CLEARANCE = 2.0 * CONDUCTOR_WIDTH

# Straight axial run length (how far the lead continues past the blend,
# along the cylinder axis). Make this long enough to exit the bore.
LEAD_AXIAL_LENGTH     = CYL_HEIGHT        # [m]

# Axial distance over which the Bezier blend transitions from
# radial-out to axial-out. Small values → tight bend near the coil;
# larger values → gentler, longer sweep.
LEAD_BLEND_LENGTH     = 0.025             # [m]

# Where each lead exits the bore:
#   '+axis'   — both leads exit via the +axis end
#   '-axis'   — both leads exit via the -axis end
#   'nearest' — each lead exits via whichever end it is closer to
LEAD_EXIT_DIRECTION   = '+axis'

# Polyline sampling density (more samples → smoother swept tube).
LEAD_BLEND_SAMPLES    = 24
LEAD_AXIAL_SAMPLES    = 12

# If True, also show a 3D preview of the coil wire_path + lead polylines
# BEFORE meshes are rendered — useful to catch wrong tangents or
# wrong exit directions.
LEAD_DEBUG_PLOT       = True

# Filename for the combined coil+leads STL (written next to the
# original `*_wire_*.stl`). `{axis}` is replaced with GRADIENT_AXIS.
LEAD_STL_FILENAME     = 'Gradient_G{axis}_wire_with_leads.stl'


# =============================================================================
# END USER PARAMETERS — do not edit below unless you know what you are doing
# =============================================================================


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

os.makedirs(RUTA_SALIDA, exist_ok=True)
os.makedirs(TARGET_DIR,  exist_ok=True)


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


target_path, target_fname = build_target_field_file(GRADIENT_AXIS)
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
    'conductor_cross_section_width': CONDUCTOR_WIDTH,
    'cross_sectional_points':        cross_sectional_points.tolist(),
    'normal_shift_length':           NORMAL_SHIFT,
    'normal_shift_smooth_factors':   NORMAL_SHIFT_SMOOTH,
    # --- post-processing ---
    'skip_postprocessing':          False,
    'skip_inductance_calculation':  True,
    'smooth_factor':                3,
    'make_cylindrical_pcb':         False,
    'save_stl_flag':                True,
    # --- optimization ---
    'tikhonov_reg_factor':          TIKHONOV_FACTOR,
    # --- output ---
    'output_directory':             RUTA_SALIDA,
    'field_shape_function':         GRADIENT_AXIS,
    'project_name':                 f'Gradient_G{GRADIENT_AXIS}'
                                    f'_tk{int(TIKHONOV_FACTOR)}_lvl{NUM_LEVELS}',
}

print("\n" + "=" * 70)
print(f"  Running pyCoilGen — G{GRADIENT_AXIS.upper()}  |  Tikhonov={TIKHONOV_FACTOR:g}"
      f"  |  levels={NUM_LEVELS}")
print("=" * 70)
solution = pyCoilGen(log, arg_dict)


# =============================================================================
# 5. METRICS — slope, RMSE, and the data needed for the plots
# =============================================================================
# Extract the realized field at 1 A and regress it against the gradient
# axis. Slope and RMSE are reported in physical engineering units.

_axis_index = {'x': 0, 'y': 1, 'z': 2}[GRADIENT_AXIS]
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

print("\n" + "-" * 70)
print("  METRICS")
print("-" * 70)
print(f"  Gradient axis          : {_axis_label}")
print(f"  Slope (realized coil)  : {slope_mTmA:.4f}  mT/(m·A)")
print(f"  RMSE of residuals      : {rmse_mTA:.4f}  mT/A")
print(f"  RMSE / coord range     : {rmse_gradient_mTmA:.4f}  mT/(m·A)")
print("-" * 70)


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


# =============================================================================
# 7. AUTOMATIC LEAD WIRES (post-processing) — new section
# =============================================================================
# Takes the finished solution, builds a polyline from each free wire end
# out of the bore, sweeps the same oval cross-section along it, and
# writes a merged STL.

if ADD_LEADS:
    # Cylinder axis in world coordinates (after the rotation applied by
    # pyCoilGen's cylinder mesh plugin).
    axis_hat = rotated_cylinder_axis(CYL_ROT_AXIS, CYL_ROT_ANGLE)
    print("\n" + "=" * 70)
    print("  Adding automatic lead wires")
    print("=" * 70)
    print(f"  cylinder axis (world)  : {axis_hat}")
    print(f"  radial clearance       : {LEAD_RADIAL_CLEARANCE*1000:.2f} mm")
    print(f"  blend length           : {LEAD_BLEND_LENGTH*1000:.2f} mm")
    print(f"  axial run length       : {LEAD_AXIAL_LENGTH*1000:.2f} mm")
    print(f"  exit direction         : {LEAD_EXIT_DIRECTION}")

    # Optional 3D preview BEFORE we commit to sweeping — catches wrong
    # tangents or wrong exit direction visually.
    if LEAD_DEBUG_PLOT:
        # Use a dummy first pass to get the polylines for preview
        # (cheap: polyline construction is O(n_samples)).
        from lead_wires import build_lead_polyline
        preview_polylines = []
        for coil_part in solution.coil_parts:
            wv = coil_part.wire_path.v
            preview_polylines.append(build_lead_polyline(
                wv[:, 0],  wv[:, 0]  - wv[:, 1], axis_hat,
                radial_clearance=LEAD_RADIAL_CLEARANCE,
                axial_length=LEAD_AXIAL_LENGTH,
                blend_length=LEAD_BLEND_LENGTH,
                exit_direction=LEAD_EXIT_DIRECTION,
                blend_samples=LEAD_BLEND_SAMPLES,
                axial_samples=LEAD_AXIAL_SAMPLES,
            ))
            preview_polylines.append(build_lead_polyline(
                wv[:, -1], wv[:, -1] - wv[:, -2], axis_hat,
                radial_clearance=LEAD_RADIAL_CLEARANCE,
                axial_length=LEAD_AXIAL_LENGTH,
                blend_length=LEAD_BLEND_LENGTH,
                exit_direction=LEAD_EXIT_DIRECTION,
                blend_samples=LEAD_BLEND_SAMPLES,
                axial_samples=LEAD_AXIAL_SAMPLES,
            ))
        plot_leads_3d(solution, preview_polylines,
                      axis_hat=axis_hat,
                      title=f'Lead wires preview — G{_axis_label}',
                      show=False)

    # Build and merge the lead tubes into a combined mesh
    combined_mesh, lead_polylines, _ = add_lead_wires(
        solution, axis_hat,
        radial_clearance=LEAD_RADIAL_CLEARANCE,
        axial_length=LEAD_AXIAL_LENGTH,
        blend_length=LEAD_BLEND_LENGTH,
        exit_direction=LEAD_EXIT_DIRECTION,
        blend_samples=LEAD_BLEND_SAMPLES,
        axial_samples=LEAD_AXIAL_SAMPLES,
    )

    # Export the combined STL next to pyCoilGen's own STL
    out_name = LEAD_STL_FILENAME.format(axis=GRADIENT_AXIS)
    out_path = os.path.join(RUTA_SALIDA, out_name)
    combined_mesh.export(out_path)
    print(f"\n  Combined STL (coil + leads) written to:")
    print(f"    {out_path}")
    print(f"  Total vertices / faces : "
          f"{len(combined_mesh.vertices)} / {len(combined_mesh.faces)}")


plt.show()
