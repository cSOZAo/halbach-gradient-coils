"""
Módulo con la función `generar_gradiente` adaptada de gradiente_belen_santi_main.py.
Permite ejecutar el diseño de la bobina como una función programática sin modificar
el script en cada corrida.
"""

import os
import shutil
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Prefer the pyCoilGen package in this workspace over any installed copy.
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from pyCoilGen.pyCoilGen_release import pyCoilGen

def resolve_fasthenry_bin(configured_path):
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

def rotx(t): return np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
def roty(t): return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
def rotz(t): return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])

_rotations_by_axis = {
    'x': rotx(0),
    'y': rotz(np.pi / 2),
    'z': roty(-np.pi / 2),
}
_fname_by_axis   = {'x': 'lin_1', 'y': 'lin_2', 'z': 'lin_3'}
_filename_by_ax  = {'x': 'OSI2_GradTarget_Lin1.npy',
                    'y': 'OSI2_GradTarget_Lin2.npy',
                    'z': 'OSI2_GradTarget_Lin3.npy'}

def generar_gradiente(gradient_axis, cyl_height, cyl_radius, tikhonov_factor, num_levels=26, base_output_dir=None, show_plots=False):
    """
    Ejecuta pyCoilGen para generar una bobina de gradiente con los parámetros especificados.
    """
    # =============================================================================
    # USER PARAMETERS INTERNOS
    # =============================================================================
    GRADIENT_AXIS   = gradient_axis.lower()
    TARGET_RX       = 0.125
    TARGET_RY       = 0.125
    TARGET_RZ       = 0.125
    RESOL_RADIAL    = 8
    RESOL_ANGULAR   = 28

    CYL_HEIGHT      = cyl_height
    CYL_RADIUS      = cyl_radius
    CYL_N_CIRC      = 200
    CYL_N_LONG      = 10
    CYL_ROT_AXIS    = (0, 1, 0)
    CYL_ROT_ANGLE   = np.pi / 2

    NUM_LEVELS          = num_levels
    TIKHONOV_FACTOR     = tikhonov_factor
    CUT_WIDTH           = 0.001
    POT_OFFSET_FACTOR   = 0.5
    MIN_LOOP_SIGNIF     = 5
    NORMAL_SHIFT        = -0.005
    NORMAL_SHIFT_SMOOTH = [7, 7, 7]

    CONDUCTOR_WIDTH     = 0.003
    CROSS_SECTION_N     = 12
    CROSS_SECTION_A_FRAC = 1.5
    CROSS_SECTION_B_FRAC = 1.0

    ENABLE_FASTHENRY = True
    FASTHENRY_BIN = r'C:\Program Files (x86)\FastFieldSolvers\FastHenry2\FastHenry2.exe'
    FASTHENRY_CONDUCTOR_WIDTH = 2.0 * CROSS_SECTION_B_FRAC * CONDUCTOR_WIDTH
    FASTHENRY_CONDUCTOR_HEIGHT = (np.pi / 2.0) * CROSS_SECTION_A_FRAC * CONDUCTOR_WIDTH
    SPECIFIC_CONDUCTIVITY_CONDUCTOR = 1.8e-8

    BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
    if base_output_dir is None:
        RUTA_SALIDA  = os.path.join(BASE_DIR, f'resultados/resultados_grande_{GRADIENT_AXIS}/final')
    else:
        RUTA_SALIDA  = base_output_dir
    TARGET_DIR   = os.path.join(RUTA_SALIDA, 'target_fields')

    # =============================================================================
    # LÓGICA PRINCIPAL
    # =============================================================================
    INTERNAL_AXIS = {'x': 'y', 'y': 'z', 'z': 'x'}[GRADIENT_AXIS]

    log = logging.getLogger(__name__)
    # Solo configuramos logging si no ha sido configurado
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    os.makedirs(RUTA_SALIDA, exist_ok=True)
    os.makedirs(TARGET_DIR,  exist_ok=True)

    FASTHENRY_BIN_RESOLVED = resolve_fasthenry_bin(FASTHENRY_BIN)
    FASTHENRY_AVAILABLE = bool(FASTHENRY_BIN_RESOLVED and os.path.isfile(FASTHENRY_BIN_RESOLVED))

    def build_target_field_file(axis):
        d1 = 1 / (RESOL_RADIAL - 1)
        d2 = np.pi / (RESOL_ANGULAR - 1)
        d3 = 2 * np.pi / (RESOL_ANGULAR - 1)
        ra    = np.arange(0, 1 + d1 / 2, d1)
        theta = np.arange(0, np.pi + d2 / 2, d2)
        phi   = np.arange(-np.pi, np.pi + d3 / 2, d3)
        R, T, P = np.meshgrid(ra, theta, phi, indexing='ij')
        r, t, p = R.ravel(), T.ravel(), P.ravel()

        x1 = TARGET_RX * r * np.sin(t) * np.cos(p)
        x2 = TARGET_RY * r * np.sin(t) * np.sin(p)
        x3 = TARGET_RZ * r * np.cos(t)
        points = np.vstack([x1, x2, x3])

        coords = _rotations_by_axis[axis] @ points
        fname  = _fname_by_axis[axis]
        path   = os.path.join(TARGET_DIR, _filename_by_ax[axis])

        data = {'coords': coords.astype(np.float64),
                fname: x1.astype(np.float64)}
        np.save(path, np.array([data], dtype=object), allow_pickle=True)
        return path, fname

    target_path, target_fname = build_target_field_file(INTERNAL_AXIS)
    print(f"[target] built {os.path.basename(target_path)}  (field = {target_fname})")

    _theta = np.linspace(0, 2 * np.pi, CROSS_SECTION_N, endpoint=True)
    cross_sectional_points = np.vstack([
        CROSS_SECTION_A_FRAC * CONDUCTOR_WIDTH * np.sin(_theta),
        CROSS_SECTION_B_FRAC * CONDUCTOR_WIDTH * np.cos(_theta),
    ])

    # Guardamos el CWD original para no afectar ejecuciones consecutivas
    old_cwd = os.getcwd()
    os.chdir(RUTA_SALIDA)

    try:
        cylinder_mesh_parameter_list = [
            CYL_HEIGHT, CYL_RADIUS,
            CYL_N_CIRC, CYL_N_LONG,
            *CYL_ROT_AXIS,
            CYL_ROT_ANGLE,
        ]

        arg_dict = {
            'target_field_definition_file':       os.path.basename(target_path),
            'target_field_definition_field_name': target_fname,
            'coil_mesh_file':               'create cylinder mesh',
            'cylinder_mesh_parameter_list': cylinder_mesh_parameter_list,
            'surface_is_cylinder_flag':     True,
            'min_loop_significance':        MIN_LOOP_SIGNIF,
            'levels':                       NUM_LEVELS,
            'pot_offset_factor':            POT_OFFSET_FACTOR,
            'interconnection_cut_width':    CUT_WIDTH,
            'conductor_cross_section_width': FASTHENRY_CONDUCTOR_WIDTH,
            'conductor_cross_section_height': FASTHENRY_CONDUCTOR_HEIGHT,
            'specific_conductivity_conductor': SPECIFIC_CONDUCTIVITY_CONDUCTOR,
            'cross_sectional_points':        cross_sectional_points.tolist(),
            'normal_shift_length':           NORMAL_SHIFT,
            'normal_shift_smooth_factors':   NORMAL_SHIFT_SMOOTH,
            'skip_postprocessing':          False,
            'skip_inductance_calculation':  not ENABLE_FASTHENRY,
            'fasthenry_bin':                FASTHENRY_BIN_RESOLVED,
            'smooth_factor':                3,
            'make_cylindrical_pcb':         False,
            'save_stl_flag':                True,
            'CAD_filename':                 '{project}_wire_{part_index}_{field_function}.stl',
            'tikhonov_reg_factor':          TIKHONOV_FACTOR,
            'output_directory':             RUTA_SALIDA,
            'field_shape_function':         INTERNAL_AXIS,
            'project_name':                 f'Gradient_G{GRADIENT_AXIS}_tk{int(TIKHONOV_FACTOR)}_lvl{NUM_LEVELS}',
        }

        print("\n" + "=" * 70)
        print(f"  Running pyCoilGen — G{GRADIENT_AXIS.upper()}  |  Tikhonov={TIKHONOV_FACTOR:g}  |  levels={NUM_LEVELS}")
        print("=" * 70)
        solution = pyCoilGen(log, arg_dict)
        
    finally:
        os.chdir(old_cwd)

    # =============================================================================
    # METRICS
    # =============================================================================
    _axis_index = {'x': 0, 'y': 1, 'z': 2}[INTERNAL_AXIS]
    _axis_label = GRADIENT_AXIS.upper()

    coords       = solution.target_field.coords
    coord_grad   = coords[_axis_index, :]
    layout_field = solution.solution_errors.combined_field_layout_per1Amp[2]
    target_field = solution.solution_errors.target_field_1A.b[2]

    layout_mT = layout_field * 1000.0
    target_mT = target_field * 1000.0

    slope_mTmA, intercept = np.polyfit(coord_grad, layout_mT, 1)
    layout_fit = slope_mTmA * coord_grad + intercept

    residuals         = layout_mT - layout_fit
    coord_range       = coord_grad.max() - coord_grad.min()
    rmse_mTA          = float(np.sqrt(np.mean(residuals ** 2)))
    rmse_gradient_mTmA = rmse_mTA / coord_range if coord_range > 0 else 0.0

    target_range = float(target_mT.max() - target_mT.min())
    if target_range > 0:
        target_scaled_mT = (target_mT - target_mT.mean()) * (
            (slope_mTmA * coord_range) / target_range
        ) + layout_mT.mean()
    else:
        target_scaled_mT = target_mT

    # Otras figuras de merito
    ferr = getattr(solution.solution_errors, 'field_error_vals', None)
    def _g(obj, name): return getattr(obj, name, None) if obj is not None else None

    max_rel_err_layout  = _g(ferr, 'max_rel_error_layout_vs_target')
    mean_rel_err_layout = _g(ferr, 'mean_rel_error_layout_vs_target')
    
    # Escritura de métricas simplificada a txt
    metrics_path = os.path.join(RUTA_SALIDA, f"{arg_dict['project_name']}_metrics.txt")
    with open(metrics_path, 'w', encoding='utf-8') as fh:
        fh.write(f"slope_mT_per_m_per_A      = {slope_mTmA}\n")
        fh.write(f"mean_rel_err_layout_vs_target_pct = {mean_rel_err_layout if mean_rel_err_layout is not None else 'n/a'}\n")
        fh.write(f"rmse_per_range_mT_per_m_per_A = {rmse_gradient_mTmA}\n")

    # =============================================================================
    # PLOTS
    # =============================================================================
    if show_plots:
        coord_cm = coord_grad * 100.0
        order    = np.argsort(coord_grad)

        fig1, ax1 = plt.subplots(figsize=(10, 6), num=f'Objective vs Generated — G{_axis_label}')
        ax1.plot(coord_cm[order], target_scaled_mT[order], 'r-', linewidth=2.0, label='Objective')
        ax1.scatter(coord_cm, layout_mT, s=14, alpha=0.55, color='steelblue', label='Generated')
        ax1.set_xlabel(f'{_axis_label} [cm]')
        ax1.set_ylabel(r'$B_z$ [mT/A]')
        ax1.legend(loc='best')

        fig2, ax2 = plt.subplots(figsize=(10, 6), num=f'Linearity — G{_axis_label}')
        ax2.scatter(coord_cm, layout_mT, s=14, alpha=0.55, color='steelblue', label='Generated field')
        ax2.plot(coord_cm[order], layout_fit[order], 'r-', linewidth=2.0, label=f'Linear fit: slope = {slope_mTmA:.4f} mT/(m·A)')
        ax2.set_xlabel(f'{_axis_label} [cm]')
        ax2.set_ylabel(r'$B_z$ [mT/A]')
        ax2.legend(loc='best')

        plt.show()
    else:
        plt.close('all')

    return {
        'slope_mTmA': slope_mTmA,
        'mean_rel_err_layout_pct': mean_rel_err_layout,
        'rmse_gradient_mTmA': rmse_gradient_mTmA,
        'output_dir': RUTA_SALIDA
    }
