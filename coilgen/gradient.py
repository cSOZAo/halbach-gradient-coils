"""
Gradient coil design step — port of ``script_belen_santi.m``.

Exposed as :func:`run_gradient` (a function, not a script that runs on import).
Builds the spherical target-field file, runs pyCoilGen on the rotated cylinder
mesh, exports the wire STL, computes metrics, and optionally checks for wire
overlaps.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Optional, Tuple

import numpy as np

from . import geometry as geo
from . import paths as _paths
from .config import Config
from .fasthenry import resolve_fasthenry_bin, fasthenry_available
from .metrics import compute_metrics, print_metrics_summary
from .overlap import detect_collisions, OverlapReport

log = logging.getLogger(__name__)


@contextmanager
def _chdir(path: str):
    """Temporarily change CWD (pyCoilGen resolves target_fields/ relative to it)."""
    old = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old)


def _setup_matplotlib(show_plots: bool):
    import matplotlib
    if not show_plots:
        matplotlib.use('Agg', force=True)
    matplotlib.rcParams['toolbar'] = 'None'
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def _make_plots(metrics: dict, cfg: Config):
    import matplotlib.pyplot as plt

    label = metrics['axis_label']
    coord_grad = metrics['coord_grad']
    layout_mT = metrics['layout_mT']
    target_scaled_mT = metrics['target_scaled_mT']
    layout_fit = metrics['layout_fit']
    slope = metrics['slope_mTmA']
    rmse = metrics['rmse_gradient_mTmA']

    coord_cm = coord_grad * 100.0
    order = np.argsort(coord_grad)

    fig1, ax1 = plt.subplots(figsize=(10, 6),
                             num=f'Objective vs Generated — {label}')
    ax1.plot(coord_cm[order], target_scaled_mT[order],
             'r-', linewidth=2.0, label='Objective (target field, scaled)')
    ax1.scatter(coord_cm, layout_mT, s=14, alpha=0.55, color='steelblue',
                label='Generated (layout @ 1 A)')
    ax1.set_xlabel(f'{label} [cm]')
    ax1.set_ylabel(r'$B_z$ [mT/A]')
    ax1.set_title(f'Objective vs Generated Field — {label}')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='best')
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(10, 6), num=f'Linearity — {label}')
    ax2.scatter(coord_cm, layout_mT, s=14, alpha=0.55, color='steelblue',
                label='Generated field')
    ax2.plot(coord_cm[order], layout_fit[order], 'r-', linewidth=2.0,
             label=f'Linear fit: slope = {slope:.4f} mT/(m.A)')
    ax2.set_xlabel(f'{label} [cm]')
    ax2.set_ylabel(r'$B_z$ [mT/A]')
    ax2.set_title(f'Linearity of Generated Gradient — {label}\n'
                  f'RMSE (residual / range) = {rmse:.4f} mT/(m.A)')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='best')
    fig2.tight_layout()

    import matplotlib.pyplot as _plt
    if _plt.get_backend().lower() == 'agg':
        _plt.close('all')
    else:
        _plt.show()


def run_gradient(
    cfg: Config,
    output_dir: Optional[str] = None,
    project_stem: Optional[str] = None,
    check_overlap: Optional[bool] = None,
) -> Tuple[object, dict, Optional[OverlapReport]]:
    """
    Run pyCoilGen for ``cfg`` and return ``(solution, metrics, overlap_report)``.

    ``output_dir`` defaults to ``cfg.output_dir`` (or a freshly allocated
    pipeline run folder). ``project_stem`` defaults to a unique stem inside
    the output dir. ``check_overlap`` defaults to ``cfg.overlap_warn``.
    """
    from pyCoilGen.pyCoilGen_release import pyCoilGen

    _setup_matplotlib(cfg.show_plots)
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO)

    # ----- Resolve output directory ---------------------------------------
    if output_dir is None:
        output_dir = cfg.output_dir or _paths.unique_run_dir(
            _paths.PIPELINE_OUTPUT_BASE, cfg.design_folder,
        )
    cfg.output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)
    target_dir = os.path.join(output_dir, 'target_fields')
    os.makedirs(target_dir, exist_ok=True)

    # ----- Target field ----------------------------------------------------
    internal_axis = cfg.internal_axis
    target_path, target_fname = geo.build_target_field_file(
        internal_axis,
        cfg.target.rx, cfg.target.ry, cfg.target.rz,
        cfg.target.resol_radial, cfg.target.resol_angular,
        target_dir,
    )
    print(f"[target] built {os.path.basename(target_path)}  (field = {target_fname})")

    # ----- FastHenry -------------------------------------------------------
    fh_resolved = resolve_fasthenry_bin(cfg.fasthenry.bin_path)
    fh_available = fasthenry_available(fh_resolved)
    if cfg.fasthenry.enabled and not fh_available:
        print("  WARNING: FastHenry2 was not found. R/L metrics will be marked as n/a.")

    # ----- Project stem (unique within the run dir) -----------------------
    if project_stem is None:
        project_stem = _paths.unique_stem(
            output_dir, cfg.project_stem_base, gradient_axis=cfg.gradient_axis,
        )
    _paths.write_active_stem(output_dir, project_stem)

    # ----- arg dict --------------------------------------------------------
    cyl = cfg.cylinder
    # Shell length = cylinder.height (GUI). Wire mesh = height × 0.95 for
    # every gradient axis (x/y/z) so grooves stay clear of the rim.
    mesh_height = float(cyl.height) * float(cyl.mesh_length_factor)
    cylinder_mesh_parameter_list = [
        mesh_height, cfg.cylinder_design_radius,
        cyl.n_circ, cyl.n_long,
        *cyl.rot_axis,
        cyl.rot_angle,
    ]

    arg_dict = {
        'target_field_definition_file':       os.path.basename(target_path),
        'target_field_definition_field_name': target_fname,
        'coil_mesh_file':               'create cylinder mesh',
        'cylinder_mesh_parameter_list': cylinder_mesh_parameter_list,
        'surface_is_cylinder_flag':     True,
        'min_loop_significance':        cfg.winding.min_loop_signif,
        'levels':                       cfg.num_levels,
        'pot_offset_factor':            cfg.winding.pot_offset_factor,
        'interconnection_cut_width':    cfg.winding.cut_width,
        'conductor_cross_section_width':  cfg.fasthenry_conductor_width,
        'conductor_cross_section_height': cfg.fasthenry_conductor_height,
        'specific_conductivity_conductor': cfg.fasthenry.specific_conductivity,
        'cross_sectional_points':        cfg.cross_sectional_points.tolist(),
        'normal_shift_length':           cfg.normal_shift_length,
        'normal_shift_smooth_factors':   cfg.winding.normal_shift_smooth,
        'skip_postprocessing':          False,
        'skip_inductance_calculation':  not cfg.fasthenry.enabled,
        'fasthenry_bin':                fh_resolved,
        'smooth_factor':                cfg.winding.smooth_factor,
        'make_cylindrical_pcb':         False,
        'save_stl_flag':                True,
        'CAD_filename':                 '{project}_wire_{part_index}_{field_function}.stl',
        'tikhonov_reg_factor':          cfg.tikhonov_factor,
        'output_directory':             output_dir,
        'field_shape_function':         internal_axis,
        'project_name':                 project_stem,
    }

    print("\n" + "=" * 70)
    print(f"  Running pyCoilGen — {cfg.axis_label}  |  Tikhonov={cfg.tikhonov_factor:g}"
          f"  |  levels={cfg.num_levels}")
    print(f"  (Internal mapped axis : {internal_axis})")
    print(f"  FastHenry enabled     : {cfg.fasthenry.enabled}")
    print(f"  FastHenry binary      : {fh_resolved or 'not configured'}")
    print(f"  FastHenry available   : {fh_available}")
    print(f"  Cable height          : {cfg.cable_height*1000:.3f} mm")
    print(f"  Layer crossing gap    : {cfg.layer_crossing_gap*1000:.3f} mm")
    print(f"  normal_shift_length   : {cfg.normal_shift_length*1000:.3f} mm")
    print(f"  R_GUI (shell outer)   : {cfg.cylinder.radius*1000:.3f} mm")
    print(f"  Cylinder height (GUI) : {cyl.height*1000:.3f} mm")
    print(f"  pyCoilGen mesh height : {mesh_height*1000:.3f} mm  "
          f"(×{cyl.mesh_length_factor:g})")
    print(f"  Groove margin/face    : {cfg.radial_peel*1000:.3f} mm")
    print(f"  Shell wall thickness  : {cfg.shell_wall_thickness*1000:.3f} mm")
    print(f"  Shell inner bore      : {cfg.shell_inner_radius*1000:.3f} mm")
    print(f"  pyCoilGen design_r    : {cfg.cylinder_design_radius*1000:.3f} mm  "
          f"(base+offset)")
    print(f"  Wire envelope         : {cfg.estimated_wire_inner_radius*1000:.3f}"
          f" -- {cfg.estimated_wire_outer_radius*1000:.3f} mm")
    print("=" * 70)

    # pyCoilGen resolves target_fields/ relative to CWD.
    with _chdir(output_dir):
        solution = pyCoilGen(log, arg_dict)

    metrics = compute_metrics(
        solution, cfg.gradient_axis, output_dir, project_stem,
        cfg.to_params_dict(),
        fasthenry_enabled=cfg.fasthenry.enabled,
        fasthenry_available=fh_available,
    )
    print_metrics_summary(metrics)

    if cfg.show_plots:
        _make_plots(metrics, cfg)

    # ----- Wire overlap check ---------------------------------------------
    overlap_report: Optional[OverlapReport] = None
    if (check_overlap if check_overlap is not None else cfg.overlap_warn):
        overlap_report = detect_collisions(solution, cfg)
        if overlap_report.n_collisions > 0:
            print(f"\n  OVERLAP WARNING: {overlap_report.n_collisions} wire pair(s) "
                  f"closer than {overlap_report.threshold_m*1000:.2f} mm "
                  f"(min distance {overlap_report.min_distance_m*1000:.3f} mm).")
        else:
            print(f"\n  Overlap check OK: min wire distance "
                  f"{overlap_report.min_distance_m*1000:.3f} mm "
                  f"(threshold {overlap_report.threshold_m*1000:.2f} mm).")
        if overlap_report.approximate:
            print("  NOTE: overlap result is approximate (subsampled fallback).")

    # ----- Wire radial extent vs analytical shell -------------------------
    from .shell import warn_wire_radial_mismatch
    wire_path = _paths.resolve_wire_stl_path(
        output_dir, cfg.gradient_axis,
        cfg.tikhonov_factor, cfg.num_levels,
    )
    warn_wire_radial_mismatch(cfg, wire_path)

    print(f"  Results directory      : {output_dir}\n")
    return solution, metrics, overlap_report


def main():
    """CLI entry: run a single gradient design from a default Config."""
    import argparse
    parser = argparse.ArgumentParser(description="Run a single gradient coil design.")
    parser.add_argument('--axis', default='y', choices=('x', 'y', 'z'))
    parser.add_argument('--tikhonov', type=float, default=2500)
    parser.add_argument('--levels', type=int, default=26)
    parser.add_argument('--radius', type=float, default=0.150)
    parser.add_argument('--height', type=float, default=0.430)
    parser.add_argument('--output-dir', default='')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()

    cfg = Config(
        gradient_axis=args.axis,
        tikhonov_factor=args.tikhonov,
        num_levels=args.levels,
        show_plots=not args.no_plots,
    )
    cfg.cylinder.radius = args.radius
    cfg.cylinder.height = args.height
    if args.output_dir:
        cfg.output_dir = args.output_dir
    run_gradient(cfg)


if __name__ == '__main__':
    main()
