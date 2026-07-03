"""
Metrics extraction for a pyCoilGen run.

Given a solved ``solution`` object and the design parameters, computes:
- slope of the realized gradient [mT/(m.A)]
- RMSE of the residuals and RMSE/range [mT/(m.A)]
- wire length (stored + recomputed) [m]
- ohmic / FastHenry resistance and inductance
- pyCoilGen field-error and gradient figures of merit

Also writes a self-describing ``{project}_metrics.txt`` next to the run.
All units SI unless noted (mT used for readability of field values).
"""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np

from .geometry import internal_field_axis


def _float_or_nan(value: Any) -> float:
    try:
        if value is None:
            return np.nan
        return float(np.asarray(value).squeeze())
    except (TypeError, ValueError):
        return np.nan


def _sum_finite_or_nan(values) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.sum(finite)) if finite else np.nan


def _g(obj: Any, name: str):
    return getattr(obj, name, None) if obj is not None else None


def _fmt(val: Any, fmt: str = '.6g') -> str:
    """Format helper that tolerates None / NaN."""
    if val is None:
        return 'n/a'
    try:
        if isinstance(val, (float, np.floating)) and not np.isfinite(val):
            return 'n/a'
        return format(val, fmt)
    except (TypeError, ValueError):
        return str(val)


def compute_metrics(solution,
                    gradient_axis: str,
                    output_dir: str,
                    project_name: str,
                    params: Dict[str, Any],
                    fasthenry_enabled: bool = True,
                    fasthenry_available: bool = False) -> Dict[str, Any]:
    """
    Compute and persist metrics for ``solution``.

    ``params`` is a flat dict of the user parameters that produced the run
    (target radii, cylinder dims, tikhonov, levels, wire cross-section, ...).
    Returns a dict with the headline numbers plus the arrays needed for plots.
    """
    internal_axis = internal_field_axis(gradient_axis)
    axis_index = {'x': 0, 'y': 1, 'z': 2}[internal_axis]
    axis_label = gradient_axis.upper()

    coords = solution.target_field.coords                       # (3, N) [m]
    coord_grad = coords[axis_index, :]                          # [m]
    # pyCoilGen optimizes Bz internally -> index 2 is the relevant component.
    layout_field = solution.solution_errors.combined_field_layout_per1Amp[2]  # [T/A]
    target_field = solution.solution_errors.target_field_1A.b[2]              # [T/A]

    layout_mT = layout_field * 1000.0
    target_mT = target_field * 1000.0

    slope_mTmA, intercept = np.polyfit(coord_grad, layout_mT, 1)
    layout_fit = slope_mTmA * coord_grad + intercept

    residuals = layout_mT - layout_fit
    coord_range = coord_grad.max() - coord_grad.min()
    rmse_mTA = float(np.sqrt(np.mean(residuals ** 2)))
    rmse_gradient_mTmA = rmse_mTA / coord_range if coord_range > 0 else 0.0

    # Scale the (arbitrarily scaled) target to the realized slope so the
    # objective-vs-generated comparison is visually interpretable.
    target_range = float(target_mT.max() - target_mT.min())
    if target_range > 0:
        target_scaled_mT = (target_mT - target_mT.mean()) * (
            (slope_mTmA * coord_range) / target_range
        ) + layout_mT.mean()
    else:
        target_scaled_mT = target_mT

    # ----- Wire length -----------------------------------------------------
    wire_lengths_stored: list[float] = []
    wire_lengths_computed: list[float] = []
    for part in solution.coil_parts:
        wp = getattr(part, 'wire_path', None)
        if wp is None or wp.v is None:
            continue
        seg = np.linalg.norm(np.diff(wp.v, axis=1), axis=0)
        wire_lengths_computed.append(float(np.sum(seg)))
        wire_lengths_stored.append(float(getattr(wp, 'v_length', np.nan)))

    total_wire_length_computed = (
        float(np.sum(wire_lengths_computed)) if wire_lengths_computed else 0.0
    )
    total_wire_length_stored = (
        float(np.nansum(wire_lengths_stored)) if wire_lengths_stored else 0.0
    )

    # ----- Electrical metrics ---------------------------------------------
    electrical_metrics = []
    for part_index, part in enumerate(solution.coil_parts):
        fh_resistance = _float_or_nan(getattr(part, 'coil_resistance', np.nan))
        fh_inductance = _float_or_nan(getattr(part, 'coil_inductance', np.nan))
        fh_cross_section = _float_or_nan(getattr(part, 'coil_cross_section', np.nan))
        if not (fasthenry_enabled and fasthenry_available):
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

    # ----- pyCoilGen figures of merit --------------------------------------
    ferr = getattr(solution.solution_errors, 'field_error_vals', None)
    max_rel_err_layout = _g(ferr, 'max_rel_error_layout_vs_target')
    mean_rel_err_layout = _g(ferr, 'mean_rel_error_layout_vs_target')
    max_rel_err_loops = _g(ferr, 'max_rel_error_unconnected_contours_vs_target')
    mean_rel_err_loops = _g(ferr, 'mean_rel_error_unconnected_contours_vs_target')
    opt_current_layout = getattr(solution.solution_errors, 'opt_current_layout', None)

    cgrad = getattr(solution, 'coil_gradient', None)
    mean_grad_target = _g(cgrad, 'mean_gradient_in_target_direction')
    std_grad_target = _g(cgrad, 'std_gradient_in_target_direction')

    # ----- Persist metrics txt --------------------------------------------
    metrics_path = os.path.join(output_dir, f"{project_name}_metrics.txt")
    with open(metrics_path, 'w', encoding='utf-8') as fh:
        fh.write("# pyCoilGen run metrics\n")
        fh.write(f"project_name              = {project_name}\n")
        fh.write(f"gradient_axis             = {gradient_axis}\n")
        fh.write(f"internal_axis             = {internal_axis}\n")
        fh.write("\n[USER PARAMETERS]\n")
        for key, value in params.items():
            fh.write(f"{key:<32} = {value}\n")

        fh.write("\n[REGRESSION ON REALIZED FIELD]\n")
        fh.write(f"slope_mT_per_m_per_A          = {_fmt(slope_mTmA)}\n")
        fh.write(f"intercept_mT_per_A            = {_fmt(intercept)}\n")
        fh.write(f"rmse_residual_mT_per_A        = {_fmt(rmse_mTA)}\n")
        fh.write(f"rmse_per_range_mT_per_m_per_A = {_fmt(rmse_gradient_mTmA)}\n")
        fh.write(f"coord_range_m                 = {_fmt(coord_range)}\n")
        fh.write(f"n_target_points               = {coord_grad.size}\n")

        fh.write("\n[pyCoilGen FIELD ERRORS]\n")
        fh.write(f"max_rel_err_layout_vs_target_pct  = {_fmt(max_rel_err_layout)}\n")
        fh.write(f"mean_rel_err_layout_vs_target_pct = {_fmt(mean_rel_err_layout)}\n")
        fh.write(f"max_rel_err_loops_vs_target_pct   = {_fmt(max_rel_err_loops)}\n")
        fh.write(f"mean_rel_err_loops_vs_target_pct  = {_fmt(mean_rel_err_loops)}\n")
        fh.write(f"opt_current_layout_A              = {_fmt(opt_current_layout)}\n")

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
        fh.write(f"total_fasthenry_inductance_H_sum_of_parts = "
                 f"{_fmt(total_fasthenry_inductance_sum)}\n")
        for item in electrical_metrics:
            idx = item['part_index']
            fh.write(f"part_{idx}_coil_length_m              = {_fmt(item['coil_length_m'])}\n")
            fh.write(f"part_{idx}_ohmian_resistance_ohm      = {_fmt(item['ohmian_resistance_ohm'])}\n")
            fh.write(f"part_{idx}_fasthenry_resistance_ohm   = {_fmt(item['fasthenry_resistance_ohm'])}\n")
            fh.write(f"part_{idx}_fasthenry_inductance_H     = {_fmt(item['fasthenry_inductance_H'])}\n")
            fh.write(f"part_{idx}_fasthenry_cross_section_m2 = {_fmt(item['fasthenry_cross_section_m2'])}\n")

    return {
        'gradient_axis': gradient_axis,
        'internal_axis': internal_axis,
        'axis_label': axis_label,
        'slope_mTmA': float(slope_mTmA),
        'intercept_mT_per_A': float(intercept),
        'rmse_mTA': rmse_mTA,
        'rmse_gradient_mTmA': rmse_gradient_mTmA,
        'coord_grad': coord_grad,
        'layout_mT': layout_mT,
        'target_scaled_mT': target_scaled_mT,
        'layout_fit': layout_fit,
        'mean_rel_err_layout_pct': mean_rel_err_layout,
        'max_rel_err_layout_pct': max_rel_err_layout,
        'mean_rel_err_loops_pct': mean_rel_err_loops,
        'max_rel_err_loops_pct': max_rel_err_loops,
        'total_wire_length_stored': total_wire_length_stored,
        'total_wire_length_computed': total_wire_length_computed,
        'wire_lengths_stored': wire_lengths_stored,
        'wire_lengths_computed': wire_lengths_computed,
        'total_ohmian_resistance': total_ohmian_resistance,
        'total_fasthenry_resistance': total_fasthenry_resistance,
        'total_fasthenry_inductance': total_fasthenry_inductance_sum,
        'electrical_metrics': electrical_metrics,
        'mean_grad_target': mean_grad_target,
        'std_grad_target': std_grad_target,
        'opt_current_layout': opt_current_layout,
        'metrics_path': metrics_path,
    }


def print_metrics_summary(metrics: Dict[str, Any]) -> None:
    """Print the headline numbers to stdout (console panel)."""
    label = metrics['axis_label']
    print("\n" + "-" * 70)
    print("  METRICS")
    print("-" * 70)
    print(f"  Gradient axis          : {label}")
    print(f"  Slope (realized coil)  : {metrics['slope_mTmA']:.4f}  mT/(m.A)")
    print(f"  RMSE of residuals      : {metrics['rmse_mTA']:.4f}  mT/A")
    print(f"  RMSE / coord range     : {metrics['rmse_gradient_mTmA']:.4f}  mT/(m.A)")
    print(f"  Wire length (stored)   : {metrics['total_wire_length_stored']:.4f}  m"
          f"   (per part: {[f'{x:.4f}' for x in metrics['wire_lengths_stored']]})")
    print(f"  Wire length (recomputed): {metrics['total_wire_length_computed']:.4f}  m"
          f"   (per part: {[f'{x:.4f}' for x in metrics['wire_lengths_computed']]})")
    print(f"  Ohmic R estimate       : {_fmt(metrics['total_ohmian_resistance'])}  ohm")
    print(f"  FastHenry R            : {_fmt(metrics['total_fasthenry_resistance'])}  ohm")
    print(f"  FastHenry L (part sum) : {_fmt(metrics['total_fasthenry_inductance'])}  H")
    print(f"  Mean rel err layout    : {_fmt(metrics['mean_rel_err_layout_pct'])} %")
    print("-" * 70)
    print(f"  Metrics written to     : {metrics['metrics_path']}")
