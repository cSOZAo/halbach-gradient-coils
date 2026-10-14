"""Evaluate the actual retained coil and terminal paths, in physical SI units."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .design_state import validate_design_state
from .field_plots import FieldPlotData, load_field_data, save_plot_data
from .geometry import b0_unit_vector
from .leads import current_path_path


def _digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def wire_field_data_path(wire_stl):
    wire = Path(wire_stl)
    stem = wire.stem.rsplit('_wire_0', 1)[0]
    return wire.with_name(stem + '_field_data.npz')


def field_at_points(path, coords):
    """Evaluate in target batches so dense ROI meshes do not exhaust memory."""
    from pyCoilGen.sub_functions.process_raw_loops import biot_savart_calc_b

    path, coords = np.asarray(path, dtype=float), np.asarray(coords, dtype=float)
    if (path.ndim != 2 or path.shape[1] != 3 or len(path) < 2
            or coords.ndim != 2 or coords.shape[0] != 3
            or not np.all(np.isfinite(path)) or not np.all(np.isfinite(coords))):
        raise ValueError('The current path and ROI coordinates must be finite XYZ points.')
    field = np.empty_like(coords)
    for start in range(0, coords.shape[1], 128):
        stop = min(start + 128, coords.shape[1])
        field[:, start:stop] = biot_savart_calc_b(
            path.T, SimpleNamespace(coords=coords[:, start:stop]))
    if not np.all(np.isfinite(field)):
        raise ValueError('The ROI includes a singular conductor point; review the target region.')
    return field


def evaluate_final_conductor(cfg, wire_stl, with_leads_stl, output_dir):
    """Persist field plots and electrical metrics for the real final conductor."""
    from pyCoilGen.sub_functions.evaluate_field_errors import optimal_layout_current
    from .final_electrical import evaluate_final_electrical

    validate_design_state(wire_stl, cfg)
    data = load_field_data(wire_field_data_path(wire_stl))
    with np.load(current_path_path(with_leads_stl), allow_pickle=False) as saved:
        if (str(saved['source_sha256'].item()) != _digest(wire_stl)
                or str(saved['with_leads_sha256'].item()) != _digest(with_leads_stl)):
            raise ValueError('The saved current path does not match its conductor STL files.')
        if (str(saved['units'].item()) != 'm'
                or str(saved['coordinate_frame'].item()) != 'physical_xyz'
                or str(saved['orientation'].item()) != 'original_coil_current'
                or bool(saved['supply_return_included'].item())):
            raise ValueError('Unsupported final current-path coordinates or orientation.')
        if (str(saved['b0_direction'].item()) != cfg.b0_direction
                or str(saved['gradient_axis'].item()) != cfg.gradient_axis
                or data.b0_direction != cfg.b0_direction
                or data.gradient_axis != cfg.gradient_axis):
            raise ValueError('Final conductor and target have different B0 or gradient axes.')
        path = np.asarray(saved['path'], dtype=float)
    generated = b0_unit_vector(cfg.b0_direction) @ field_at_points(path, data.coords_m) * 1000
    final_data = FieldPlotData(data.coords_m, generated, data.target_mT_per_A,
                               data.gradient_axis, data.b0_direction,
                               Path(with_leads_stl).stem, circuit_stage='final')
    plots = save_plot_data(final_data, output_dir)
    slope, _fit, _residual, rms = final_data.linearity()
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    target_scale = float(np.max(np.abs(data.target_mT_per_A)))
    error = np.abs(generated - data.target_mT_per_A)
    ideal_current = optimal_layout_current(data.target_mT_per_A, generated)
    electrical = evaluate_final_electrical(cfg, path, output_dir)
    result = {
        'circuit_stage': 'final', 'coordinate_frame': 'physical_xyz',
        'b0_direction': cfg.b0_direction, 'gradient_axis': cfg.gradient_axis,
        'supply_return_included': False,
        'current_path': str(Path(current_path_path(with_leads_stl)).resolve()),
        'wire_length_m': length,
        'resistance_ohm_analytic': cfg.fasthenry.specific_conductivity * length / cfg.electrical_cross_section_area,
        'slope_mT_per_m_per_A': float(slope), 'residual_rms_mT_per_A': float(rms),
        'mean_target_error_pct': float(np.mean(error) / target_scale * 100) if target_scale else None,
        'max_target_error_pct': float(np.max(error) / target_scale * 100) if target_scale else None,
        'optimal_current_A': float(ideal_current) if np.isfinite(ideal_current) else None,
        'fasthenry': electrical, 'field_plots': plots,
    }
    metric_path = Path(output_dir) / (Path(with_leads_stl).stem + '_final_metrics.json')
    metric_path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'  Final conductor metrics: {metric_path}')
    print(f'  Final conductor length: {length:.6g} m; gradient: {slope:.6g} mT/(m A)')
    print(f"  Final ohmic resistance: {result['resistance_ohm_analytic']:.6g} ohm")
    if electrical['status'] == 'success':
        print(f"  Final FastHenry R/L: {electrical['resistance_ohm']:.6g} ohm / "
              f"{electrical['inductance_H']:.6g} H")
    else:
        print(f"  Final FastHenry: {electrical['status']} "
              f"{electrical.get('reason', '')}")
    result['metrics_path'] = str(metric_path)
    return result
