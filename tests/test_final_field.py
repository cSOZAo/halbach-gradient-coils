import hashlib
from pathlib import Path

import numpy as np
import pytest

from halbach_coils.coilgen.config import Config
from halbach_coils.coilgen.design_state import save_design_state
from halbach_coils.coilgen.field_plots import FieldPlotData, load_field_data, save_plot_data
from halbach_coils.coilgen.final_field import evaluate_final_conductor, field_at_points
from halbach_coils.coilgen.leads import current_path_path


def setup_circuit(tmp_path):
    cfg = Config(gradient_axis='x', b0_direction='+y')
    cfg.fasthenry.enabled = False
    stem = 'Gradient_Gx_test'
    wire = tmp_path / (stem + '_wire_0.stl')
    final = tmp_path / (stem + '_wire_0_with_leads.stl')
    wire.write_bytes(b'original-conductor-STL')
    final.write_bytes(b'final-conductor-STL')
    coords = np.vstack([np.linspace(-.01, .01, 9), np.zeros(9), np.zeros(9)])
    data = FieldPlotData(coords, coords[0]*2, coords[0]*3, 'x', '+y', stem)
    artifact = save_plot_data(data, tmp_path)
    save_design_state(wire, cfg, [artifact['data_path']])
    path = np.array([[-.04, 0, -.08], [-.04, 0, -.04], [.04, 0, -.04],
                     [.04, 0, .04], [-.04, 0, .04], [-.04, 0, .08]])
    np.savez_compressed(current_path_path(final), path=path,
                        source_sha256=hashlib.sha256(wire.read_bytes()).hexdigest(),
                        with_leads_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),
                        b0_direction='+y', gradient_axis='x', units='m',
                        coordinate_frame='physical_xyz',
                        orientation='original_coil_current', supply_return_included=False)
    return cfg, wire, final, path, coords


def test_final_metrics_use_retained_coil_and_both_terminals(tmp_path):
    cfg, wire, final, path, coords = setup_circuit(tmp_path)
    result = evaluate_final_conductor(cfg, wire, final, tmp_path)
    data = load_field_data(result['field_plots']['data_path'])
    assert data.circuit_stage == 'final'
    np.testing.assert_allclose(data.generated_mT_per_A, field_at_points(path, coords)[1]*1000)
    expected_length = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
    assert np.isclose(result['wire_length_m'], expected_length)
    assert np.isclose(result['resistance_ohm_analytic'],
                      cfg.fasthenry.specific_conductivity * expected_length / cfg.electrical_cross_section_area)
    assert result['fasthenry']['status'] == 'disabled'
    assert all(Path(p).is_file() for p in result['field_plots']['png_paths'].values())
    assert result['supply_return_included'] is False


def test_final_current_path_rejects_modified_conductor_stl(tmp_path):
    cfg, wire, final, *_ = setup_circuit(tmp_path)
    final.write_bytes(b'modified-STL')
    with pytest.raises(ValueError, match='does not match'):
        evaluate_final_conductor(cfg, wire, final, tmp_path)


def test_biot_savart_batches_equal_unbatched_reference():
    from types import SimpleNamespace
    from pyCoilGen.sub_functions.process_raw_loops import biot_savart_calc_b
    path = np.array([[-.04, 0, -.04], [.04, 0, -.04], [.04, 0, .04]])
    coords = np.vstack([np.linspace(-.02, .02, 275), np.full(275, .01), np.zeros(275)])
    expected = biot_savart_calc_b(path.T, SimpleNamespace(coords=coords))
    np.testing.assert_allclose(field_at_points(path, coords), expected)
