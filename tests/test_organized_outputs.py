import hashlib
import json
from pathlib import Path

import pytest

from halbach_coils.coilgen import paths
from halbach_coils.coilgen.config import Config
from halbach_coils.coilgen.design_state import save_design_state, validate_design_state, copy_design_state
from halbach_coils.coilgen.outputs import organize_outputs, relocate_paths, write_result_summary
from halbach_coils.gui.field_viewer import electrical_result_text


STEM = 'Gradient_Gx_tk2500_lvl12'


def test_arrangement_preserves_provenance_reuse_and_field_artifacts(tmp_path):
    cfg = Config(gradient_axis='x', num_levels=12)
    wire = tmp_path / f'{STEM}_wire_0.stl'
    wire.write_bytes(b'original-wire')
    sidecar = tmp_path / f'{STEM}_field_data.npz'
    sidecar.write_bytes(b'field-data')
    save_design_state(wire, cfg, [sidecar])
    final = tmp_path / f'{wire.stem}_with_leads.stl'
    final.write_bytes(b'final-wire')
    shell = tmp_path / f'{STEM}_shell_0.stl'
    shell.write_bytes(b'shell')
    png = tmp_path / f'{final.stem}_field_comparison.png'
    png.write_bytes(b'png')
    paths.write_active_stem(str(tmp_path), STEM)
    report = tmp_path / f'{final.stem}_final_metrics.json'
    report.write_text(json.dumps({'field_plots': {'data_path': str(sidecar),
                                                 'png_paths': {'comparison': str(png)}}}))
    unrelated = tmp_path / 'my-notes.txt'
    unrelated.write_text('keep')
    original_digest = hashlib.sha256(wire.read_bytes()).hexdigest()
    moves = organize_outputs(tmp_path)
    resolved = Path(paths.resolve_wire_stl_path(str(tmp_path), 'x', 2500, 12))
    assert resolved == tmp_path / 'Bobina' / wire.name
    validate_design_state(resolved, cfg)
    assert hashlib.sha256(resolved.read_bytes()).hexdigest() == original_digest
    assert unrelated.read_text() == 'keep'
    assert paths.resolve_lead_stl_paths(str(resolved))[0] == str(tmp_path / 'Leads' / final.name)
    assert paths.derive_align_wire_path(str(tmp_path / 'Leads' / final.name)) == str(resolved)
    saved = json.loads((tmp_path / 'Leads' / report.name).read_text())
    assert saved['field_plots']['png_paths']['comparison'] == str(tmp_path / 'Graficos' / png.name)
    assert Path(saved['field_plots']['data_path']).is_file()
    assert relocate_paths({'file': str(shell)}, moves)['file'] == str(tmp_path / 'Molde' / shell.name)
    assert paths.unique_stem(str(tmp_path), STEM, gradient_axis='x') == STEM + '(2)'
    assert organize_outputs(tmp_path) == {}  # Reopening a completed result is harmless.
    copied = tmp_path / 'reused' / wire.name
    copied.parent.mkdir()
    copied.write_bytes(resolved.read_bytes())
    copy_design_state(resolved, copied)
    validate_design_state(copied, cfg)


def test_diagnostics_paths_in_final_json_are_updated(tmp_path):
    folder = tmp_path / 'final_fasthenry_example'
    folder.mkdir()
    record = {'evidence_directory': str(folder), 'input_file': str(folder / 'final_circuit.inp')}
    (folder / 'result.json').write_text(json.dumps(record))
    organize_outputs(tmp_path)
    saved = json.loads((tmp_path / 'Diagnostico' / folder.name / 'result.json').read_text())
    assert saved['evidence_directory'] == str(tmp_path / 'Diagnostico' / folder.name)
    assert saved['input_file'].startswith(saved['evidence_directory'])


def test_existing_files_are_not_overwritten_or_partially_moved(tmp_path):
    file = tmp_path / f'{STEM}_wire_0.stl'
    file.write_bytes(b'new')
    (tmp_path / 'Bobina').mkdir()
    existing = tmp_path / 'Bobina' / file.name
    existing.write_bytes(b'previous')
    with pytest.raises(FileExistsError):
        organize_outputs(tmp_path)
    assert file.read_bytes() == b'new'
    assert existing.read_bytes() == b'previous'


def test_partial_leads_and_shells_reserve_organized_names(tmp_path):
    (tmp_path / 'Leads').mkdir()
    (tmp_path / 'Leads' / f'{STEM}_wire_0_coil_open.stl').touch()
    result = paths.unique_lead_output_paths(str(tmp_path / f'{STEM}_wire_0.stl'))
    assert all('(2)' in path for path in result)
    (tmp_path / 'Molde').mkdir()
    (tmp_path / 'Molde' / f'{STEM}_shell_0.stl').touch()
    assert paths.unique_path(str(tmp_path / f'{STEM}_shell_0.stl')).endswith('_shell_0(2).stl')


@pytest.mark.parametrize('status', ['success', 'failed', 'disabled'])
def test_visible_summary_and_gui_use_final_solver_values(tmp_path, status):
    (tmp_path / 'Leads').mkdir()
    result = {'gradient_axis': 'x', 'b0_direction': '-y',
              'slope_mT_per_m_per_A': 5.062, 'wire_length_m': 6.783,
              'resistance_ohm_analytic': .0362747,
              'fasthenry': {'status': status, 'resistance_ohm': .0362738 if status == 'success' else None,
                           'inductance_H': 1.477785e-5 if status == 'success' else None,
                           'reason': 'Solver timeout' if status == 'failed' else ''}}
    filename = tmp_path / 'Leads' / f'{STEM}_wire_0_with_leads_final_metrics.json'
    filename.write_text(json.dumps(result))
    metrics = write_result_summary(tmp_path, stem=STEM)
    summary = (tmp_path / 'Resumen.txt').read_text(encoding='utf-8')
    gui_text = electrical_result_text(metrics)
    if status == 'success':
        assert '0.0362738' in summary and '14.77785' in summary
        assert '0.0362738' in gui_text and '14.77785' in gui_text
    else:
        assert status in summary and status in gui_text
        assert '14.77785' not in summary and '14.77785' not in gui_text
    if status == 'failed':
        assert 'Solver timeout' in summary and 'Solver timeout' in gui_text
