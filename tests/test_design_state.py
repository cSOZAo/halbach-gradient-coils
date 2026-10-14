"""Reuse requires matching physics/configuration and unchanged binary inputs."""

import copy
import json
import shutil
from pathlib import Path

import pytest

from halbach_coils.coilgen.config import Config
from halbach_coils.coilgen.design_state import (
    save_design_state, validate_design_state, copy_design_state,
)


def _saved_design(tmp_path):
    wire = tmp_path / 'coil_wire_0.stl'
    wire.write_bytes(b'binary STL samples')
    sidecar = tmp_path / 'coil_lead_reference.npz'
    sidecar.write_bytes(b'binary wire centerline arrays')
    cfg = Config(gradient_axis='z', b0_direction='-y')
    state = Path(save_design_state(wire, cfg, (sidecar,)))
    return wire, sidecar, cfg, state


def test_saved_state_is_deterministic_and_validates_binary_sidecar(tmp_path):
    wire, sidecar, cfg, state = _saved_design(tmp_path)
    first = state.read_bytes()
    save_design_state(wire, cfg, (sidecar.name,))
    assert state.read_bytes() == first
    document = json.loads(first)
    assert document['schema_version'] == 2
    assert document['coordinate_frame'] == 'physical_xyz'
    assert document['sidecars'][0]['filename'] == sidecar.name
    validate_design_state(wire, cfg)


@pytest.mark.parametrize('field,value', [
    ('b0_direction', '+y'), ('gradient_axis', 'x'),
    ('tikhonov_factor', 2000), ('num_levels', 12),
    ('target.rx', .02), ('target.resol_radial', 4),
    ('cylinder.radius', .07), ('cylinder.rot_axis', (1, 0, 0)),
    ('cylinder.rot_angle', 1.2), ('winding.smooth_factor', 2),
    ('wire.conductor_width', .003), ('advanced.gauss_order', 4),
    ('fasthenry.specific_conductivity', 2.82e-8),
    ('shell.auto_margin_pct', .1),
])
def test_coil_or_effective_geometry_changes_reject_reuse(tmp_path, field, value):
    wire, _, cfg, _ = _saved_design(tmp_path)
    block, _, name = field.rpartition('.')
    setattr(getattr(cfg, block) if block else cfg, name if block else field, value)
    with pytest.raises(ValueError, match='parameters do not match'):
        validate_design_state(wire, cfg)


def test_runtime_leads_and_fasthenry_flags_do_not_invalidate_coil(tmp_path):
    wire, _, cfg, _ = _saved_design(tmp_path)
    cfg.output_dir = '/other/results'
    cfg.show_plots = False
    cfg.control.run_gradient = False
    cfg.control.run_leads = False
    cfg.leads.lead_length = .05
    cfg.fasthenry.enabled = False
    cfg.fasthenry.bin_path = '/other/fasthenry'
    cfg.shell.smooth_iterations = 20
    cfg.winding.normal_shift_height_factor = 20
    (wire.parent / '.statusrunning').touch()
    validate_design_state(wire, cfg)


@pytest.mark.parametrize('file_to_change', ('wire', 'sidecar', 'state'))
def test_missing_or_altered_files_are_rejected(tmp_path, file_to_change):
    wire, sidecar, cfg, state = _saved_design(tmp_path)
    selected = {'wire': wire, 'sidecar': sidecar, 'state': state}[file_to_change]
    selected.write_bytes(b'changed')
    with pytest.raises(ValueError):
        validate_design_state(wire, cfg)
    selected.unlink()
    with pytest.raises(ValueError):
        validate_design_state(wire, cfg)


def test_state_is_not_written_until_all_sidecars_exist(tmp_path):
    wire = tmp_path / 'coil.stl'
    wire.write_bytes(b'wire')
    with pytest.raises(ValueError, match='missing'):
        save_design_state(wire, Config(), ('missing.npz',))
    assert not (tmp_path / 'coil_design.json').exists()


def test_old_frame_and_path_traversal_entries_are_rejected(tmp_path):
    wire, _, cfg, state = _saved_design(tmp_path)
    valid = json.loads(state.read_text())
    for key, value in [('schema_version', 1), ('coordinate_frame', 'legacy')]:
        document = copy.deepcopy(valid)
        document[key] = value
        state.write_text(json.dumps(document))
        with pytest.raises(ValueError, match='Unsupported'):
            validate_design_state(wire, cfg)
    valid['sidecars'][0]['filename'] = '../outside.npz'
    state.write_text(json.dumps(valid))
    with pytest.raises(ValueError, match='Invalid sidecar'):
        validate_design_state(wire, cfg)


def test_copy_validates_wire_and_copies_required_sidecars(tmp_path):
    source_dir = tmp_path / 'source'
    source_dir.mkdir()
    wire, sidecar, cfg, _ = _saved_design(source_dir)
    dest_dir = tmp_path / 'destination'
    dest_dir.mkdir()
    destination = dest_dir / 'renamed_wire.stl'
    shutil.copy2(wire, destination)
    copied_state = copy_design_state(wire, destination)
    assert Path(copied_state).name == 'renamed_wire_design.json'
    assert (dest_dir / sidecar.name).read_bytes() == sidecar.read_bytes()
    validate_design_state(destination, cfg)
    destination.write_bytes(b'other wire')
    with pytest.raises(ValueError, match='Destination STL'):
        copy_design_state(wire, destination)
