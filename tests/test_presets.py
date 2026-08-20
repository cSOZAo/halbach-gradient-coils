import json
from pathlib import Path
import sys

import numpy as np
import pytest

from halbach_coils.coilgen.config import Config
from halbach_coils.coilgen.presets import (
    config_from_preset,
    config_to_preset,
    load_config_preset,
    save_config_preset,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'halbach_coils'))
from halbach_coils.gui.sweep_panel import SweepPanel  # noqa: E402


def test_config_preset_round_trip_preserves_design_parameters(tmp_path):
    cfg = Config(gradient_axis='z', tikhonov_factor=4321, num_levels=17)
    cfg.target.rx, cfg.target.ry, cfg.target.rz = 0.01, 0.02, 0.03
    cfg.cylinder.radius, cfg.cylinder.height = 0.04, 0.18
    cfg.wire.conductor_width = 0.0015
    cfg.wire.cross_section_a_frac = 1.7
    cfg.fasthenry.material = 'Al'
    cfg.fasthenry.specific_conductivity = 2.82e-8
    cfg.winding.layer_gap_mm = 0.0004
    cfg.control.run_shell = False
    path = tmp_path / 'aluminium-z.json'

    save_config_preset(cfg, str(path), name='Aluminium Z')
    loaded = load_config_preset(str(path))

    assert loaded.gradient_axis == 'z'
    assert loaded.tikhonov_factor == 4321
    assert loaded.num_levels == 17
    assert (loaded.target.rx, loaded.target.ry, loaded.target.rz) == (0.01, 0.02, 0.03)
    assert loaded.cylinder.radius == 0.04
    assert loaded.wire.conductor_width == 0.0015
    assert loaded.fasthenry.material == 'Al'
    assert loaded.fasthenry.specific_conductivity == pytest.approx(2.82e-8)
    assert loaded.control.run_shell is False
    assert np.array_equal(loaded.lead_preset().lead_direction,
                          Config(gradient_axis='z').lead_preset().lead_direction)


def test_preset_json_omits_runtime_paths_and_numpy_values(tmp_path):
    cfg = Config()
    cfg.output_dir = 'machine-specific-output'
    cfg.fasthenry.bin_path = 'machine-specific-fasthenry.exe'
    preset = config_to_preset(cfg)

    encoded = json.dumps(preset)

    assert 'machine-specific-output' not in encoded
    assert 'machine-specific-fasthenry' not in encoded


def test_unknown_preset_schema_is_rejected():
    with pytest.raises(ValueError, match='Unsupported preset schema'):
        config_from_preset({'schema_version': 999, 'config': {}})


def test_selected_sweep_row_is_applied_to_pipeline():
    applied = []
    selected_tabs = []
    pipeline_panel = object()
    root = type('Root', (), {
        'pipeline_panel': pipeline_panel,
        'notebook': type('Notebook', (), {
            'select': lambda _self, panel: selected_tabs.append(panel),
        })(),
    })()
    panel = object.__new__(SweepPanel)
    panel.root = root
    panel.tree = type('Tree', (), {'selection': lambda _self: ('row-1',)})()
    panel._rows_by_item = {'row-1': {'Tikhonov': 9876.0}}
    panel._last_sweep_cfg = Config(gradient_axis='z', tikhonov_factor=1)
    root.pipeline_panel = type('Pipeline', (), {
        'apply_config': lambda _self, cfg: applied.append(cfg),
    })()

    panel._use_selected()

    assert applied[0].gradient_axis == 'z'
    assert applied[0].tikhonov_factor == 9876.0
    assert selected_tabs == [root.pipeline_panel]
