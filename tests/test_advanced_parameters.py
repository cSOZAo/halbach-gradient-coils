import ast
import json
from types import SimpleNamespace
from pathlib import Path
import sys
import tkinter as tk

import pytest

from halbach_coils.coilgen.advanced import (
    AdvancedConfig, PARAMETERS, apply_advanced_values, config_values,
    density_summary, validate_advanced_config,
)
from halbach_coils.coilgen.config import Config
from halbach_coils.coilgen.presets import config_to_preset, config_from_preset
from pyCoilGen.sub_functions.parse_input import parse_input, create_input

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'halbach_coils'))
from halbach_coils.gui.advanced_controls import AdvancedControls  # noqa: E402


def test_advanced_defaults_match_current_engine_and_existing_geometry():
    cfg = Config()
    _, engine = parse_input(False)
    args = AdvancedConfig().to_coilgen_args()
    for key, value in args.items():
        expected = getattr(engine, key)
        if key.startswith('minimize_method_'):
            assert ast.literal_eval(value) == ast.literal_eval(expected)
        else:
            assert value == expected
    before = json.dumps(config_to_preset(cfg), sort_keys=True)
    apply_advanced_values(cfg, config_values(cfg))
    assert json.dumps(config_to_preset(cfg), sort_keys=True) == before


def test_advanced_changes_map_to_existing_blocks_and_valid_engine_arguments():
    cfg = Config()
    apply_advanced_values(cfg, {
        'cylinder.n_circ': '96', 'cylinder.n_long': '12',
        'target.resol_radial': '6', 'target.resol_angular': '18',
        'winding.cut_width': 0.0025, 'winding.normal_shift_smooth': '5, 9, 7',
        'advanced.gauss_order': '4', 'advanced.sf_opt_method': 'minimize',
        'advanced.minimize_method': 'L-BFGS-B', 'advanced.maxiter': '250',
        'advanced.tol': '1e-8',
    })
    assert cfg.cylinder.n_circ == 96
    assert cfg.target.resol_angular == 18
    assert cfg.winding.cut_width == pytest.approx(0.0025)
    assert cfg.winding.normal_shift_smooth == [5, 9, 7]
    args = cfg.advanced.to_coilgen_args()
    assert not any(key.startswith(('cylinder.', 'target.', 'winding.')) for key in args)
    _, engine = create_input(args)
    assert engine.gauss_order == 4
    assert ast.literal_eval(engine.minimize_method_options)['maxiter'] == 250
    assert ast.literal_eval(engine.minimize_method_parameters)['tol'] == 1e-8


@pytest.mark.parametrize(('path', 'invalid'), [
    ('cylinder.n_circ', 7), ('cylinder.n_long', '2.5'),
    ('cylinder.mesh_length_factor', 0), ('target.resol_radial', 1),
    ('target.resol_angular', float('inf')), ('winding.cut_width', float('nan')),
    ('winding.min_loop_signif', -1), ('winding.pot_offset_factor', 1.01),
    ('winding.normal_shift_smooth', '7, 0, 7'),
    ('winding.normal_shift_smooth', '7, 7'), ('advanced.gauss_order', True),
    ('advanced.level_set_method', 'unsupported'), ('advanced.tol', 0),
])
def test_invalid_advanced_change_is_rejected_atomically(path, invalid):
    cfg = Config()
    before = config_values(cfg)
    with pytest.raises(ValueError):
        apply_advanced_values(cfg, {'cylinder.n_circ': 64, path: invalid})
    assert config_values(cfg) == before


def test_density_summary_matches_refined_cylinder_and_stored_target_samples():
    cfg = Config()
    apply_advanced_values(cfg, {
        'cylinder.n_circ': 16, 'cylinder.n_long': 3,
        'advanced.iteration_num_mesh_refinement': 2,
        'target.resol_radial': 4, 'target.resol_angular': 8,
    })
    counts = density_summary(cfg)
    assert counts['triangles'] == 2 * 16 * 3 * 4 ** 2
    assert counts['vertices'] == (16 * 4) * (3 * 4 + 1)
    assert counts['target_points'] == 4 * 8 ** 2


def test_advanced_preset_round_trip_and_bilingual_hover_catalog():
    cfg = Config()
    apply_advanced_values(cfg, {'advanced.gauss_order': 3, 'cylinder.n_circ': 72})
    loaded = config_from_preset(config_to_preset(cfg))
    validate_advanced_config(loaded)
    assert config_values(loaded) == config_values(cfg)
    for spec in PARAMETERS:
        assert spec.label('en') and spec.label('es')
        assert spec.help('en') and spec.help('es')
        assert spec.help('en') != spec.help('es')


def test_controls_display_mm_preserve_defaults_and_reject_hidden_invalid_values():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f'Tk display unavailable: {exc}')
    root.withdraw()
    try:
        cfg = Config()
        controls = AdvancedControls(root, cfg)
        before = config_values(cfg)
        controls.apply_to(cfg)
        assert config_values(cfg) == before
        assert float(controls.variables['winding.cut_width'].get()) == 1.0
        controls.variables['winding.cut_width'].set('2.5')
        controls.apply_to(cfg)
        assert cfg.winding.cut_width == pytest.approx(0.0025)
        assert not controls.expanded.get()
        controls.variables['cylinder.n_circ'].set('bad')
        with pytest.raises(ValueError, match='Circumferential'):
            controls.apply_to(cfg)
    finally:
        root.destroy()


def test_controls_debounce_preview_changes_and_translate_labels():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f'Tk display unavailable: {exc}')
    root.withdraw()
    try:
        root.localizer = SimpleNamespace(language='es')
        changes = []
        controls = AdvancedControls(root, Config(), on_change=lambda: changes.append(True))
        assert controls.toggle.cget('text') == 'Mostrar más parámetros'
        controls.expanded.set(True)
        controls._toggle()
        controls.variables['cylinder.n_circ'].set('64')
        controls.variables['cylinder.n_long'].set('8')
        root.after(400, root.quit)
        root.mainloop()
        assert changes == [True]
        assert controls.summary.cget('text').startswith('Malla:')
    finally:
        root.destroy()
