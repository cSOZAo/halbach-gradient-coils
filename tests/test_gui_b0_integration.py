"""Exercise the actual GUI entry point and its physical configuration contract."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'halbach_coils'))


@pytest.fixture
def app():
    import tkinter as tk
    from gui.main_window import CoilGenApp

    try:
        window = CoilGenApp()
    except tk.TclError as exc:
        if 'display' in str(exc).lower():
            pytest.skip(f'Tk display unavailable: {exc}')
        raise
    window.withdraw()
    window.update()
    yield window
    window.destroy()


def test_gui_preset_restores_b0_and_hidden_advanced_values(app):
    from coilgen.presets import config_from_preset, config_to_preset

    panel = app.pipeline_panel
    app.b0_var.set('-x')
    panel.coilgen_advanced.variables['cylinder.n_circ'].set('64')
    panel.coilgen_advanced.variables['advanced.gauss_order'].set('3')
    cfg = panel._build_cfg()
    saved = config_to_preset(cfg)
    app.b0_var.set('+z')
    panel.coilgen_advanced.variables['cylinder.n_circ'].set('32')
    panel.apply_config(config_from_preset(saved))
    restored = panel._build_cfg()
    assert restored.b0_direction == '-x'
    assert restored.cylinder.n_circ == 64
    assert restored.advanced.gauss_order == 3
    assert not panel.coilgen_advanced.expanded.get()
    app.localizer.set_language('es')
    app._refresh_language()
    assert app.menu_bar.entrycget(1, 'label') == 'B₀: -X'
    assert app.menu_bar.entrycget(2, 'label') == 'Resultados'


def test_preview_roi_uses_bore_x_and_physical_labels(app):
    panel = app.pipeline_panel
    # A long X semiaxis fits the bore; the same radial semiaxis does not.
    assert not panel._roi_cylinder_collision(.04, .01, .01, .035, .085)
    assert panel._roi_cylinder_collision(.01, .04, .01, .035, .085)
    assert panel._roi_cylinder_collision(.09, .01, .01, .035, .085)
    app.b0_var.set('-z')
    assert panel.geo3d_ax.get_xlabel() == 'X [mm]'
    assert panel.geo3d_ax.get_ylabel() == 'Y [mm]'
    assert panel.geo3d_ax.get_zlabel() == 'Z [mm]'
    assert 'B₀ -Z' in panel.geo3d_caption.get_text()
