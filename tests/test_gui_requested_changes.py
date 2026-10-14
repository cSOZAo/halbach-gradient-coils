from types import SimpleNamespace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'halbach_coils'))

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from halbach_coils.coilgen.config import Config
from halbach_coils.gui.plot_controls import zoom_3d
from halbach_coils.coilgen.plot_arrows import ViewScaledArrow


def test_bore_and_all_default_terminal_directions_are_z():
    for axis in 'xyz':
        cfg = Config(gradient_axis=axis)
        np.testing.assert_allclose(cfg.rotated_cylinder_axis, [0, 0, 1], atol=1e-14)
        np.testing.assert_allclose(cfg.lead_preset().lead_direction, [0, 0, -1])
        assert np.dot(cfg.lead_preset().sector_ref_dir, cfg.rotated_cylinder_axis) == 0


def test_wheel_zoom_preserves_camera_and_other_axes():
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(121, projection='3d')
    other = fig.add_subplot(122, projection='3d')
    ax.set_xlim(-20, 40)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-85, 85)
    camera = (ax.elev, ax.azim)
    other_bounds = (other.get_xlim(), other.get_ylim(), other.get_zlim())
    before = (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())
    zoom_3d(SimpleNamespace(inaxes=ax, canvas=canvas, step=1, button='up'))
    for old, new in zip(before, (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())):
        assert np.isclose(new[1]-new[0], .8*(old[1]-old[0]))
        assert np.isclose(sum(old), sum(new))
    assert camera == (ax.elev, ax.azim)
    assert other_bounds == (other.get_xlim(), other.get_ylim(), other.get_zlim())
    zoom_3d(SimpleNamespace(inaxes=ax, canvas=canvas, step=-1, button='down'))
    np.testing.assert_allclose(before, (ax.get_xlim(), ax.get_ylim(), ax.get_zlim()))


def test_orientation_arrow_keeps_its_screen_size_and_direction_during_zoom():
    from mpl_toolkits.mplot3d import proj3d
    figure = Figure()
    canvas = FigureCanvasAgg(figure)
    ax = figure.add_subplot(projection='3d')
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_zlim(-100, 100)
    arrow = ViewScaledArrow(ax, [0, 0, 0], [0, -1, 0], fraction=.33,
                            label='B0 -Y', color='purple')

    def screen_length():
        origin, end = arrow.endpoints()
        x, y, _ = proj3d.proj_transform(*np.column_stack((origin, end)), ax.get_proj())
        pixels = ax.transData.transform(np.column_stack((x, y)))
        return np.linalg.norm(pixels[1] - pixels[0])

    canvas.draw()
    initial_screen_length = screen_length()
    initial_world_length = np.linalg.norm(arrow.endpoints()[1])
    initial_artist_count = len(ax.get_children())
    for _ in range(12):
        zoom_3d(SimpleNamespace(inaxes=ax, canvas=canvas, step=1, button='up'))
    canvas.draw()
    assert np.isclose(screen_length(), initial_screen_length)
    assert np.isclose(np.linalg.norm(arrow.endpoints()[1]), initial_world_length * .8**12)
    assert arrow.endpoints()[1][1] < 0
    assert arrow.get_mutation_scale() == 9
    assert len(ax.get_children()) == initial_artist_count
    for _ in range(12):
        zoom_3d(SimpleNamespace(inaxes=ax, canvas=canvas, step=-1, button='down'))
    canvas.draw()
    assert np.isclose(screen_length(), initial_screen_length)


def test_advanced_reset_is_limited_to_advanced_parameters():
    import tkinter as tk
    import pytest
    from halbach_coils.gui.advanced_controls import AdvancedControls
    root = None
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        if 'display' in str(exc).lower():
            pytest.skip(str(exc))
        raise
    root.withdraw()
    cfg = Config(gradient_axis='z', b0_direction='-x')
    cfg.cylinder.radius = .15
    cfg.target.rx = .07
    updates = []
    controls = AdvancedControls(root, cfg, on_change=lambda: updates.append(True))
    controls.variables['cylinder.n_circ'].set('400')
    controls.variables['winding.cut_width'].set('invalid')
    controls.variables['advanced.gauss_order'].set('7')
    controls.reset_button.invoke()
    controls.apply_to(cfg)
    assert cfg.cylinder.n_circ == 200
    assert cfg.winding.cut_width == .001
    assert cfg.advanced.gauss_order == 2
    assert cfg.cylinder.radius == .15 and cfg.target.rx == .07
    assert cfg.gradient_axis == 'z' and cfg.b0_direction == '-x'
    controls._notify()
    assert updates
    root.destroy()
