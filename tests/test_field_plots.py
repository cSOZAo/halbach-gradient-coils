from types import SimpleNamespace

import numpy as np
import pytest

from halbach_coils.coilgen.field_plots import (
    FIELD_PLOT_TRANSLATIONS_ES, FieldPlotData, PLOT_KINDS, create_field_figure,
    field_data_from_solution, load_field_data, save_field_plots,
)


def _solution():
    # Independent nonzero components detect accidental use of Bz or a rotated
    # gradient coordinate. The field is linear in physical X with an offset.
    coord = np.linspace(-0.05, 0.05, 31)
    coords = np.vstack((coord, np.sin(coord * 40) * 0.02, np.cos(coord * 50) * 0.01))
    fields = np.vstack((0.7 + coord, 0.003 * coord + 0.00002, 0.3 - 2 * coord))
    target = fields.copy()
    target[1] = 0.003 * coord
    errors = SimpleNamespace(combined_field_layout_per1Amp=fields,
                             target_field_1A=SimpleNamespace(b=target))
    return SimpleNamespace(target_field=SimpleNamespace(coords=coords), solution_errors=errors)


def test_signed_b0_projection_and_gradient_use_physical_coordinates():
    data = field_data_from_solution(_solution(), "x", "-y")
    np.testing.assert_allclose(data.generated_mT_per_A,
                               -(_solution().solution_errors.combined_field_layout_per1Amp[1] * 1000))
    slope, fit, residual, rms = data.linearity()
    assert slope == pytest.approx(-3.0)
    np.testing.assert_allclose(fit, data.generated_mT_per_A, atol=1e-12)
    np.testing.assert_allclose(residual, 0, atol=1e-12)
    assert rms < 1e-12
    # Preserve the true target amplitude and offset; no hidden target scaling.
    np.testing.assert_allclose(data.target_mT_per_A, -3 * data.coords_m[0])


@pytest.mark.parametrize("direction,component,sign", [
    ("+x", 0, 1), ("-x", 0, -1), ("+y", 1, 1),
    ("-y", 1, -1), ("+z", 2, 1), ("-z", 2, -1),
])
def test_all_signed_components(direction, component, sign):
    solution = _solution()
    data = field_data_from_solution(solution, "x", direction)
    np.testing.assert_allclose(data.generated_mT_per_A,
                               solution.solution_errors.combined_field_layout_per1Amp[component] * sign * 1000)


def test_saved_plots_work_headless_and_npz_reopens_without_pickle(tmp_path, monkeypatch):
    # Export must not delegate to pyplot.show or create a global GUI figure.
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *a, **k: pytest.fail("show called from exporter"))
    existing_figures = plt.get_fignums()
    artifact = save_field_plots(_solution(), tmp_path, "x", "-y", "synthetic")
    reopened = load_field_data(artifact["data_path"])
    assert reopened.b0_direction == "-y"
    assert reopened.gradient_axis == "x"
    assert reopened.project_name == "synthetic"
    np.testing.assert_allclose(reopened.coords_m, _solution().target_field.coords)
    with np.load(artifact["data_path"], allow_pickle=False) as arrays:
        assert all(not arrays[key].dtype.hasobject for key in arrays.files)
    assert set(artifact["png_paths"]) == set(PLOT_KINDS)
    for path in artifact["png_paths"].values():
        with open(path, "rb") as image:
            assert image.read(8) == b"\x89PNG\r\n\x1a\n"
    assert plt.get_fignums() == existing_figures


def test_plot_uses_target_amplitude_and_translated_captions():
    data = field_data_from_solution(_solution(), "x", "+y")
    tr = lambda text, **values: FIELD_PLOT_TRANSLATIONS_ES.get(text, text).format(**values)
    figure = create_field_figure(data, "comparison", tr)
    np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(), data.target_mT_per_A)
    assert "Campo objetivo" in figure._suptitle.get_text()
    assert figure.axes[0].get_xlabel() == "Posición X [mm]"
    figure.clear()


def test_reopened_spatial_map_keeps_saved_b0_and_signed_samples(tmp_path):
    solution = _solution()
    artifact = save_field_plots(solution, tmp_path, "x", "-y", "saved_b0")
    expected = -solution.solution_errors.combined_field_layout_per1Amp[1] * 1000
    # A later design or menu change must not reinterpret a previous result.
    solution.solution_errors.combined_field_layout_per1Amp[:] = 999
    reopened = load_field_data(artifact["data_path"])
    figure = create_field_figure(reopened, "spatial")
    assert "B0 -Y" in figure._suptitle.get_text()
    np.testing.assert_allclose(figure.axes[0].collections[0].get_array(), expected)
    # The second spatial panel is a physical target error, not a fit residual.
    np.testing.assert_allclose(figure.axes[2].collections[0].get_array(),
                               expected - reopened.target_mT_per_A)
    figure.clear()


def test_residual_detects_cross_axis_field_variation():
    solution = _solution()
    solution.solution_errors.combined_field_layout_per1Amp[1] += solution.target_field.coords[2] * 0.001
    data = field_data_from_solution(solution, "x", "+y")
    assert data.linearity()[3] > 0


def test_invalid_samples_and_unknown_formats_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="span"):
        FieldPlotData(np.zeros((3, 2)), np.zeros(2), np.zeros(2), "x", "+y")
    with pytest.raises(ValueError, match="non-finite"):
        FieldPlotData(np.array([[0., 1.], [0., 0.], [0., 0.]]),
                      np.array([0., np.nan]), np.zeros(2), "x", "+y")
    with pytest.raises(ValueError, match="B0"):
        field_data_from_solution(_solution(), "x", "+q")
    np.savez(tmp_path / "future.npz", format_version=np.asarray(99))
    with pytest.raises(ValueError, match="Unsupported"):
        load_field_data(tmp_path / "future.npz")
