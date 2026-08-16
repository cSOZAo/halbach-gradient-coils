"""Unit tests for :mod:`coilgen.metrics`."""

from types import SimpleNamespace

import numpy as np
import pytest

from coilgen import metrics


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('value,expected', [
    (None, np.nan),
    (np.nan, np.nan),
    ('abc', np.nan),
    (3, 3.0),
    (2.5, 2.5),
    (np.array([[4.0]]), 4.0),
])
def test_float_or_nan(value, expected):
    result = metrics._float_or_nan(value)

    if np.isnan(expected):
        assert np.isnan(result)
    else:
        assert result == expected


def test_sum_finite_or_nan_ignores_non_finite():
    assert metrics._sum_finite_or_nan([1.0, np.nan, 2.0, np.inf]) == 3.0


def test_sum_finite_or_nan_is_nan_without_finite_values():
    assert np.isnan(metrics._sum_finite_or_nan([np.nan, np.inf]))
    assert np.isnan(metrics._sum_finite_or_nan([]))


def test_g_reads_attributes_and_tolerates_none():
    obj = SimpleNamespace(a=1)

    assert metrics._g(obj, 'a') == 1
    assert metrics._g(obj, 'missing') is None
    assert metrics._g(None, 'a') is None


@pytest.mark.parametrize('value,expected', [
    (None, 'n/a'),
    (np.nan, 'n/a'),
    (np.inf, 'n/a'),
    (1.5, '1.5'),
    ('text', 'text'),
    ([1, 2], '[1, 2]'),
])
def test_fmt(value, expected):
    assert metrics._fmt(value) == expected


def test_fmt_honours_the_format_spec():
    assert metrics._fmt(1.23456789, '.3f') == '1.235'


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

def _fake_solution(slope_T_per_m=0.001, noise=0.0, n=41, target_range=1.0,
                   n_parts=1, with_extras=True):
    """
    Build a minimal stand-in for a solved pyCoilGen solution.

    The realized field is linear along the internal axis with slope
    ``slope_T_per_m`` [T/(m.A)] plus optional alternating ``noise`` [T/A].
    """
    coords = np.zeros((3, n))
    z = np.linspace(-0.1, 0.1, n)
    coords[2, :] = z                                  # Gy -> internal axis z
    layout = slope_T_per_m * z
    if noise:
        layout = layout + noise * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    target = target_range * z

    parts = []
    for _ in range(n_parts):
        v = np.array([[0.0, 0.03, 0.03], [0.0, 0.0, 0.04], [0.0, 0.0, 0.0]])
        parts.append(SimpleNamespace(
            wire_path=SimpleNamespace(v=v, v_length=0.06),
            coil_length=0.07,
            ohmian_resistance=0.5,
            coil_resistance=0.6,
            coil_inductance=1e-5,
            coil_cross_section=4e-6,
        ))

    field_errors = SimpleNamespace(
        max_rel_error_layout_vs_target=5.0,
        mean_rel_error_layout_vs_target=1.0,
        max_rel_error_unconnected_contours_vs_target=4.0,
        mean_rel_error_unconnected_contours_vs_target=0.8,
    ) if with_extras else None

    solution_errors = SimpleNamespace(
        combined_field_layout_per1Amp=np.vstack([np.zeros(n), np.zeros(n), layout]),
        target_field_1A=SimpleNamespace(b=np.vstack([np.zeros(n), np.zeros(n), target])),
        opt_current_layout=1.0 if with_extras else None,
    )
    if with_extras:
        solution_errors.field_error_vals = field_errors

    solution = SimpleNamespace(
        target_field=SimpleNamespace(coords=coords),
        solution_errors=solution_errors,
        coil_parts=parts,
    )
    if with_extras:
        solution.coil_gradient = SimpleNamespace(
            mean_gradient_in_target_direction=1.0,
            std_gradient_in_target_direction=0.05,
        )
    return solution


def test_compute_metrics_slope_and_axis_labels(tmp_path):
    solution = _fake_solution(slope_T_per_m=0.002)      # 2 mT/(m.A)

    result = metrics.compute_metrics(
        solution, 'y', str(tmp_path), 'Proj', {'tikhonov_factor': 2500})

    assert result['gradient_axis'] == 'y'
    assert result['internal_axis'] == 'z'
    assert result['axis_label'] == 'Y'
    assert np.isclose(result['slope_mTmA'], 2.0)
    assert np.isclose(result['intercept_mT_per_A'], 0.0)
    assert np.isclose(result['rmse_mTA'], 0.0, atol=1e-9)
    assert np.isclose(result['rmse_gradient_mTmA'], 0.0, atol=1e-9)
    assert np.allclose(result['layout_fit'], result['layout_mT'], atol=1e-9)


def test_compute_metrics_rmse_matches_the_injected_residual(tmp_path):
    noise = 1e-5                                        # 0.01 mT/A alternating
    solution = _fake_solution(slope_T_per_m=0.002, noise=noise)

    result = metrics.compute_metrics(
        solution, 'y', str(tmp_path), 'Proj', {})

    assert np.isclose(result['rmse_mTA'], noise * 1000.0, rtol=1e-3)
    assert np.isclose(result['rmse_gradient_mTmA'], result['rmse_mTA'] / 0.2, rtol=1e-9)


def test_compute_metrics_scales_the_target_onto_the_realized_slope(tmp_path):
    solution = _fake_solution(slope_T_per_m=0.002, target_range=1.0)

    result = metrics.compute_metrics(solution, 'y', str(tmp_path), 'Proj', {})

    scaled = result['target_scaled_mT']
    assert np.isclose(scaled.max() - scaled.min(),
                      result['slope_mTmA'] * 0.2, rtol=1e-9)
    assert np.isclose(scaled.mean(), result['layout_mT'].mean())


def test_compute_metrics_leaves_a_flat_target_unscaled(tmp_path):
    solution = _fake_solution(slope_T_per_m=0.002, target_range=0.0)

    result = metrics.compute_metrics(solution, 'y', str(tmp_path), 'Proj', {})

    assert np.allclose(result['target_scaled_mT'], 0.0)


def test_compute_metrics_wire_lengths_sum_over_parts(tmp_path):
    solution = _fake_solution(n_parts=2)

    result = metrics.compute_metrics(solution, 'y', str(tmp_path), 'Proj', {})

    # Each part polyline is 0.03 m + 0.04 m long, stored length 0.06 m.
    assert np.isclose(result['total_wire_length_computed'], 2 * 0.07)
    assert np.isclose(result['total_wire_length_stored'], 2 * 0.06)
    assert len(result['wire_lengths_computed']) == 2


def test_compute_metrics_ignores_parts_without_a_wire_path(tmp_path):
    solution = _fake_solution()
    solution.coil_parts.append(SimpleNamespace(wire_path=None))

    result = metrics.compute_metrics(solution, 'y', str(tmp_path), 'Proj', {})

    assert len(result['wire_lengths_computed']) == 1
    # Electrical metrics still cover every part, including the pathless one.
    assert len(result['electrical_metrics']) == 2


def test_compute_metrics_keeps_fasthenry_numbers_when_available(tmp_path):
    solution = _fake_solution(n_parts=2)

    result = metrics.compute_metrics(
        solution, 'y', str(tmp_path), 'Proj', {},
        fasthenry_enabled=True, fasthenry_available=True)

    assert np.isclose(result['total_fasthenry_resistance'], 1.2)
    assert np.isclose(result['total_fasthenry_inductance'], 2e-5)
    assert np.isclose(result['total_ohmian_resistance'], 1.0)
    assert np.isclose(result['electrical_metrics'][0]['fasthenry_cross_section_m2'], 4e-6)


@pytest.mark.parametrize('enabled,available', [(False, True), (True, False), (False, False)])
def test_compute_metrics_blanks_fasthenry_when_unavailable(tmp_path, enabled, available):
    solution = _fake_solution()

    result = metrics.compute_metrics(
        solution, 'y', str(tmp_path), 'Proj', {},
        fasthenry_enabled=enabled, fasthenry_available=available)

    assert np.isnan(result['total_fasthenry_resistance'])
    assert np.isnan(result['total_fasthenry_inductance'])
    assert np.isnan(result['electrical_metrics'][0]['fasthenry_resistance_ohm'])
    # The ohmic estimate does not depend on FastHenry.
    assert np.isclose(result['total_ohmian_resistance'], 0.5)


def test_compute_metrics_tolerates_missing_pycoilgen_figures(tmp_path):
    solution = _fake_solution(with_extras=False)

    result = metrics.compute_metrics(solution, 'y', str(tmp_path), 'Proj', {})

    assert result['mean_rel_err_layout_pct'] is None
    assert result['max_rel_err_loops_pct'] is None
    assert result['mean_grad_target'] is None
    assert result['opt_current_layout'] is None
    assert 'n/a' in open(result['metrics_path'], encoding='utf-8').read()


@pytest.mark.parametrize('axis,internal', [('x', 'y'), ('y', 'z'), ('z', 'x')])
def test_compute_metrics_reads_the_coordinate_row_of_the_internal_axis(tmp_path, axis, internal):
    solution = _fake_solution()
    solution.target_field.coords = np.vstack([
        np.full(41, 7.0), np.full(41, 8.0), np.linspace(-0.1, 0.1, 41)])
    row = {'x': 0, 'y': 1, 'z': 2}[internal]
    solution.target_field.coords[row] = np.linspace(-0.1, 0.1, 41)

    result = metrics.compute_metrics(solution, axis, str(tmp_path), 'Proj', {})

    assert result['internal_axis'] == internal
    assert np.allclose(result['coord_grad'], np.linspace(-0.1, 0.1, 41))


def test_compute_metrics_writes_a_self_describing_metrics_file(tmp_path):
    solution = _fake_solution(slope_T_per_m=0.002, n_parts=2)
    params = {'tikhonov_factor': 2500, 'num_levels': 26}

    result = metrics.compute_metrics(
        solution, 'y', str(tmp_path), 'Gradient_Gy', params,
        fasthenry_enabled=True, fasthenry_available=True)

    assert result['metrics_path'] == str(tmp_path / 'Gradient_Gy_metrics.txt')
    text = open(result['metrics_path'], encoding='utf-8').read()

    assert 'project_name              = Gradient_Gy' in text
    assert 'gradient_axis             = y' in text
    assert 'internal_axis             = z' in text
    for section in ('[USER PARAMETERS]', '[REGRESSION ON REALIZED FIELD]',
                    '[pyCoilGen FIELD ERRORS]', '[pyCoilGen GRADIENT (target direction)]',
                    '[WIRE LENGTH]', '[ELECTRICAL METRICS]'):
        assert section in text
    assert 'tikhonov_factor' in text and '2500' in text
    assert 'slope_mT_per_m_per_A          = 2' in text
    assert 'n_target_points               = 41' in text
    assert 'part_1_wire_length_m_stored' in text
    assert 'part_1_fasthenry_inductance_H' in text


def test_compute_metrics_handles_a_two_point_target(tmp_path):
    """Smallest usable target grid: the regression is exact through both points."""
    solution = _fake_solution(n=2)

    result = metrics.compute_metrics(solution, 'y', str(tmp_path), 'Proj', {})

    assert result['coord_grad'].size == 2
    assert np.isclose(result['rmse_mTA'], 0.0, atol=1e-9)
    assert np.isclose(result['slope_mTmA'], 1.0)


# ---------------------------------------------------------------------------
# print_metrics_summary
# ---------------------------------------------------------------------------

def test_print_metrics_summary_reports_headline_numbers(tmp_path, capsys):
    solution = _fake_solution(slope_T_per_m=0.002)
    result = metrics.compute_metrics(solution, 'y', str(tmp_path), 'Proj', {})

    metrics.print_metrics_summary(result)
    out = capsys.readouterr().out

    assert 'METRICS' in out
    assert 'Gradient axis          : Y' in out
    assert 'Slope (realized coil)  : 2.0000' in out
    assert 'FastHenry R            : n/a' in out
    assert result['metrics_path'] in out
