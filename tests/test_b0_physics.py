"""B0 selection must change the optimized physics, not just the UI label."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from halbach_coils.coilgen.config import Config
from halbach_coils.coilgen.geometry import build_target_field_file, b0_unit_vector
from halbach_coils.coilgen.metrics import compute_metrics
from pyCoilGen.sub_functions.data_structures import TargetField
from pyCoilGen.sub_functions.field_component import field_direction
from pyCoilGen.sub_functions.process_raw_loops import biot_savart_calc_b, evaluate_loop_significance
from pyCoilGen.sub_functions.calculate_gradient import calculate_gradient
import pyCoilGen.sub_functions.stream_function_optimization as optimizer
import pyCoilGen.sub_functions.evaluate_field_errors as evaluator
import pyCoilGen.sub_functions.find_group_cut_position as cut_position


DIRECTIONS = ('+x', '-x', '+y', '-y', '+z', '-z')


@pytest.mark.parametrize('b0', DIRECTIONS)
@pytest.mark.parametrize('axis', ('x', 'y', 'z'))
def test_target_uses_physical_axis_and_preserves_ellipsoid_radii(tmp_path, axis, b0):
    path, key = build_target_field_file(axis, .03, .04, .05, 3, 9, str(tmp_path), b0)
    data = np.load(path, allow_pickle=True)[0]
    coords = data['coords']
    np.testing.assert_allclose(np.max(np.abs(coords), axis=1), [.03, .04, .05], atol=1e-14)
    np.testing.assert_allclose(b0_unit_vector(b0) @ data[key], coords['xyz'.index(axis)])
    np.testing.assert_allclose(data[key], b0_unit_vector(b0)[:, None] * coords['xyz'.index(axis)])
    cfg = Config(gradient_axis=axis, b0_direction=b0)
    np.testing.assert_allclose(cfg.rotated_cylinder_axis, [0, 0, 1], atol=1e-14)
    assert cfg.internal_axis == axis


@pytest.mark.parametrize('b0', DIRECTIONS)
def test_optimizer_uses_selected_sensitivity_component_and_polarity(monkeypatch, b0):
    # A known three-component linear field operator: Bx=sf, By=2sf, Bz=4sf.
    # Its exact inverse changes with the requested B0; an implementation still
    # hardcoding Bz cannot pass this check for transverse directions.
    sensitivity = np.array([np.eye(3), 2 * np.eye(3), 4 * np.eye(3)])
    part = NS(sensitivity_matrix=sensitivity, resistance_matrix=np.eye(3),
              current_density_mat=np.zeros((3, 2, 3)))
    mesh = NS(vertices=np.zeros((3, 3)), boundary=[np.array([0])],
              mesh_part_vertex_ind=np.ones(3))
    monkeypatch.setattr(optimizer, 'generate_combined_mesh', lambda _parts: mesh)
    b0_vector = b0_unit_vector(b0)
    scalar_target = np.array([-1., .5, 1.])
    target = NS(b=b0_vector[:, None] * scalar_target)
    args = NS(tikhonov_reg_factor=0, sf_opt_method='tikhonov', b_0_direction=b0_vector)
    parts, _, actual = optimizer.stream_function_optimization([part], target, args)
    np.testing.assert_allclose(actual @ b0_vector, scalar_target, atol=1e-14)
    component_scale = b0_vector @ np.array([1, 2, 4])
    np.testing.assert_allclose(parts[0].stream_function, scalar_target / component_scale)


def _loop_and_target():
    angle = np.linspace(0, 2 * np.pi, 401)
    # Loop normal is physical Y; Bz at all axial sample points is zero.
    loop = np.vstack((.07 * np.cos(angle), np.zeros_like(angle), .07 * np.sin(angle)))
    coords = np.vstack((np.zeros(5), np.linspace(.01, .03, 5), np.zeros(5)))
    return loop, TargetField(coords=coords)


@pytest.mark.parametrize('b0', ('+y', '-y'))
def test_gradient_matches_finite_difference_of_signed_biot_savart(b0):
    loop, target = _loop_and_target()
    part = NS(wire_path=NS(v=loop))
    args = NS(field_shape_function='y', b_0_direction=b0_unit_vector(b0))
    result = calculate_gradient([part], args, target)
    epsilon = 1e-6
    offset = np.array([0., epsilon, 0.])[:, None]
    plus = biot_savart_calc_b(loop, TargetField(coords=target.coords + offset))
    minus = biot_savart_calc_b(loop, TargetField(coords=target.coords - offset))
    numerical = b0_unit_vector(b0) @ (plus - minus) / (2 * epsilon) * 1000
    np.testing.assert_allclose(result.gradient_in_target_direction, numerical, rtol=1e-8)


def test_loop_significance_uses_transverse_component():
    loop, target = _loop_and_target()
    target.b = np.zeros_like(target.coords)
    part = NS(contour_lines=[NS(v=loop)], contour_step=1)
    evaluate_loop_significance([part], target, NS(b_0_direction=[0, 1, 0]))
    assert np.isfinite(part.loop_significance[0])
    assert part.loop_significance[0] >= 100


@pytest.mark.parametrize('b0', DIRECTIONS)
def test_metrics_regression_is_in_physical_coordinates(tmp_path, b0):
    coords = np.array([[-.01, 0., .01], [-.02, 0., .02], [-.03, 0., .03]])
    vector = b0_unit_vector(b0)
    layout = vector[:, None] * (.006 * coords[0] + .001)
    # Large unwanted perpendicular fields must not enter scalar metrics.
    perpendicular = np.roll(np.abs(vector), 1)
    layout += perpendicular[:, None] * np.array([2., 4., 2.])
    solution = NS(target_field=NS(coords=coords), coil_parts=[],
                  solution_errors=NS(combined_field_layout_per1Amp=layout,
                                     target_field_1A=NS(b=vector[:, None] * coords[0])))
    metrics = compute_metrics(solution, 'x', str(tmp_path), 'test', {'b0_direction': b0}, False)
    assert metrics['slope_mTmA'] == pytest.approx(6.)
    assert metrics['intercept_mT_per_A'] == pytest.approx(1.)
    assert metrics['rmse_mTA'] < 1e-12
    assert metrics['coordinate_frame'] == 'physical_xyz'


@pytest.mark.parametrize('b0', DIRECTIONS)
def test_error_and_polarity_evaluation_uses_selected_component(monkeypatch, b0):
    vector = b0_unit_vector(b0)
    scalar = np.array([-1., .5, 1.])
    desired = vector[:, None] * scalar
    perpendicular = np.roll(np.abs(vector), 1)
    measured = -desired + 100 * perpendicular[:, None]
    loop = NS(v=np.zeros((3, 4)), uv=np.zeros((2, 4)))
    part = NS(contour_lines=[loop], contour_step=1)
    monkeypatch.setattr(evaluator, 'biot_savart_calc_b', lambda *_: measured)
    args = NS(b_0_direction=vector, skip_postprocessing=True)
    _, errors = evaluator.evaluate_field_errors([part], args, NS(b=desired.copy()), desired.T)
    np.testing.assert_allclose(vector @ errors.combined_field_layout_per1Amp, scalar)
    assert errors.field_error_vals.max_rel_error_layout_vs_target == pytest.approx(0)
    assert errors.opt_current_layout == pytest.approx(1)


def test_defaults_and_invalid_b0():
    assert Config().b0_direction == '+y'
    np.testing.assert_array_equal(field_direction(), [0, 0, 1])
    assert Config(b0_direction='Y').b0_direction == '+y'
    with pytest.raises(ValueError, match='B0'):
        Config(b0_direction='diagonal')
    with pytest.raises(ValueError, match='nonzero'):
        field_direction(NS(b_0_direction=[0, 0, 0]))


@pytest.mark.parametrize('b0', DIRECTIONS)
def test_b0_aligned_cuts_have_finite_plane_for_both_polarities(monkeypatch, b0):
    vector = b0_unit_vector(b0)
    group = NS(loops=[NS(v=np.zeros((3, 2)))])
    monkeypatch.setattr(cut_position, 'calc_mean_loop_normal', lambda *_: vector)

    class PlaneReached(Exception):
        pass

    def inspect_plane(normal, *_):
        assert np.all(np.isfinite(normal))
        assert np.linalg.norm(normal) == pytest.approx(1.)
        assert normal @ vector == pytest.approx(0.)
        raise PlaneReached

    monkeypatch.setattr(cut_position, 'plane_line_intersect', inspect_plane)
    with pytest.raises(PlaneReached):
        cut_position.find_group_cut_position(group, np.zeros(3), None, vector, 'B0')
