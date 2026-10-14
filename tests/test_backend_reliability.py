"""Regressions for unavailable current metrics and rejected solver results."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from halbach_coils.coilgen import fasthenry
from pyCoilGen.sub_functions import evaluate_field_errors as evaluator
from pyCoilGen.sub_functions import stream_function_optimization as optimizer


def test_fasthenry_finds_standard_windows_installation(monkeypatch, tmp_path):
    executable = tmp_path / 'FastHenry2.exe'
    executable.touch()
    monkeypatch.setattr(fasthenry.platform, 'system', lambda: 'Windows')
    monkeypatch.setattr(fasthenry.shutil, 'which', lambda _name: None)
    monkeypatch.setattr(fasthenry, 'DEFAULT_WINDOWS_PATH', str(executable))
    assert fasthenry.resolve_fasthenry_bin() == str(executable)


def test_fasthenry_preserves_explicit_and_path_precedence(monkeypatch, tmp_path):
    explicit = tmp_path / 'selected.exe'
    from_path = tmp_path / 'on_path.exe'
    standard = tmp_path / 'standard.exe'
    for executable in (explicit, from_path, standard):
        executable.touch()
    monkeypatch.setattr(fasthenry.platform, 'system', lambda: 'Windows')
    monkeypatch.setattr(fasthenry.shutil, 'which', lambda _name: str(from_path))
    monkeypatch.setattr(fasthenry, 'DEFAULT_WINDOWS_PATH', str(standard))
    assert fasthenry.resolve_fasthenry_bin(str(explicit)) == str(explicit)
    assert fasthenry.resolve_fasthenry_bin() == str(from_path)


def test_fasthenry_missing_candidates_return_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(fasthenry.platform, 'system', lambda: 'Windows')
    monkeypatch.setattr(fasthenry.shutil, 'which', lambda _name: None)
    monkeypatch.setattr(fasthenry, 'DEFAULT_WINDOWS_PATH', str(tmp_path / 'absent.exe'))
    assert fasthenry.resolve_fasthenry_bin(str(tmp_path / 'also_absent.exe')) == ''


@pytest.mark.parametrize('scale', (1., 1e-200, 1e200))
def test_optimal_current_handles_exact_zero_samples_and_field_units(scale):
    # Pointwise target/layout produced NaN from the origin even for exact fields.
    layout = np.array([0., -1., 2.]) * scale
    target = 3 * layout
    with np.errstate(all='raise'):
        assert evaluator.optimal_layout_current(target, layout) == pytest.approx(3.)


def test_optimal_current_minimizes_total_residual_instead_of_mean_ratios():
    layout = np.array([0., 1., 2.])
    target = np.array([1., 1., 1.])
    current = evaluator.optimal_layout_current(target, layout)
    assert current == pytest.approx(.6)
    assert np.linalg.norm(current * layout - target) < np.linalg.norm(.75 * layout - target)


def test_null_layout_current_is_explicitly_unavailable_without_infinity():
    with np.errstate(all='raise'):
        assert np.isnan(evaluator.optimal_layout_current([0, 1, 2], [0, 0, 0]))
        assert evaluator.optimal_layout_current([0, 0, 0], [0, 0, 0]) == 0
        assert evaluator.optimal_layout_current([0, 0, 0], [0, 1, 2]) == 0


@pytest.mark.parametrize('scalar_target', (np.array([0., 1., 2.]), np.zeros(3)))
def test_error_evaluation_handles_null_field_without_division_warnings(monkeypatch, scalar_target):
    generated = np.zeros((3, 3))
    target = np.vstack((np.zeros(3), scalar_target, np.zeros(3)))
    part = NS(contour_lines=[], contour_step=1.)
    args = NS(skip_postprocessing=True, b_0_direction=[0, 1, 0])
    with np.errstate(all='raise'):
        _, result = evaluator.evaluate_field_errors([part], args, NS(b=target), generated.T)
    if np.any(scalar_target):
        assert np.isnan(result.opt_current_layout)
        assert result.field_error_vals.max_rel_error_layout_vs_target == 100
    else:
        assert result.opt_current_layout == 0
        assert result.field_error_vals.max_rel_error_layout_vs_target == 0


def _iterative_problem(monkeypatch, target=None, method='SLSQP'):
    sensitivity = np.array([np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3)])
    part = NS(sensitivity_matrix=sensitivity, resistance_matrix=np.eye(3),
              current_density_mat=np.zeros((3, 2, 3)))
    mesh = NS(vertices=np.zeros((3, 3)), boundary=[np.array([0])],
              mesh_part_vertex_ind=np.ones(3))
    monkeypatch.setattr(optimizer, 'generate_combined_mesh', lambda _: mesh)
    target = np.array([-1., .5, 1.]) if target is None else np.asarray(target)
    field = NS(b=np.vstack((np.zeros(3), np.zeros(3), target)))
    args = NS(tikhonov_reg_factor=.1, sf_opt_method='minimize',
              minimize_method=method, minimize_method_parameters="{'tol': 1e-10}",
              minimize_method_options="{'maxiter': 100}", b_0_direction=[0, 0, 1])
    return [part], field, args


def test_nonconverged_iterative_solution_is_not_used(monkeypatch):
    problem = _iterative_problem(monkeypatch)
    monkeypatch.setattr(optimizer, 'minimize', lambda *_a, **_k: NS(
        success=False, message='Iteration limit reached', x=np.zeros(3)))
    with pytest.raises(RuntimeError, match='did not converge.*Iteration limit'):
        optimizer.stream_function_optimization(*problem)


@pytest.mark.parametrize('coefficients', ([np.nan, 0, 0], [np.inf, 0, 0], [0, 0]))
def test_claimed_success_with_invalid_coefficients_is_rejected(monkeypatch, coefficients):
    problem = _iterative_problem(monkeypatch)
    monkeypatch.setattr(optimizer, 'minimize', lambda *_a, **_k: NS(
        success=True, message='success', x=np.array(coefficients)))
    with pytest.raises(RuntimeError, match='invalid or non-finite coefficients'):
        optimizer.stream_function_optimization(*problem)


def test_claimed_success_with_worse_objective_is_rejected(monkeypatch):
    problem = _iterative_problem(monkeypatch)
    monkeypatch.setattr(optimizer, 'minimize', lambda *_a, **_k: NS(
        success=True, message='success', x=np.array([1., 1., 1.])))
    with pytest.raises(RuntimeError, match='worse field fit'):
        optimizer.stream_function_optimization(*problem)


@pytest.mark.parametrize('method', ('SLSQP', 'L-BFGS-B'))
def test_real_iterative_optimizer_accepts_finite_optimum_and_negative_start(monkeypatch, method):
    # An all-negative start previously constructed reversed bounds.
    problem = _iterative_problem(monkeypatch, target=[-1., -.5, -.75], method=method)
    _, _, field = optimizer.stream_function_optimization(*problem)
    np.testing.assert_allclose(field[:, 2], np.array([-1., -.5, -.75]) / 1.1, rtol=1e-5)
