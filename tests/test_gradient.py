"""
Unit tests for :mod:`coilgen.gradient`.

``run_gradient`` imports ``pyCoilGen`` lazily, which lets these tests inject a
fake solver and assert on the ``arg_dict`` contract documented in the README
(every key must exist in pyCoilGen's parser, ``field_shape_function`` is the
*internal* axis, the mesh length is ``height * mesh_length_factor``, ...).
No real coil is optimized.
"""

import os
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from coilgen import gradient
from coilgen.config import Config


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def _fake_solution(n=21):
    z = np.linspace(-0.1, 0.1, n)
    coords = np.vstack([z, z, z])       # any gradient axis yields a valid fit
    part = SimpleNamespace(
        wire_path=SimpleNamespace(
            v=np.array([[0.0, 0.03], [0.0, 0.0], [0.0, 0.0]]), v_length=0.03),
        coil_length=0.03,
        ohmian_resistance=0.4,
        coil_resistance=0.5,
        coil_inductance=1e-5,
        coil_cross_section=4e-6,
    )
    return SimpleNamespace(
        target_field=SimpleNamespace(coords=coords),
        solution_errors=SimpleNamespace(
            combined_field_layout_per1Amp=np.vstack(
                [np.zeros(n), np.zeros(n), 0.002 * z]),
            target_field_1A=SimpleNamespace(
                b=np.vstack([np.zeros(n), np.zeros(n), z])),
            opt_current_layout=1.0,
            field_error_vals=SimpleNamespace(
                max_rel_error_layout_vs_target=4.0,
                mean_rel_error_layout_vs_target=1.0,
                max_rel_error_unconnected_contours_vs_target=3.0,
                mean_rel_error_unconnected_contours_vs_target=0.9,
            ),
        ),
        coil_parts=[part],
        coil_gradient=SimpleNamespace(
            mean_gradient_in_target_direction=1.0,
            std_gradient_in_target_direction=0.1,
        ),
    )


@pytest.fixture
def fake_pycoilgen(monkeypatch):
    """Install a fake ``pyCoilGen`` package and capture the arg_dict / CWD."""
    captured = {}

    def fake_solver(log, arg_dict):
        captured['arg_dict'] = dict(arg_dict)
        captured['cwd'] = os.getcwd()
        return _fake_solution()

    release = types.ModuleType('pyCoilGen.pyCoilGen_release')
    release.pyCoilGen = fake_solver
    package = types.ModuleType('pyCoilGen')
    package.pyCoilGen_release = release
    monkeypatch.setitem(sys.modules, 'pyCoilGen', package)
    monkeypatch.setitem(sys.modules, 'pyCoilGen.pyCoilGen_release', release)

    # coilgen.shell needs manifold3d, which is optional for this step.
    shell_stub = types.ModuleType('coilgen.shell')
    shell_stub.warn_wire_radial_mismatch = lambda cfg, path: captured.setdefault(
        'warned_path', path)
    monkeypatch.setitem(sys.modules, 'coilgen.shell', shell_stub)

    return captured


def _cfg(axis='y', **kwargs):
    cfg = Config(gradient_axis=axis, tikhonov_factor=2500, num_levels=26,
                 show_plots=False, **kwargs)
    cfg.fasthenry.enabled = False
    return cfg


# ---------------------------------------------------------------------------
# _chdir
# ---------------------------------------------------------------------------

def test_chdir_restores_the_previous_directory(tmp_path):
    before = os.getcwd()

    with gradient._chdir(str(tmp_path)):
        assert os.path.realpath(os.getcwd()) == os.path.realpath(str(tmp_path))

    assert os.getcwd() == before


def test_chdir_restores_the_directory_after_an_error(tmp_path):
    before = os.getcwd()

    with pytest.raises(ValueError):
        with gradient._chdir(str(tmp_path)):
            raise ValueError('boom')

    assert os.getcwd() == before


# ---------------------------------------------------------------------------
# _setup_matplotlib
# ---------------------------------------------------------------------------

def test_setup_matplotlib_forces_agg_when_plots_are_disabled():
    import matplotlib

    gradient._setup_matplotlib(show_plots=False)

    assert matplotlib.get_backend().lower() == 'agg'
    assert matplotlib.rcParams['toolbar'] == 'None'


# ---------------------------------------------------------------------------
# run_gradient
# ---------------------------------------------------------------------------

def test_run_gradient_builds_the_documented_arg_dict(tmp_path, fake_pycoilgen):
    cfg = _cfg('y')
    cfg.cylinder.height = 0.4
    cfg.cylinder.mesh_length_factor = 0.95

    solution, metrics, overlap = gradient.run_gradient(
        cfg, output_dir=str(tmp_path), check_overlap=False)

    args = fake_pycoilgen['arg_dict']
    assert args['field_shape_function'] == 'z'                  # Gy -> internal z
    assert args['target_field_definition_field_name'] == 'lin_3'
    assert args['target_field_definition_file'] == 'OSI2_GradTarget_Lin3.npy'
    assert args['coil_mesh_file'] == 'create cylinder mesh'
    assert args['tikhonov_reg_factor'] == 2500
    assert args['levels'] == 26
    assert args['output_directory'] == str(tmp_path)
    assert args['save_stl_flag'] is True
    assert args['skip_inductance_calculation'] is True          # FastHenry disabled
    assert args['normal_shift_length'] == cfg.normal_shift_length
    assert isinstance(args['cross_sectional_points'], list)

    mesh_params = args['cylinder_mesh_parameter_list']
    assert mesh_params[0] == pytest.approx(0.4 * 0.95)
    assert mesh_params[1] == pytest.approx(cfg.cylinder_design_radius)
    assert mesh_params[2:4] == [cfg.cylinder.n_circ, cfg.cylinder.n_long]
    assert tuple(mesh_params[4:7]) == cfg.cylinder.rot_axis
    assert mesh_params[7] == pytest.approx(cfg.cylinder.rot_angle)

    # 'smooth_flag' does not exist in pyCoilGen; only smooth_factor is passed.
    assert 'smooth_flag' not in args
    assert args['smooth_factor'] == cfg.winding.smooth_factor

    assert solution is not None
    assert metrics['internal_axis'] == 'z'
    assert overlap is None


@pytest.mark.parametrize('axis,internal,fname', [
    ('x', 'y', 'lin_2'), ('y', 'z', 'lin_3'), ('z', 'x', 'lin_1')])
def test_run_gradient_writes_the_target_file_for_each_axis(
        tmp_path, fake_pycoilgen, axis, internal, fname):
    cfg = _cfg(axis)
    cfg.target.resol_radial = 4
    cfg.target.resol_angular = 6

    gradient.run_gradient(cfg, output_dir=str(tmp_path), check_overlap=False)

    args = fake_pycoilgen['arg_dict']
    assert args['field_shape_function'] == internal
    assert args['target_field_definition_field_name'] == fname
    target_file = tmp_path / 'target_fields' / args['target_field_definition_file']
    assert target_file.is_file()
    # pyCoilGen resolves target_fields/ relative to the CWD it is called in.
    assert os.path.realpath(fake_pycoilgen['cwd']) == os.path.realpath(str(tmp_path))


def test_run_gradient_records_the_active_project_stem(tmp_path, fake_pycoilgen):
    cfg = _cfg('y')

    _, metrics, _ = gradient.run_gradient(
        cfg, output_dir=str(tmp_path), check_overlap=False)

    stem = 'Gradient_Gy_tk2500_lvl26'
    assert fake_pycoilgen['arg_dict']['project_name'] == stem
    assert (tmp_path / '.active_project_stem').read_text() == stem
    assert metrics['metrics_path'] == str(tmp_path / f'{stem}_metrics.txt')
    assert os.path.isfile(metrics['metrics_path'])


def test_run_gradient_allocates_a_unique_stem_on_a_second_run(tmp_path, fake_pycoilgen):
    cfg = _cfg('y')

    gradient.run_gradient(cfg, output_dir=str(tmp_path), check_overlap=False)
    gradient.run_gradient(cfg, output_dir=str(tmp_path), check_overlap=False)

    assert fake_pycoilgen['arg_dict']['project_name'] == 'Gradient_Gy_tk2500_lvl26(2)'


def test_run_gradient_honours_an_explicit_project_stem(tmp_path, fake_pycoilgen):
    gradient.run_gradient(_cfg('y'), output_dir=str(tmp_path),
                          project_stem='MyRun', check_overlap=False)

    assert fake_pycoilgen['arg_dict']['project_name'] == 'MyRun'


def test_run_gradient_uses_cfg_output_dir_and_creates_it(tmp_path, fake_pycoilgen):
    cfg = _cfg('y')
    cfg.output_dir = str(tmp_path / 'nested' / 'run')

    gradient.run_gradient(cfg, check_overlap=False)

    assert os.path.isdir(cfg.output_dir)
    assert fake_pycoilgen['arg_dict']['output_directory'] == cfg.output_dir


def test_run_gradient_warns_when_fasthenry_is_missing(tmp_path, fake_pycoilgen,
                                                     monkeypatch, capsys):
    monkeypatch.setattr(gradient, 'resolve_fasthenry_bin', lambda p: '')
    cfg = _cfg('y')
    cfg.fasthenry.enabled = True

    _, metrics, _ = gradient.run_gradient(
        cfg, output_dir=str(tmp_path), check_overlap=False)

    out = capsys.readouterr().out
    assert 'FastHenry2 was not found' in out
    assert fake_pycoilgen['arg_dict']['fasthenry_bin'] == ''
    assert fake_pycoilgen['arg_dict']['skip_inductance_calculation'] is False
    assert np.isnan(metrics['total_fasthenry_resistance'])


def test_run_gradient_passes_a_resolved_fasthenry_binary(tmp_path, fake_pycoilgen,
                                                        monkeypatch):
    exe = tmp_path / 'fasthenry'
    exe.write_text('bin')
    monkeypatch.setattr(gradient, 'resolve_fasthenry_bin', lambda p: str(exe))
    cfg = _cfg('y')
    cfg.fasthenry.enabled = True

    _, metrics, _ = gradient.run_gradient(
        cfg, output_dir=str(tmp_path), check_overlap=False)

    assert fake_pycoilgen['arg_dict']['fasthenry_bin'] == str(exe)
    assert np.isclose(metrics['total_fasthenry_resistance'], 0.5)


def test_run_gradient_runs_the_overlap_check_when_requested(tmp_path, fake_pycoilgen,
                                                           capsys):
    report = gradient.run_gradient(
        _cfg('y'), output_dir=str(tmp_path), check_overlap=True)[2]

    assert report is not None
    assert report.n_collisions == 0
    assert 'Overlap check OK' in capsys.readouterr().out


def test_run_gradient_reports_detected_collisions(tmp_path, fake_pycoilgen,
                                                 monkeypatch, capsys):
    from coilgen.overlap import OverlapReport
    monkeypatch.setattr(gradient, 'detect_collisions',
                        lambda solution, cfg: OverlapReport(
                            2, 0.0005, 0.002, [(0, 5, 0.0005), (1, 9, 0.0009)]))

    report = gradient.run_gradient(
        _cfg('y'), output_dir=str(tmp_path), check_overlap=True)[2]

    assert report.n_collisions == 2
    assert 'OVERLAP WARNING: 2 wire pair(s)' in capsys.readouterr().out


def test_check_overlap_defaults_to_the_config_flag(tmp_path, fake_pycoilgen):
    cfg = _cfg('y')
    cfg.overlap_warn = False

    assert gradient.run_gradient(cfg, output_dir=str(tmp_path))[2] is None


def test_run_gradient_makes_plots_when_enabled(tmp_path, fake_pycoilgen, monkeypatch):
    plotted = []
    monkeypatch.setattr(gradient, '_make_plots',
                        lambda metrics, cfg: plotted.append(metrics['axis_label']))
    cfg = _cfg('y')
    cfg.show_plots = True

    gradient.run_gradient(cfg, output_dir=str(tmp_path), check_overlap=False)

    assert plotted == ['Y']


def test_run_gradient_checks_the_wire_radial_extent(tmp_path, fake_pycoilgen):
    gradient.run_gradient(_cfg('y'), output_dir=str(tmp_path), check_overlap=False)

    assert fake_pycoilgen['warned_path'].endswith('_wire_0_z.stl')


def test_make_plots_is_headless_with_the_agg_backend(tmp_path, fake_pycoilgen):
    """The plotting helper must not block when matplotlib runs headless."""
    import matplotlib
    matplotlib.use('Agg', force=True)
    cfg = _cfg('y')

    _, metrics, _ = gradient.run_gradient(
        cfg, output_dir=str(tmp_path), check_overlap=False)

    gradient._make_plots(metrics, cfg)          # must return without showing

    import matplotlib.pyplot as plt
    assert plt.get_fignums() == []
