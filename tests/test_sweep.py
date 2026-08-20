"""
Unit tests for :mod:`coilgen.sweep`.

pyCoilGen is never invoked: ``run_gradient`` is replaced by a stub so the
sweep's grid construction, best-point selection and report writing can be
tested deterministically.
"""

import csv
import os

import numpy as np
import pytest

from coilgen import sweep
from coilgen.config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _metrics(slope=1.0, err=2.0, rmse=0.1, length=12.0):
    return {
        'slope_mTmA': slope,
        'mean_rel_err_layout_pct': err,
        'rmse_gradient_mTmA': rmse,
        'total_wire_length_computed': length,
    }


def _stub_run_gradient(monkeypatch, slope_of_tk, err_of_tk=None, fail_on=()):
    """Patch ``sweep.run_gradient`` with a deterministic fake. Records calls."""
    calls = []

    def fake(cfg, output_dir=None, check_overlap=True):
        calls.append((cfg.tikhonov_factor, output_dir, check_overlap))
        if cfg.tikhonov_factor in fail_on:
            raise RuntimeError('pyCoilGen exploded')
        tk = cfg.tikhonov_factor
        err = err_of_tk(tk) if err_of_tk else 1.0
        return object(), _metrics(slope=slope_of_tk(tk), err=err), None

    monkeypatch.setattr(sweep, 'run_gradient', fake)
    return calls


def _cfg(axis='y'):
    cfg = Config(gradient_axis=axis, num_levels=26)
    cfg.show_plots = True
    cfg.overlap_warn = True
    return cfg


# ---------------------------------------------------------------------------
# Row / formatting helpers
# ---------------------------------------------------------------------------

def test_row_from_metrics_maps_the_spanish_column_names():
    row = sweep._row_from_metrics(2500.0, 'Grueso', _metrics(slope=3.5, err=1.25))

    assert row == {
        'Fase': 'Grueso',
        'Tikhonov': 2500.0,
        'Pendiente_mT_per_m_per_A': 3.5,
        'Error_Medio_pct': 1.25,
        'RMSE_per_range_mT_per_m_per_A': 0.1,
        'Wire_length_m': 12.0,
    }


def test_row_from_metrics_uses_nan_for_missing_or_none_values():
    row = sweep._row_from_metrics(1.0, 'Fino', {'mean_rel_err_layout_pct': None})

    assert np.isnan(row['Pendiente_mT_per_m_per_A'])
    assert np.isnan(row['Error_Medio_pct'])
    assert np.isnan(row['RMSE_per_range_mT_per_m_per_A'])
    assert np.isnan(row['Wire_length_m'])


@pytest.mark.parametrize('value,expected', [
    (None, 'n/a'),
    (float('nan'), 'n/a'),
    (float('inf'), 'n/a'),
    (2.5, '2.5'),
    ('x', 'x'),
])
def test_fmt_val(value, expected):
    assert sweep._fmt_val(value) == expected


def test_fmt_val_honours_the_format_spec():
    assert sweep._fmt_val(2.34567, '.2f') == '2.35'


def test_default_ranges_widen_the_gz_upper_bound():
    assert sweep.DEFAULT_RANGES['z'][1] > sweep.DEFAULT_RANGES['y'][1]
    assert sweep.DEFAULT_RANGES['z'] == (1.0, 1_000_000.0, 12)


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def test_write_csv_writes_a_header_and_one_row_per_entry(tmp_path):
    rows = [sweep._row_from_metrics(1.0, 'Grueso', _metrics(slope=1.0)),
            sweep._row_from_metrics(10.0, 'Fino', _metrics(slope=2.0))]
    path = str(tmp_path / 'out.csv')

    sweep._write_csv(path, rows)

    with open(path, newline='', encoding='utf-8') as fh:
        parsed = list(csv.DictReader(fh))
    assert [r['Tikhonov'] for r in parsed] == ['1.0', '10.0']
    assert parsed[1]['Fase'] == 'Fino'


def test_write_csv_skips_empty_rows(tmp_path):
    path = str(tmp_path / 'out.csv')

    sweep._write_csv(path, [])

    assert not os.path.exists(path)


def test_write_txt_highlights_both_best_points(tmp_path):
    rows = [sweep._row_from_metrics(1.0, 'Grueso', _metrics(slope=1.0, err=9.0)),
            sweep._row_from_metrics(10.0, 'Fino', _metrics(slope=5.0, err=0.5))]
    path = str(tmp_path / 'out.txt')

    sweep._write_txt(path, _cfg('y'), rows, rows[1], rows[1])
    text = open(path, encoding='utf-8').read()

    assert 'EJE GY' in text
    assert 'MEJOR PENDIENTE ABSOLUTA' in text
    assert 'MENOR ERROR MEDIO' in text
    assert 'Tikhonov = 10.0' in text
    assert text.count('Grueso') >= 1 and text.count('Fino') >= 2


def test_write_txt_renders_failed_rows_as_na(tmp_path):
    failed = sweep._row_from_metrics(1.0, 'Grueso', {})
    good = sweep._row_from_metrics(10.0, 'Fino', _metrics())
    path = str(tmp_path / 'out.txt')

    sweep._write_txt(path, _cfg(), [failed, good], good, good)

    assert 'n/a' in open(path, encoding='utf-8').read()


# ---------------------------------------------------------------------------
# _run_one
# ---------------------------------------------------------------------------

def test_run_one_isolates_the_config_and_disables_plots(tmp_path, monkeypatch):
    calls = _stub_run_gradient(monkeypatch, slope_of_tk=lambda tk: 1.0)
    cfg = _cfg()

    row = sweep._run_one(cfg, 2500.0, str(tmp_path), 'Grueso')

    assert row['Tikhonov'] == 2500.0
    assert row['Fase'] == 'Grueso'
    tk_used, out_dir, check_overlap = calls[0]
    assert tk_used == 2500.0
    assert out_dir == str(tmp_path / 'Tk_Grueso_2500.0')
    assert os.path.isdir(out_dir)
    assert check_overlap is False
    # The caller's config is left untouched.
    assert cfg.tikhonov_factor == 2500
    assert cfg.show_plots is True
    assert cfg.overlap_warn is True


def test_run_one_returns_nan_row_when_the_run_fails(tmp_path, monkeypatch, capsys):
    _stub_run_gradient(monkeypatch, slope_of_tk=lambda tk: 1.0, fail_on=(7.0,))

    row = sweep._run_one(_cfg(), 7.0, str(tmp_path), 'Grueso')

    assert row['Fase'] == 'Grueso' and row['Tikhonov'] == 7.0
    assert np.isnan(row['Pendiente_mT_per_m_per_A'])
    assert np.isnan(row['Error_Medio_pct'])
    assert 'FAILED tk=7.0' in capsys.readouterr().out


def test_run_one_reports_progress_events(tmp_path, monkeypatch):
    _stub_run_gradient(monkeypatch, slope_of_tk=lambda tk: 4.0)
    events = []

    sweep._run_one(_cfg(), 5.0, str(tmp_path), 'Fino',
                   on_progress=lambda *a: events.append(a))

    assert events[0] == ('Fino', 5.0, 'start')
    assert events[1][:3] == ('Fino', 5.0, 'done')
    assert events[1][3]['Pendiente_mT_per_m_per_A'] == 4.0


# ---------------------------------------------------------------------------
# run_tikhonov_sweep
# ---------------------------------------------------------------------------

def test_sweep_coarse_grid_is_log_spaced_and_writes_reports(tmp_path, monkeypatch):
    calls = _stub_run_gradient(monkeypatch, slope_of_tk=lambda tk: tk)

    result = sweep.run_tikhonov_sweep(
        _cfg('y'), tk_min=1.0, tk_max=100.0, n_coarse=3, fine=False,
        output_base_dir=str(tmp_path))

    assert [tk for tk, _, _ in calls] == [1.0, 10.0, 100.0]
    assert result.axis == 'y'
    assert [r['Tikhonov'] for r in result.rows] == [1.0, 10.0, 100.0]
    assert all(r['Fase'] == 'Grueso' for r in result.rows)
    assert result.csv_path == str(tmp_path / 'Eje_GY' / 'Resumen_Completo_Eje_GY.csv')
    assert result.txt_path == str(tmp_path / 'Eje_GY' / 'Resumen_Completo_Eje_GY.txt')
    assert os.path.isfile(result.csv_path) and os.path.isfile(result.txt_path)


def test_sweep_picks_the_largest_absolute_slope(tmp_path, monkeypatch):
    # Slope is most negative at the middle point -> best by |slope|.
    slopes = {1.0: 1.0, 10.0: -9.0, 100.0: 2.0}
    _stub_run_gradient(monkeypatch, slope_of_tk=lambda tk: slopes[tk])

    result = sweep.run_tikhonov_sweep(
        _cfg(), tk_min=1.0, tk_max=100.0, n_coarse=3, fine=False,
        output_base_dir=str(tmp_path))

    assert result.best_slope['Tikhonov'] == 10.0


def test_sweep_picks_the_lowest_mean_error(tmp_path, monkeypatch):
    errors = {1.0: 5.0, 10.0: 0.25, 100.0: 3.0}
    _stub_run_gradient(monkeypatch, slope_of_tk=lambda tk: 1.0,
                       err_of_tk=lambda tk: errors[tk])

    result = sweep.run_tikhonov_sweep(
        _cfg(), tk_min=1.0, tk_max=100.0, n_coarse=3, fine=False,
        output_base_dir=str(tmp_path))

    assert result.best_error['Tikhonov'] == 10.0
    assert result.best_error['Error_Medio_pct'] == 0.25


def test_fine_pass_samples_between_the_neighbours_of_the_best_point(tmp_path, monkeypatch):
    slopes = {1.0: 1.0, 10.0: 9.0, 100.0: 2.0}
    _stub_run_gradient(monkeypatch, slope_of_tk=lambda tk: slopes.get(tk, 1.0))

    result = sweep.run_tikhonov_sweep(
        _cfg(), tk_min=1.0, tk_max=100.0, n_coarse=3, fine=True, n_fine=3,
        output_base_dir=str(tmp_path))

    fine_tks = [r['Tikhonov'] for r in result.rows if r['Fase'] == 'Fino']
    # Linear pass across [1, 100] with 3 points; grid duplicates are dropped.
    assert fine_tks == [50.5]
    assert [r['Tikhonov'] for r in result.rows] == sorted(
        r['Tikhonov'] for r in result.rows)


def test_fine_pass_extrapolates_when_the_best_point_is_the_last_grid_value(tmp_path, monkeypatch):
    _stub_run_gradient(monkeypatch, slope_of_tk=lambda tk: tk)      # best = highest tk

    result = sweep.run_tikhonov_sweep(
        _cfg(), tk_min=1.0, tk_max=100.0, n_coarse=3, fine=True, n_fine=3,
        output_base_dir=str(tmp_path))

    fine_tks = [r['Tikhonov'] for r in result.rows if r['Fase'] == 'Fino']
    # Neighbours of 100 are 10 (below) and 2*100 (extrapolated above).
    assert fine_tks == [105.0, 200.0]


def test_sweep_skips_the_fine_pass_when_every_coarse_run_fails(tmp_path, monkeypatch, capsys):
    _stub_run_gradient(monkeypatch, slope_of_tk=lambda tk: 1.0,
                       fail_on=(1.0, 10.0, 100.0))

    result = sweep.run_tikhonov_sweep(
        _cfg(), tk_min=1.0, tk_max=100.0, n_coarse=3, fine=True,
        output_base_dir=str(tmp_path))

    assert all(r['Fase'] == 'Grueso' for r in result.rows)
    assert np.isnan(result.best_slope['Pendiente_mT_per_m_per_A'])
    assert result.best_error is result.rows[0]
    assert 'skipping fine pass' in capsys.readouterr().out


def test_sweep_uses_the_per_axis_defaults(tmp_path, monkeypatch):
    calls = _stub_run_gradient(monkeypatch, slope_of_tk=lambda tk: 1.0)
    cfg = _cfg('z')
    cfg.sweep.fine = False

    sweep.run_tikhonov_sweep(cfg, output_base_dir=str(tmp_path))

    tk_min, tk_max, n_coarse = sweep.DEFAULT_RANGES['z']
    tks = [tk for tk, _, _ in calls]
    assert len(tks) == n_coarse
    assert tks[0] == pytest.approx(tk_min)
    assert tks[-1] == pytest.approx(tk_max)
    assert os.path.isdir(tmp_path / 'Eje_GZ')


def test_sweep_forwards_progress_events_for_the_gui(tmp_path, monkeypatch):
    _stub_run_gradient(monkeypatch, slope_of_tk=lambda tk: 1.0)
    events = []

    sweep.run_tikhonov_sweep(
        _cfg(), tk_min=1.0, tk_max=10.0, n_coarse=2, fine=False,
        output_base_dir=str(tmp_path),
        on_progress=lambda *a: events.append(a))

    assert [e[2] for e in events] == ['start', 'done', 'start', 'done']
