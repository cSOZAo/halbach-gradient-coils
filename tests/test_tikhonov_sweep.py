import csv

import pytest

from halbach_coils.coilgen.config import Config
from halbach_coils.coilgen import sweep


@pytest.mark.parametrize(
    ('tk_min', 'tk_max', 'n_coarse', 'message'),
    [
        (0, 100, 3, 'min'),
        (float('nan'), 100, 3, 'min'),
        (100, 10, 3, 'max'),
        (1, float('inf'), 3, 'max'),
        (1, 100, 1, 'Coarse points'),
    ],
)
def test_invalid_sweep_ranges_are_rejected(tk_min, tk_max, n_coarse, message):
    with pytest.raises(ValueError, match=message):
        sweep.validate_sweep_parameters(tk_min, tk_max, n_coarse)


def test_sweep_runs_coarse_and_fine_passes_and_writes_summaries(
        monkeypatch, tmp_path):
    calls = []
    progress = []

    def fake_run_gradient(cfg, output_dir, check_overlap):
        tk = cfg.tikhonov_factor
        calls.append((tk, output_dir, check_overlap))
        # Make tk=10 the best coarse point so a fine pass is generated around it.
        slope = 100.0 - abs(tk - 10.0)
        metrics = {
            'slope_mTmA': slope,
            'mean_rel_err_layout_pct': abs(tk - 20.0),
            'rmse_gradient_mTmA': tk / 100.0,
            'total_wire_length_computed': 1.5,
        }
        return object(), metrics, None

    monkeypatch.setattr(sweep, 'run_gradient', fake_run_gradient)
    cfg = Config(gradient_axis='x', show_plots=False)

    result = sweep.run_tikhonov_sweep(
        cfg,
        tk_min=1,
        tk_max=100,
        n_coarse=3,
        fine=True,
        n_fine=3,
        output_base_dir=str(tmp_path),
        on_progress=lambda *args: progress.append(args),
    )

    assert [row['Tikhonov'] for row in result.rows] == [1.0, 10.0, 50.5, 100.0]
    assert result.best_slope['Tikhonov'] == 10.0
    assert result.best_error['Tikhonov'] == 10.0
    assert len(calls) == 4
    assert all(check_overlap is False for _, _, check_overlap in calls)
    assert len([event for event in progress if event[2] == 'start']) == 4
    assert len([event for event in progress if event[2] == 'done']) == 4

    with open(result.csv_path, newline='', encoding='utf-8') as fh:
        csv_rows = list(csv.DictReader(fh))
    assert len(csv_rows) == 4
    assert 'MEJOR PENDIENTE' in open(
        result.txt_path, encoding='utf-8').read()


def test_sweep_keeps_going_when_one_run_fails(monkeypatch, tmp_path):
    def fake_run_gradient(cfg, output_dir, check_overlap):
        if cfg.tikhonov_factor == 1.0:
            raise RuntimeError('synthetic failure')
        return object(), {
            'slope_mTmA': 2.0,
            'mean_rel_err_layout_pct': 3.0,
            'rmse_gradient_mTmA': 4.0,
            'total_wire_length_computed': 5.0,
        }, None

    monkeypatch.setattr(sweep, 'run_gradient', fake_run_gradient)

    result = sweep.run_tikhonov_sweep(
        Config(show_plots=False), tk_min=1, tk_max=10, n_coarse=2,
        fine=False, output_base_dir=str(tmp_path),
    )

    assert len(result.rows) == 2
    assert result.best_slope['Tikhonov'] == 10.0
