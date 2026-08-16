"""
Tikhonov regularization sweep.

Runs pyCoilGen headless across a coarse (log-spaced) Tikhonov grid, then an
optional fine pass around the best coarse point. Reports both the realized
slope [mT/(m.A)] and the mean relative layout-vs-target error -- the latter is
the figure of merit the Gz design was failing on, so the sweep surfaces it
explicitly instead of ranking by slope alone.

Refactored from the former ``obsolete/barrido_tikhonov_v2.py`` and
``obsolete/gradiente_belen_santi_func.py`` into a function that reuses
:func:`coilgen.gradient.run_gradient` (no plots, no overlap check) and
:func:`coilgen.metrics.compute_metrics`.

Outputs: per-axis ``Resumen_Completo_Eje_<AXIS>.csv`` and ``.txt`` with the
best-by-slope and best-by-error (lowest mean rel err) points highlighted.
"""

from __future__ import annotations

import copy
import csv
import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .config import Config
from .formatting import fmt_value
from .gradient import run_gradient
from .paths import unique_run_dir


@dataclass
class SweepResult:
    axis: str
    rows: List[dict] = field(default_factory=list)
    csv_path: str = ''
    txt_path: str = ''
    best_slope: Optional[dict] = None
    best_error: Optional[dict] = None


# Per-axis default sweep ranges. Gz needs a much wider high-Tikhonov regime
# (its best point currently has a large target-field error), so the default
# upper bound is extended to 1e6 for Z. The GUI exposes these as editable
# starting points.
DEFAULT_RANGES = {
    'x': (1.0, 100_000.0, 10),
    'y': (1.0, 100_000.0, 10),
    'z': (1.0, 1_000_000.0, 12),
}


def _row_from_metrics(tk: float, phase: str, metrics: dict) -> dict:
    return {
        'Fase': phase,
        'Tikhonov': tk,
        'Pendiente_mT_per_m_per_A': float(metrics.get('slope_mTmA', float('nan'))),
        'Error_Medio_pct': (
            float(metrics['mean_rel_err_layout_pct'])
            if metrics.get('mean_rel_err_layout_pct') is not None
            else float('nan')
        ),
        'RMSE_per_range_mT_per_m_per_A': float(
            metrics.get('rmse_gradient_mTmA', float('nan'))
        ),
        'Wire_length_m': float(
            metrics.get('total_wire_length_computed', float('nan'))
        ),
    }


def _write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _write_txt(path: str, cfg: Config, rows: List[dict],
               best_slope: dict, best_error: dict) -> None:
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(f"Resumen Barrido Tikhonov - EJE {cfg.axis_label}\n")
        fh.write(f"Construccion: radio_externo={cfg.cylinder.radius}m, "
                 f"largo={cfg.cylinder.height}m, niveles={cfg.num_levels}\n")
        fh.write("-" * 70 + "\n")
        fh.write("Fase\tTikhonov\tPendiente[mT/(m.A)]\tErrorMedio[%]\tRMSE/range\n")
        for r in rows:
            fh.write(f"{r['Fase']}\t{r['Tikhonov']}\t"
                     f"{fmt_value(r['Pendiente_mT_per_m_per_A'])}\t"
                     f"{fmt_value(r['Error_Medio_pct'])}\t"
                     f"{fmt_value(r['RMSE_per_range_mT_per_m_per_A'])}\n")
        fh.write("-" * 70 + "\n")
        fh.write(f"[*] MEJOR PENDIENTE ABSOLUTA:\n"
                 f"    Tikhonov = {best_slope['Tikhonov']} "
                 f"(Fase: {best_slope['Fase']})\n"
                 f"    Pendiente = {fmt_value(best_slope['Pendiente_mT_per_m_per_A'])} "
                 f"mT/(m.A)\n\n")
        fh.write(f"[*] MENOR ERROR MEDIO vs CAMPO OBJETIVO:\n"
                 f"    Tikhonov = {best_error['Tikhonov']} "
                 f"(Fase: {best_error['Fase']})\n"
                 f"    Error = {fmt_value(best_error['Error_Medio_pct'])} %\n")


def _run_one(cfg: Config, tk: float, base_dir: str, phase: str,
             on_progress: Optional[Callable] = None) -> dict:
    """Run pyCoilGen for one Tikhonov value; return a sweep row dict."""
    run_cfg = copy.deepcopy(cfg)
    run_cfg.tikhonov_factor = float(tk)
    run_cfg.show_plots = False
    run_cfg.overlap_warn = False
    run_cfg.control.run_gradient = True

    tk_dir = os.path.join(base_dir, f"Tk_{phase}_{tk}")
    os.makedirs(tk_dir, exist_ok=True)
    run_cfg.output_dir = tk_dir

    if on_progress:
        on_progress(phase, tk, 'start')
    try:
        _, metrics, _ = run_gradient(run_cfg, output_dir=tk_dir, check_overlap=False)
        row = _row_from_metrics(tk, phase, metrics)
    except Exception as exc:  # keep sweeping on a single failure
        print(f"  [sweep] FAILED tk={tk}: {exc}")
        row = {
            'Fase': phase, 'Tikhonov': tk,
            'Pendiente_mT_per_m_per_A': float('nan'),
            'Error_Medio_pct': float('nan'),
            'RMSE_per_range_mT_per_m_per_A': float('nan'),
            'Wire_length_m': float('nan'),
        }
    if on_progress:
        on_progress(phase, tk, 'done', row)
    return row


def run_tikhonov_sweep(
    cfg: Config,
    tk_min: Optional[float] = None,
    tk_max: Optional[float] = None,
    n_coarse: Optional[int] = None,
    fine: Optional[bool] = None,
    n_fine: int = 7,
    output_base_dir: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> SweepResult:
    """
    Sweep Tikhonov regularization for ``cfg.gradient_axis``.

    Coarse grid is log-spaced from ``tk_min`` to ``tk_max`` (``n_coarse``
    points). If ``fine`` is True, a second linear pass of ``n_fine`` points
    runs between the neighbours of the best coarse point (by max |slope|).
    Defaults come from :data:`DEFAULT_RANGES` (Gz widened to 1e6).

    ``on_progress(phase, tk, event, row=None)`` is called for GUI updates.
    """
    axis = cfg.gradient_axis.lower()
    d_min, d_max, d_n = DEFAULT_RANGES.get(axis, (1.0, 100_000.0, 10))
    tk_min = d_min if tk_min is None else tk_min
    tk_max = d_max if tk_max is None else tk_max
    n_coarse = d_n if n_coarse is None else n_coarse
    fine = cfg.sweep.fine if fine is None else fine

    if output_base_dir is None:
        output_base_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'resultados', 'barrido_tikhonov',
        )
    axis_dir = os.path.join(output_base_dir, f"Eje_{cfg.axis_label}")
    os.makedirs(axis_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"  BARRIDO TIKHONOV - {cfg.axis_label} "
          f"(radio externo {cfg.cylinder.radius} m)")
    print("=" * 60)

    # ----- Phase 1: coarse (log-spaced) -----------------------------------
    coarse_tks = np.logspace(np.log10(tk_min), np.log10(tk_max), n_coarse)
    coarse_tks = [round(float(tk), 4) for tk in coarse_tks]
    rows: List[dict] = []
    for tk in coarse_tks:
        print(f"\n  -> [Grueso] Tikhonov = {tk}")
        rows.append(_run_one(cfg, tk, axis_dir, 'Grueso', on_progress))

    # ----- Phase 2: fine (linear around best coarse) ----------------------
    finite = [r for r in rows if np.isfinite(r['Pendiente_mT_per_m_per_A'])]
    if not finite:
        print("  [sweep] no successful coarse runs; skipping fine pass.")
        best_slope = rows[0]
    else:
        best_slope = max(finite, key=lambda r: abs(r['Pendiente_mT_per_m_per_A']))
        if fine:
            best_tk = best_slope['Tikhonov']
            # neighbours in the coarse grid
            sorted_coarse = sorted(coarse_tks)
            idx = sorted_coarse.index(best_tk) if best_tk in sorted_coarse else min(
                range(len(sorted_coarse)),
                key=lambda i: abs(sorted_coarse[i] - best_tk),
            )
            tk_prev = sorted_coarse[idx - 1] if idx > 0 else max(tk_min, best_tk / 2)
            tk_next = sorted_coarse[idx + 1] if idx < len(sorted_coarse) - 1 else best_tk * 2

            print(f"\n  [*] Mejor grueso: {best_tk}; barrido fino en "
                  f"[{tk_prev}, {tk_next}]")
            fine_tks = [round(float(v), 4) for v in
                        np.linspace(tk_prev, tk_next, n_fine)]
            fine_tks = [tk for tk in fine_tks if tk not in coarse_tks]
            for tk in fine_tks:
                print(f"\n  -> [Fino] Tikhonov = {tk}")
                rows.append(_run_one(cfg, tk, axis_dir, 'Fino', on_progress))

    # ----- Consolidate ----------------------------------------------------
    rows.sort(key=lambda r: r['Tikhonov'])
    finite_rows = [r for r in rows if np.isfinite(r['Error_Medio_pct'])]
    best_error = (
        min(finite_rows, key=lambda r: r['Error_Medio_pct'])
        if finite_rows else rows[0]
    )
    finite_slope = [r for r in rows if np.isfinite(r['Pendiente_mT_per_m_per_A'])]
    if finite_slope:
        best_slope = max(finite_slope, key=lambda r: abs(r['Pendiente_mT_per_m_per_A']))

    csv_path = os.path.join(axis_dir, f"Resumen_Completo_Eje_{cfg.axis_label}.csv")
    txt_path = os.path.join(axis_dir, f"Resumen_Completo_Eje_{cfg.axis_label}.txt")
    _write_csv(csv_path, rows)
    _write_txt(txt_path, cfg, rows, best_slope, best_error)

    print(f"\n  Resumen guardado en: {axis_dir}")
    print(f"  Mejor pendiente: tk={best_slope['Tikhonov']} "
          f"({fmt_value(best_slope['Pendiente_mT_per_m_per_A'])} mT/(m.A))")
    print(f"  Menor error:     tk={best_error['Tikhonov']} "
          f"({fmt_value(best_error['Error_Medio_pct'])} %)")

    return SweepResult(
        axis=axis, rows=rows, csv_path=csv_path, txt_path=txt_path,
        best_slope=best_slope, best_error=best_error,
    )


def main():
    """CLI entry: run a Tikhonov sweep for one axis."""
    import argparse
    parser = argparse.ArgumentParser(description="Tikhonov regularization sweep.")
    parser.add_argument('--axis', default='z', choices=('x', 'y', 'z'))
    parser.add_argument('--tikhonov-min', type=float, default=None)
    parser.add_argument('--tikhonov-max', type=float, default=None)
    parser.add_argument('--n-coarse', type=int, default=None)
    parser.add_argument('--no-fine', action='store_true')
    parser.add_argument('--levels', type=int, default=26)
    parser.add_argument('--radius', type=float, default=None)
    parser.add_argument('--height', type=float, default=0.430)
    args = parser.parse_args()

    cfg = Config(gradient_axis=args.axis, num_levels=args.levels,
                 show_plots=False, overlap_warn=False)
    cfg.cylinder.height = args.height
    if args.radius is not None:
        cfg.cylinder.radius = args.radius
    cfg.sweep.fine = not args.no_fine
    run_tikhonov_sweep(
        cfg, tk_min=args.tikhonov_min, tk_max=args.tikhonov_max,
        n_coarse=args.n_coarse,
    )


if __name__ == '__main__':
    main()
