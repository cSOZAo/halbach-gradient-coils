r"""
End-to-end negative-mold pipeline for belen_santi gradient coils.

Runs (optionally) pyCoilGen, adds lead wires, and carves the Fusion 360 shell
halves -- all driven by a single :class:`coilgen.config.Config`.

Usage:
    .\.venv\Scripts\python.exe halbach_coils\run_pipeline.py                       # Gy defaults
    .\.venv\Scripts\python.exe halbach_coils\run_pipeline.py --axis z --tikhonov 10000 --layer 3
    .\.venv\Scripts\python.exe halbach_coils\run_pipeline.py --skip-gradient       # re-use a previous wire STL
    .\.venv\Scripts\python.exe halbach_coils\run_pipeline.py --skip-leads --skip-shell

Outputs go to ``resultados/pipeline/G{axis}_tk{N}_lvl{M}/`` (or ``...(2)`` on
re-run). Each full run gets its own subfolder.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

# Make the coilgen package importable when run from the workspace root.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from coilgen import paths as _paths
from coilgen.config import Config
from coilgen.gradient import run_gradient
from coilgen.leads import run_leads
from coilgen.shell import run_shell


def _coil_only_wire_paths(folder: str) -> list[str]:
    """Coil-only wire STLs in *folder* (any internal field suffix x/y/z)."""
    paths: list[str] = []
    for wire in glob.glob(os.path.join(folder, 'Gradient_*_wire_0_?.stl')):
        if not _paths._is_derived_wire_stl(wire):
            paths.append(wire)
    return paths


def _seed_wire_from_previous_run(cfg: Config, run_dir: str) -> bool:
    """
    Copy coil-only wire STL (+ stem marker + metrics) from the newest sibling
    run folder. Used when --skip-gradient but a fresh run directory was
    allocated. Stem/metrics are copied once, outside the wire loop.
    """
    design = cfg.design_folder
    best_dir = ''
    best_mtime = 0.0
    for folder in glob.glob(os.path.join(_paths.PIPELINE_OUTPUT_BASE, f'{design}*')):
        if not os.path.isdir(folder):
            continue
        if os.path.normpath(folder) == os.path.normpath(run_dir):
            continue
        for wire in _coil_only_wire_paths(folder):
            mtime = os.path.getmtime(wire)
            if mtime > best_mtime:
                best_mtime = mtime
                best_dir = folder

    if not best_dir:
        return False

    os.makedirs(run_dir, exist_ok=True)
    wires = _coil_only_wire_paths(best_dir)
    for wire in wires:
        shutil.copy2(wire, os.path.join(run_dir, os.path.basename(wire)))

    # Seed stem + metrics once (fix: previously copied per-wire inside the loop).
    stem_path = os.path.join(best_dir, _paths.ACTIVE_STEM_FILE)
    if os.path.isfile(stem_path):
        shutil.copy2(stem_path, os.path.join(run_dir, _paths.ACTIVE_STEM_FILE))
    for m in glob.glob(os.path.join(best_dir, '*_metrics.txt')):
        shutil.copy2(m, os.path.join(run_dir, os.path.basename(m)))

    print(f"  Seeded wire from: {best_dir}")
    return True


def _ensure_wire_exists(cfg: Config, run_dir: str, run_gradient: bool) -> str:
    """Return the coil-only wire STL path, seeding from a previous run if needed."""
    path = _paths.resolve_wire_stl_path(
        run_dir, cfg.gradient_axis, cfg.tikhonov_factor, cfg.num_levels,
    )
    if os.path.isfile(path):
        return path
    if not run_gradient and _seed_wire_from_previous_run(cfg, run_dir):
        path = _paths.resolve_wire_stl_path(
            run_dir, cfg.gradient_axis, cfg.tikhonov_factor, cfg.num_levels,
        )
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"Wire STL missing: {path}\n"
        f"Expected Gradient_G{cfg.gradient_axis}_tk{int(cfg.tikhonov_factor)}"
        f"_lvl{cfg.num_levels}_wire_0_<internal_axis>.stl in {run_dir}\n"
        f"Run with the gradient step enabled, or --axis/--tikhonov/--levels "
        f"matching an existing run.")


def run_pipeline(cfg: Config, should_stop=None) -> str:
    """Run the configured pipeline steps; returns the run directory.

    Respects ``cfg.output_dir`` when set (GUI / ``--output-dir``). Only
    allocates a fresh ``unique_run_dir`` under ``resultados/pipeline/`` when
    no output directory was supplied.
    """
    if cfg.output_dir:
        run_dir = os.path.abspath(cfg.output_dir)
        os.makedirs(run_dir, exist_ok=True)
    else:
        run_dir = _paths.unique_run_dir(_paths.PIPELINE_OUTPUT_BASE, cfg.design_folder)
        cfg.output_dir = run_dir

    print("=" * 70)
    print("  Coil negative-mold pipeline")
    print("=" * 70)
    print(f"  Gradient axis : {cfg.axis_label}")
    if cfg.shell.use_custom_stl:
        a, b = cfg.shell_half_paths()
        print(f"  Shell         : custom halves")
        print(f"  Half A/B      : {os.path.basename(a)} / {os.path.basename(b)}")
    else:
        print(f"  Shell         : auto hollow cylinder")
    print(f"  Radius        : outer={cfg.cylinder.radius*1000:.2f} mm  "
          f"inner={cfg.shell_inner_radius*1000:.2f} mm  "
          f"design_r={cfg.cylinder_design_radius*1000:.2f} mm")
    print(f"  Results dir   : {run_dir}")
    print(f"  Steps         : gradient={cfg.control.run_gradient}  "
          f"leads={cfg.control.run_leads}  shell={cfg.control.run_shell}")
    print()

    def _stop_requested() -> bool:
        return bool(should_stop and should_stop())

    if cfg.control.run_gradient:
        if _stop_requested():
            print("  Stop requested before gradient step.")
            return run_dir
        print("[1/3] Running pyCoilGen (gradient)...")
        run_gradient(cfg, output_dir=run_dir, check_overlap=cfg.overlap_warn)
        print()
    else:
        _ensure_wire_exists(cfg, run_dir, cfg.control.run_gradient)

    if cfg.control.run_leads:
        if _stop_requested():
            print("  Stop requested before leads step.")
            return run_dir
        wire_stl = _ensure_wire_exists(cfg, run_dir, cfg.control.run_gradient)
        print("[2/3] Adding lead wires...")
        run_leads(cfg, input_stl=wire_stl)
        print()

    if cfg.control.run_shell:
        if _stop_requested():
            print("  Stop requested before shell step.")
            return run_dir
        print("[3/3] Carving shell halves...")
        run_shell(cfg, output_dir=run_dir)
        print()

    print("=" * 70)
    print("  Pipeline complete.")
    print("=" * 70)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Run the full coil-mold pipeline.")
    parser.add_argument('--axis', default='y', choices=('x', 'y', 'z'))
    parser.add_argument('--tikhonov', type=float, default=2500)
    parser.add_argument('--levels', type=int, default=26)
    parser.add_argument('--radius', type=float, default=0.150)
    parser.add_argument('--height', type=float, default=0.430)
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--output-dir', default='')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--skip-gradient', action='store_true')
    parser.add_argument('--skip-leads', action='store_true')
    parser.add_argument('--skip-shell', action='store_true')
    parser.add_argument('--no-overlap-warn', action='store_true')
    args = parser.parse_args()

    cfg = Config(
        gradient_axis=args.axis,
        tikhonov_factor=args.tikhonov,
        num_levels=args.levels,
        show_plots=not args.no_plots,
        overlap_warn=not args.no_overlap_warn,
    )
    cfg.cylinder.radius = args.radius
    cfg.cylinder.height = args.height
    cfg.shell.layer = args.layer
    cfg.control.run_gradient = not args.skip_gradient
    cfg.control.run_leads = not args.skip_leads
    cfg.control.run_shell = not args.skip_shell
    if args.output_dir:
        cfg.output_dir = args.output_dir

    run_pipeline(cfg)


if __name__ == '__main__':
    main()
