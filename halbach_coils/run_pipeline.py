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
from dataclasses import dataclass

# Make the coilgen package importable when run from the workspace root.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from coilgen import paths as _paths
from coilgen.config import Config
from coilgen.gradient import run_gradient
from coilgen.leads import run_leads
from coilgen.shell import run_shell
from coilgen.design_state import validate_design_state, copy_design_state


@dataclass(frozen=True)
class PipelineResult:
    """Outcome returned to the GUI after the configured pipeline steps."""

    output_dir: str
    discarded: bool = False
    field_plots: dict | None = None


def _coil_only_wire_paths(folder: str) -> list[str]:
    """Canonical and legacy coil-only wire STLs in *folder*."""
    paths: set[str] = set()
    patterns = ('Gradient_*_wire_0.stl', 'Gradient_*_wire_0_?.stl')
    for pattern in patterns:
        for wire in glob.glob(os.path.join(folder, pattern)):
            if not _paths._is_derived_wire_stl(wire):
                paths.add(wire)
    return sorted(paths)


def _seed_wire_from_previous_run(cfg: Config, run_dir: str) -> bool:
    """
    Copy coil-only wire STL (+ stem marker + metrics) from the newest sibling
    run folder. Used when --skip-gradient but a fresh run directory was
    allocated. Stem/metrics are copied once, outside the wire loop.
    """
    design = cfg.design_folder
    best_wire = ''
    best_mtime = 0.0
    for folder in glob.glob(os.path.join(_paths.PIPELINE_OUTPUT_BASE, f'{design}*')):
        if not os.path.isdir(folder):
            continue
        if os.path.normpath(folder) == os.path.normpath(run_dir):
            continue
        for wire in _coil_only_wire_paths(folder):
            try:
                validate_design_state(wire, cfg)
            except (ValueError, OSError) as exc:
                print(f"  Skipping incompatible previous wire: {wire}: {exc}")
                continue
            mtime = os.path.getmtime(wire)
            if mtime > best_mtime:
                best_mtime = mtime
                best_wire = wire

    if not best_wire:
        return False

    os.makedirs(run_dir, exist_ok=True)
    destination = os.path.join(run_dir, os.path.basename(best_wire))
    shutil.copy2(best_wire, destination)
    copy_design_state(best_wire, destination)
    stem = os.path.basename(best_wire).rsplit('_wire_0', 1)[0]
    _paths.write_active_stem(run_dir, stem)
    metrics_path = os.path.join(os.path.dirname(best_wire), f'{stem}_metrics.txt')
    if os.path.isfile(metrics_path):
        shutil.copy2(metrics_path, os.path.join(run_dir, os.path.basename(metrics_path)))
    print(f"  Seeded verified wire from: {os.path.dirname(best_wire)}")
    return True


def _ensure_wire_exists(cfg: Config, run_dir: str, run_gradient: bool) -> str:
    """Return the coil-only wire STL path, seeding from a previous run if needed."""
    path = _paths.resolve_wire_stl_path(
        run_dir, cfg.gradient_axis, cfg.tikhonov_factor, cfg.num_levels,
    )
    if os.path.isfile(path):
        if not run_gradient:
            validate_design_state(path, cfg)
        return path
    if not run_gradient and _seed_wire_from_previous_run(cfg, run_dir):
        path = _paths.resolve_wire_stl_path(
            run_dir, cfg.gradient_axis, cfg.tikhonov_factor, cfg.num_levels,
        )
        if os.path.isfile(path):
            validate_design_state(path, cfg)
            return path
    raise FileNotFoundError(
        f"Wire STL missing: {path}\n"
        f"Expected Gradient_G{cfg.gradient_axis}_tk{int(cfg.tikhonov_factor)}"
        f"_lvl{cfg.num_levels}_wire_0.stl in {run_dir}\n"
        f"Run with the gradient step enabled, or --axis/--tikhonov/--levels "
        f"matching an existing run.")


def run_pipeline(cfg: Config, should_stop=None, ask_user=None,
                 question_language: str = "es") -> PipelineResult:
    """Run the configured pipeline steps and return its outcome.

    ``question_language`` controls the interactive congestion warning emitted
    for a GUI callback; Spanish is retained as the default for legacy callers.

    Respects ``cfg.output_dir`` when set (GUI / ``--output-dir``). Only
    allocates a fresh ``unique_run_dir`` under ``resultados/pipeline/`` when
    no output directory was supplied.
    """
    field_plots = None
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
            return PipelineResult(run_dir)
        print("[1/3] Running pyCoilGen (gradient)...")
        _solution, _metrics, collision_report = run_gradient(
            cfg, output_dir=run_dir, check_overlap=cfg.overlap_warn)
        field_plots = _metrics.get('field_plots')
        del _solution
        print()
        if collision_report is not None and collision_report.n_collisions > 0:
            question = collision_report.user_question(question_language)
            print(question)
            if ask_user is not None and not ask_user(question):
                if question_language == "es":
                    print("\n  Resultado descartado por el usuario.")
                    print("  No se ejecutarán las etapas de leads ni shell.")
                else:
                    print("\n  The user discarded the result.")
                    print("  The leads and shell stages will not run.")
                return PipelineResult(run_dir, discarded=True, field_plots=field_plots)
            if ask_user is not None:
                if question_language == "es":
                    print("\n  El usuario aceptó continuar con el resultado.")
                else:
                    print("\n  The user accepted the result and continued.")
    else:
        _ensure_wire_exists(cfg, run_dir, cfg.control.run_gradient)

    if cfg.control.run_leads:
        if _stop_requested():
            print("  Stop requested before leads step.")
            return PipelineResult(run_dir, field_plots=field_plots)
        wire_stl = _ensure_wire_exists(cfg, run_dir, cfg.control.run_gradient)
        print("[2/3] Adding lead wires...")
        lead_paths = run_leads(cfg, input_stl=wire_stl)
        if lead_paths:
            from coilgen.final_field import evaluate_final_conductor
            final = evaluate_final_conductor(cfg, wire_stl, lead_paths[0], run_dir)
            field_plots = final['field_plots']
        print()

    if cfg.control.run_shell:
        if _stop_requested():
            print("  Stop requested before shell step.")
            return PipelineResult(run_dir, field_plots=field_plots)
        print("[3/3] Carving shell halves...")
        run_shell(cfg, output_dir=run_dir)
        print()

    print("=" * 70)
    print("  Pipeline complete.")
    print("=" * 70)
    return PipelineResult(run_dir, field_plots=field_plots)


def main():
    parser = argparse.ArgumentParser(description="Run the full coil-mold pipeline.")
    parser.add_argument('--axis', default='y', choices=('x', 'y', 'z'))
    parser.add_argument('--b0', default='+y', choices=('+x', '-x', '+y', '-y', '+z', '-z'),
                        help='Physical B0 direction; use --b0=-y for a negative direction')
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
        b0_direction=args.b0,
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
