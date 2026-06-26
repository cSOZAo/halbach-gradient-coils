"""
End-to-end negative-mold pipeline for belen_santi gradient coils.

Runs (optionally) pyCoilGen, adds lead wires, and carves the Fusion 360
shell halves — all from the shared settings in ``coil_mold_common.py``.

Usage:
    python run_coil_mold_pipeline.py

Edit ``coil_mold_common.py`` to change geometry, paths, and which steps run.
"""

from __future__ import annotations

import os
import sys

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

import coil_mold_common as cfg


def _seed_wire_from_previous_run(run_dir: str) -> bool:
    """
    Copy coil-only wire STL (+ stem marker) from the newest sibling run folder.

    Used when RUN_GRADIENT=False but a fresh run directory was allocated.
    """
    import glob
    import shutil

    from output_utils import ACTIVE_STEM_FILE, PIPELINE_OUTPUT_BASE

    design = cfg.design_run_stem()
    best_dir = ''
    best_mtime = 0.0
    for folder in glob.glob(os.path.join(PIPELINE_OUTPUT_BASE, f'{design}*')):
        if not os.path.isdir(folder):
            continue
        if os.path.normpath(folder) == os.path.normpath(run_dir):
            continue
        for wire in glob.glob(os.path.join(folder, '*_wire_0_z.stl')):
            base = os.path.basename(wire)
            if any(tag in base for tag in ('_with_leads', '_coil_open', '_leads_only')):
                continue
            mtime = os.path.getmtime(wire)
            if mtime > best_mtime:
                best_mtime = mtime
                best_dir = folder

    if not best_dir:
        return False

    os.makedirs(run_dir, exist_ok=True)
    for wire in glob.glob(os.path.join(best_dir, '*_wire_0_z.stl')):
        base = os.path.basename(wire)
        if any(tag in base for tag in ('_with_leads', '_coil_open', '_leads_only')):
            continue
        shutil.copy2(wire, os.path.join(run_dir, base))
        stem_path = os.path.join(best_dir, ACTIVE_STEM_FILE)
        if os.path.isfile(stem_path):
            shutil.copy2(stem_path, os.path.join(run_dir, ACTIVE_STEM_FILE))
        metrics = glob.glob(os.path.join(best_dir, '*_metrics.txt'))
        for m in metrics:
            shutil.copy2(m, os.path.join(run_dir, os.path.basename(m)))
        print(f"  Seeded wire from: {best_dir}")
        cfg.sync_project_stem_from_disk()
        return True
    return False


def _ensure_wire_exists(run_dir: str) -> None:
    path = cfg.wire_stl_path(with_leads=False)
    if not os.path.isfile(path):
        if not cfg.RUN_GRADIENT and _seed_wire_from_previous_run(run_dir):
            path = cfg.wire_stl_path(with_leads=False)
        if not os.path.isfile(path):
            print(f"Wire STL missing: {path}")
            if not cfg.RUN_GRADIENT:
                print("Set RUN_GRADIENT = True in coil_mold_common.py, or run "
                      "gradiente_belen_santi_main.py first.")
                sys.exit(1)


def _run_gradient_script(run_dir: str) -> None:
    """Run gradiente_belen_santi_main.py as a subprocess (script has no main())."""
    import subprocess

    cfg.set_results_dir(run_dir)
    script = os.path.join(cfg.PIPELINE_DIR, 'gradiente_belen_santi_main.py')
    print(f"  Executing: {script}")
    print("  NOTE: edit coil_mold_common.py before relying on this step.")
    env = {**os.environ, cfg.RESULTS_DIR_ENV: run_dir}
    result = subprocess.run(
        [sys.executable, script],
        cwd=cfg.PIPELINE_DIR,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        sys.exit(result.returncode)
    cfg.set_results_dir(run_dir)
    cfg.sync_project_stem_from_disk()


def _sync_leads_params() -> None:
    import add_coil_leads as leads

    leads.INPUT_STL = cfg.wire_stl_path(with_leads=False)
    leads.LEAD_DIRECTION = cfg.LEAD_DIRECTION
    leads.SECTOR_MIN_Z = cfg.SECTOR_MIN_Z
    leads.SECTOR_MAX_ABS_Y = cfg.SECTOR_MAX_ABS_Y
    leads.CYL_AXIS = cfg.CYL_AXIS
    leads.SHELL_RADIUS = cfg.SHELL_RADIUS
    leads.CONDUCTOR_WIDTH = cfg.CONDUCTOR_WIDTH
    leads.CROSS_SECTION_A_FRAC = cfg.CROSS_SECTION_A_FRAC
    leads.CROSS_SECTION_B_FRAC = cfg.CROSS_SECTION_B_FRAC
    leads.CROSS_SECTION_N = cfg.CROSS_SECTION_N
    leads.CUT_LOOP_LENGTH = cfg.CUT_LOOP_LENGTH
    leads.GAP_AXIAL_LENGTH = cfg.GAP_AXIAL_LENGTH
    leads.WIRE_ISOLATE_HALF = cfg.WIRE_ISOLATE_HALF
    leads.TANGENT_RADIUS = cfg.TANGENT_RADIUS
    leads.WIRE_TANGENT_RUN = cfg.WIRE_TANGENT_RUN
    leads.FACE_TOWARD_GAP = cfg.FACE_TOWARD_GAP
    leads.PEEL_OUT = cfg.PEEL_OUT
    leads.LEAD_JUNCTION_COIL_BACKSET = cfg.LEAD_JUNCTION_COIL_BACKSET
    leads.LEAD_JUNCTION_GAP_BACKSET = cfg.LEAD_JUNCTION_GAP_BACKSET
    leads.LEAD_LENGTH = cfg.LEAD_LENGTH
    leads.LEAD_BLEND = cfg.LEAD_BLEND
    leads.TIP_FAN = cfg.TIP_FAN
    leads.LEAD_STEPS = cfg.LEAD_STEPS
    leads.LEAD_0_SPREAD_SIGN = cfg.LEAD_0_SPREAD_SIGN
    leads.LEAD_1_SPREAD_SIGN = cfg.LEAD_1_SPREAD_SIGN
    leads.EXIT_DIRECTION = cfg.EXIT_DIRECTION
    leads.CS_BLEND_RINGS = cfg.CS_BLEND_RINGS
    leads.JUNCTION_RIGID_STEPS = cfg.JUNCTION_RIGID_STEPS
    leads.JUNCTION_PLANE_RINGS = cfg.JUNCTION_PLANE_RINGS


def main() -> None:
    run_dir = cfg.init_pipeline_run()
    cfg.set_results_dir(run_dir)
    print("=" * 70)
    print("  Coil negative-mold pipeline")
    print("=" * 70)
    print(f"  Gradient axis : G{cfg.GRADIENT_AXIS}")
    print(f"  Layer         : {cfg.GRADIENT_LAYER}")
    print(f"  Results dir   : {run_dir}")
    print(f"  Steps         : gradient={cfg.RUN_GRADIENT}  "
          f"leads={cfg.RUN_LEADS}  shell={cfg.RUN_SHELL}")
    print()

    if cfg.RUN_GRADIENT:
        print("[1/3] Running pyCoilGen (gradiente_belen_santi_main)...")
        _run_gradient_script(run_dir)
        print()
    else:
        _ensure_wire_exists(run_dir)

    if cfg.RUN_LEADS:
        print("[2/3] Adding lead wires (add_coil_leads)...")
        _sync_leads_params()
        import add_coil_leads as leads
        leads.main()
        print()

    if cfg.RUN_SHELL:
        print("[3/3] Carving shell halves (generate_coil_shell_split)...")
        cfg.refresh_stl_paths()
        import generate_coil_shell_split as shell
        shell.main()
        print()

    print("=" * 70)
    print("  Pipeline complete.")
    print("=" * 70)


if __name__ == '__main__':
    main()
