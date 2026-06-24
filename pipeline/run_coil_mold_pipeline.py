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


def _ensure_wire_exists() -> None:
    path = cfg.wire_stl_path(with_leads=False)
    if not os.path.isfile(path):
        print(f"Wire STL missing: {path}")
        if not cfg.RUN_GRADIENT:
            print("Set RUN_GRADIENT = True in coil_mold_common.py, or run "
                  "gradiente_belen_santi_main.py first.")
            sys.exit(1)


def _run_gradient_script() -> None:
    """Run gradiente_belen_santi_main.py as a subprocess (script has no main())."""
    import subprocess

    script = os.path.join(cfg.PIPELINE_DIR, 'gradiente_belen_santi_main.py')
    print(f"  Executing: {script}")
    print("  NOTE: edit coil_mold_common.py before relying on this step.")
    result = subprocess.run([sys.executable, script], cwd=cfg.PIPELINE_DIR, check=False)
    if result.returncode != 0:
        sys.exit(result.returncode)


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
    print("=" * 70)
    print("  Coil negative-mold pipeline")
    print("=" * 70)
    print(f"  Gradient axis : G{cfg.GRADIENT_AXIS}")
    print(f"  Layer         : {cfg.GRADIENT_LAYER}")
    print(f"  Results dir   : {cfg.RESULTS_DIR}")
    print(f"  Steps         : gradient={cfg.RUN_GRADIENT}  "
          f"leads={cfg.RUN_LEADS}  shell={cfg.RUN_SHELL}")
    print()

    if cfg.RUN_GRADIENT:
        print("[1/3] Running pyCoilGen (gradiente_belen_santi_main)...")
        _run_gradient_script()
        print()
    else:
        _ensure_wire_exists()

    if cfg.RUN_LEADS:
        print("[2/3] Adding lead wires (add_coil_leads)...")
        _sync_leads_params()
        import add_coil_leads as leads
        leads.main()
        print()

    if cfg.RUN_SHELL:
        print("[3/3] Carving shell halves (generate_coil_shell_split)...")
        import generate_coil_shell_split as shell
        shell.main()
        print()

    print("=" * 70)
    print("  Pipeline complete.")
    print("=" * 70)


if __name__ == '__main__':
    main()
