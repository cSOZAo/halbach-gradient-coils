"""
Shell split from a wire STL with leads.

Input:  path to the cable mesh with leads.
Output: ``<stem>_shell_g<layer>a.stl`` and ``<stem>_shell_g<layer>b.stl`` in the
        same folder as the input.

Sibling STLs in that folder are picked up automatically when present:
  - ``<stem>.stl``            coil-only (alignment)
  - ``<stem>_coil_open.stl``  open loop (component mode)
  - ``<stem>_leads_only.stl`` lead tubes (component / legacy mode)

Usage:
    1. Set WIRE_WITH_LEADS_STL below.
    2. Run:  python generate_shell_split_from_stl.py

    Or pass the path on the command line:
    python generate_shell_split_from_stl.py path/to/wire_with_leads.stl

Fusion shell halves and groove settings come from ``coil_mold_common.py``.
"""

from __future__ import annotations

import argparse
import os
import sys

import coil_mold_common as cfg
from generate_coil_shell_split import run_shell_split

# =============================================================================
# USER PARAMETER — set your wire STL path here
# =============================================================================

WIRE_WITH_LEADS_STL = ''  # set path, or pass on command line

# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Carve Fusion shell halves from a wire STL with leads.',
    )
    parser.add_argument(
        'wire_stl',
        nargs='?',
        default=None,
        help='Optional: overrides WIRE_WITH_LEADS_STL above.',
    )
    args = parser.parse_args()

    wire_path = args.wire_stl or WIRE_WITH_LEADS_STL
    if not wire_path:
        print('ERROR: set WIRE_WITH_LEADS_STL at the top of this script, or pass the path on the command line.')
        sys.exit(1)

    wire_path = os.path.normpath(os.path.abspath(wire_path))
    if not os.path.isfile(wire_path):
        print(f'ERROR: wire STL not found:\n  {wire_path}')
        sys.exit(1)

    run_shell_split(
        wire_path,
        output_dir=os.path.dirname(wire_path),
        gradient_layer=cfg.GRADIENT_LAYER,
        shell_stl_dir=cfg.SHELL_STL_DIR,
    )


if __name__ == '__main__':
    main()
