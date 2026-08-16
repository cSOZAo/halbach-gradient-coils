"""
coilgen — consolidated config-driven package for designing MRI gradient
coils with pyCoilGen and carving printable molds.

Replaces the former ``pipeline/``, ``pipeline_gz/`` and ``standalone/``
duplicates with a single source of truth driven by :class:`coilgen.config.Config`.

Coordinate frame
----------------
The scanner B0 is parallel to +Y (transverse to the bore). pyCoilGen optimizes
Bz internally, so the cylinder mesh is rotated R_y(pi/2) to align the optimized
component with the physical gradient axis. ``Config.internal_axis`` maps a
physical gradient axis (x/y/z) to the pyCoilGen internal axis (y/z/x).
"""

from __future__ import annotations

import os
import sys

# pyCoilGen lives one level above this workspace (sibling of ``pruebas/``).
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_PACKAGE_DIR)          # .../pruebas
_PYCOILGEN_ROOT = os.path.dirname(_PROJECT_ROOT)       # .../pyCoilGen-0.2.4
if _PYCOILGEN_ROOT not in sys.path:
    sys.path.insert(0, _PYCOILGEN_ROOT)

__all__ = ["config", "geometry", "fasthenry", "paths", "metrics"]
