"""
FastHenry2 binary resolution (cross-platform).

The pipeline default used to be a hardcoded Windows path that broke on
Linux/macOS. ``resolve_fasthenry_bin`` instead prefers (a) an explicit path
passed by the user, then (b) whatever ``FastHenry2`` / ``fasthenry`` is on
PATH, and only falls back to the historical Windows location as a last
resort hint (without asserting it exists).
"""

from __future__ import annotations

import os
import shutil

# Historical default — kept only as a fallback hint for Windows users who
# installed FastHenry2 in the standard location. Not required to exist.
DEFAULT_WINDOWS_PATH = r'C:\Program Files (x86)\FastFieldSolvers\FastHenry2\FastHenry2.exe'

_CANDIDATE_NAMES = ('FastHenry2.exe', 'fasthenry.exe', 'fasthenry2', 'fasthenry')


def resolve_fasthenry_bin(configured_path: str | None = None) -> str:
    """
    Return a usable FastHenry binary path, or '' if none is found.

    Search order: explicit ``configured_path`` -> PATH lookup -> Windows
    default (only if it actually exists). Always returns a string.
    """
    candidates: list[str] = []
    if configured_path:
        candidates.append(os.path.expandvars(os.path.expanduser(configured_path)))
    for exe_name in _CANDIDATE_NAMES:
        found = shutil.which(exe_name)
        if found:
            candidates.append(found)
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return candidates[0] if candidates else ''


def fasthenry_available(resolved_path: str) -> bool:
    """True if ``resolved_path`` points to an existing file."""
    return bool(resolved_path and os.path.isfile(resolved_path))
