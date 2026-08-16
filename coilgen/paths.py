"""
Output path helpers for the coil-mold workflow.

- ``resultados/pipeline/``  — one subfolder per full pipeline run
- ``resultados/standalone/`` — shared folder per design (axis / Tikhonov / levels)

When a file or run folder already exists, the next free name uses a ``(n)``
suffix (``(2)``, ``(3)``, ...) instead of overwriting.
"""

from __future__ import annotations

import os
import re

from .geometry import internal_field_axis

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_MODULE_DIR)

STANDALONE_OUTPUT_BASE = os.path.join(PROJECT_ROOT, 'resultados', 'standalone')
PIPELINE_OUTPUT_BASE = os.path.join(PROJECT_ROOT, 'resultados', 'pipeline')

ACTIVE_STEM_FILE = '.active_project_stem'

_SUFFIX_RE = re.compile(r'\(\d+\)$')


def dir_has_outputs(path: str) -> bool:
    """True if *path* is a directory that contains at least one file."""
    if not os.path.isdir(path):
        return False
    for _root, _dirs, files in os.walk(path):
        if files:
            return True
    return False


def design_folder_name(axis: str, tikhonov: float, num_levels: int) -> str:
    """Short folder label: ``Gy_tk2500_lvl26``."""
    return f'G{axis}_tk{int(tikhonov)}_lvl{num_levels}'


def gradient_project_stem(axis: str, tikhonov: float, num_levels: int) -> str:
    """pyCoilGen project / STL prefix: ``Gradient_Gy_tk2500_lvl26``."""
    return f'Gradient_G{axis}_tk{int(tikhonov)}_lvl{num_levels}'


def standalone_design_dir(axis: str, tikhonov: float, num_levels: int) -> str:
    """Fixed working directory for a standalone design (all three steps)."""
    return os.path.join(
        STANDALONE_OUTPUT_BASE,
        design_folder_name(axis, tikhonov, num_levels),
    )


def unique_path(path: str) -> str:
    """Return *path* or ``stem(n).ext`` if *path* already exists."""
    if not os.path.exists(path):
        return path
    directory, basename = os.path.split(path)
    stem, ext = os.path.splitext(basename)
    base = _SUFFIX_RE.sub('', stem)
    n = 2
    while True:
        candidate = os.path.join(directory, f'{base}({n}){ext}')
        if not os.path.exists(candidate):
            return candidate
        n += 1


def unique_stem(
    directory: str,
    stem: str,
    marker_templates: tuple[str, ...] | None = None,
    *,
    gradient_axis: str = 'y',
) -> str:
    """
    Return *stem* or ``stem(n)`` when marker files already exist in *directory*.

    Occupancy is decided by the marker templates (wire STL + metrics txt by
    default, keyed to the internal field axis for the requested gradient axis).
    """
    if marker_templates is None:
        field_axis = internal_field_axis(gradient_axis)
        marker_templates = (
            f'{{stem}}_wire_0_{field_axis}.stl',
            '{stem}_metrics.txt',
        )
    os.makedirs(directory, exist_ok=True)

    def _occupied(s: str) -> bool:
        return any(
            os.path.exists(os.path.join(directory, tpl.format(stem=s)))
            for tpl in marker_templates
        )

    if not _occupied(stem):
        return stem
    n = 2
    while _occupied(f'{stem}({n})'):
        n += 1
    return f'{stem}({n})'


def unique_run_dir(base_dir: str, stem: str) -> str:
    """
    Return a new run folder under *base_dir*.

    Each pipeline invocation gets the next free name (``stem``, ``stem(2)``...).
    Existing folders are never reused, even when empty, so every iteration lands
    in its own directory.
    """
    os.makedirs(base_dir, exist_ok=True)
    if not os.path.isdir(os.path.join(base_dir, stem)):
        path = os.path.join(base_dir, stem)
        os.makedirs(path, exist_ok=True)
        return path
    n = 2
    while os.path.isdir(os.path.join(base_dir, f'{stem}({n})')):
        n += 1
    path = os.path.join(base_dir, f'{stem}({n})')
    os.makedirs(path, exist_ok=True)
    return path


def write_active_stem(directory: str, stem: str) -> None:
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, ACTIVE_STEM_FILE), 'w', encoding='utf-8') as fh:
        fh.write(stem)


def read_active_stem(directory: str, default: str = '') -> str:
    path = os.path.join(directory, ACTIVE_STEM_FILE)
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as fh:
            return fh.read().strip()
    return default


def wire_stl_name(stem: str, gradient_axis: str = 'y') -> str:
    field_axis = internal_field_axis(gradient_axis)
    return f'{stem}_wire_0_{field_axis}.stl'


def _is_derived_wire_stl(path: str) -> bool:
    base = os.path.basename(path)
    return any(
        tag in base for tag in ('_with_leads', '_coil_open', '_leads_only')
    )


def resolve_lead_stl_paths(wire_stl: str) -> tuple[str, str, str]:
    """
    Return ``(with_leads, coil_open, leads_only)`` for the newest matching
    files derived from *wire_stl* (handles optional ``(n)`` suffixes).
    """
    import glob

    base, ext = os.path.splitext(wire_stl)
    directory = os.path.dirname(wire_stl) or '.'

    def _newest(pattern: str) -> str:
        matches = glob.glob(os.path.join(directory, pattern))
        if matches:
            return max(matches, key=os.path.getmtime)
        return ''

    with_leads = _newest(os.path.basename(base) + '_with_leads*.stl')
    coil_open = _newest(os.path.basename(base) + '_coil_open*.stl')
    leads_only = _newest(os.path.basename(base) + '_leads_only*.stl')

    if not with_leads:
        with_leads = base + '_with_leads' + ext
    if not coil_open:
        coil_open = base + '_coil_open' + ext
    if not leads_only:
        leads_only = base + '_leads_only' + ext
    return with_leads, coil_open, leads_only


def resolve_wire_stl_path(
    output_dir: str,
    axis: str,
    tikhonov: float,
    num_levels: int,
) -> str:
    """Wire STL from ``.active_project_stem`` or the default naming convention."""
    import glob

    default_stem = gradient_project_stem(axis, tikhonov, num_levels)
    stem = read_active_stem(output_dir, default_stem)
    field_axis = internal_field_axis(axis)
    for candidate_stem in (stem, default_stem):
        path = os.path.join(output_dir, wire_stl_name(candidate_stem, axis))
        if os.path.isfile(path):
            return path

    patterns = (
        os.path.join(
            output_dir,
            f'Gradient_G{axis}_tk{int(tikhonov)}_lvl{num_levels}'
            f'*_wire_0_{field_axis}.stl',
        ),
        os.path.join(
            output_dir,
            f'Gradient_G{axis}_tk{int(tikhonov)}_lvl{num_levels}*_wire_0_?.stl',
        ),
    )
    matches: list[str] = []
    for pattern in patterns:
        for path in glob.glob(pattern):
            if not _is_derived_wire_stl(path):
                matches.append(path)
    if matches:
        return max(matches, key=os.path.getmtime)
    return os.path.join(output_dir, wire_stl_name(stem, axis))


def derive_align_wire_path(subtract_path: str) -> str:
    """Return the coil-only STL path paired with a *_with_leads* subtract STL."""
    import re
    base, ext = os.path.splitext(subtract_path)
    coil_only = re.sub(r'_with_leads(?:\(\d+\))?$', '', base) + ext
    if os.path.isfile(coil_only):
        return coil_only
    legacy = base[:-len('_with_leads')] + ext if base.endswith('_with_leads') else base
    if os.path.isfile(legacy):
        return legacy
    return subtract_path


def unique_lead_output_paths(input_stl: str) -> tuple[str, str, str]:
    """
    Return ``(with_leads, coil_open, leads_only)`` with a shared ``(n)`` suffix
    when any of the trio would overwrite an existing file.
    """
    base, ext = os.path.splitext(input_stl)
    with_leads = unique_path(base + '_with_leads' + ext)
    wl_base, wl_ext = os.path.splitext(with_leads)
    coil_open = wl_base.replace('_with_leads', '_coil_open') + wl_ext
    leads_only = wl_base.replace('_with_leads', '_leads_only') + wl_ext
    if os.path.exists(coil_open) or os.path.exists(leads_only):
        with_leads = unique_path(with_leads)
        wl_base, wl_ext = os.path.splitext(with_leads)
        coil_open = wl_base.replace('_with_leads', '_coil_open') + wl_ext
        leads_only = wl_base.replace('_with_leads', '_leads_only') + wl_ext
    return with_leads, coil_open, leads_only
