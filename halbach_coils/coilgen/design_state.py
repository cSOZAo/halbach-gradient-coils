"""Per-wire provenance and integrity checks for safe reuse of a coil design."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 2
COORDINATE_FRAME = 'physical_xyz'


def _state_path(wire: Path) -> Path:
    return wire.with_name(wire.stem + '_design.json')


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _config_snapshot(cfg) -> dict:
    winding = asdict(cfg.winding)
    # This legacy multiplier is ignored by the current normal-shift formula.
    winding.pop('normal_shift_height_factor', None)
    return _json_value({
        'gradient_axis': cfg.gradient_axis.lower(),
        'b0_direction': cfg.b0_direction.strip().lower(),
        'tikhonov_factor': cfg.tikhonov_factor,
        'num_levels': cfg.num_levels,
        'target': asdict(cfg.target),
        'cylinder': asdict(cfg.cylinder),
        'winding': winding,
        'wire': asdict(cfg.wire),
        'advanced': asdict(cfg.advanced),
        'resistivity_ohm_m': cfg.fasthenry.specific_conductivity,
        'derived_geometry': {
            'cylinder_design_radius_m': cfg.cylinder_design_radius,
            'normal_shift_length_m': cfg.normal_shift_length,
            'conductor_semi_a_m': cfg.conductor_semi_a,
            'conductor_semi_b_m': cfg.conductor_semi_b,
            'electrical_cross_section_area_m2': cfg.electrical_cross_section_area,
        },
    })


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f'Design file is missing: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _write_state(path: Path, data: dict) -> str:
    contents = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False,
                          allow_nan=False) + '\n'
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                                         dir=path.parent, prefix='.' + path.name,
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return str(path)


def save_design_state(wire_stl, cfg, sidecar_paths=()) -> str:
    """Atomically save a state only after the STL and all sibling sidecars exist.

    Relative sidecar paths are resolved beside the STL. Sidecars are stored as
    filenames so a complete validated design can be copied to another run.
    """
    wire = Path(wire_stl).resolve()
    sidecars = []
    names = set()
    for provided in sidecar_paths:
        path = Path(provided)
        path = (wire.parent / path).resolve() if not path.is_absolute() else path.resolve()
        if path.parent != wire.parent or path in (wire, _state_path(wire)):
            raise ValueError(f'Design sidecar must be a separate file beside the STL: {path}')
        if path.name in names:
            raise ValueError(f'Duplicate design sidecar: {path.name}')
        names.add(path.name)
        sidecars.append({'filename': path.name, 'sha256': _sha256(path)})
    data = {
        'schema_version': SCHEMA_VERSION,
        'coordinate_frame': COORDINATE_FRAME,
        'config': _config_snapshot(cfg),
        'wire': {'filename': wire.name, 'sha256': _sha256(wire)},
        'sidecars': sorted(sidecars, key=lambda item: item['filename']),
    }
    return _write_state(_state_path(wire), data)


def _checked_state(wire: Path) -> dict:
    path = _state_path(wire)
    try:
        with path.open(encoding='utf-8') as stream:
            data = json.load(stream)
    except (OSError, ValueError) as exc:
        raise ValueError(f'Design state is missing or unreadable for {wire.name}; run the gradient design again.') from exc
    if (not isinstance(data, dict) or data.get('schema_version') != SCHEMA_VERSION
            or data.get('coordinate_frame') != COORDINATE_FRAME
            or not isinstance(data.get('config'), dict)
            or not isinstance(data.get('wire'), dict)
            or not isinstance(data.get('sidecars'), list)):
        raise ValueError(f'Unsupported or incomplete design state: {path}')
    if data['wire'].get('filename') != wire.name:
        raise ValueError(f'Design state belongs to a different STL: {path}')
    if _sha256(wire) != data['wire'].get('sha256'):
        raise ValueError(f'Wire STL changed since the design was saved: {wire}')
    names = set()
    for item in data['sidecars']:
        filename = item.get('filename') if isinstance(item, dict) else None
        if (not isinstance(filename, str) or Path(filename).name != filename
                or filename in ('', '.', '..', wire.name, path.name) or filename in names):
            raise ValueError(f'Invalid sidecar entry in design state: {path}')
        names.add(filename)
        if _sha256(wire.parent / filename) != item.get('sha256'):
            raise ValueError(f'Design sidecar changed since the design was saved: {filename}')
    return data


def _changed_fields(previous, current, prefix='') -> list[str]:
    if isinstance(previous, dict) and isinstance(current, dict):
        changes = []
        for key in sorted(previous.keys() | current.keys()):
            label = prefix + '.' + key if prefix else key
            if key not in previous or key not in current:
                changes.append(label)
            else:
                changes.extend(_changed_fields(previous[key], current[key], label))
        return changes
    return [] if previous == current else [prefix]


def validate_design_state(wire_stl, cfg) -> None:
    """Reject missing/changed files, old physical models and incompatible config."""
    wire = Path(wire_stl).resolve()
    data = _checked_state(wire)
    changes = _changed_fields(data['config'], _config_snapshot(cfg))
    if changes:
        raise ValueError('Wire design parameters do not match the current configuration: '
                         + ', '.join(changes[:8]) + '. Run the gradient design again.')


def copy_design_state(sourcewire, destwire) -> str:
    """Copy verified sidecars/state after the caller has copied the matching STL."""
    source, destination = Path(sourcewire).resolve(), Path(destwire).resolve()
    data = _checked_state(source)
    if _sha256(destination) != data['wire']['sha256']:
        raise ValueError('Destination STL does not match the validated source design')
    for item in data['sidecars']:
        src = source.parent / item['filename']
        dst = destination.parent / item['filename']
        if src != dst:
            shutil.copy2(src, dst)
    data['wire']['filename'] = destination.name
    return _write_state(_state_path(destination), data)
