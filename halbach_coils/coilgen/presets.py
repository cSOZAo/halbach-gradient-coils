"""Versioned JSON presets for reproducible coil-design configurations."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np

from .config import Config


SCHEMA_VERSION = 1
PRESET_EXTENSION = '.json'


def default_preset_dir() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'resultados', 'presets',
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def config_to_preset(cfg: Config, *, name: str = '', source: str = 'manual') -> dict:
    config_data = asdict(cfg)
    # Runtime and machine-specific state is intentionally not portable.
    config_data.pop('output_dir', None)
    config_data.pop('show_plots', None)
    config_data['fasthenry'].pop('bin_path', None)
    # The per-axis lead preset is derived again from gradient_axis on load.
    config_data['leads'].pop('preset', None)
    return {
        'schema_version': SCHEMA_VERSION,
        'name': name,
        'source': source,
        'config': _json_value(config_data),
    }


def _apply_values(target: Any, values: dict) -> None:
    for key, value in values.items():
        if not hasattr(target, key):
            continue
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_values(current, value)
        elif isinstance(current, np.ndarray):
            setattr(target, key, np.asarray(value, dtype=float))
        elif isinstance(current, tuple):
            setattr(target, key, tuple(value))
        else:
            setattr(target, key, value)


def config_from_preset(preset: dict) -> Config:
    if preset.get('schema_version') != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported preset schema: {preset.get('schema_version')!r}")
    values = preset.get('config')
    if not isinstance(values, dict):
        raise ValueError('Preset does not contain a config object.')
    axis = str(values.get('gradient_axis', 'x')).lower()
    cfg = Config(gradient_axis=axis)
    _apply_values(cfg, values)
    # Rebuild the derived preset in case gradient_axis was loaded from JSON.
    cfg.leads.preset = None
    cfg.lead_preset()
    return cfg


def save_config_preset(cfg: Config, path: str, *, name: str = '',
                       source: str = 'manual') -> str:
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(config_to_preset(cfg, name=name, source=source), fh,
                  indent=2, ensure_ascii=False)
        fh.write('\n')
    return path


def load_config_preset(path: str) -> Config:
    with open(path, encoding='utf-8') as fh:
        preset = json.load(fh)
    return config_from_preset(preset)
