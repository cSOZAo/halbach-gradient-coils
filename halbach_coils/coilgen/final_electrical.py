"""FastHenry on the final oriented conductor, with bounded path simplification.

Inductance is partial: supply/return wiring is outside this model. Windows
FastHenry uses COM; the client is bounded, but an unresponsive COM server
cannot be guaranteed to stop when its client times out.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import tempfile
from pathlib import Path
from uuid import uuid4

import numpy as np

from .fasthenry import resolve_fasthenry_bin, fasthenry_available


def simplify_current_path(path, tolerance):
    """RDP in 3D: omitted points stay within tolerance of retained segments."""
    path = np.asarray(path, dtype=float)
    if (path.ndim != 2 or path.shape[1] != 3 or len(path) < 2
            or not np.all(np.isfinite(path)) or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError('Expected a finite N x 3 current path and positive tolerance.')
    path = path[np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-12]]
    if len(path) < 2:
        raise ValueError('The current path has no nonzero segments.')
    keep = {0, len(path) - 1}
    pending = [(0, len(path) - 1)]
    while pending:
        first, last = pending.pop()
        if last <= first + 1:
            continue
        delta = path[last] - path[first]
        denominator = float(delta @ delta)
        offsets = path[first + 1:last] - path[first]
        along = np.clip(offsets @ delta / denominator, 0, 1) if denominator > 1e-24 else np.zeros(len(offsets))
        distances = np.linalg.norm(offsets - along[:, None] * delta, axis=1)
        local = int(np.argmax(distances))
        if distances[local] > tolerance:
            middle = first + 1 + local
            keep.add(middle)
            pending.extend(((first, middle), (middle, last)))
    return path[sorted(keep)]


def registered_windows_server():
    """Read the COM server path without creating or attaching to the server."""
    import winreg
    with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, r'FastHenry2.Document\CLSID') as key:
        clsid = winreg.QueryValue(key, None)
    with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, rf'CLSID\{clsid}\LocalServer32') as key:
        command = winreg.QueryValue(key, None).strip()
    match = re.match(r'"([^"]+)"|(.+?\.exe)(?:\s|$)', command, flags=re.I)
    if not match:
        raise RuntimeError('Cannot identify the registered FastHenry COM executable.')
    return match.group(1) or match.group(2)


def _windows_server_running(binary):
    """Conservatively avoid attaching to an existing user's FastHenry process."""
    process_list = subprocess.run(
        ['tasklist.exe', '/FO', 'CSV', '/NH'], capture_output=True, text=True,
        timeout=5, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if process_list.returncode:
        raise RuntimeError('Cannot check whether a FastHenry COM server is already open.')
    name = Path(binary).name.casefold()
    return any(row and row[0].casefold() == name for row in csv.reader(process_list.stdout.splitlines()))


def _atomic_text(filename, contents, encoding='utf-8'):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding=encoding, newline='\n',
                                         dir=filename.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, filename)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _write_input(path, filename, width, height, resistivity):
    lines = ['* Final conductor; supply and return wires omitted', '.Units m',
             f'.Default sigma={1 / resistivity:.17g}']
    lines.extend(f'N{i} x={p[0]:.17g} y={p[1]:.17g} z={p[2]:.17g}'
                 for i, p in enumerate(path, 1))
    lines.extend(f'E{i} N{i} N{i + 1} w={width:.17g} h={height:.17g}'
                 for i in range(1, len(path)))
    lines.extend((f'.external N1 N{len(path)}', '.freq fmin=1 fmax=1 ndec=1', '.end'))
    _atomic_text(filename, '\n'.join(lines) + '\n')


def _parse_impedance(filename):
    lines = Path(filename).read_text(encoding='ascii').splitlines()
    number = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?'
    match = re.fullmatch(rf'\s*({number})\s+([+-]?)\s*({number})[jJ]\s*',
                         lines[2] if len(lines) > 2 else '')
    if match is None:
        raise ValueError('FastHenry did not produce a valid one-port impedance matrix.')
    resistance = float(match.group(1).replace('D', 'E').replace('d', 'e'))
    imaginary = float(match.group(3).replace('D', 'E').replace('d', 'e'))
    if match.group(2) == '-':
        imaginary = -imaginary
    inductance = imaginary / (2 * np.pi)
    if not np.isfinite(resistance) or not np.isfinite(inductance) or resistance <= 0 or inductance <= 0:
        raise ValueError('FastHenry returned invalid resistance or inductance.')
    return resistance, inductance


def evaluate_final_electrical(cfg, path_N3, output_dir, tolerance=2e-5, timeout=60) -> dict:
    """Return solver status/R/L and save compact evidence for an attempted run.

    subprocess cwd applies only to the owned solver client. Existing Windows
    FastHenry instances are never intentionally reused or terminated. On a COM
    timeout the client is stopped; the server may remain alive, and no L is used.
    """
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError('FastHenry timeout must be a positive finite number.')
    simplified = simplify_current_path(path_N3, tolerance)
    path = np.asarray(path_N3, dtype=float)
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    area, resistivity = float(cfg.electrical_cross_section_area), float(cfg.fasthenry.specific_conductivity)
    if not np.isfinite(area) or not np.isfinite(resistivity) or area <= 0 or resistivity <= 0:
        raise ValueError('Conductor area and resistivity must be positive finite numbers.')
    solver_length = float(np.linalg.norm(np.diff(simplified, axis=0), axis=1).sum())
    result = {
        'status': 'disabled' if not cfg.fasthenry.enabled else 'unavailable',
        'resistance_ohm': None, 'inductance_H': None, 'frequency_Hz': 1,
        'section_model': 'square_of_equal_area_to_round_conductor',
        'supply_return_included': False, 'path_tolerance_m': float(tolerance),
        'original_point_count': len(path), 'solver_point_count': len(simplified),
        'original_length_m': length, 'solver_length_m': solver_length,
        'relative_length_change': abs(solver_length - length) / length,
        'cross_section_m2': area, 'analytic_resistance_ohm': resistivity * length / area,
        'note': 'Low-frequency partial inductance of the modeled conductor; supply and return wiring excluded.',
    }
    if not cfg.fasthenry.enabled:
        return result
    binary = resolve_fasthenry_bin(cfg.fasthenry.bin_path)
    result['configured_binary'] = binary
    if not fasthenry_available(binary):
        result['reason'] = 'FastHenry executable unavailable.'
        return result
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='final_fasthenry_', dir=output))
    token = uuid4().hex[:12]
    try:
        windows = platform.system() == 'Windows'
        actual_binary = registered_windows_server() if windows else binary
        if os.path.normcase(os.path.realpath(actual_binary)) != os.path.normcase(os.path.realpath(binary)):
            raise RuntimeError('The selected executable differs from the registered FastHenry COM server.')
        result['actual_binary'] = actual_binary
        with open(actual_binary, 'rb') as stream:
            result['binary_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
        input_file = folder / 'final_circuit.inp'
        side = float(np.sqrt(area))
        _write_input(simplified, input_file, side, side, resistivity)
        if windows:
            if _windows_server_running(binary):
                result.update(status='unavailable', reason='FastHenry is already open; close it before requesting this calculation.')
                return result
            command = f'"{input_file}" -S {token}'.replace('"', '""')
            script = folder / 'solve.vbs'
            _atomic_text(script,
                'Set solver = CreateObject("FastHenry2.Document")\n'
                f'accepted = solver.Run("{command}")\n'
                'If Not accepted Then\n  solver.Quit\n  WScript.Quit 2\nEnd If\n'
                'started = Timer\nDo While solver.IsRunning\n  WScript.Sleep 100\n'
                '  elapsed = Timer - started\n  If elapsed < 0 Then elapsed = elapsed + 86400\n'
                f'  If elapsed > {timeout:g} Then\n    WScript.Quit 3\n  End If\n'
                'Loop\nsolver.Quit\nSet solver = Nothing\n', encoding='utf-16')
            arguments = ['cscript.exe', '//Nologo', str(script)]
            flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
            result['com_timeout_cleanup'] = 'Client bounded; a hung external COM server may remain. No user processes are killed.'
        else:
            arguments, flags = [binary, str(input_file), '-S', token], 0
        completed = subprocess.run(arguments, cwd=folder, capture_output=True, text=True,
                                   timeout=timeout + 2, creationflags=flags)
        _atomic_text(folder / 'solver.log', completed.stdout + completed.stderr)
        if completed.returncode:
            raise RuntimeError(f'FastHenry launcher exit code: {completed.returncode}')
        resistance, inductance = _parse_impedance(folder / f'Zc{token}.mat')
        analytic = result['analytic_resistance_ohm']
        result.update(status='success', resistance_ohm=resistance, inductance_H=inductance,
                      relative_resistance_difference=abs(resistance - analytic) / analytic)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        result.update(status='failed', reason=str(exc))
        log_path = folder / 'solver.log'
        previous = log_path.read_text(encoding='utf-8') if log_path.exists() else ''
        _atomic_text(log_path, previous + '\n' + str(exc) + '\n')
    finally:
        result['evidence_directory'] = str(folder)
        _atomic_text(folder / 'result.json', json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result
