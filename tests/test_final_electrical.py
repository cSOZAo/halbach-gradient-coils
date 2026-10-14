"""Final conductor solver evidence, bounded execution and physical section."""

from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from halbach_coils.coilgen.config import Config
from halbach_coils.coilgen import final_electrical as electrical


def test_simplification_preserves_direction_endpoints_and_error_bound():
    angle = np.linspace(0, 2 * np.pi, 1001)
    path = np.column_stack((.03 * np.cos(angle), .03 * np.sin(angle), .005 * angle))
    tolerance = .00005
    simplified = electrical.simplify_current_path(path, tolerance)
    assert len(simplified) < len(path)
    np.testing.assert_array_equal(simplified[[0, -1]], path[[0, -1]])
    delta = np.diff(simplified, axis=0)
    offset = path[:, None, :] - simplified[None, :-1, :]
    along = np.clip(np.einsum('msi,si->ms', offset, delta) / (delta * delta).sum(1), 0, 1)
    distances = np.linalg.norm(offset - along[:, :, None] * delta, axis=2).min(1)
    assert distances.max() <= tolerance * (1 + 1e-10)


def test_disabled_solver_preserves_analytic_resistance_without_fake_inductance(tmp_path):
    cfg = Config()
    cfg.fasthenry.enabled = False
    result = electrical.evaluate_final_electrical(cfg, [[0, 0, 0], [.1, 0, 0]], tmp_path)
    assert result['status'] == 'disabled'
    assert result['resistance_ohm'] is result['inductance_H'] is None
    assert result['analytic_resistance_ohm'] == pytest.approx(
        cfg.fasthenry.specific_conductivity * .1 / cfg.electrical_cross_section_area)
    assert result['supply_return_included'] is False
    assert not list(tmp_path.iterdir())


def test_solver_unavailable_is_explicit(monkeypatch, tmp_path):
    monkeypatch.setattr(electrical, 'resolve_fasthenry_bin', lambda _: '')
    result = electrical.evaluate_final_electrical(Config(), [[0, 0, 0], [.1, 0, 0]], tmp_path)
    assert result['status'] == 'unavailable'
    assert result['inductance_H'] is None


def _fake_binary(monkeypatch, tmp_path, system='Linux'):
    binary = tmp_path / 'FastHenry2.exe'
    binary.write_bytes(b'fake executable; subprocess is mocked')
    monkeypatch.setattr(electrical.platform, 'system', lambda: system)
    monkeypatch.setattr(electrical, 'resolve_fasthenry_bin', lambda _: str(binary))
    return binary


def test_success_uses_absolute_input_path_own_cwd_equal_area_and_persists_result(monkeypatch, tmp_path):
    binary = _fake_binary(monkeypatch, tmp_path)
    cwd = Path.cwd()
    output = tmp_path / 'output with spaces'
    captured = {}

    def run(arguments, **options):
        captured.update(arguments=arguments, **options)
        folder = Path(options['cwd'])
        assert Path(arguments[1]).is_absolute()
        assert arguments[0] == str(binary)
        assert options['timeout'] == 5 + 2
        (folder / f'Zc{arguments[-1]}.mat').write_text('one port\nfrequency 1\n0.025 +0.00005j\n')
        return NS(returncode=0, stdout='solved', stderr='')

    monkeypatch.setattr(electrical.subprocess, 'run', run)
    cfg = Config()
    cfg.wire.cross_section_a_frac = 3
    result = electrical.evaluate_final_electrical(cfg, [[0, 0, 0], [.1, 0, 0]], output, timeout=5)
    assert result['status'] == 'success'
    assert result['resistance_ohm'] == .025
    assert result['inductance_H'] == pytest.approx(.00005 / (2 * np.pi))
    assert Path.cwd() == cwd
    folder = Path(result['evidence_directory'])
    saved = json.loads((folder / 'result.json').read_text())
    assert saved['status'] == 'success'
    assert (folder / 'solver.log').read_text() == 'solved'
    text = (folder / 'final_circuit.inp').read_text()
    side = np.sqrt(cfg.electrical_cross_section_area)
    assert f'w={side:.17g} h={side:.17g}' in text
    assert '.Units m' in text and '.external N1 N2' in text


def test_timeout_persists_failure_and_never_publishes_r_or_l(monkeypatch, tmp_path):
    _fake_binary(monkeypatch, tmp_path)

    def timeout(*args, **options):
        raise subprocess.TimeoutExpired(args[0], options['timeout'])

    monkeypatch.setattr(electrical.subprocess, 'run', timeout)
    result = electrical.evaluate_final_electrical(Config(), [[0, 0, 0], [.1, 0, 0]], tmp_path)
    assert result['status'] == 'failed'
    assert result['resistance_ohm'] is result['inductance_H'] is None
    assert (Path(result['evidence_directory']) / 'solver.log').is_file()


def test_windows_rejects_different_com_server_before_any_launch(monkeypatch, tmp_path):
    _fake_binary(monkeypatch, tmp_path, 'Windows')
    monkeypatch.setattr(electrical, 'registered_windows_server', lambda: 'C:/other/FastHenry2.exe')
    monkeypatch.setattr(electrical.subprocess, 'run', lambda *_a, **_k: pytest.fail('COM must not launch'))
    result = electrical.evaluate_final_electrical(Config(), [[0, 0, 0], [.1, 0, 0]], tmp_path)
    assert result['status'] == 'failed'
    assert 'differs' in result['reason']


def test_windows_does_not_attach_to_users_open_server(monkeypatch, tmp_path):
    binary = _fake_binary(monkeypatch, tmp_path, 'Windows')
    monkeypatch.setattr(electrical, 'registered_windows_server', lambda: str(binary))
    monkeypatch.setattr(electrical, '_windows_server_running', lambda _: True)
    monkeypatch.setattr(electrical.subprocess, 'run', lambda *_a, **_k: pytest.fail('COM must not launch'))
    result = electrical.evaluate_final_electrical(Config(), [[0, 0, 0], [.1, 0, 0]], tmp_path)
    assert result['status'] == 'unavailable'
    assert 'already open' in result['reason']
    assert (Path(result['evidence_directory']) / 'result.json').is_file()


def test_windows_script_quotes_input_with_spaces_and_is_bounded(monkeypatch, tmp_path):
    binary = _fake_binary(monkeypatch, tmp_path, 'Windows')
    monkeypatch.setattr(electrical, 'registered_windows_server', lambda: str(binary))
    monkeypatch.setattr(electrical, '_windows_server_running', lambda _: False)
    captured = {}

    def run(arguments, **options):
        script = Path(arguments[-1]).read_text(encoding='utf-16')
        captured.update(script=script, **options)
        assert 'solver.Run("""' in script
        assert 'If elapsed > 5 Then' in script
        assert arguments[0] == 'cscript.exe'
        return NS(returncode=3, stdout='', stderr='timeout')

    monkeypatch.setattr(electrical.subprocess, 'run', run)
    result = electrical.evaluate_final_electrical(Config(), [[0, 0, 0], [.1, 0, 0]],
                                                  tmp_path / 'output with spaces', timeout=5)
    assert result['status'] == 'failed'
    assert captured['timeout'] == 7
    assert 'COM server may remain' in result['com_timeout_cleanup']
    assert 'timeout' in (Path(result['evidence_directory']) / 'solver.log').read_text()


@pytest.mark.parametrize('quoted', (True, False))
def test_registered_server_path_handles_windows_spaces(monkeypatch, quoted):
    executable = r'C:\Program Files (x86)\FastHenry2\FastHenry2.exe'
    command = ('"' + executable + '"' if quoted else executable) + ' /Automation'
    registry = NS(HKEY_CLASSES_ROOT=1,
                  OpenKey=lambda _root, key: nullcontext(key),
                  QueryValue=lambda key, _name: '{abc}' if key.startswith('FastHenry2.Document') else command)
    monkeypatch.setitem(sys.modules, 'winreg', registry)
    assert electrical.registered_windows_server() == executable


@pytest.mark.parametrize('row', ('0.025 +0.00005j', '2.5D-2 + 5.0d-5j'))
def test_impedance_parser_recovers_resistance_and_inductance(tmp_path, row):
    filename = tmp_path / 'Zc.mat'
    filename.write_text('one port\nfrequency 1\n' + row + '\n')
    resistance, inductance = electrical._parse_impedance(filename)
    assert resistance == .025
    assert inductance == pytest.approx(.00005 / (2 * np.pi))


@pytest.mark.parametrize('row', ('nan +0j', '0.025 -0.00005j', '0 +0j', 'not an impedance'))
def test_impedance_parser_rejects_invalid_values(tmp_path, row):
    filename = tmp_path / 'Zc.mat'
    filename.write_text('one port\nfrequency 1\n' + row + '\n')
    with pytest.raises(ValueError):
        electrical._parse_impedance(filename)
