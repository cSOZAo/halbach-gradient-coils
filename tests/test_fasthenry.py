"""Unit tests for :mod:`coilgen.fasthenry`."""

import os

import pytest

from coilgen import fasthenry


@pytest.fixture
def no_path_binaries(monkeypatch):
    """Make ``shutil.which`` find nothing, whatever the host has installed."""
    monkeypatch.setattr(fasthenry.shutil, 'which', lambda name: None)


def _make_exe(tmp_path, name='FastHenry2.exe'):
    path = tmp_path / name
    path.write_text('binary')
    return str(path)


def test_configured_existing_path_wins(tmp_path, monkeypatch):
    configured = _make_exe(tmp_path)
    on_path = _make_exe(tmp_path, 'fasthenry')
    monkeypatch.setattr(fasthenry.shutil, 'which',
                        lambda name: on_path if name == 'fasthenry' else None)

    assert fasthenry.resolve_fasthenry_bin(configured) == configured


def test_configured_path_expands_user_and_vars(tmp_path, monkeypatch, no_path_binaries):
    configured = _make_exe(tmp_path)
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('FH_DIR', str(tmp_path))

    assert fasthenry.resolve_fasthenry_bin('~/FastHenry2.exe') == configured
    assert fasthenry.resolve_fasthenry_bin('$FH_DIR/FastHenry2.exe') == configured


def test_path_lookup_used_when_configured_path_missing(tmp_path, monkeypatch):
    on_path = _make_exe(tmp_path, 'fasthenry2')
    monkeypatch.setattr(fasthenry.shutil, 'which',
                        lambda name: on_path if name == 'fasthenry2' else None)

    resolved = fasthenry.resolve_fasthenry_bin(str(tmp_path / 'missing.exe'))

    assert resolved == on_path


def test_candidate_names_are_probed_in_order(tmp_path, monkeypatch):
    """The first name in ``_CANDIDATE_NAMES`` found on PATH is preferred."""
    found = {
        'fasthenry.exe': _make_exe(tmp_path, 'fasthenry.exe'),
        'fasthenry': _make_exe(tmp_path, 'fasthenry'),
    }
    monkeypatch.setattr(fasthenry.shutil, 'which', lambda name: found.get(name))

    assert fasthenry.resolve_fasthenry_bin() == found['fasthenry.exe']


def test_returns_first_candidate_hint_when_nothing_exists(tmp_path, no_path_binaries):
    missing = str(tmp_path / 'nowhere' / 'FastHenry2.exe')

    # Nothing resolvable: the (non-existent) configured path is returned as a hint.
    assert fasthenry.resolve_fasthenry_bin(missing) == missing


def test_returns_empty_string_without_candidates(no_path_binaries):
    assert fasthenry.resolve_fasthenry_bin() == ''
    assert fasthenry.resolve_fasthenry_bin('') == ''
    assert fasthenry.resolve_fasthenry_bin(None) == ''


def test_windows_default_is_not_asserted_to_exist(no_path_binaries):
    """The historical Windows path must not leak in on non-Windows hosts."""
    resolved = fasthenry.resolve_fasthenry_bin()

    assert fasthenry.DEFAULT_WINDOWS_PATH.endswith('FastHenry2.exe')
    assert resolved != fasthenry.DEFAULT_WINDOWS_PATH


@pytest.mark.parametrize('value', ['', None])
def test_fasthenry_available_false_for_empty(value):
    assert fasthenry.fasthenry_available(value) is False


def test_fasthenry_available_checks_file(tmp_path):
    exe = _make_exe(tmp_path)

    assert fasthenry.fasthenry_available(exe) is True
    assert fasthenry.fasthenry_available(str(tmp_path)) is False          # a directory
    assert fasthenry.fasthenry_available(os.path.join(exe, 'x')) is False
