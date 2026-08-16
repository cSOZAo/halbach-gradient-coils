"""Unit tests for :mod:`gui.units`."""

import pytest

from gui import units


def test_mm_per_m_constant():
    assert units.MM_PER_M == 1000.0


@pytest.mark.parametrize('mm,m', [(0, 0.0), (1, 0.001), (430, 0.430), (-2.5, -0.0025)])
def test_mm_to_m(mm, m):
    assert units.mm_to_m(mm) == pytest.approx(m)


@pytest.mark.parametrize('m,mm', [(0, 0.0), (0.002, 2.0), (0.039, 39.0)])
def test_m_to_mm(m, mm):
    assert units.m_to_mm(m) == pytest.approx(mm)


def test_conversions_accept_strings_and_return_floats():
    assert units.mm_to_m('2000') == 2.0
    assert units.m_to_mm('0.5') == 500.0
    assert isinstance(units.mm_to_m(1), float)
    assert isinstance(units.m_to_mm(1), float)


@pytest.mark.parametrize('value', [0.0, 0.0125, 123.456])
def test_roundtrip(value):
    assert units.mm_to_m(units.m_to_mm(value)) == pytest.approx(value)
