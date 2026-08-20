"""
Unit tests for the pure preview helpers of :mod:`gui.pipeline_panel`.

``_roi_cylinder_collision``, ``_annular_sector_poly`` and
``_radial_ellipse_polys`` are static methods with no widget state, so they can
be tested without building a Tk panel (which needs a display and a main loop).
"""

import math
import os

import numpy as np
import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')
pytest.importorskip('tkinter')

from gui.pipeline_panel import PipelinePanel                        # noqa: E402


collision = PipelinePanel._roi_cylinder_collision
sector = PipelinePanel._annular_sector_poly
ellipse = PipelinePanel._radial_ellipse_polys


# ---------------------------------------------------------------------------
# _roi_cylinder_collision
# ---------------------------------------------------------------------------

def test_roi_inside_the_bore_reports_no_collision():
    assert collision(0.02, 0.02, 0.05, r_inner=0.035, half_h=0.2) == []


def test_roi_exactly_on_the_bore_limits_is_accepted():
    """The tolerance keeps a flush fit from being reported as a collision."""
    assert collision(0.035, 0.035, 0.2, r_inner=0.035, half_h=0.2) == []


def test_roi_wider_than_the_bore_is_flagged_radially():
    reasons = collision(0.05, 0.02, 0.05, r_inner=0.035, half_h=0.2)

    assert len(reasons) == 1
    assert 'ROI radial' in reasons[0]
    assert '50.0 mm' in reasons[0] and '35.0 mm' in reasons[0]


def test_roi_radial_check_uses_the_larger_of_rx_and_ry():
    assert collision(0.02, 0.05, 0.05, 0.035, 0.2)[0].startswith('ROI radial')


def test_roi_longer_than_the_cylinder_is_flagged_axially():
    reasons = collision(0.02, 0.02, 0.30, r_inner=0.035, half_h=0.2)

    assert len(reasons) == 1
    assert 'ROI axial' in reasons[0]
    assert 'H/2=200.0 mm' in reasons[0]


def test_roi_outside_on_both_counts_reports_both_reasons():
    reasons = collision(0.05, 0.06, 0.30, r_inner=0.035, half_h=0.2)

    assert len(reasons) == 2
    assert 'ROI radial' in reasons[0]
    assert 'ROI axial' in reasons[1]


# ---------------------------------------------------------------------------
# _annular_sector_poly
# ---------------------------------------------------------------------------

def test_annular_sector_poly_spans_both_radii():
    poly = sector(0.03, 0.04, 0.0, math.pi, n=36)

    assert poly.shape == (72, 2)
    radii = np.linalg.norm(poly, axis=1)
    assert radii[:36] == pytest.approx(np.full(36, 0.04))
    assert radii[36:] == pytest.approx(np.full(36, 0.03))


def test_annular_sector_poly_walks_out_then_back_in():
    poly = sector(0.03, 0.04, 0.0, math.pi / 2, n=8)

    assert poly[0] == pytest.approx([0.04, 0.0])
    assert poly[7] == pytest.approx([0.0, 0.04], abs=1e-12)
    assert poly[8] == pytest.approx([0.0, 0.03], abs=1e-12)
    assert poly[-1] == pytest.approx([0.03, 0.0])


def test_annular_sector_poly_covers_the_requested_angles():
    t0, t1 = 0.4, 1.3

    poly = sector(0.03, 0.04, t0, t1, n=16)

    angles = np.arctan2(poly[:16, 1], poly[:16, 0])
    assert angles.min() == pytest.approx(t0)
    assert angles.max() == pytest.approx(t1)


# ---------------------------------------------------------------------------
# _radial_ellipse_polys
# ---------------------------------------------------------------------------

def test_radial_ellipse_outline_is_centred_on_the_groove_radius():
    _, _, outline = ellipse(center_radius=0.037, theta=0.0, semi_radial=0.002,
                            semi_tangent=0.001, peel_toward='out',
                            peel_frac=0.0, n=64)

    assert outline.shape == (64, 2)
    assert outline[:, 0].max() == pytest.approx(0.039)
    assert outline[:, 0].min() == pytest.approx(0.035)
    assert outline[:, 1].max() == pytest.approx(0.001)
    assert outline.mean(axis=0) == pytest.approx([0.037, 0.0], abs=1e-9)


def test_radial_ellipse_is_rotated_to_the_requested_angle():
    theta = math.pi / 2

    _, _, outline = ellipse(0.037, theta, 0.002, 0.001, 'out', 0.0, n=64)

    assert outline.mean(axis=0) == pytest.approx([0.0, 0.037], abs=1e-9)
    # radial semi-axis now points along +Y
    assert outline[:, 1].max() == pytest.approx(0.039)
    assert outline[:, 0].max() == pytest.approx(0.001)


def test_zero_peel_fraction_keeps_the_whole_ellipse():
    kept, peel, outline = ellipse(0.037, 0.0, 0.002, 0.001, 'out', 0.0, n=48)

    assert peel.shape == (0, 2)
    assert kept == pytest.approx(outline)


def test_peel_toward_out_removes_the_outer_tip():
    kept, peel, _ = ellipse(0.037, 0.0, 0.002, 0.001, 'out', 0.25, n=48)

    assert len(peel) >= 8
    assert len(kept) >= 16
    # peel sits beyond the split radius, kept stays inside it
    split = 0.037 + 0.002 - 2 * 0.25 * 0.002
    assert peel[:, 0].min() == pytest.approx(split)
    assert peel[:, 0].max() == pytest.approx(0.039, abs=1e-5)
    assert kept[:, 0].max() == pytest.approx(split)
    assert kept[:, 0].min() == pytest.approx(0.035, abs=1e-5)


def test_peel_toward_in_removes_the_inner_tip():
    kept, peel, _ = ellipse(0.037, 0.0, 0.002, 0.001, 'in', 0.25, n=48)

    split = 0.037 - 0.002 + 2 * 0.25 * 0.002
    assert peel[:, 0].max() == pytest.approx(split)
    assert peel[:, 0].min() == pytest.approx(0.035, abs=1e-5)
    assert kept[:, 0].min() == pytest.approx(split)
    assert kept[:, 0].max() == pytest.approx(0.039, abs=1e-5)


def test_a_half_peel_fraction_splits_at_the_groove_centre():
    kept, peel, _ = ellipse(0.037, 0.0, 0.002, 0.001, 'out', 0.25, n=48)
    kept_h, peel_h, _ = ellipse(0.037, 0.0, 0.002, 0.001, 'out', 0.5, n=48)

    assert peel_h[:, 0].min() == pytest.approx(0.037)
    assert peel_h[:, 0].min() < peel[:, 0].min()
    assert kept_h[:, 0].max() == pytest.approx(0.037)


def test_a_full_peel_fraction_removes_the_whole_ellipse():
    kept, peel, outline = ellipse(0.037, 0.0, 0.002, 0.001, 'out', 1.0, n=48)

    assert peel[:, 0].min() == pytest.approx(outline[:, 0].min(), abs=1e-4)
    assert peel[:, 0].max() == pytest.approx(outline[:, 0].max(), abs=1e-4)
    assert kept[:, 0].min() == pytest.approx(kept[:, 0].max())


def test_peel_fraction_is_clamped_into_the_unit_range():
    over = ellipse(0.037, 0.0, 0.002, 0.001, 'out', 4.0, n=48)
    full = ellipse(0.037, 0.0, 0.002, 0.001, 'out', 1.0, n=48)
    under = ellipse(0.037, 0.0, 0.002, 0.001, 'out', -3.0, n=48)

    assert over[1] == pytest.approx(full[1])
    assert under[1].shape == (0, 2)


def test_a_degenerate_radial_semi_axis_yields_no_peel():
    kept, peel, outline = ellipse(0.037, 0.0, 0.0, 0.001, 'out', 0.3, n=48)

    assert peel.shape == (0, 2)
    assert kept == pytest.approx(outline)
