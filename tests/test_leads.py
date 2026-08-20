"""
Unit tests for the axis-agnostic geometry helpers in :mod:`coilgen.leads`.

These helpers are the part of the lead-attachment step that carries the
axis-awareness fix (apex selection in cylindrical coordinates, tangent/normal
frames built from ``axis_hat`` instead of world axes). They are pure and can be
exercised without pyCoilGen output, so they are tested directly with small
synthetic cylinders instead of a real wire STL.
"""

import numpy as np
import pytest
import trimesh

from coilgen import leads
from coilgen.config import Config


X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])


def _cylinder_points(axis_hat, radius=0.04, n_ang=48, n_ax=9, length=0.2):
    """Points on a cylindrical shell around *axis_hat*."""
    axis_hat = axis_hat / np.linalg.norm(axis_hat)
    e1 = leads._radial_hat(np.array([0.13, 0.29, 0.47]), axis_hat)
    e2 = np.cross(axis_hat, e1)
    ang = np.linspace(0.0, 2 * np.pi, n_ang, endpoint=False)
    ax = np.linspace(-length / 2, length / 2, n_ax)
    ring = radius * (np.outer(np.cos(ang), e1) + np.outer(np.sin(ang), e2))
    return np.vstack([ring + a * axis_hat for a in ax])


# ---------------------------------------------------------------------------
# _axial_coord / _radial_vec / _radial_hat
# ---------------------------------------------------------------------------

def test_axial_coord_of_a_single_point():
    assert leads._axial_coord(np.array([1.0, 2.0, 3.0]), Z) == pytest.approx(3.0)


def test_axial_coord_of_a_point_array():
    pts = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -5.0]])

    assert leads._axial_coord(pts, Z) == pytest.approx([2.0, -5.0])


def test_radial_vec_removes_the_axial_component():
    pts = np.array([[3.0, 4.0, 7.0], [0.0, 0.0, 2.0]])

    radial = leads._radial_vec(pts, Z)

    assert radial.flatten() == pytest.approx([3.0, 4.0, 0.0, 0.0, 0.0, 0.0])


def test_radial_vec_accepts_a_single_point():
    assert leads._radial_vec(np.array([3.0, 4.0, 7.0]), Z) == pytest.approx(
        [3.0, 4.0, 0.0])


def test_radial_vec_for_a_bore_along_x():
    assert leads._radial_vec(np.array([9.0, 0.0, 2.0]), X) == pytest.approx(
        [0.0, 0.0, 2.0])


def test_radial_hat_is_a_unit_radial_direction():
    r_hat = leads._radial_hat(np.array([3.0, 4.0, 10.0]), Z)

    assert np.linalg.norm(r_hat) == pytest.approx(1.0)
    assert r_hat == pytest.approx([0.6, 0.8, 0.0])
    assert np.dot(r_hat, Z) == pytest.approx(0.0)


def test_radial_hat_falls_back_on_the_axis_for_an_on_axis_point():
    """A point on the bore axis has no radial direction; a seed is used."""
    r_hat = leads._radial_hat(np.array([0.0, 0.0, 5.0]), Z)

    assert np.linalg.norm(r_hat) == pytest.approx(1.0)
    assert np.dot(r_hat, Z) == pytest.approx(0.0, abs=1e-12)


def test_radial_hat_fallback_avoids_a_seed_parallel_to_the_axis():
    r_hat = leads._radial_hat(np.array([5.0, 0.0, 0.0]), Z * 1.0)
    assert np.dot(r_hat, Z) == pytest.approx(0.0)

    # axis == +Z means the +Z seed is unusable and +Y must be picked instead
    on_axis = leads._radial_hat(np.zeros(3), Z)
    assert on_axis == pytest.approx([0.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# _snap_to_shell
# ---------------------------------------------------------------------------

def test_snap_to_shell_rescales_only_the_radial_component():
    snapped = leads._snap_to_shell(np.array([0.03, 0.0, 0.12]), Z, 0.05)

    assert snapped == pytest.approx([0.05, 0.0, 0.12])


def test_snap_to_shell_pushes_outward_and_inward():
    out = leads._snap_to_shell(np.array([0.01, 0.0, 0.0]), Z, 0.05)
    inward = leads._snap_to_shell(np.array([0.20, 0.0, 0.0]), Z, 0.05)

    assert np.linalg.norm(out) == pytest.approx(0.05)
    assert np.linalg.norm(inward) == pytest.approx(0.05)


def test_snap_to_shell_handles_an_on_axis_point():
    snapped = leads._snap_to_shell(np.array([0.0, 0.0, 0.1]), Z, 0.05)

    assert leads._axial_coord(snapped, Z) == pytest.approx(0.1)
    assert np.linalg.norm(leads._radial_vec(snapped, Z)) == pytest.approx(0.05)


def test_snap_to_shell_for_a_bore_along_x():
    snapped = leads._snap_to_shell(np.array([0.1, 0.0, 0.02]), X, 0.04)

    assert snapped == pytest.approx([0.1, 0.0, 0.04])


# ---------------------------------------------------------------------------
# _tangent_offset / _unit
# ---------------------------------------------------------------------------

def test_tangent_offset_removes_the_radial_component():
    at = np.array([0.05, 0.0, 0.0])

    tangent = leads._tangent_offset(np.array([1.0, 2.0, 3.0]), at, Z)

    assert tangent == pytest.approx([0.0, 2.0, 3.0])


def test_tangent_offset_of_a_purely_radial_vector_is_zero():
    at = np.array([0.05, 0.0, 0.0])

    assert leads._tangent_offset(np.array([2.0, 0.0, 0.0]), at, Z) == pytest.approx(
        np.zeros(3))


def test_unit_normalizes():
    assert leads._unit(np.array([0.0, 3.0, 4.0])) == pytest.approx([0.0, 0.6, 0.8])


def test_unit_uses_the_fallback_for_a_degenerate_vector():
    assert leads._unit(np.zeros(3), fallback=np.array([0.0, 0.0, 2.0])) == pytest.approx(
        [0.0, 0.0, 1.0])


def test_unit_raises_without_a_fallback():
    with pytest.raises(ValueError, match='without fallback'):
        leads._unit(np.zeros(3))


def test_unit_raises_when_the_fallback_is_also_degenerate():
    with pytest.raises(ValueError, match='Fallback vector is also degenerate'):
        leads._unit(np.zeros(3), fallback=np.zeros(3))


# ---------------------------------------------------------------------------
# _circumferential_unit / _project_exit_tangent
# ---------------------------------------------------------------------------

def test_circumferential_unit_is_tangent_to_the_cylinder():
    at = np.array([0.05, 0.0, 0.0])

    circ = leads._circumferential_unit(np.array([0.2, 1.0, 0.0]), at, Z)

    assert np.linalg.norm(circ) == pytest.approx(1.0)
    assert circ == pytest.approx([0.0, 1.0, 0.0])


def test_circumferential_unit_rejects_a_purely_radial_tangent():
    with pytest.raises(ValueError, match='no component along the loop'):
        leads._circumferential_unit(np.array([1.0, 0.0, 0.0]),
                                    np.array([0.05, 0.0, 0.0]), Z)


def test_project_exit_tangent_keeps_an_axial_exit_axial():
    tangent = leads._project_exit_tangent(Z, np.array([0.05, 0.0, 0.0]), Z)

    assert tangent == pytest.approx([0.0, 0.0, 1.0])


def test_project_exit_tangent_falls_back_when_the_exit_is_radial():
    tangent = leads._project_exit_tangent(np.array([1.0, 0.0, 0.0]),
                                          np.array([0.05, 0.0, 0.0]), Z)

    assert tangent == pytest.approx([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# _estimate_shell_radius / _pca_tangent
# ---------------------------------------------------------------------------

def test_estimate_shell_radius_recovers_the_cylinder_radius():
    verts = _cylinder_points(Z, radius=0.037)
    apex = verts[np.argmax(verts @ Z)]

    assert leads._estimate_shell_radius(verts, apex, Z) == pytest.approx(0.037)


def test_estimate_shell_radius_falls_back_to_nearest_vertices():
    """With a tiny sample radius fewer than 8 neighbours are found."""
    verts = _cylinder_points(X, radius=0.05)
    apex = verts[0]

    r = leads._estimate_shell_radius(verts, apex, X, sample_radius=1e-6)

    assert r == pytest.approx(0.05)


def test_pca_tangent_finds_the_local_wire_direction():
    line = np.column_stack([np.linspace(-0.01, 0.01, 21),
                            np.zeros(21), np.zeros(21)])

    tangent = leads._pca_tangent(line, np.zeros(3), 0.005)

    assert abs(np.dot(tangent, X)) == pytest.approx(1.0)


def test_pca_tangent_falls_back_to_the_12_nearest_points():
    line = np.column_stack([np.linspace(-0.01, 0.01, 21),
                            np.zeros(21), np.zeros(21)])

    tangent = leads._pca_tangent(line, np.zeros(3), 1e-9)

    assert abs(np.dot(tangent, X)) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _find_apex (the axis-awareness fix)
# ---------------------------------------------------------------------------

def test_find_apex_picks_the_outermost_vertex_inside_the_sector():
    verts = _cylinder_points(X, radius=0.04, length=0.2)

    apex_v, apex = leads._find_apex(verts, lead_dir=X, axis_hat=X,
                                    sector_ref_dir=Z, sector_angular_half=0.4)

    assert apex is verts[apex_v] or np.allclose(apex, verts[apex_v])
    assert leads._axial_coord(apex, X) == pytest.approx(0.1)     # furthest along +X
    radial_hat = leads._radial_hat(apex, X)
    assert np.dot(radial_hat, Z) > np.cos(0.4)                   # inside the wedge


def test_find_apex_respects_the_reference_direction():
    verts = _cylinder_points(X, radius=0.04)

    _, apex_plus = leads._find_apex(verts, X, X, Z, 0.4)
    _, apex_minus = leads._find_apex(verts, X, X, -Z, 0.4)

    assert np.dot(leads._radial_hat(apex_plus, X), Z) > 0.9
    assert np.dot(leads._radial_hat(apex_minus, X), Z) < -0.9


def test_find_apex_uses_the_lead_direction_sign():
    verts = _cylinder_points(X, radius=0.04, length=0.2)

    _, apex = leads._find_apex(verts, lead_dir=-X, axis_hat=X,
                               sector_ref_dir=Z, sector_angular_half=0.4)

    assert leads._axial_coord(apex, X) == pytest.approx(-0.1)


def test_find_apex_warns_and_falls_back_when_the_sector_is_empty(capsys):
    verts = _cylinder_points(X, radius=0.04, n_ang=4)

    _, apex = leads._find_apex(verts, X, X, sector_ref_dir=Z,
                               sector_angular_half=1e-6)

    assert 'apex sector empty' in capsys.readouterr().out
    assert leads._axial_coord(apex, X) == pytest.approx(0.1)


def test_find_apex_ignores_an_axial_component_of_the_reference_direction():
    """sector_ref_dir is projected into the radial plane before comparison."""
    verts = _cylinder_points(X, radius=0.04)

    _, straight = leads._find_apex(verts, X, X, Z, 0.4)
    _, tilted = leads._find_apex(verts, X, X, Z + 5.0 * X, 0.4)

    assert tilted == pytest.approx(straight)


# ---------------------------------------------------------------------------
# _perp_distance_to_loop
# ---------------------------------------------------------------------------

def test_perp_distance_to_loop_ignores_travel_along_the_loop():
    apex = np.zeros(3)
    pts = np.array([[0.0, 0.01, 0.0], [0.003, 0.05, 0.0], [0.0, 0.0, 0.004]])

    d = leads._perp_distance_to_loop(pts, apex, Y)

    assert d == pytest.approx([0.0, 0.003, 0.004])


# ---------------------------------------------------------------------------
# _loop_plane / ring helpers
# ---------------------------------------------------------------------------

def test_loop_plane_of_a_planar_ring():
    ang = np.linspace(0.0, 2 * np.pi, 12, endpoint=False)
    ring = np.column_stack([0.002 * np.cos(ang), 0.002 * np.sin(ang),
                            np.full(12, 0.05)])

    center, normal, ax1, ax2 = leads._loop_plane(ring)

    assert center == pytest.approx([0.0, 0.0, 0.05])
    assert abs(np.dot(normal, Z)) == pytest.approx(1.0)
    for a, b in ((ax1, ax2), (normal, ax1), (normal, ax2)):
        assert np.dot(a, b) == pytest.approx(0.0, abs=1e-9)
    assert np.linalg.norm(ax1) == pytest.approx(1.0)


def test_loop_plane_falls_back_to_svd_for_a_degenerate_newell_normal():
    """Collinear points give a zero Newell normal; SVD supplies the plane."""
    ring = np.column_stack([np.linspace(-1.0, 1.0, 5), np.zeros(5), np.zeros(5)])

    _, normal, ax1, ax2 = leads._loop_plane(ring)

    assert np.linalg.norm(normal) == pytest.approx(1.0)
    assert np.dot(normal, X) == pytest.approx(0.0, abs=1e-9)


def test_ordered_ring_2d_sorts_by_polar_angle():
    ang = np.array([2.0, 0.1, -1.0, 3.0])
    ring = np.column_stack([np.cos(ang), np.sin(ang), np.zeros(4)])

    pts = leads._ordered_ring_2d(ring, np.zeros(3), X, Y)

    angles = np.arctan2(pts[:, 1], pts[:, 0])
    assert np.all(np.diff(angles) > 0)


def test_rotate_ordered_ring_preserves_connectivity_order():
    ang = np.linspace(0.0, 2 * np.pi, 8, endpoint=False)
    ring = np.column_stack([np.cos(ang), np.sin(ang), np.zeros(8)])
    idx = list(range(100, 108))

    new_idx, new_pts = leads._rotate_ordered_ring(idx, ring, np.zeros(3), X, Y)

    assert set(new_idx) == set(idx)
    assert len(new_pts) == len(ring)
    # rotation only, so the cyclic order is unchanged
    k = new_idx.index(100)
    assert new_idx[k:] + new_idx[:k] == idx
    assert new_pts[0] == pytest.approx(ring[idx.index(new_idx[0])])


def test_resample_closed_2d_keeps_the_perimeter_and_point_count():
    square = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])

    out = leads._resample_closed_2d(square, 16)

    assert out.shape == (16, 2)
    assert np.max(np.abs(out)) == pytest.approx(1.0)


def test_resample_closed_2d_handles_fewer_than_three_points():
    out = leads._resample_closed_2d(np.array([[1.0, 0.0], [0.0, 1.0]]), 6)

    assert out.shape == (6, 2)


def test_resample_closed_2d_handles_a_zero_length_ring():
    out = leads._resample_closed_2d(np.zeros((5, 2)), 7)

    assert out.shape == (7, 2)
    assert out == pytest.approx(np.zeros((7, 2)))


def test_align_oval_to_cut_matches_the_first_vertex_angle():
    ang = np.linspace(0.0, 2 * np.pi, 24, endpoint=False)
    oval = np.column_stack([0.002 * np.sin(ang), 0.001 * np.cos(ang)])
    cut = np.column_stack([0.002 * np.cos(ang + 0.7), 0.002 * np.sin(ang + 0.7)])

    aligned = leads._align_oval_to_cut(oval, cut)

    assert len(aligned) == len(cut)
    # Rotation is rigid, so radii stay inside the original oval's envelope.
    radii = np.linalg.norm(aligned, axis=1)
    assert radii.min() >= np.linalg.norm(oval, axis=1).min() - 1e-12
    assert radii.max() <= np.linalg.norm(oval, axis=1).max() + 1e-12
    # ``oval @ [[c, -s], [s, c]]`` rotates row vectors by -rot, so the first
    # vertex lands at 2*a_oval - a_cut instead of on a_cut itself.
    resampled = leads._resample_closed_2d(oval, len(cut))
    a_cut = np.arctan2(cut[0, 1], cut[0, 0])
    a_oval = np.arctan2(resampled[0, 1], resampled[0, 0])
    expected = 2 * a_oval - a_cut
    assert np.arctan2(aligned[0, 1], aligned[0, 0]) == pytest.approx(
        np.arctan2(np.sin(expected), np.cos(expected)))


# ---------------------------------------------------------------------------
# _rotation_minimizing_frames
# ---------------------------------------------------------------------------

def test_rmf_frames_are_orthonormal_along_a_curved_path():
    t = np.linspace(0.0, 1.0, 25)
    path = np.column_stack([t, t ** 2, np.zeros_like(t)])

    T, N, B = leads._rotation_minimizing_frames(path)

    assert len(T) == len(path)
    for i in range(len(path)):
        assert np.linalg.norm(T[i]) == pytest.approx(1.0)
        assert np.linalg.norm(N[i]) == pytest.approx(1.0)
        assert np.linalg.norm(B[i]) == pytest.approx(1.0)
        assert np.dot(T[i], N[i]) == pytest.approx(0.0, abs=1e-9)
        assert np.dot(T[i], B[i]) == pytest.approx(0.0, abs=1e-9)


def test_rmf_frames_are_twist_free_on_a_straight_path():
    path = np.column_stack([np.linspace(0.0, 1.0, 10), np.zeros(10), np.zeros(10)])

    _, N, _ = leads._rotation_minimizing_frames(path)

    for i in range(1, len(N)):
        assert np.dot(N[i], N[0]) == pytest.approx(1.0)


def test_rmf_frames_repeat_the_frame_across_duplicate_points():
    path = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                     [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    _, N, B = leads._rotation_minimizing_frames(path)

    assert N[1] == pytest.approx(N[0])
    assert B[1] == pytest.approx(B[0])


# ---------------------------------------------------------------------------
# _shell_bezier / _common_tip / _shell_radius_of_points
# ---------------------------------------------------------------------------

def test_shell_bezier_stays_on_the_shell_and_hits_both_endpoints():
    p0 = np.array([0.04, 0.0, 0.0])
    p3 = np.array([0.0, 0.04, 0.05])

    path = leads._shell_bezier(p0, p0 + Z * 0.01, p3 - Z * 0.01, p3, Z, 0.04, 25)

    assert len(path) == 25
    assert path[0] == pytest.approx(p0)
    assert path[-1] == pytest.approx(p3)
    assert leads._shell_radius_of_points(path, Z) == pytest.approx(
        np.full(25, 0.04))


def test_common_tip_lands_on_the_exit_plane_at_the_shell_radius():
    anchor = np.array([0.04, 0.0, 0.02])

    tip = leads._common_tip(0.12, anchor, Z, 0.04, fan_offset=np.array([0.0, 0.003, 0.0]))

    assert leads._axial_coord(tip, Z) == pytest.approx(0.12)
    assert np.linalg.norm(leads._radial_vec(tip, Z)) == pytest.approx(0.04)
    assert tip[1] > 0.0                                   # fanned along +Y


def test_common_tip_without_a_fan_offset_keeps_the_anchor_angle():
    anchor = np.array([0.04, 0.0, 0.02])

    tip = leads._common_tip(0.1, anchor, Z, 0.04, fan_offset=np.zeros(3))

    assert tip == pytest.approx([0.04, 0.0, 0.1])


def test_common_tip_ignores_a_radial_fan_offset():
    anchor = np.array([0.04, 0.0, 0.0])

    tip = leads._common_tip(0.1, anchor, Z, 0.04, fan_offset=np.array([0.01, 0.0, 0.0]))

    assert tip == pytest.approx([0.04, 0.0, 0.1])


def test_two_leads_share_the_same_exit_plane():
    """Equal-length leads require both tips on one axial plane."""
    a0 = np.array([0.04, 0.0, 0.01])
    a1 = np.array([0.039, 0.008, -0.01])

    t0 = leads._common_tip(0.15, a0, Z, 0.04, np.array([0.0, 0.004, 0.0]))
    t1 = leads._common_tip(0.15, a1, Z, 0.04, np.array([0.0, -0.004, 0.0]))

    assert leads._axial_coord(t0, Z) == pytest.approx(leads._axial_coord(t1, Z))


# ---------------------------------------------------------------------------
# _surface_frame / _route_frame
# ---------------------------------------------------------------------------

def test_surface_frame_is_orthonormal_and_points_away_from_the_gap():
    verts = _cylinder_points(Z, radius=0.04, n_ang=64, n_ax=1)
    center = np.array([0.04, 0.0, 0.0])
    ref_center = np.array([0.04 * np.cos(-0.1), 0.04 * np.sin(-0.1), 0.0])

    n_hat, t_hat, b_hat = leads._surface_frame(
        center, ref_center, verts, Z, tangent_radius=0.01)

    assert n_hat == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)
    assert np.dot(t_hat, n_hat) == pytest.approx(0.0, abs=1e-6)
    assert np.dot(b_hat, t_hat) == pytest.approx(0.0, abs=1e-6)
    assert np.linalg.norm(b_hat) == pytest.approx(1.0)
    away = leads._tangent_offset(center - ref_center, center, Z)
    assert np.dot(t_hat, away) > 0.0


def test_surface_frame_uses_the_fallback_tangent_when_the_gap_is_degenerate():
    verts = _cylinder_points(Z, radius=0.04, n_ang=64, n_ax=1)
    center = np.array([0.04, 0.0, 0.0])

    _, t_hat, _ = leads._surface_frame(center, center, verts, Z, 0.01,
                                       fallback_tangent=np.array([0.0, -1.0, 0.0]))

    assert np.dot(t_hat, np.array([0.0, -1.0, 0.0])) > 0.0


def test_route_frame_separates_the_route_and_side_directions():
    center = np.array([0.04, 0.0, 0.02])
    ref_center = np.array([0.04 * np.cos(-0.2), 0.04 * np.sin(-0.2), 0.02])

    route_dir, side_dir = leads._route_frame(center, ref_center, exit_dir=Z,
                                             axis_hat=Z,
                                             wire_tangent=np.array([0.0, 1.0, 0.0]))

    assert route_dir == pytest.approx([0.0, 0.0, 1.0])
    assert np.dot(route_dir, side_dir) == pytest.approx(0.0, abs=1e-9)
    assert np.linalg.norm(side_dir) == pytest.approx(1.0)


def test_route_frame_uses_the_wire_tangent_when_the_faces_coincide():
    center = np.array([0.04, 0.0, 0.0])

    route_dir, side_dir = leads._route_frame(center, center, Z, Z,
                                             wire_tangent=np.array([0.0, 1.0, 0.0]))

    assert route_dir == pytest.approx([0.0, 0.0, 1.0])
    assert side_dir == pytest.approx([0.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# _pycoilgen_oval_profile / _populate_params
# ---------------------------------------------------------------------------

def test_populate_params_copies_the_config_into_the_module_scalars():
    cfg = Config(gradient_axis='y')

    leads._populate_params(cfg)

    assert leads._P.conductor_width == cfg.wire.conductor_width
    assert leads._P.conductor_semi_a == cfg.conductor_semi_a
    assert leads._P.conductor_semi_b == cfg.conductor_semi_b
    assert leads._P.cross_section_n == cfg.wire.cross_section_n
    assert leads._P.cut_loop_length == cfg.leads.cut_loop_length
    assert leads._P.tip_fan == cfg.leads.tip_fan
    assert leads._P.lead_steps == cfg.leads.lead_steps


def test_oval_profile_matches_the_configured_conductor_cross_section():
    cfg = Config(gradient_axis='y')
    leads._populate_params(cfg)

    profile = leads._pycoilgen_oval_profile()

    template = profile['template_2d']
    assert template.shape == (cfg.wire.cross_section_n, 2)
    assert profile['semi_a'] == cfg.conductor_semi_a
    assert profile['semi_b'] == cfg.conductor_semi_b
    assert np.max(np.abs(template[:, 0])) <= cfg.conductor_semi_a + 1e-12
    assert np.max(np.abs(template[:, 1])) <= cfg.conductor_semi_b + 1e-12
    radii = np.linalg.norm(template, axis=1)
    assert profile['cs_mean'] == pytest.approx(radii.mean())
    assert profile['cs_span'] == pytest.approx(radii.max() - radii.min())
    # closed ring: first and last samples coincide
    assert template[0] == pytest.approx(template[-1])


def test_ring_cross_sections_returns_matched_cut_and_oval_profiles():
    cfg = Config(gradient_axis='y')
    leads._populate_params(cfg)
    n = cfg.wire.cross_section_n
    ang = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    ring = np.column_stack([0.0015 * np.cos(ang), 0.0015 * np.sin(ang),
                            np.full(n, 0.04)])

    cut_2d, oval_2d, cs_mean, cs_span = leads._ring_cross_sections(
        ring, leads._pycoilgen_oval_profile())

    assert len(cut_2d) == n
    assert len(oval_2d) == n
    assert cs_mean > 0.0
    assert cs_span >= 0.0


def test_profile_ring_2d_resamples_to_the_requested_point_count():
    cfg = Config(gradient_axis='y')
    leads._populate_params(cfg)
    ang = np.linspace(0.0, 2 * np.pi, 9, endpoint=False)
    ring = np.column_stack([0.0015 * np.cos(ang), 0.0015 * np.sin(ang),
                            np.zeros(9)])

    oval_2d, cs_mean, cs_span, cut_2d = leads._profile_ring_2d(
        leads._pycoilgen_oval_profile(), ring, 20)

    assert oval_2d.shape == (20, 2)
    assert cut_2d.shape == (20, 2)
    assert cs_mean > 0.0


def test_profile_ring_2d_in_frame_expresses_the_cut_in_the_local_frame():
    cfg = Config(gradient_axis='y')
    leads._populate_params(cfg)
    ang = np.linspace(0.0, 2 * np.pi, 12, endpoint=False)
    ring = np.column_stack([np.full(12, 0.04),
                            0.0015 * np.cos(ang), 0.0015 * np.sin(ang)])

    oval_2d, mean_r, span_r, cut_2d = leads._profile_ring_2d_in_frame(
        leads._pycoilgen_oval_profile(), ring, normal=Y, binormal=Z, n_pts=12)

    assert oval_2d.shape == (12, 2)
    assert cut_2d.shape == (12, 2)
    assert np.linalg.norm(cut_2d, axis=1) == pytest.approx(np.full(12, 0.0015))
    assert mean_r > 0.0
    assert span_r >= 0.0


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

def test_boundary_adjacency_of_a_closed_mesh_has_no_boundary_edges():
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))

    be, adj = leads._boundary_adjacency(mesh)

    assert len(be) == 0
    assert len(adj) == 0


def test_boundary_loops_finds_one_loop_per_hole():
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    open_mesh = mesh.submesh([np.arange(len(mesh.faces) - 2)], append=True)

    loops, adj = leads._boundary_loops(open_mesh)

    assert len(loops) == 1
    assert len(loops[0]) >= 3
    assert all(len(adj[v]) >= 1 for v in loops[0])


def test_boundary_loops_returns_nothing_for_a_watertight_mesh():
    loops, _ = leads._boundary_loops(trimesh.creation.box(extents=(1.0, 1.0, 1.0)))

    assert loops == []


def test_order_loop_walks_the_boundary_in_ring_order():
    adj = {0: {1, 3}, 1: {0, 2}, 2: {1, 3}, 3: {2, 0}}

    ordered = leads._order_loop([0, 1, 2, 3], adj)

    assert len(ordered) == 4
    assert set(ordered) == {0, 1, 2, 3}
    for a, b in zip(ordered, ordered[1:]):
        assert b in adj[a]


def test_order_loop_returns_short_loops_unchanged():
    assert leads._order_loop([7], {}) == [7]


def test_order_loop_stops_at_a_dead_end():
    adj = {0: {1}, 1: {0}, 2: set()}

    ordered = leads._order_loop([0, 1, 2], adj)

    assert ordered == [0, 1]


def test_cap_ring_builds_a_fan_around_the_ring_centroid():
    ang = np.linspace(0.0, 2 * np.pi, 8, endpoint=False)
    ring = np.column_stack([np.cos(ang), np.sin(ang), np.zeros(8)])

    verts, faces = leads._cap_ring(ring)

    assert len(verts) == 9
    assert verts[-1] == pytest.approx(ring.mean(axis=0))
    assert faces.shape == (8, 3)
    assert set(faces[:, 0]) == {8}
    assert sorted(faces[:, 1]) == list(range(8))


def test_shell_radius_of_points_matches_the_cylinder_radius():
    pts = _cylinder_points(X, radius=0.033, n_ang=8, n_ax=3)

    radii = leads._shell_radius_of_points(pts, X)

    assert radii == pytest.approx(np.full(len(pts), 0.033))
