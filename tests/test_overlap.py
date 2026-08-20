"""Unit tests for :mod:`coilgen.overlap`."""

import builtins
from types import SimpleNamespace

import numpy as np
import pytest

from coilgen import overlap
from coilgen.config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _part(points):
    """Fake coil part whose ``wire_path.v`` is a (3, N) polyline."""
    v = None if points is None else np.asarray(points, dtype=float).T
    return SimpleNamespace(wire_path=SimpleNamespace(v=v))


def _solution(*parts):
    return SimpleNamespace(coil_parts=list(parts))


def _cfg(clearance=1.0, conductor_width=0.002):
    cfg = Config(gradient_axis='y')
    cfg.overlap_clearance = clearance
    cfg.wire.conductor_width = conductor_width
    return cfg


def _brute_force_distance(p0, p1, q0, q1, n=2001):
    s = np.linspace(0, 1, n)[:, None]
    a = p0 + s * (p1 - p0)
    b = q0 + s * (q1 - q0)
    return float(np.min(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)))


# ---------------------------------------------------------------------------
# _segment_segment_distance
# ---------------------------------------------------------------------------

def _d(p0, p1, q0, q1):
    return overlap._segment_segment_distance(
        *[np.asarray(v, dtype=float) for v in (p0, p1, q0, q1)])


def test_distance_between_degenerate_segments_is_point_distance():
    assert np.isclose(_d([0, 0, 0], [0, 0, 0], [3, 4, 0], [3, 4, 0]), 5.0)


def test_distance_point_to_segment_projects_inside():
    # Degenerate first segment above the middle of the second.
    assert np.isclose(_d([0.5, 1.0, 0], [0.5, 1.0, 0], [0, 0, 0], [1, 0, 0]), 1.0)


def test_distance_point_to_segment_clamps_past_the_end():
    assert np.isclose(_d([3.0, 0, 0], [3.0, 0, 0], [0, 0, 0], [1, 0, 0]), 2.0)


def test_distance_segment_to_degenerate_second_segment():
    assert np.isclose(_d([0, 0, 0], [1, 0, 0], [0.5, 0, 2.0], [0.5, 0, 2.0]), 2.0)


def test_distance_between_parallel_segments_is_the_offset():
    assert np.isclose(_d([0, 0, 0], [1, 0, 0], [0, 0.25, 0], [1, 0.25, 0]), 0.25)


def test_distance_between_collinear_disjoint_segments():
    assert np.isclose(_d([0, 0, 0], [1, 0, 0], [3, 0, 0], [4, 0, 0]), 2.0)


def test_distance_is_zero_for_intersecting_segments():
    assert np.isclose(_d([-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]), 0.0)


def test_distance_between_skew_segments_is_the_common_perpendicular():
    assert np.isclose(_d([-1, 0, 0], [1, 0, 0], [0, -1, 0.5], [0, 1, 0.5]), 0.5)


def test_distance_is_symmetric_and_clamped_to_endpoints():
    p0, p1 = [0, 0, 0], [1, 0, 0]
    q0, q1 = [2, 1, 0], [3, 2, 0]

    d = _d(p0, p1, q0, q1)

    assert np.isclose(d, np.hypot(1.0, 1.0))                       # (1,0,0) -> (2,1,0)
    assert np.isclose(d, _d(q0, q1, p0, p1))


@pytest.mark.parametrize('q0,q1', [
    ([0.2, 0.3, 0.1], [0.9, -0.4, 0.6]),
    ([2.0, 0.1, 0.0], [2.5, 0.1, 0.9]),
    ([0.0, 0.5, 0.0], [1.0, 0.5, 0.0]),
    ([-1.0, -1.0, -1.0], [0.5, 0.2, 0.3]),
])
def test_distance_matches_brute_force_sampling(q0, q1):
    p0, p1 = np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
    q0, q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)

    assert np.isclose(_d(p0, p1, q0, q1),
                      _brute_force_distance(p0, p1, q0, q1), atol=1e-3)


# ---------------------------------------------------------------------------
# _collect_segments
# ---------------------------------------------------------------------------

def test_collect_segments_builds_segments_and_adjacency():
    solution = _solution(_part([[0, 0, 0], [1, 0, 0], [2, 0, 0]]),
                         _part([[0, 1, 0], [1, 1, 0]]))

    starts, ends, adjacency = overlap._collect_segments(solution)

    assert starts.shape == (3, 3)
    assert ends.shape == (3, 3)
    assert np.allclose(starts[0], [0, 0, 0])
    assert np.allclose(ends[2], [1, 1, 0])
    # Only the two segments of the first part are adjacent; parts do not chain.
    assert adjacency == [(0, 1)]


def test_collect_segments_skips_empty_and_too_short_parts():
    solution = _solution(_part(None),
                         _part([[0, 0, 0]]),
                         SimpleNamespace(),                        # no wire_path attribute
                         _part([[0, 0, 0], [1, 0, 0]]))

    starts, ends, adjacency = overlap._collect_segments(solution)

    assert starts.shape == (1, 3)
    assert adjacency == []


def test_collect_segments_returns_empty_arrays_without_usable_parts():
    starts, ends, adjacency = overlap._collect_segments(_solution(_part(None)))

    assert starts.shape == (0, 3)
    assert ends.shape == (0, 3)
    assert adjacency == []


# ---------------------------------------------------------------------------
# detect_collisions
# ---------------------------------------------------------------------------

def test_threshold_is_clearance_times_conductor_width():
    cfg = _cfg(clearance=1.5, conductor_width=0.002)
    solution = _solution(_part([[0, 0, 0], [0.05, 0, 0]]))

    report = overlap.detect_collisions(solution, cfg)

    assert np.isclose(report.threshold_m, 0.003)
    assert report.n_collisions == 0
    assert report.min_distance_m == float('inf')
    assert report.pairs == []


def test_no_collision_reported_for_well_separated_wires():
    cfg = _cfg(clearance=1.0, conductor_width=0.002)
    solution = _solution(_part([[0, 0, 0], [0.05, 0, 0]]),
                         _part([[0, 0.02, 0], [0.05, 0.02, 0]]))

    report = overlap.detect_collisions(solution, cfg)

    assert report.n_collisions == 0
    assert np.isclose(report.min_distance_m, 0.02)


def test_collision_reported_for_wires_closer_than_the_conductor_width():
    cfg = _cfg(clearance=1.0, conductor_width=0.002)          # threshold 2 mm
    solution = _solution(_part([[0, 0, 0], [0.05, 0, 0]]),
                         _part([[0, 0.0005, 0], [0.05, 0.0005, 0]]))

    report = overlap.detect_collisions(solution, cfg)

    assert report.n_collisions == 1
    assert np.isclose(report.min_distance_m, 0.0005)
    i, j, distance = report.pairs[0]
    assert (i, j) == (0, 1)
    assert np.isclose(distance, 0.0005)


def test_adjacent_segments_of_one_part_are_never_collisions():
    cfg = _cfg(clearance=1.0, conductor_width=0.01)           # threshold 10 mm
    # A tight hairpin: the two consecutive segments touch at the shared vertex.
    solution = _solution(_part([[0, 0, 0], [0.05, 0, 0], [0, 0.0005, 0]]))

    report = overlap.detect_collisions(solution, cfg)

    assert report.n_collisions == 0
    assert report.min_distance_m == float('inf')


def test_pairs_are_sorted_by_increasing_distance():
    cfg = _cfg(clearance=1.0, conductor_width=0.01)           # threshold 10 mm
    solution = _solution(_part([[0, 0, 0], [0.05, 0, 0]]),
                         _part([[0, 0.004, 0], [0.05, 0.004, 0]]),
                         _part([[0, 0.001, 0], [0.05, 0.001, 0]]))

    report = overlap.detect_collisions(solution, cfg)

    distances = [d for _, _, d in report.pairs]
    assert distances == sorted(distances)
    assert np.isclose(distances[0], 0.001)
    assert np.isclose(report.min_distance_m, 0.001)


def test_report_is_empty_when_fewer_than_two_segments():
    cfg = _cfg()

    report = overlap.detect_collisions(_solution(_part([[0, 0, 0], [1, 0, 0]])), cfg)

    assert (report.n_collisions, report.pairs) == (0, [])
    assert report.min_distance_m == float('inf')
    assert np.isclose(report.threshold_m, cfg.overlap_clearance * cfg.wire.conductor_width)


def test_report_is_empty_for_a_solution_without_parts():
    report = overlap.detect_collisions(_solution(), _cfg())

    assert report.n_collisions == 0
    assert report.min_distance_m == float('inf')


def test_fallback_without_scipy_finds_the_same_collision(monkeypatch):
    """The O(n^2) path is used when scipy is unavailable."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'scipy.spatial':
            raise ImportError('no scipy for this test')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)

    cfg = _cfg(clearance=1.0, conductor_width=0.002)
    solution = _solution(_part([[0, 0, 0], [0.05, 0, 0]]),
                         _part([[0, 0.0005, 0], [0.05, 0.0005, 0]]))

    report = overlap.detect_collisions(solution, cfg)

    assert report.n_collisions == 1
    assert np.isclose(report.min_distance_m, 0.0005)
