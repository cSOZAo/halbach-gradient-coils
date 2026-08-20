from types import SimpleNamespace

import numpy as np

from halbach_coils.coilgen.config import Config
from halbach_coils.coilgen.overlap import detect_collisions
from pyCoilGen.sub_functions.shift_return_paths import intersection_points_for_pairs


def _solution(points, segments, positions, layers):
    part = SimpleNamespace(
        crossing_points_uv=np.asarray(points, dtype=float),
        crossing_segments=np.asarray(segments, dtype=int),
        crossing_path_positions=np.asarray(positions, dtype=float),
        crossing_layer_factors=np.asarray(layers, dtype=float),
    )
    return SimpleNamespace(coil_parts=[part])


def test_intersection_points_correspond_to_segment_pairs():
    track = np.array([
        [-1.0, 1.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, -1.0, 1.0],
    ])
    points = intersection_points_for_pairs(track, np.array([[0, 3]]))
    np.testing.assert_allclose(points, [[0.0, 0.0]])


def test_ordinary_two_cable_crossing_does_not_alert():
    solution = _solution(
        points=[[0.0, 0.0]],
        segments=[[10, 100]],
        positions=[[0.1, 1.0]],
        layers=[[0.0, 1.0]],
    )
    report = detect_collisions(solution, Config())
    assert report.n_collisions == 0


def test_same_inner_layer_x_reports_one_three_cable_collision():
    solution = _solution(
        points=[[0.0, 0.0]],
        segments=[[100, 200]],
        positions=[[1.0, 2.0]],
        layers=[[1.0, 1.0]],
    )
    report = detect_collisions(solution, Config())

    assert report.n_collisions == 1
    assert report.max_cables == 3
    assert report.sites[0].cable_count == 3
    assert report.sites[0].inner_layer_count == 2
    question = report.user_question()
    assert "Hay 3 cables intentando pasar por el mismo lugar" in question
    assert "1 quedará en la capa externa y 2 quedarán en la interna" in question
    assert "¿Desea continuar o descartar el resultado?" in question

    english_question = report.user_question("en")
    assert "1 will remain in the outer layer and 2 in the inner layer" in english_question
    assert "Do you want to continue or discard the result?" in english_question


def test_nearby_projected_crossings_only_count_actual_same_layer_x():
    solution = _solution(
        # Regression for the default Gx result: several projected crossings
        # are close, but only the pair assigned to the same inner layer is a
        # physical collision.
        points=[[0.0, 0.0], [0.0095, 0.0], [0.0190, 0.0]],
        segments=[[10, 100], [11, 200], [12, 300]],
        positions=[[0.100, 1.0], [0.102, 2.0], [0.104, 3.0]],
        layers=[[0.0, 1.0], [1.0, 1.0], [0.0, 1.0]],
    )
    report = detect_collisions(solution, Config())
    assert report.n_collisions == 1
    assert report.sites[0].cable_count == 3


def test_separate_two_cable_crossings_do_not_form_a_false_multiwire_site():
    solution = _solution(
        points=[[0.0, 0.0], [0.020, 0.0]],
        segments=[[10, 100], [200, 300]],
        positions=[[0.1, 1.0], [2.0, 3.0]],
        layers=[[0.0, 1.0], [0.0, 1.0]],
    )
    report = detect_collisions(solution, Config())
    assert report.n_collisions == 0
