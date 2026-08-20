"""Unit tests for :mod:`coilgen.geometry`."""

import numpy as np
import pytest
import trimesh

from coilgen import geometry as geo


# ---------------------------------------------------------------------------
# Elementary rotations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('rot', [geo.rotx, geo.roty, geo.rotz])
def test_rotations_are_orthonormal_with_unit_determinant(rot):
    R = rot(0.7)

    assert np.allclose(R @ R.T, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1.0)


@pytest.mark.parametrize('rot', [geo.rotx, geo.roty, geo.rotz])
def test_zero_angle_is_identity(rot):
    assert np.allclose(rot(0.0), np.eye(3))


def test_rotation_axes_and_sense():
    half = np.pi / 2
    assert np.allclose(geo.rotx(half) @ [0, 1, 0], [0, 0, 1])
    assert np.allclose(geo.roty(half) @ [0, 0, 1], [1, 0, 0])
    assert np.allclose(geo.rotz(half) @ [1, 0, 0], [0, 1, 0])


def test_rodrigues_matches_elementary_rotations():
    angle = 0.37
    assert np.allclose(geo.rodrigues_rotation_matrix((1, 0, 0), angle), geo.rotx(angle))
    assert np.allclose(geo.rodrigues_rotation_matrix((0, 1, 0), angle), geo.roty(angle))
    assert np.allclose(geo.rodrigues_rotation_matrix((0, 0, 1), angle), geo.rotz(angle))


def test_rodrigues_normalizes_the_axis():
    scaled = geo.rodrigues_rotation_matrix((0, 0, 5), 0.9)

    assert np.allclose(scaled, geo.rotz(0.9))


def test_rodrigues_leaves_its_axis_invariant():
    axis = np.array([1.0, 2.0, -3.0])
    R = geo.rodrigues_rotation_matrix(axis, 1.1)

    assert np.allclose(R @ axis, axis)
    assert np.isclose(np.linalg.det(R), 1.0)


# ---------------------------------------------------------------------------
# Axis mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('physical,internal', [('x', 'y'), ('y', 'z'), ('z', 'x')])
def test_internal_field_axis_mapping(physical, internal):
    assert geo.internal_field_axis(physical) == internal
    assert geo.internal_field_axis(physical.upper()) == internal


def test_internal_field_axis_rejects_unknown_axis():
    with pytest.raises(KeyError):
        geo.internal_field_axis('w')


def test_rotated_cylinder_axis_is_plus_x_by_default():
    assert np.allclose(geo.rotated_cylinder_axis(), [1.0, 0.0, 0.0])


def test_rotated_cylinder_axis_follows_custom_rotation():
    axis = geo.rotated_cylinder_axis(rot_axis=(1, 0, 0), rot_angle=np.pi / 2)

    assert np.allclose(axis, [0.0, -1.0, 0.0])
    assert np.isclose(np.linalg.norm(axis), 1.0)


# ---------------------------------------------------------------------------
# Target field file
# ---------------------------------------------------------------------------

def _load_target(path):
    return np.load(path, allow_pickle=True)[0]


@pytest.mark.parametrize('axis,fname,filename', [
    ('x', 'lin_1', 'OSI2_GradTarget_Lin1.npy'),
    ('y', 'lin_2', 'OSI2_GradTarget_Lin2.npy'),
    ('z', 'lin_3', 'OSI2_GradTarget_Lin3.npy'),
])
def test_build_target_field_file_names(tmp_path, axis, fname, filename):
    path, field_name = geo.build_target_field_file(
        axis, 0.1, 0.1, 0.1, 4, 6, str(tmp_path))

    assert field_name == fname
    assert path == str(tmp_path / filename)

    data = _load_target(path)
    assert set(data) == {'coords', fname}
    assert data['coords'].shape == (3, data[fname].size)
    assert data['coords'].dtype == np.float64
    assert data[fname].dtype == np.float64


def test_build_target_field_grid_size_matches_resolution(tmp_path):
    resol_radial, resol_angular = 5, 7
    path, fname = geo.build_target_field_file(
        'x', 0.1, 0.1, 0.1, resol_radial, resol_angular, str(tmp_path))

    data = _load_target(path)
    assert data[fname].size == resol_radial * resol_angular * resol_angular


def test_target_points_stay_inside_the_requested_ellipsoid(tmp_path):
    rx, ry, rz = 0.125, 0.100, 0.075
    path, _ = geo.build_target_field_file('x', rx, ry, rz, 5, 9, str(tmp_path))

    coords = _load_target(path)['coords']          # axis 'x' -> identity rotation
    normalized = (coords[0] / rx) ** 2 + (coords[1] / ry) ** 2 + (coords[2] / rz) ** 2

    assert normalized.max() <= 1.0 + 1e-9
    assert np.isclose(np.abs(coords[0]).max(), rx)


def test_rotation_relabels_the_linear_field_axis(tmp_path):
    """The stored field is linear in X; the rotation moves X onto the axis."""
    rx = 0.12
    paths = {axis: geo.build_target_field_file(axis, rx, rx, rx, 5, 9, str(tmp_path))
             for axis in ('x', 'y', 'z')}

    reference = _load_target(paths['x'][0])['lin_1']

    for axis, row in (('x', 0), ('y', 1), ('z', 2)):
        path, fname = paths[axis]
        data = _load_target(path)
        # Field values are identical across axes; only the coordinate cloud rotates.
        assert np.allclose(data[fname], reference)
        assert np.allclose(data['coords'][row], reference, atol=1e-12)


def test_build_target_field_rejects_unknown_axis(tmp_path):
    with pytest.raises(KeyError):
        geo.build_target_field_file('w', 0.1, 0.1, 0.1, 4, 6, str(tmp_path))


# ---------------------------------------------------------------------------
# STL measurement
# ---------------------------------------------------------------------------

def _write_tube_stl(path, radius, height, z_center=0.0, axis='z'):
    """Write an annulus-free tube (open cylinder shell) STL at *path*."""
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=32)
    if axis == 'x':
        mesh.apply_transform(trimesh.transformations.rotation_matrix(
            np.pi / 2, [0, 1, 0]))
        mesh.apply_translation([z_center, 0, 0])
    else:
        mesh.apply_translation([0, 0, z_center])
    mesh.export(str(path))
    return str(path)


def test_detect_fusion_cylinder_dims_spans_both_halves(tmp_path):
    # Fusion halves are in millimetres, cylinder axis along +Z.
    a = _write_tube_stl(tmp_path / 'g_2a.stl', radius=40.0, height=100.0, z_center=50.0)
    b = _write_tube_stl(tmp_path / 'g_2b.stl', radius=40.0, height=100.0, z_center=-50.0)

    dims = geo.detect_fusion_cylinder_dims(a, b)

    assert np.isclose(dims['z_min_mm'], -100.0)
    assert np.isclose(dims['z_max_mm'], 100.0)
    assert np.isclose(dims['axial_min_m'], -0.100)
    assert np.isclose(dims['axial_max_m'], 0.100)
    assert np.isclose(dims['axial_center_m'], 0.0)
    assert np.isclose(dims['axial_length_m'], 0.200)
    assert np.isclose(dims['outer_r_m'], 0.040, atol=1e-6)
    assert dims['inner_r_m'] < dims['outer_r_m']


def test_measure_wire_dims_uses_the_rotated_bore_axis(tmp_path):
    # After R_y(pi/2) the bore axis is +X, so build the tube along X.
    stl = _write_tube_stl(tmp_path / 'wire.stl', radius=0.04, height=0.30,
                          z_center=0.01, axis='x')

    dims = geo.measure_wire_dims(stl)

    assert np.isclose(dims['axial_min'], 0.01 - 0.15, atol=1e-6)
    assert np.isclose(dims['axial_max'], 0.01 + 0.15, atol=1e-6)
    assert np.isclose(dims['axial_center'], 0.01, atol=1e-6)
    assert np.isclose(dims['axial_extent'], 0.30, atol=1e-6)
    assert np.isclose(dims['outer_r'], 0.04, atol=1e-6)
    assert dims['path'] == stl


def test_format_dims_mm_reports_millimetres():
    dims = {
        'axial_min': -0.1, 'axial_max': 0.1, 'axial_center': 0.0,
        'inner_r': 0.0355, 'outer_r': 0.0395,
    }

    text = geo.format_dims_mm(dims, prefix='wire: ')

    assert text.startswith('wire: axial [-100.0, 100.0] mm')
    assert 'centre 0.00 mm' in text
    assert 'radial [35.50, 39.50]' in text
