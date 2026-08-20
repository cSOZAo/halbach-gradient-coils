"""
Unit tests for :mod:`coilgen.shell`.

Covers the mesh/manifold conversions, the voxel-pass cropping helpers, the
analytical limiter/tube builders and the Fusion-half transform (mm -> m,
recentre, rotate onto the bore axis). The full ``run_shell`` orchestration is
left out: it needs real pyCoilGen wire STLs and minutes of boolean work.
"""

import numpy as np
import pytest
import trimesh

m3d = pytest.importorskip('manifold3d')

from coilgen import shell                                          # noqa: E402
from coilgen.config import Config                                  # noqa: E402


X = np.array([1.0, 0.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])


@pytest.fixture(autouse=True)
def default_params():
    """``_P`` is module-level state normally filled by run_shell()."""
    shell._populate_params(Config(gradient_axis='y'))


def _tube_stl(tmp_path, name='wire.stl', inner_r=0.033, outer_r=0.041,
              length=0.3, axis=X):
    """A hollow cylinder STL standing in for a pyCoilGen wire layout."""
    tube = trimesh.creation.annulus(r_min=inner_r, r_max=outer_r, height=length,
                                    sections=32)
    if not np.allclose(axis, Z):
        rot = trimesh.geometry.align_vectors(Z, axis)
        tube.apply_transform(rot)
    path = tmp_path / name
    tube.export(path)
    return str(path)


# ---------------------------------------------------------------------------
# manifold conversions
# ---------------------------------------------------------------------------

def test_manifold_round_trip_preserves_the_volume():
    box = trimesh.creation.box(extents=(0.02, 0.03, 0.04))

    man = shell.manifold_from_trimesh(box)
    back = shell.trimesh_from_manifold(man)

    assert back.volume == pytest.approx(box.volume, rel=1e-4)
    assert back.vertices.shape[1] == 3
    assert back.is_watertight


def test_trimesh_from_manifold_drops_extra_vertex_properties():
    man = m3d.Manifold.cube([1.0, 1.0, 1.0], center=True)

    tm = shell.trimesh_from_manifold(man)

    assert tm.vertices.shape[1] == 3
    assert tm.volume == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# expand_wire_mesh
# ---------------------------------------------------------------------------

def test_expand_wire_mesh_grows_the_mesh_along_the_vertex_normals():
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.01)

    grown = shell.expand_wire_mesh(sphere, 0.001)

    radii = np.linalg.norm(grown.vertices, axis=1)
    assert radii == pytest.approx(np.full(len(radii), 0.011), rel=1e-3)
    assert np.array_equal(grown.faces, sphere.faces)


def test_expand_wire_mesh_with_zero_expansion_is_a_no_op():
    sphere = trimesh.creation.icosphere(subdivisions=1, radius=0.01)

    same = shell.expand_wire_mesh(sphere, 0.0)

    assert same.vertices == pytest.approx(np.asarray(sphere.vertices))


def test_expand_wire_mesh_shrinks_with_a_negative_expansion():
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.01)

    shrunk = shell.expand_wire_mesh(sphere, -0.002)

    assert np.linalg.norm(shrunk.vertices, axis=1) == pytest.approx(
        np.full(len(sphere.vertices), 0.008), rel=1e-3)


# ---------------------------------------------------------------------------
# crop_mesh_to_bounds / slab_axial_bounds
# ---------------------------------------------------------------------------

def test_crop_mesh_to_bounds_keeps_only_faces_touching_the_box():
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))

    cropped = shell.crop_mesh_to_bounds(
        box, (np.array([-1.0, -1.0, 0.4]), np.array([1.0, 1.0, 1.0])))

    assert len(cropped.faces) < len(box.faces)
    assert cropped.vertices[:, 2].max() == pytest.approx(0.5)


def test_crop_mesh_to_bounds_returns_the_input_when_nothing_is_inside():
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))

    cropped = shell.crop_mesh_to_bounds(
        box, (np.array([10.0, 10.0, 10.0]), np.array([11.0, 11.0, 11.0])))

    assert cropped is box


def test_slab_axial_bounds_brackets_the_selected_slab_with_padding():
    box = trimesh.creation.box(extents=(0.02, 0.02, 0.4))

    bounds = shell.slab_axial_bounds(box, Z, -0.25, 0.0, pad=0.001)

    assert bounds.shape == (2, 3)
    # A box only has vertices on its end planes, so the -Z face is selected.
    assert bounds[0][2] == pytest.approx(-0.2 - 0.001)
    assert bounds[1][2] == pytest.approx(-0.2 + 0.001)
    assert bounds[0][0] == pytest.approx(-0.01 - 0.001)


def test_slab_axial_bounds_returns_none_for_an_empty_slab():
    box = trimesh.creation.box(extents=(0.02, 0.02, 0.4))

    assert shell.slab_axial_bounds(box, Z, 5.0, 6.0) is None


def test_slab_axial_bounds_follows_the_given_axis():
    box = trimesh.creation.box(extents=(0.4, 0.02, 0.02))

    bounds = shell.slab_axial_bounds(box, X, -0.3, 0.0, pad=0.0)

    assert bounds[1][0] == pytest.approx(-0.2)              # only the -X face


# ---------------------------------------------------------------------------
# pad_shell_outer
# ---------------------------------------------------------------------------

def _hollow_tube_mesh(inner_r=0.03, outer_r=0.04, length=0.1):
    outer = trimesh.creation.cylinder(radius=outer_r, height=length, sections=32)
    inner = trimesh.creation.cylinder(radius=inner_r, height=length, sections=32)
    return trimesh.util.concatenate([outer, inner])


def test_pad_shell_outer_moves_only_the_outer_wall():
    mesh = _hollow_tube_mesh()
    dims = {'inner_r_m': 0.03, 'outer_r_m': 0.04}

    padded = shell.pad_shell_outer(mesh, dims, Z, 0.002)

    r_before = np.linalg.norm(np.asarray(mesh.vertices)[:, :2], axis=1)
    r_after = np.linalg.norm(np.asarray(padded.vertices)[:, :2], axis=1)
    inner = r_before < 0.035
    assert r_after[inner] == pytest.approx(r_before[inner])
    outer = r_before > 0.035
    assert r_after[outer] == pytest.approx(r_before[outer] + 0.002)


def test_pad_shell_outer_is_a_no_op_for_a_non_positive_pad():
    mesh = _hollow_tube_mesh()

    assert shell.pad_shell_outer(mesh, {'inner_r_m': 0.03, 'outer_r_m': 0.04},
                                 Z, 0.0) is mesh


# ---------------------------------------------------------------------------
# limiters / hollow cylinder
# ---------------------------------------------------------------------------

def test_build_radius_limiter_trims_the_fusion_outer_radius():
    dims = {'outer_r_m': 0.05, 'inner_r_m': 0.04, 'axial_length_m': 0.2}

    limiter = shell.build_radius_limiter(dims, 0.0, 0.001,
                                         rot_axis=(0.0, 1.0, 0.0),
                                         rot_angle=np.pi / 2)

    tm = shell.trimesh_from_manifold(limiter)
    # Rotated onto +X: axial extent along X, radial radius in the YZ plane.
    assert np.ptp(tm.vertices[:, 0]) == pytest.approx(0.202, rel=1e-3)
    radial = np.linalg.norm(tm.vertices[:, 1:], axis=1)
    assert radial.max() == pytest.approx(0.049, rel=1e-2)


def test_build_radius_limiter_is_centred_on_the_alignment_centre():
    dims = {'outer_r_m': 0.05, 'inner_r_m': 0.04, 'axial_length_m': 0.2}

    limiter = shell.build_radius_limiter(dims, 0.01, 0.0, (0.0, 1.0, 0.0),
                                         np.pi / 2)

    tm = shell.trimesh_from_manifold(limiter)
    assert tm.vertices[:, 0].mean() == pytest.approx(0.01, abs=1e-6)


def test_build_auto_outer_limiter_uses_the_gui_outer_radius():
    cfg = Config(gradient_axis='y')
    cfg.cylinder.radius = 0.045
    cfg.cylinder.height = 0.3

    limiter = shell.build_auto_outer_limiter(cfg, {}, cfg.cylinder.rot_axis,
                                            cfg.cylinder.rot_angle)

    tm = shell.trimesh_from_manifold(limiter)
    assert np.ptp(tm.vertices[:, 0]) == pytest.approx(0.302, rel=1e-3)
    assert np.linalg.norm(tm.vertices[:, 1:], axis=1).max() == pytest.approx(
        0.045, rel=1e-2)


def test_build_auto_radius_limiter_is_an_alias():
    cfg = Config(gradient_axis='y')

    a = shell.trimesh_from_manifold(shell.build_auto_radius_limiter(
        cfg, {}, cfg.cylinder.rot_axis, cfg.cylinder.rot_angle))
    b = shell.trimesh_from_manifold(shell.build_auto_outer_limiter(
        cfg, {}, cfg.cylinder.rot_axis, cfg.cylinder.rot_angle))

    assert a.volume == pytest.approx(b.volume)


def test_build_auto_inner_peel_cylinder_opens_the_bore_by_the_radial_peel():
    cfg = Config(gradient_axis='y')

    peeler = shell.build_auto_inner_peel_cylinder(
        cfg, {}, cfg.cylinder.rot_axis, cfg.cylinder.rot_angle)

    tm = shell.trimesh_from_manifold(peeler)
    expected = cfg.shell_build_inner_radius + cfg.radial_peel
    assert np.linalg.norm(tm.vertices[:, 1:], axis=1).max() == pytest.approx(
        expected, rel=1e-2)


def test_build_auto_hollow_cylinder_matches_the_analytical_dimensions(tmp_path,
                                                                     capsys):
    cfg = Config(gradient_axis='y')
    cfg.cylinder.radius = 0.045
    cfg.cylinder.height = 0.2
    wire = _tube_stl(tmp_path, inner_r=cfg.estimated_wire_inner_radius,
                     outer_r=cfg.estimated_wire_outer_radius, length=0.19)

    tube = shell.build_auto_hollow_cylinder(cfg, wire)

    tm = shell.trimesh_from_manifold(tube)
    radial = np.linalg.norm(tm.vertices[:, 1:], axis=1)
    assert radial.max() == pytest.approx(cfg.shell_outer_radius, rel=1e-2)
    assert radial.min() == pytest.approx(cfg.shell_inner_radius, rel=1e-2)
    assert np.ptp(tm.vertices[:, 0]) == pytest.approx(0.2, rel=1e-3)
    assert tm.vertices[:, 0].mean() == pytest.approx(0.0, abs=1e-6)
    assert 'Auto hollow-cylinder' in capsys.readouterr().out


def test_build_auto_hollow_cylinder_rejects_a_non_positive_height(tmp_path):
    cfg = Config(gradient_axis='y')
    cfg.cylinder.height = 0.0
    wire = _tube_stl(tmp_path)

    with pytest.raises(RuntimeError, match='height'):
        shell.build_auto_hollow_cylinder(cfg, wire)


def test_build_auto_hollow_cylinder_rejects_a_collapsed_bore(tmp_path):
    """A cable stack thicker than the outer radius leaves no inner bore."""
    cfg = Config(gradient_axis='y')
    cfg.cylinder.radius = 0.004
    wire = _tube_stl(tmp_path)

    with pytest.raises(RuntimeError, match='inner radius'):
        shell.build_auto_hollow_cylinder(cfg, wire)


# ---------------------------------------------------------------------------
# boolean wrappers
# ---------------------------------------------------------------------------

def test_subtract_wire_from_shell_removes_the_wire_volume(capsys):
    box = m3d.Manifold.cube([1.0, 1.0, 1.0], center=True)
    cutter = m3d.Manifold.cube([0.5, 0.5, 2.0], center=True)

    result = shell.subtract_wire_from_shell(box, cutter, 'a')

    assert shell.trimesh_from_manifold(result).volume == pytest.approx(
        1.0 - 0.25, rel=1e-4)
    assert 'Subtracting from half A' in capsys.readouterr().out


def test_subtract_wires_from_shell_applies_every_subtractor():
    box = m3d.Manifold.cube([1.0, 1.0, 1.0], center=True)
    cutters = [m3d.Manifold.cube([0.2, 0.2, 2.0], center=True).translate([0.3, 0, 0]),
               m3d.Manifold.cube([0.2, 0.2, 2.0], center=True).translate([-0.3, 0, 0])]

    result = shell.subtract_wires_from_shell(box, cutters, 'a')

    assert shell.trimesh_from_manifold(result).volume == pytest.approx(
        1.0 - 2 * 0.04, rel=1e-4)


def test_subtract_wires_from_shell_with_no_subtractors_returns_the_shell():
    box = m3d.Manifold.cube([1.0, 1.0, 1.0], center=True)

    assert shell.subtract_wires_from_shell(box, [], 'a') is box


def test_intersect_with_limiter_keeps_the_overlap(capsys):
    box = m3d.Manifold.cube([1.0, 1.0, 1.0], center=True)
    limiter = m3d.Manifold.cube([0.5, 1.0, 1.0], center=True)

    trimmed = shell.intersect_with_limiter(box, limiter, 'a', 'Trimming radius')

    assert shell.trimesh_from_manifold(trimmed).volume == pytest.approx(0.5,
                                                                       rel=1e-4)
    assert 'Trimming radius on a' in capsys.readouterr().out


def test_peel_inner_skin_subtracts_the_bore_cylinder():
    tube = (m3d.Manifold.cylinder(0.1, 0.04, center=True)
            - m3d.Manifold.cylinder(0.2, 0.03, center=True))
    bore = m3d.Manifold.cylinder(0.2, 0.032, center=True)

    peeled = shell.peel_inner_skin(tube, bore, 'a', 0.002)

    radial = np.linalg.norm(
        shell.trimesh_from_manifold(peeled).vertices[:, :2], axis=1)
    assert radial.min() == pytest.approx(0.032, rel=1e-2)


# ---------------------------------------------------------------------------
# Fusion half loading / naming
# ---------------------------------------------------------------------------

def test_load_fusion_half_mesh_converts_mm_to_m_and_rotates_onto_the_bore(tmp_path):
    half = trimesh.creation.cylinder(radius=50.0, height=200.0, sections=24)
    half.apply_translation([0.0, 0.0, 100.0])            # Fusion: 0..200 mm in +Z
    path = tmp_path / 'half_a.stl'
    half.export(path)
    dims = {'inner_r_m': 0.04, 'outer_r_m': 0.05, 'axial_length_m': 0.2,
            'axial_center_m': 0.1}

    tm = shell.load_fusion_half_mesh(str(path), dims, 0.0,
                                     rot_axis=(0.0, 1.0, 0.0),
                                     rot_angle=np.pi / 2)

    assert np.ptp(tm.vertices[:, 0]) == pytest.approx(0.2, rel=1e-3)
    assert tm.vertices[:, 0].mean() == pytest.approx(0.0, abs=1e-6)
    radial = np.linalg.norm(tm.vertices[:, 1:], axis=1)
    assert radial.max() == pytest.approx(0.05, rel=1e-2)


def test_load_fusion_half_mesh_shifts_to_the_alignment_centre(tmp_path):
    half = trimesh.creation.cylinder(radius=50.0, height=200.0, sections=24)
    path = tmp_path / 'half_b.stl'
    half.export(path)
    dims = {'inner_r_m': 0.04, 'outer_r_m': 0.05, 'axial_length_m': 0.2,
            'axial_center_m': 0.0}

    tm = shell.load_fusion_half_mesh(str(path), dims, 0.015, (0.0, 1.0, 0.0),
                                     np.pi / 2)

    assert tm.vertices[:, 0].mean() == pytest.approx(0.015, abs=1e-6)


@pytest.mark.parametrize('name,expected', [
    ('Gradient_Gy_tk2500_lvl26_wire_0_z_with_leads.stl',
     'Gradient_Gy_tk2500_lvl26_shell_0_z'),
    ('Gradient_Gy_tk2500_lvl26_wire_0_z_with_leads(2).stl',
     'Gradient_Gy_tk2500_lvl26_shell_0_z'),
    ('Gradient_Gy_tk2500_lvl26_wire_0_z.stl',
     'Gradient_Gy_tk2500_lvl26_shell_0_z'),
    ('custom_part.stl', 'custom_part_shell'),
])
def test_shell_output_base_strips_the_with_leads_suffix(name, expected):
    assert shell.shell_output_base(f'/tmp/out/{name}') == expected


# ---------------------------------------------------------------------------
# warn_wire_radial_mismatch / print_wire_dims / _populate_params
# ---------------------------------------------------------------------------

def test_warn_wire_radial_mismatch_is_silent_for_a_missing_file(capsys):
    shell.warn_wire_radial_mismatch(Config(gradient_axis='y'), '/no/such.stl')

    assert capsys.readouterr().out == ''


def test_warn_wire_radial_mismatch_accepts_a_matching_wire(tmp_path, capsys):
    cfg = Config(gradient_axis='y')
    wire = _tube_stl(tmp_path, inner_r=cfg.estimated_wire_inner_radius,
                     outer_r=cfg.estimated_wire_outer_radius, length=0.2)

    shell.warn_wire_radial_mismatch(cfg, wire, tol_m=0.001)

    out = capsys.readouterr().out
    assert 'within tolerance' in out
    assert 'WARNING' not in out


def test_warn_wire_radial_mismatch_flags_a_wire_off_the_shell(tmp_path, capsys):
    cfg = Config(gradient_axis='y')
    wire = _tube_stl(tmp_path, inner_r=0.01, outer_r=0.02, length=0.2)

    shell.warn_wire_radial_mismatch(cfg, wire, tol_m=0.0003)

    assert 'WARNING: wire radial extent differs' in capsys.readouterr().out


def test_print_wire_dims_reports_millimetres(capsys):
    shell.print_wire_dims({'inner_r': 0.033, 'outer_r': 0.041,
                           'axial_min': -0.1, 'axial_max': 0.1,
                           'axial_extent': 0.2, 'axial_center': 0.0}, 'wire')

    out = capsys.readouterr().out
    assert '33.00 -- 41.00 mm' in out
    assert 'length = 200.0 mm' in out


def test_populate_params_uses_the_derived_trim_in_auto_mode():
    cfg = Config(gradient_axis='y')
    cfg.shell.use_custom_stl = False

    shell._populate_params(cfg)

    assert shell._P.outer_skin_trim == cfg.outer_skin_trim
    assert shell._P.groove_expansion == cfg.shell.groove_expansion
    assert shell._P.circular_segments == cfg.shell.circular_segments


def test_populate_params_uses_the_configured_trim_for_custom_stls():
    cfg = Config(gradient_axis='y')
    cfg.shell.use_custom_stl = True
    cfg.shell.outer_skin_trim = 0.0007

    shell._populate_params(cfg)

    assert shell._P.outer_skin_trim == 0.0007
