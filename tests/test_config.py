"""Unit tests for :mod:`coilgen.config`."""

import os

import numpy as np
import pytest
import trimesh

from coilgen import config as cfgmod
from coilgen.config import Config


# ---------------------------------------------------------------------------
# __post_init__ / presets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('axis', ['x', 'y', 'z', 'X', 'Y', 'Z'])
def test_valid_axes_are_accepted(axis):
    assert Config(gradient_axis=axis).gradient_axis == axis


def test_invalid_axis_raises():
    with pytest.raises(ValueError, match="gradient_axis must be"):
        Config(gradient_axis='w')


def test_post_init_fills_preset_stl_dir_and_outer_pad():
    cfg = Config(gradient_axis='z')

    assert cfg.leads.preset is cfgmod.preset_for('z')
    assert cfg.shell.stl_dir == cfgmod.default_shell_stl_dir()
    assert cfg.shell.shell_outer_pad == max(0.0005, 0.65 * cfg.conductor_semi_a)


def test_post_init_keeps_explicit_preset_and_stl_dir(tmp_path):
    custom = cfgmod.LeadAxisPreset(
        lead_direction=np.array([1.0, 0.0, 0.0]),
        sector_ref_dir=np.array([0.0, 1.0, 0.0]),
        sector_angular_half=0.5,
    )
    cfg = Config(gradient_axis='y',
                 leads=cfgmod.LeadsConfig(preset=custom),
                 shell=cfgmod.ShellConfig(stl_dir=str(tmp_path),
                                          shell_outer_pad=0.004))

    assert cfg.leads.preset is custom
    assert cfg.shell.stl_dir == str(tmp_path)
    assert cfg.shell.shell_outer_pad == 0.004


@pytest.mark.parametrize('axis,ref_dir', [
    ('x', [0.0, 1.0, 0.0]),
    ('y', [0.0, 0.0, 1.0]),
    ('z', [0.0, 1.0, 0.0]),
])
def test_preset_for_matches_documented_stations(axis, ref_dir):
    preset = cfgmod.preset_for(axis.upper())

    assert np.allclose(preset.lead_direction, [-1.0, 0.0, 0.0])
    assert np.allclose(preset.sector_ref_dir, ref_dir)
    assert preset.sector_angular_half == 0.35
    assert preset.spread_signs == (1, -1)
    assert preset.exit_direction is None


def test_lead_preset_recreates_a_cleared_preset():
    cfg = Config(gradient_axis='x')
    cfg.leads.preset = None

    assert cfg.lead_preset() is cfgmod.preset_for('x')
    assert cfg.leads.preset is cfgmod.preset_for('x')


# ---------------------------------------------------------------------------
# Axis / naming derived values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('axis,internal', [('x', 'y'), ('y', 'z'), ('z', 'x')])
def test_internal_axis_and_label(axis, internal):
    cfg = Config(gradient_axis=axis)

    assert cfg.internal_axis == internal
    assert cfg.axis_label == f'G{axis.upper()}'


def test_rotated_cylinder_axis_is_the_bore_axis():
    assert np.allclose(Config().rotated_cylinder_axis, [1.0, 0.0, 0.0])


def test_design_folder_and_project_stem():
    cfg = Config(gradient_axis='y', tikhonov_factor=2500.0, num_levels=26)

    assert cfg.design_folder == 'Gy_tk2500_lvl26'
    assert cfg.project_stem_base == 'Gradient_Gy_tk2500_lvl26'


# ---------------------------------------------------------------------------
# Conductor geometry
# ---------------------------------------------------------------------------

def test_conductor_semi_axes_and_cable_height():
    cfg = Config(wire=cfgmod.WireConfig(conductor_width=0.002,
                                        cross_section_a_frac=2.0,
                                        cross_section_b_frac=1.0))

    assert cfg.conductor_semi_a == 0.002
    assert cfg.conductor_semi_b == 0.001
    assert cfg.cable_height == 0.004


def test_fasthenry_cross_section_dimensions():
    cfg = Config()

    assert cfg.fasthenry_conductor_width == 2.0 * cfg.conductor_semi_b
    assert np.isclose(cfg.fasthenry_conductor_height,
                      (np.pi / 2.0) * cfg.conductor_semi_a)


def test_cross_sectional_points_trace_the_conductor_ellipse():
    cfg = Config()
    pts = cfg.cross_sectional_points

    assert pts.shape == (2, cfg.wire.cross_section_n)
    # Every sample sits exactly on the semi_a x semi_b ellipse.
    ellipse = ((pts[0] / cfg.conductor_semi_a) ** 2
               + (pts[1] / cfg.conductor_semi_b) ** 2)
    assert np.allclose(ellipse, 1.0)
    assert np.abs(pts[0]).max() <= cfg.conductor_semi_a + 1e-12
    assert np.isclose(np.abs(pts[1]).max(), cfg.conductor_semi_b)
    # Closed contour (endpoint=True).
    assert np.allclose(pts[:, 0], pts[:, -1])


# ---------------------------------------------------------------------------
# Radii / wall formulas (auto shell)
# ---------------------------------------------------------------------------

def test_auto_mode_design_radius_and_wall():
    cfg = Config(cylinder=cfgmod.CylinderConfig(radius=0.039),
                 winding=cfgmod.WindingConfig(layer_gap_mm=0.0005))

    peel = cfg.shell.auto_margin_pct * cfg.cable_height
    assert np.isclose(cfg.radial_peel, peel)
    assert np.isclose(cfg.outer_skin_trim, peel)
    assert np.isclose(cfg.groove_margin, peel)
    assert np.isclose(cfg.cylinder_design_radius,
                      0.039 - cfg.conductor_semi_a + peel)
    assert np.isclose(cfg.shell_wall_thickness,
                      2.0 * cfg.cable_height + 0.0005 - 2.0 * peel)
    assert np.isclose(cfg.shell_outer_radius, 0.039)
    assert np.isclose(cfg.shell_inner_radius, 0.039 - cfg.shell_wall_thickness)


def test_radius_aliases_track_the_canonical_properties():
    cfg = Config()

    assert cfg.shell_build_outer_radius == cfg.shell_outer_radius
    assert cfg.estimated_shell_outer_radius == cfg.shell_outer_radius
    assert cfg.shell_build_inner_radius == cfg.shell_inner_radius
    assert cfg.shell_final_inner_radius == cfg.shell_inner_radius
    assert cfg.estimated_shell_inner_radius == cfg.shell_inner_radius
    assert cfg.estimated_shell_thickness == cfg.shell_wall_thickness
    assert np.isclose(cfg.shell_radial_center,
                      0.5 * (cfg.shell_inner_radius + cfg.shell_outer_radius))


def test_normal_shift_defaults_to_negative_pack_height():
    cfg = Config(winding=cfgmod.WindingConfig(layer_gap_mm=0.001))

    assert np.isclose(cfg.normal_shift_length, -(cfg.cable_height + 0.001))
    assert cfg.layer_crossing_gap == 0.001


def test_normal_shift_override_is_used_verbatim():
    cfg = Config(winding=cfgmod.WindingConfig(normal_shift=-0.0075))

    assert cfg.normal_shift_length == -0.0075


def test_wire_radii_bracket_both_layers():
    cfg = Config()
    center = cfg.cylinder_design_radius
    shift = cfg.normal_shift_length

    assert np.isclose(cfg.estimated_wire_outer_radius,
                      center + cfg.conductor_semi_a)
    assert np.isclose(cfg.estimated_wire_inner_radius,
                      center + shift - cfg.conductor_semi_a)
    assert np.isclose(cfg.estimated_wire_radial_center, center + 0.5 * shift)
    assert cfg.estimated_wire_inner_radius < cfg.estimated_wire_outer_radius


def test_wire_radii_with_a_positive_normal_shift():
    cfg = Config(winding=cfgmod.WindingConfig(normal_shift=0.003))
    center = cfg.cylinder_design_radius

    assert np.isclose(cfg.estimated_wire_inner_radius,
                      center - cfg.conductor_semi_a)
    assert np.isclose(cfg.estimated_wire_outer_radius,
                      center + 0.003 + cfg.conductor_semi_a)


# ---------------------------------------------------------------------------
# Custom (measured) shell mode
# ---------------------------------------------------------------------------

def _custom_cfg(inner_r=0.0355, outer_r=0.039, gap=0.0005):
    return Config(
        cylinder=cfgmod.CylinderConfig(radius=outer_r),
        winding=cfgmod.WindingConfig(layer_gap_mm=gap),
        shell=cfgmod.ShellConfig(use_custom_stl=True, measured_inner_r=inner_r),
    )


def test_custom_mode_uses_measured_inner_radius_and_wall():
    cfg = _custom_cfg()

    assert cfg._custom_shell_fixed is True
    assert np.isclose(cfg.shell_inner_radius, 0.0355)
    assert np.isclose(cfg.shell_wall_thickness, 0.039 - 0.0355)


def test_custom_mode_centres_the_wire_pack_in_the_wall():
    cfg = _custom_cfg()
    pack_half = 0.5 * (cfg.cable_height + cfg.layer_crossing_gap)

    assert np.isclose(cfg.cylinder_design_radius,
                      cfg.shell_radial_center + pack_half)
    # The pack midplane then coincides with the wall midplane.
    assert np.isclose(cfg.estimated_wire_radial_center, cfg.shell_radial_center)


def test_custom_flag_without_measurement_falls_back_to_auto_formulas():
    cfg = Config(shell=cfgmod.ShellConfig(use_custom_stl=True))

    assert cfg._custom_shell_fixed is False
    assert np.isclose(cfg.cylinder_design_radius,
                      cfg.cylinder.radius - cfg.conductor_semi_a + cfg.radial_peel)


# ---------------------------------------------------------------------------
# Shell asset discovery
# ---------------------------------------------------------------------------

def _write_stl(path):
    trimesh.creation.box(extents=(1, 1, 1)).export(str(path))
    return str(path)


def test_default_shell_stl_dir_points_at_the_assets_folder():
    assert cfgmod.default_shell_stl_dir().endswith(os.path.join('assets', 'shells'))


def test_shell_half_paths_uses_layer_assets_by_default():
    cfg = Config(shell=cfgmod.ShellConfig(layer=3, stl_dir='/tmp/shells'))

    assert cfg.shell_half_paths() == ('/tmp/shells/g_3a.stl', '/tmp/shells/g_3b.stl')


def test_shell_half_paths_returns_existing_custom_halves(tmp_path):
    a = _write_stl(tmp_path / 'half_a.stl')
    b = _write_stl(tmp_path / 'half_b.stl')
    cfg = Config(shell=cfgmod.ShellConfig(use_custom_stl=True,
                                          custom_stl_a=a, custom_stl_b=b,
                                          stl_dir=str(tmp_path)))

    assert cfg.shell_half_paths() == (a, b)


def test_shell_half_paths_falls_back_when_a_custom_half_is_missing(tmp_path):
    a = _write_stl(tmp_path / 'half_a.stl')
    cfg = Config(shell=cfgmod.ShellConfig(use_custom_stl=True, layer=2,
                                          custom_stl_a=a,
                                          custom_stl_b=str(tmp_path / 'gone.stl'),
                                          stl_dir=str(tmp_path)))

    assert cfg.shell_half_paths() == (str(tmp_path / 'g_2a.stl'),
                                      str(tmp_path / 'g_2b.stl'))


def test_legacy_single_custom_stl_maps_to_half_a(tmp_path):
    single = _write_stl(tmp_path / 'single.stl')
    b = _write_stl(tmp_path / 'half_b.stl')
    cfg = Config(shell=cfgmod.ShellConfig(use_custom_stl=True,
                                          custom_stl=single, custom_stl_b=b,
                                          stl_dir=str(tmp_path)))

    assert cfg.shell_half_paths() == (single, b)


def test_list_shell_pairs_sorts_by_layer_index(tmp_path):
    for name in ('g_1a.stl', 'g_1b.stl', 'g_10a.stl', 'g_10b.stl', 'g_2a.stl', 'g_2b.stl'):
        _write_stl(tmp_path / name)

    pairs = cfgmod.list_shell_pairs(str(tmp_path))

    assert [label for label, _, _ in pairs] == ['Capa 1', 'Capa 2', 'Capa 10']
    assert pairs[0][1] == str(tmp_path / 'g_1a.stl')
    assert pairs[0][2] == str(tmp_path / 'g_1b.stl')


def test_list_shell_pairs_skips_unmatched_and_malformed_names(tmp_path):
    _write_stl(tmp_path / 'g_1a.stl')                     # no matching b half
    _write_stl(tmp_path / 'g_xa.stl')                     # non-numeric index
    _write_stl(tmp_path / 'g_xb.stl')
    _write_stl(tmp_path / 'other.stl')                    # not a g_ half
    _write_stl(tmp_path / 'g_3a.stl')
    _write_stl(tmp_path / 'g_3b.stl')

    pairs = cfgmod.list_shell_pairs(str(tmp_path))

    assert [label for label, _, _ in pairs] == ['Capa 3']


def test_list_shell_pairs_accepts_an_uppercase_b_half(tmp_path):
    _write_stl(tmp_path / 'g_4a.stl')
    _write_stl(tmp_path / 'g_4B.stl')

    pairs = cfgmod.list_shell_pairs(str(tmp_path))

    assert pairs == [('Capa 4', str(tmp_path / 'g_4a.stl'), str(tmp_path / 'g_4B.stl'))]


def test_list_shell_pairs_empty_for_a_missing_directory(tmp_path):
    assert cfgmod.list_shell_pairs(str(tmp_path / 'nope')) == []


def test_list_shell_pairs_defaults_to_the_repo_assets():
    pairs = cfgmod.list_shell_pairs()

    assert all(p[1].startswith(cfgmod.default_shell_stl_dir()) for p in pairs)


# ---------------------------------------------------------------------------
# apply_custom_shell_dims
# ---------------------------------------------------------------------------

def test_apply_custom_shell_dims_writes_measured_geometry(tmp_path, monkeypatch):
    dims = {
        'axial_min_m': -0.2, 'axial_max_m': 0.2, 'axial_center_m': 0.0,
        'axial_length_m': 0.4, 'inner_r_m': 0.0355, 'outer_r_m': 0.039,
        'z_min_mm': -200.0, 'z_max_mm': 200.0,
    }
    monkeypatch.setattr(cfgmod.geo, 'detect_fusion_cylinder_dims',
                        lambda a, b: dims)
    cfg = Config()

    returned = cfgmod.apply_custom_shell_dims(cfg, 'a.stl', 'b.stl')

    assert returned is dims
    assert cfg.shell.use_custom_stl is True
    assert (cfg.shell.custom_stl_a, cfg.shell.custom_stl_b) == ('a.stl', 'b.stl')
    assert cfg.cylinder.radius == 0.039
    assert cfg.cylinder.height == 0.4
    assert cfg.shell.measured_inner_r == 0.0355
    # And the derived radii now follow the custom (fixed-wall) branch.
    assert cfg._custom_shell_fixed is True
    assert np.isclose(cfg.shell_inner_radius, 0.0355)


def test_apply_custom_shell_dims_measures_real_stl_halves(tmp_path):
    def tube(path, z_center):
        mesh = trimesh.creation.cylinder(radius=40.0, height=200.0, sections=32)
        mesh.apply_translation([0, 0, z_center])
        mesh.export(str(path))
        return str(path)

    a = tube(tmp_path / 'g_9a.stl', 100.0)
    b = tube(tmp_path / 'g_9b.stl', -100.0)
    cfg = Config()

    dims = cfgmod.apply_custom_shell_dims(cfg, a, b)

    assert np.isclose(cfg.cylinder.height, 0.4)
    assert np.isclose(cfg.cylinder.radius, 0.04, atol=1e-6)
    assert cfg.shell.measured_inner_r == dims['inner_r_m']


# ---------------------------------------------------------------------------
# to_params_dict
# ---------------------------------------------------------------------------

def test_to_params_dict_exposes_inputs_and_derived_values():
    cfg = Config(gradient_axis='y', tikhonov_factor=2500, num_levels=26)

    params = cfg.to_params_dict()

    assert params['gradient_axis'] == 'y'
    assert params['tikhonov_factor'] == 2500
    assert params['num_levels'] == 26
    assert np.isclose(params['cyl_mesh_height_m'],
                      cfg.cylinder.height * cfg.cylinder.mesh_length_factor)
    assert params['cyl_design_radius_m'] == cfg.cylinder_design_radius
    assert params['normal_shift_m'] == cfg.normal_shift_length
    assert params['shell_wall_thickness_m'] == cfg.shell_wall_thickness
    assert params['shell_inner_radius_m'] == cfg.shell_inner_radius
    assert params['outer_skin_trim_m'] == cfg.radial_peel
    assert params['cable_height_m'] == cfg.cable_height
    assert params['fasthenry_conductor_height_m'] == cfg.fasthenry_conductor_height
    assert params['enable_fasthenry'] is True


def test_to_params_dict_values_are_plain_scalars_for_the_metrics_header():
    params = Config().to_params_dict()

    for key, value in params.items():
        assert isinstance(value, (int, float, bool, str, tuple, list)), key


def test_to_params_dict_tracks_config_edits():
    cfg = Config()
    cfg.cylinder.radius = 0.05
    cfg.winding.layer_gap_mm = 0.002

    params = cfg.to_params_dict()

    assert params['cyl_radius_m'] == 0.05
    assert params['layer_gap_mm'] == 0.002
    assert params['layer_crossing_gap_m'] == 0.002
    assert params['shell_outer_radius_m'] == 0.05
