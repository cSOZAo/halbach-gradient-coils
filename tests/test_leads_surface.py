"""Regression tests for the preserved profile/connection fixes, in SI units."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import trimesh

from halbach_coils.coilgen.config import Config, LeadAxisPreset
from halbach_coils.coilgen.lead_surface import (
    GeometryRejected, Options, Reference, clearance_check, plan,
    segment_distances, surface_lead, tube_mesh, unit, validate_mesh, vector_areas,
)
from halbach_coils.coilgen.leads import (
    _automatic_options, _plan_with_sector_fallback,
    current_path_path, final_current_path, lead_reference_path,
    load_lead_reference, run_leads, save_lead_reference,
)


def saddle_reference(n=256, corners=32, semi_a=.002, semi_b=.001):
    phi = np.linspace(0, 2*np.pi, n, endpoint=False)
    radius = .039
    arc, axial = .035*np.sin(phi), .075*np.cos(phi)
    theta = arc/radius
    normal = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(n)])
    circum = np.column_stack([-np.sin(theta), np.cos(theta), np.zeros(n)])
    path = radius*normal + axial[:, None]*np.array([0, 0, 1])
    tangent = unit(.035*np.cos(phi)[:, None]*circum
                   - .075*np.sin(phi)[:, None]*np.array([0, 0, 1]))
    binormal = np.cross(tangent, normal)
    angle = np.linspace(0, 2*np.pi, corners, endpoint=False)
    rings = (path[:, None] + semi_a*np.sin(angle)[None, :, None]*normal[:, None]
             + semi_b*np.cos(angle)[None, :, None]*binormal[:, None])
    a = (np.arange(n)[:, None]*corners + np.arange(corners)).ravel()
    b = (np.arange(n)[:, None]*corners + (np.arange(corners)+1) % corners).ravel()
    c, d = (a+corners) % (n*corners), (b+corners) % (n*corners)
    faces = np.vstack([np.column_stack([a, b, d]), np.column_stack([a, d, c])])
    return Reference(path, rings, faces, np.array([0., 0., 1.]), np.zeros(3),
                     semi_a, semi_b, .035, .039, .200)


def fixture_config():
    cfg = Config(gradient_axis='x', b0_direction='-y', show_plots=False)
    cfg.cylinder.rot_angle = 0
    cfg.cylinder.height = .2
    cfg.cylinder.radius = .039
    cfg.wire.conductor_width = .0016
    cfg.wire.cross_section_a_frac = 2.5
    cfg.wire.cross_section_b_frac = 1.25
    cfg.wire.cross_section_n = 33
    cfg.shell.use_custom_stl = True
    cfg.shell.measured_inner_r = .035
    cfg.leads.preset = LeadAxisPreset(
        lead_direction=np.array([0., 0., 1.]), sector_ref_dir=np.array([1., 0., 0.]),
        sector_angular_half=.35)
    return cfg


def solution_for_reference(ref, reverse_current=False):
    path = ref.path[::-1] if reverse_current else ref.path
    layout = NS(get_vertices=lambda: ref.rings.reshape(-1, 3), get_faces=lambda: ref.faces)
    return NS(coil_parts=[NS(wire_path=NS(v=path.T), layout_surface_mesh=layout)])


@pytest.mark.parametrize('reverse,shift', [(False, 0), (False, 7), (True, 3)])
def test_section_order_and_phase_preserve_area_and_radial_height(reverse, shift):
    ref = saddle_reference()
    ring = np.roll(ref.rings[0], shift, axis=0)
    if reverse:
        ring = ring[::-1]
    lead = surface_lead(ring, np.array([0, 1, 0]), ref.axis, ref.origin, .115, .6,
                        1., Options())
    assert np.array_equal(lead.rings[0], ring)
    np.testing.assert_allclose(vector_areas(lead.rings), float(vector_areas(ring[None])[0]), rtol=1e-9)
    assert lead.metrics['height_in_frame_mm'] == pytest.approx(4.)
    assert lead.metrics['section_area_relative_error'] < 1e-9
    validate_mesh(tube_mesh(lead.rings))


@pytest.mark.parametrize('exit_sign', [-1, 1])
def test_cut_preserves_original_mesh_and_leads_clear_the_shell(exit_sign):
    ref = saddle_reference()
    result, leads, opened, report = plan(ref, [1, 0, 0], exit_sign)
    assert report['retained_vertices_exact']
    np.testing.assert_array_equal(result.vertices[:len(opened.vertices)], opened.vertices)
    assert result.is_watertight and result.is_winding_consistent
    for lead in leads:
        assert lead.min_radius >= .006
        assert lead.metrics['section_area_relative_error'] < .02
        assert exit_sign*lead.path[-1, 2] > ref.shell_height/2


def test_rotating_physical_frame_does_not_change_cut_or_current_orientation():
    ref = saddle_reference()
    _, leads, _, report = plan(ref, [1, 0, 0], 1)
    rotation = Rotation.from_rotvec([.43, -1.1, .71]).as_matrix()
    offset = np.array([.25, -.13, .08])
    transformed = replace(ref, path=ref.path@rotation.T+offset,
                          rings=ref.rings@rotation.T+offset,
                          axis=rotation@ref.axis, origin=offset)
    _, new_leads, _, new_report = plan(transformed, rotation@np.array([1, 0, 0]), 1)
    assert report['cut_indices'] == new_report['cut_indices']
    before = final_current_path(ref, leads, report)
    after = final_current_path(transformed, new_leads, new_report)
    np.testing.assert_allclose(before@rotation.T+offset, after, atol=1e-10)


def test_current_direction_follows_optimized_path_if_engine_reverses_mesh_polarity():
    ref = saddle_reference()
    _, leads, _, report = plan(ref, [1, 0, 0], 1)
    forward = final_current_path(ref, leads, dict(report))
    reversed_report = dict(report)
    reverse = final_current_path(replace(ref, path=ref.path[::-1]), leads, reversed_report)
    np.testing.assert_allclose(reverse, forward[::-1])
    assert reversed_report['sweep_reversed_relative_to_current']


def test_crossing_segments_and_coincident_leads_are_rejected():
    a = np.array([[0., 0., 0.]])
    b = np.array([[2., 0., 0.]])
    c = np.array([[1., -1., 0.]])
    d = np.array([[1., 1., 0.]])
    np.testing.assert_allclose(segment_distances(a, b, c, d), [0])
    ref = saddle_reference()
    lead = surface_lead(ref.rings[0], [0, 1, 0], ref.axis, ref.origin, .115, .6, 1., Options())
    with pytest.raises(GeometryRejected, match='Lead-lead'):
        clearance_check([lead, lead], ref.path, .002, .0003)


def test_exported_current_path_is_numeric_signed_and_matches_connected_geometry(tmp_path):
    ref = saddle_reference()
    cfg = fixture_config()
    cfg.output_dir = str(tmp_path)
    wire = str(tmp_path/'wire.stl')
    ref.mesh().export(wire)
    assert save_lead_reference(solution_for_reference(ref, True), cfg, wire) == lead_reference_path(wire)
    outputs = run_leads(cfg, wire)
    for filename in (outputs[0], outputs[2]):
        mesh = trimesh.load_mesh(filename)
        assert mesh.is_watertight and mesh.is_winding_consistent
        assert np.all(mesh.area_faces > 0)
    with np.load(current_path_path(outputs[0]), allow_pickle=False) as saved:
        assert saved['path'].shape[1] == 3
        assert saved['b0_direction'].item() == '-y'
        assert saved['units'].item() == 'm'
        assert saved['coordinate_frame'].item() == 'physical_xyz'
        assert saved['with_leads_sha256'].item() == sha256(Path(outputs[0]).read_bytes()).hexdigest()
        assert not saved['supply_return_included'].item()
        assert all(not saved[key].dtype.hasobject for key in saved.files)
        # There is no artificial last-to-first closing conductor.
        assert np.linalg.norm(saved['path'][-1] - saved['path'][0]) > .004
        assert np.max(np.linalg.norm(np.diff(saved['path'], axis=0), axis=1)) < .004
    report = json.loads((tmp_path/'wire_with_leads_lead_report.json').read_text())
    assert report['sweep_reversed_relative_to_current']
    assert report['leads'][0]['height_in_frame_mm'] == pytest.approx(4.)


def test_stale_or_missing_reference_never_falls_back_to_deforming_legacy_method(tmp_path):
    ref = saddle_reference()
    wire = str(tmp_path/'wire.stl')
    ref.mesh().export(wire)
    cfg = fixture_config()
    with pytest.raises(FileNotFoundError, match='Regenerate'):
        run_leads(cfg, wire)
    save_lead_reference(solution_for_reference(ref), cfg, wire)
    with open(wire, 'ab') as stream:
        stream.write(b'changed')
    with pytest.raises(GeometryRejected, match='does not match'):
        load_lead_reference(wire)
    assert not list(tmp_path.glob('*with_leads*'))


def test_impossible_bend_never_produces_a_candidate():
    with pytest.raises(GeometryRejected, match='No feasible'):
        plan(saddle_reference(), [1, 0, 0], 1, Options(min_bend_radius=1., candidates=1))


def test_gz_default_sector_routes_the_saddle_without_relaxing_constraints():
    reference = saddle_reference()
    cfg = Config(gradient_axis='z')
    with pytest.raises(GeometryRejected, match='No stations'):
        plan(reference, [0, 1, 0], -1, Options())
    result, leads, opened, report = plan(reference, cfg.lead_preset().sector_ref_dir, -1, Options())
    assert result.is_watertight and report['retained_vertices_exact']
    np.testing.assert_array_equal(result.vertices[:len(opened.vertices)], opened.vertices)
    assert min(lead.min_radius for lead in leads) >= .006
    assert all(lead.path[-1, 2] < -reference.shell_height/2 for lead in leads)
    assert max(abs(volume) for volume in report['lead_coil_intersection_volumes_mm3']) <= 1e-6


def test_extended_search_is_bounded_and_never_relaxes_geometric_limits(monkeypatch):
    import halbach_coils.coilgen.leads as adapter

    attempts = []

    def reject(_ref, _sector, _sign, options):
        attempts.append(options)
        raise GeometryRejected('No feasible lead pair: synthetic impossible case')

    monkeypatch.setattr(adapter, 'plan', reject)
    ref = saddle_reference()
    cfg = fixture_config()
    options = _automatic_options(ref, cfg)
    with pytest.raises(GeometryRejected, match='routing attempts rejected'):
        _plan_with_sector_fallback(ref, cfg.lead_preset(), 1, options, expand_cut_search=True)
    assert len(attempts) <= 10
    assert all(a.min_bend_radius == options.min_bend_radius and a.clearance == options.clearance
               and a.section_tolerance == options.section_tolerance for a in attempts)
    assert max(a.candidates for a in attempts) == 96
    assert max(max(a.cut_lengths) for a in attempts) == pytest.approx(.064)
