"""Generate surface-constrained leads and persist the real final current path.

The corrected geometry uses the original ordered sweep sections. It never
flattens a section onto the cylinder or blends unmatched profile corners.
All STL and NPZ coordinates remain metres in the physical XYZ frame.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
import os
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from typing import Optional

import numpy as np
import trimesh

from .config import Config
from .paths import unique_lead_output_paths
from .lead_surface import GeometryRejected, Options, Reference, plan, tube_mesh


def lead_reference_path(wire_stl: str) -> str:
    return os.path.splitext(str(wire_stl))[0] + '_lead_reference.npz'


def current_path_path(with_leads_stl: str) -> str:
    return os.path.splitext(str(with_leads_stl))[0] + '_current_path.npz'


def _file_sha256(path: str) -> str:
    digest = sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _surface_arrays(solution, cfg: Config) -> dict:
    parts = list(getattr(solution, 'coil_parts', []))
    if len(parts) != 1:
        raise GeometryRejected(
            'Surface leads currently require one connected pyCoilGen circuit; '
            f'this design has {len(parts)} parts.')
    part = parts[0]
    path = np.asarray(part.wire_path.v, dtype=float).T
    layout = part.layout_surface_mesh
    vertices = np.asarray(layout.get_vertices(), dtype=float)
    faces = np.asarray(layout.get_faces(), dtype=np.int64)
    if len(path) < 12 or len(vertices) % len(path):
        raise GeometryRejected('pyCoilGen did not provide a usable path/ring correspondence.')
    return {
        'path': path,
        'rings': vertices.reshape(len(path), -1, 3),
        'faces': faces,
        'axis': np.asarray(cfg.rotated_cylinder_axis, dtype=float),
        'origin': np.zeros(3),
        'semi_a': float(cfg.conductor_semi_a),
        'semi_b': float(cfg.conductor_semi_b),
        'shell_inner': float(cfg.shell_inner_radius),
        'shell_outer': float(cfg.shell_outer_radius),
        'shell_height': float(cfg.cylinder.height),
    }


def save_lead_reference(solution, cfg: Config, wire_stl: str) -> Optional[str]:
    """Persist the geometry required by surface mode alongside *wire_stl*.

    The metadata contains numeric arrays only, never an untrusted pickle. A
    design with more than one circuit cannot be routed until that topology is
    supported by the validated planner.
    """
    try:
        arrays = _surface_arrays(solution, cfg)
        reference = Reference(**arrays)
        reference.validate()
        _validate_reference_matches_wire(reference, wire_stl)
    except (AttributeError, GeometryRejected, ValueError) as exc:
        print(f'  Surface-leads metadata unavailable: {exc}')
        return None
    path = lead_reference_path(wire_stl)
    np.savez_compressed(path, **arrays, format_version=np.asarray(1),
                        source_sha256=np.asarray(_file_sha256(wire_stl)),
                        gradient_axis=np.asarray(cfg.gradient_axis),
                        b0_direction=np.asarray(cfg.b0_direction),
                        coordinate_frame=np.asarray('physical_xyz'), units=np.asarray('m'))
    print(f'  Surface-leads metadata: {path}')
    return path


def _automatic_options(reference: Reference, cfg: Optional[Config] = None) -> Options:
    """Scale all route constraints from the actual profile in *reference*."""
    radial_height = 2.0 * reference.semi_a
    transverse_width = 2.0 * reference.semi_b
    options = Options(
        cut_lengths=tuple(radial_height * factor for factor in (5.0, 6.5, 8.0, 10.0)),
        tip_separations=tuple(radial_height * factor for factor in (3.75, 5.0, 6.5)),
        max_step=min(0.0003, 0.30 * min(reference.semi_a, reference.semi_b)),
        min_bend_radius=3.0 * reference.semi_a,
        clearance=0.15 * transverse_width,
        exit_margin=2.5 * max(reference.semi_a, reference.semi_b),
        candidates=16,
        radius_tolerance=0.0375 * radial_height,
    )
    if cfg is not None:
        options.min_bend_radius = (getattr(cfg.leads, 'min_bend_radius', None) if getattr(cfg.leads, 'min_bend_radius', None) is not None
                                  else max(options.min_bend_radius, 3 * cfg.wire.conductor_width))
        if getattr(cfg.leads, 'clearance', None) is not None:
            options.clearance = cfg.leads.clearance
        if getattr(cfg.leads, 'exit_margin', None) is not None:
            options.exit_margin = cfg.leads.exit_margin
        # Keep the two useful legacy controls meaningful while the remaining
        # mesh/PCA controls are no longer needed by an ordered-section cut.
        gap = float(cfg.leads.cut_loop_length)
        separation = float(cfg.leads.tip_fan)
        if gap > 0:
            options.cut_lengths = tuple(gap * factor for factor in (1., 1.3, 1.6, 2.))
        if separation > 0:
            options.tip_separations = tuple(separation * factor for factor in (1., 4/3, 26/15))
    return options


def _validate_reference_matches_wire(reference: Reference, input_stl: str) -> None:
    """Reject a stale sidecar, allowing STL vertex/face reordering.

    STL's actual precision is float32. Compare complete triangles at that
    precision, independently of indexing and winding; bounds alone cannot
    distinguish two different windings on the same design cylinder.
    """
    actual = trimesh.load_mesh(input_stl, process=False)
    expected = reference.mesh()

    def canonical_triangles(mesh):
        triangles = np.asarray(mesh.triangles, dtype=np.float32).copy()
        triangles[triangles == 0] = 0  # Normalize signed zero from STL writers.
        corners = triangles.view(np.dtype((np.void, 12))).reshape(-1, 3)
        corners.sort(axis=1)
        return np.sort(corners.reshape(-1).view(np.dtype((np.void, 36))))

    if (not isinstance(actual, trimesh.Trimesh)
            or len(actual.faces) != len(expected.faces)
            or not np.array_equal(canonical_triangles(actual),
                                  canonical_triangles(expected))):
        raise GeometryRejected(
            'Surface metadata does not match the selected wire STL. '
            'Use the wire and lead_reference.npz from the same generation.')


def _plan_with_sector_fallback(reference, preset, exit_sign, options, allow_opposite=True,
                               extend_exit=True, expand_cut_search=False):
    """Search more routing space without relaxing curvature or clearance."""
    preferred = np.asarray(preset.sector_ref_dir, dtype=float)
    sectors = [(False, preferred)]
    if allow_opposite:
        sectors.append((True, -preferred))
    variants = [options]
    if extend_exit and isinstance(options, Options):
        for factor in (2, 3, 4):
            margin = max(options.exit_margin, factor * options.min_bend_radius)
            if margin > variants[-1].exit_margin:
                variants.append(replace(options, exit_margin=margin))
        if expand_cut_search:
            variants.append(replace(
                variants[-1], candidates=max(96, options.candidates),
                cut_lengths=tuple(sorted(set((*options.cut_lengths,
                                               24*reference.semi_a,
                                               32*reference.semi_a))))))
    failures = []
    for candidate_options, used_fallback, sector in (
            (variant, fallback, sector) for variant in variants for fallback, sector in sectors):
        try:
            final, lead_parts, open_mesh, report = plan(
                reference, sector, exit_sign, candidate_options)
            report['requested_sector_ref_dir'] = preferred.tolist()
            report['used_sector_ref_dir'] = np.asarray(sector).tolist()
            report['used_opposite_sector_fallback'] = used_fallback
            report['routing_attempt_rejections'] = failures
            if failures:
                report['preferred_sector_rejection'] = failures[0]
            return final, lead_parts, open_mesh, report
        except GeometryRejected as exc:
            if not str(exc).startswith(('No feasible lead pair', 'No stations in requested exit sector')):
                raise
            failures.append(str(exc))
            print(f'  Routing attempt rejected: {exc}')
    raise GeometryRejected(
        'No feasible lead pair within the allowed sectors and exit space. '
        f'{len(failures)} routing attempts rejected; last reason: {failures[-1]}')


def final_current_path(reference, lead_parts, report):
    """Orient both leads consistently with the original coil current."""
    centers = reference.rings.mean(axis=1)
    direct_error = np.linalg.norm(centers - reference.path, axis=1)
    reverse_error = np.linalg.norm(centers - reference.path[::-1], axis=1)
    reversed_sweep = np.mean(reverse_error) < np.mean(direct_error)
    aligned = reference.path[::-1] if reversed_sweep else reference.path
    errors = reverse_error if reversed_sweep else direct_error
    # pyCoilGen may reverse the mesh traversal after creating the wire path.
    # A duplicated section endpoint also biases the polygon mean slightly.
    # Reject other reorderings rather than creating fictitious long segments.
    if np.max(errors) > .15 * reference.semi_b:
        raise GeometryRejected('Current path does not correspond to the saved sweep sections.')
    left, right = report['cut_indices']
    order = np.arange(right, right + (left - right) % len(reference.path) + 1) % len(reference.path)
    path = np.concatenate((lead_parts[1].path[::-1], aligned[order], lead_parts[0].path))
    if reversed_sweep:
        path = path[::-1]
    report['sweep_reversed_relative_to_current'] = bool(reversed_sweep)
    path = path[np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-12]]
    if not np.all(np.isfinite(path)):
        raise GeometryRejected('Final current path contains non-finite points.')
    return path


def _export_m(mesh: trimesh.Trimesh, path: str, *, must_be_closed: bool) -> None:
    """Export metres for the existing shell stage and verify the reloaded STL."""
    if os.path.exists(path):
        raise FileExistsError(path)
    if not len(mesh.faces) or not np.all(np.isfinite(mesh.vertices)):
        raise GeometryRejected(f'Cannot export empty or non-finite geometry: {path}')
    payload = mesh.export(file_type='stl')
    reloaded = trimesh.load_mesh(BytesIO(payload), file_type='stl', process=True)
    if must_be_closed and not reloaded.is_watertight:
        raise GeometryRejected(f'Exported STL is not watertight: {path}')
    if must_be_closed and (not reloaded.is_winding_consistent or reloaded.volume <= 0):
        raise GeometryRejected(f'Exported STL is not an oriented solid: {path}')
    if np.any(reloaded.area_faces <= 0):
        raise GeometryRejected(f'Exported STL has degenerate faces: {path}')
    with open(path, 'xb') as stream:
        stream.write(payload)


def load_lead_reference(input_stl: str) -> Reference:
    """Load numeric sections only when they belong to this exact wire STL."""
    filename = lead_reference_path(input_stl)
    if not os.path.isfile(filename):
        raise FileNotFoundError(
            'Corrected leads require ordered path data. Regenerate the wire '
            f'with the updated generator. Missing: {filename}')
    with np.load(filename, allow_pickle=False) as saved:
        if int(saved['format_version']) != 1:
            raise GeometryRejected('Unsupported lead reference format. Regenerate the wire.')
        if str(saved['source_sha256'].item()) != _file_sha256(input_stl):
            raise GeometryRejected('Lead reference does not match the selected wire STL. Regenerate the wire.')
        if str(saved['coordinate_frame'].item()) != 'physical_xyz' or str(saved['units'].item()) != 'm':
            raise GeometryRejected('Lead reference must use physical XYZ coordinates in metres.')
    reference = Reference.load(filename)
    _validate_reference_matches_wire(reference, input_stl)
    return reference


def run_leads(cfg: Config, input_stl: Optional[str] = None) -> tuple[str, str, str]:
    """Return (with_leads, coil_open, leads_only); also save a current-path NPZ.

    The NPZ at ``current_path_path(with_leads)`` has ``path`` shaped (N, 3),
    in metres, traversed consistently with the optimized coil current. It
    represents the retained coil plus both terminals, without a supply return.
    """
    if input_stl is None:
        from .paths import resolve_wire_stl_path
        input_stl = resolve_wire_stl_path(
            cfg.output_dir, cfg.gradient_axis, cfg.tikhonov_factor, cfg.num_levels)
    if not input_stl or not os.path.isfile(input_stl):
        raise FileNotFoundError(f'Wire STL file not found: {input_stl!r}')
    input_stl = os.path.abspath(input_stl)
    reference = load_lead_reference(input_stl)
    preset = cfg.lead_preset()
    direction = (preset.exit_direction if preset.exit_direction is not None
                 else preset.lead_direction)
    projection = float(np.dot(direction, reference.axis))
    if abs(projection) < 1e-9:
        raise GeometryRejected('Lead exit direction is perpendicular to the saved cylinder axis.')
    exit_sign = 1 if projection > 0 else -1
    options = _automatic_options(reference, cfg)
    options.sector_angular_half = float(preset.sector_angular_half)
    print('\nGenerating surface leads from the original sweep sections...')
    print(f'  Radial x transverse profile: {2*reference.semi_a*1000:.3f} x '
          f'{2*reference.semi_b*1000:.3f} mm')
    print(f'  Minimum bend radius: {options.min_bend_radius*1000:.3f} mm')
    final, lead_parts, open_mesh, report = _plan_with_sector_fallback(
        reference, preset, exit_sign, options,
        expand_cut_search=cfg.gradient_axis.lower() == 'z')
    current_path = final_current_path(reference, lead_parts, report)
    source_hash = _file_sha256(input_stl)
    report.update(current_path_units='m', coordinate_frame='physical_xyz',
                  current_path_source=os.path.abspath(lead_reference_path(input_stl)),
                  source_sha256=source_hash, supply_return_included=False)
    output_dir = os.path.abspath(cfg.output_dir or os.path.dirname(input_stl))
    os.makedirs(output_dir, exist_ok=True)
    output_stl, coil_open_stl, leads_only_stl = unique_lead_output_paths(
        os.path.join(output_dir, os.path.basename(input_stl)))
    report_path = os.path.splitext(output_stl)[0] + '_lead_report.json'
    final_path_file = current_path_path(output_stl)
    leads_only = trimesh.util.concatenate([tube_mesh(lead.rings) for lead in lead_parts])
    # Validate every file in a temporary directory before publishing this run.
    published = []
    with TemporaryDirectory(prefix='.leads-', dir=output_dir) as staging:
        staged = []
        for mesh, destination, closed in (
            (open_mesh, coil_open_stl, False),
            (leads_only, leads_only_stl, True),
            (final, output_stl, True),
        ):
            temporary = os.path.join(staging, os.path.basename(destination))
            _export_m(mesh, temporary, must_be_closed=closed)
            staged.append((temporary, destination))
        with_leads_hash = _file_sha256(staged[-1][0])
        report['with_leads_sha256'] = with_leads_hash
        temporary = os.path.join(staging, os.path.basename(report_path))
        with open(temporary, 'x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        staged.append((temporary, report_path))
        temporary = os.path.join(staging, os.path.basename(final_path_file))
        with np.load(lead_reference_path(input_stl), allow_pickle=False) as saved:
            gradient_axis = str(saved['gradient_axis'].item())
            b0_direction = str(saved['b0_direction'].item())
        np.savez_compressed(
            temporary, format_version=np.asarray(1), path=current_path,
            original_path=reference.path, axis=reference.axis, origin=reference.origin,
            cut_indices=np.asarray(report['cut_indices'], dtype=np.int64),
            units=np.asarray('m'), coordinate_frame=np.asarray('physical_xyz'),
            orientation=np.asarray('original_coil_current'),
            supply_return_included=np.asarray(False), source_sha256=np.asarray(source_hash),
            with_leads_sha256=np.asarray(with_leads_hash),
            gradient_axis=np.asarray(gradient_axis), b0_direction=np.asarray(b0_direction),
        )
        staged.append((temporary, final_path_file))
        try:
            for temporary, destination in staged:
                with open(destination, 'xb') as stream:
                    published.append(destination)
                    with open(temporary, 'rb') as source:
                        copyfileobj(source, stream)
        except BaseException:
            for destination in published:
                os.unlink(destination)
            raise
    print(f'  Cut removed: {report["cut_removed_arc_mm"]:.3f} mm')
    print(f'  Leads STL: {output_stl}')
    print(f'  Final current path: {final_path_file}')
    return output_stl, coil_open_stl, leads_only_stl


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Add surface-constrained lead wires.')
    parser.add_argument('input_stl', nargs='?', default='')
    parser.add_argument('--axis', default='y', choices=('x', 'y', 'z'))
    parser.add_argument('--output-dir', default='')
    args = parser.parse_args()
    cfg = Config(gradient_axis=args.axis)
    cfg.output_dir = args.output_dir
    run_leads(cfg, args.input_stl or None)


if __name__ == '__main__':
    main()
