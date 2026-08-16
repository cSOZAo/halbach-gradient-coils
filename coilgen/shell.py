"""
Split coil former -- carve wire-path grooves into Fusion 360 half-cylinder
STLs.

Loads pre-designed half-cylinder STLs exported from Fusion 360 and subtracts
the pyCoilGen wire path from each half. Fusion dimensions are read from the
STL files (no hard-coded length/radius).

Alignment uses the coil-only wire STL (no leads) so lead extensions do not
shift the gradient axially on the cylinder. Subtraction uses the open cable
with leads (``*_with_leads.stl``) so the cut gap is not carved as a groove.

Dependencies: numpy, trimesh, manifold3d, scikit-image (for marching cubes).
"""

from __future__ import annotations

import os
import re
import time
from types import SimpleNamespace

import numpy as np
import trimesh
import trimesh.smoothing
import manifold3d as m3d

from . import geometry as geo
from .config import Config, apply_custom_shell_dims
from .paths import unique_path, derive_align_wire_path, resolve_lead_stl_paths


_P = SimpleNamespace()
VOXEL_CROP_PAD = 0.005  # [m] margin around shell bbox for voxel pass


# ---------------------------------------------------------------------------
# Mesh / manifold conversion
# ---------------------------------------------------------------------------

def manifold_from_trimesh(tm):
    mesh = m3d.Mesh(
        vert_properties=np.asarray(tm.vertices, dtype=np.float32),
        tri_verts=np.asarray(tm.faces, dtype=np.uint32),
    )
    return m3d.Manifold(mesh)


def trimesh_from_manifold(manifold):
    mesh = manifold.to_mesh()
    verts = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
    faces = np.asarray(mesh.tri_verts, dtype=np.int64)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def expand_wire_mesh(wire_tm, expansion):
    verts = np.array(wire_tm.vertices, dtype=np.float64)
    normals = np.array(wire_tm.vertex_normals, dtype=np.float64)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-12)
    expanded_verts = verts + expansion * normals
    return trimesh.Trimesh(vertices=expanded_verts,
                           faces=wire_tm.faces.copy(),
                           process=False)


# ---------------------------------------------------------------------------
# Voxel remesh (optional)
# ---------------------------------------------------------------------------

def crop_mesh_to_bounds(wire_tm, bounds):
    bmin, bmax = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    vmask = np.all((wire_tm.vertices >= bmin) & (wire_tm.vertices <= bmax), axis=1)
    fmask = vmask[wire_tm.faces].any(axis=1)
    if not np.any(fmask):
        return wire_tm
    cropped = trimesh.Trimesh(vertices=wire_tm.vertices.copy(),
                              faces=wire_tm.faces[fmask], process=False)
    cropped.remove_unreferenced_vertices()
    return cropped


def slab_axial_bounds(wire_tm, axis, a_lo, a_hi, pad=0.005):
    axis = np.asarray(axis, dtype=float)
    axial = wire_tm.vertices @ axis
    mask = (axial >= a_lo) & (axial <= a_hi)
    if not np.any(mask):
        return None
    pts = wire_tm.vertices[mask]
    return np.array([pts.min(axis=0) - pad, pts.max(axis=0) + pad])


def resolve_self_intersections(wire_tm, pitch):
    print(f"    Voxelizing at {pitch * 1000:.2f} mm pitch...")
    vg = wire_tm.voxelized(pitch)
    print(f"    Surface voxels : {vg.filled_count}  "
          f"(grid {vg.shape[0]}x{vg.shape[1]}x{vg.shape[2]})")

    vg_filled = vg.fill()
    print(f"    After fill     : {vg_filled.filled_count} voxels")

    print("    Running marching cubes...")
    mc = vg_filled.marching_cubes
    mc.vertices = mc.vertices * vg_filled.pitch + vg_filled.translation
    print(f"    Clean mesh     : {len(mc.vertices)} verts, {len(mc.faces)} faces")

    if _P.smooth_iterations > 0:
        print(f"    Taubin smoothing ({_P.smooth_iterations} iterations)...")
        trimesh.smoothing.filter_taubin(
            mc, lamb=0.5, nu=0.53, iterations=_P.smooth_iterations)
        print(f"    Smoothed mesh  : {len(mc.vertices)} verts, "
              f"{len(mc.faces)} faces")

    return mc


def resolve_self_intersections_slabbed(wire_tm, axis, pitch):
    """Voxel-remesh the wire in overlapping axial slabs and union the solids."""
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    axial = wire_tm.vertices @ axis
    a_min, a_max = float(axial.min()), float(axial.max())

    combined = None
    pos = a_min
    slab_idx = 0
    print(f"    Slabbed voxel pass: pitch={pitch*1000:.2f} mm, "
          f"slab={_P.voxel_slab_length*1000:.0f} mm, "
          f"overlap={_P.voxel_slab_overlap*1000:.0f} mm")

    while pos < a_max - 1e-9:
        a_lo = pos - (_P.voxel_slab_overlap if slab_idx > 0 else 0.0)
        a_hi = min(pos + _P.voxel_slab_length, a_max) + _P.voxel_slab_overlap
        bounds = slab_axial_bounds(wire_tm, axis, a_lo, a_hi)
        if bounds is not None:
            cropped = crop_mesh_to_bounds(wire_tm, bounds)
            if len(cropped.faces) > 0:
                t0 = time.perf_counter()
                print(f"    Slab {slab_idx + 1}: axial [{a_lo*1000:.0f}, {a_hi*1000:.0f}] mm, "
                      f"{len(cropped.faces)} faces...", flush=True)
                mc = resolve_self_intersections(cropped, pitch)
                slab_man = manifold_from_trimesh(mc)
                combined = slab_man if combined is None else combined + slab_man
                print(f"      slab done ({time.perf_counter() - t0:.1f} s)", flush=True)
        pos += _P.voxel_slab_length
        slab_idx += 1

    if combined is None:
        raise RuntimeError("Slabbed voxel pass produced no geometry.")
    return combined


# ---------------------------------------------------------------------------
# Wire / lead preparation
# ---------------------------------------------------------------------------

def print_wire_dims(dims, label):
    print(f"    {label} radial range    : {dims['inner_r']*1000:.2f} -- "
          f"{dims['outer_r']*1000:.2f} mm")
    print(f"    {label} axial extent    : "
          f"{dims['axial_min']*1000:.1f} -- {dims['axial_max']*1000:.1f} mm "
          f"(length = {dims['axial_extent']*1000:.1f} mm)")
    print(f"    {label} axial centre    : {dims['axial_center']*1000:.2f} mm")


def prepare_wire_manifold(stl_path, expansion, axis, label='wire'):
    print(f"  Loading {label}: {stl_path}", flush=True)
    wire_tm = trimesh.load(stl_path)

    if not wire_tm.is_watertight:
        print(f"  WARNING: {label} mesh is NOT watertight. Attempting repair...")
        wire_tm.fill_holes()
        if not wire_tm.is_watertight:
            print(f"  WARNING: {label} mesh is still not watertight after repair -- "
                  f"the boolean subtraction may drop or distort grooves.")

    wire_tm.fix_normals()
    print(f"  {label}: {len(wire_tm.vertices)} verts, {len(wire_tm.faces)} faces")

    if expansion > 0:
        print(f"  Fattening {label} by {expansion * 1000:.2f} mm (per side)...")
        wire_tm = expand_wire_mesh(wire_tm, expansion)

    if _P.resolve_self_intersections:
        print(f"  WARNING: voxel remesh enabled for {label} -- grooves may look blocky.")
        print("  Resolving self-intersections (slabbed voxel pass)...")
        t_vox = time.perf_counter()
        wire_man = resolve_self_intersections_slabbed(wire_tm, axis, _P.voxel_pitch)
        print(f"  Voxel union ready ({time.perf_counter() - t_vox:.1f} s)", flush=True)
        return wire_man

    return manifold_from_trimesh(wire_tm)


def prepare_open_coil_manifold(stl_path, expansion, axis, label='coil_open'):
    """Load open gradient loop; do not fill cut-face holes."""
    print(f"  Loading {label}: {stl_path}", flush=True)
    wire_tm = trimesh.load(stl_path)
    wire_tm.fix_normals()
    if wire_tm.is_watertight:
        print(f"  NOTE: {label} is watertight (expected open mesh at cut faces)")
    else:
        print(f"  {label}: open mesh at weld cut (holes not filled)")
    print(f"  {label}: {len(wire_tm.vertices)} verts, {len(wire_tm.faces)} faces")
    if expansion > 0:
        print(f"  Fattening {label} by {expansion * 1000:.2f} mm (per side)...")
        wire_tm = expand_wire_mesh(wire_tm, expansion)
    if _P.resolve_self_intersections:
        print(f"  WARNING: voxel remesh enabled for {label} -- grooves may look blocky.")
        t_vox = time.perf_counter()
        wire_man = resolve_self_intersections_slabbed(wire_tm, axis, _P.voxel_pitch)
        print(f"  Voxel union ready ({time.perf_counter() - t_vox:.1f} s)", flush=True)
        return wire_man
    return manifold_from_trimesh(wire_tm)


def prepare_lead_components(stl_path, expansion, label='leads'):
    """Load leads STL and return one manifold per connected component."""
    print(f"  Loading {label}: {stl_path}", flush=True)
    wire_tm = trimesh.load(stl_path)
    if expansion > 0:
        print(f"  Fattening {label} by {expansion * 1000:.2f} mm (per side)...")
        wire_tm = expand_wire_mesh(wire_tm, expansion)
    components = wire_tm.split(only_watertight=False)
    print(f"  {label}: {len(components)} component(s), "
          f"{sum(len(c.faces) for c in components)} faces total")
    mans = []
    open_components: list[int] = []
    for i, comp in enumerate(components):
        if not comp.is_watertight:
            print(f"    WARNING: component {i + 1} not watertight -- attempting repair")
            comp.fill_holes()
            comp.fix_normals()
        if not comp.is_watertight:
            open_components.append(i + 1)
            continue
        mans.append(manifold_from_trimesh(comp))
    if open_components:
        raise RuntimeError(
            f"{len(open_components)} of {len(components)} {label} component(s) "
            f"{open_components} "
            f"are not watertight after repair, so their grooves would be missing "
            f"from the shell:\n    {stl_path}\n  Re-run the leads step (e.g. higher "
            f"cs_blend_rings / junction_rigid_steps) before carving the shell.")
    if not mans:
        raise RuntimeError(f"no {label} component found in:\n    {stl_path}")
    return mans


# ---------------------------------------------------------------------------
# Boolean subtraction
# ---------------------------------------------------------------------------

def subtract_wire_from_shell(shell_man, wire_man, label):
    print(f"    Subtracting from half {label.upper()}...", flush=True)
    t0 = time.perf_counter()
    result = shell_man - wire_man
    print(f"      Done ({time.perf_counter() - t0:.2f} s)")
    return result


def subtract_wires_from_shell(shell_man, wire_mans, half_label):
    result = shell_man
    for i, wire_man in enumerate(wire_mans, start=1):
        result = subtract_wire_from_shell(result, wire_man, f'{half_label} #{i}')
    return result


# ---------------------------------------------------------------------------
# Shell padding / radius limiter
# ---------------------------------------------------------------------------

def pad_shell_outer(mesh_tm, fusion_dims, axis_hat, pad_m):
    """Push outer-wall vertices radially outward by *pad_m*."""
    if pad_m <= 0:
        return mesh_tm
    axis_hat = np.asarray(axis_hat, dtype=float)
    axis_hat /= np.linalg.norm(axis_hat)
    v = np.asarray(mesh_tm.vertices, dtype=float)
    axial = v @ axis_hat
    radial = v - np.outer(axial, axis_hat)
    r = np.linalg.norm(radial, axis=1)
    mid_r = 0.5 * (fusion_dims['inner_r_m'] + fusion_dims['outer_r_m'])
    near_outer = r >= mid_r
    rhat = radial / np.maximum(r[:, None], 1e-12)
    v_new = v.copy()
    v_new[near_outer] += pad_m * rhat[near_outer]
    return trimesh.Trimesh(vertices=v_new, faces=mesh_tm.faces.copy(), process=False)


def build_radius_limiter(fusion_dims, align_axial_center, trim_m,
                         rot_axis, rot_angle):
    """Solid cylinder r = outer_r - trim, aligned with the Fusion shell."""
    outer_r = fusion_dims['outer_r_m'] - trim_m
    height = fusion_dims['axial_length_m'] + 0.002
    m3d.set_circular_segments(_P.circular_segments)
    cyl = m3d.Manifold.cylinder(height, outer_r, center=True)
    R = geo.rodrigues_rotation_matrix(rot_axis, rot_angle)
    axis_hat = R @ np.array([0.0, 0.0, 1.0])
    T = np.zeros((3, 4), dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = align_axial_center * axis_hat
    return cyl.transform(T)


def build_auto_outer_limiter(cfg: Config, align_dims: dict,
                             rot_axis, rot_angle):
    """Solid cylinder R = final outer radius X (clips padded shell)."""
    outer_r = cfg.cylinder.radius
    height = float(cfg.cylinder.height) + 0.002
    m3d.set_circular_segments(_P.circular_segments)
    cyl = m3d.Manifold.cylinder(height, outer_r, center=True)
    R = geo.rodrigues_rotation_matrix(rot_axis, rot_angle)
    T = np.zeros((3, 4), dtype=np.float64)
    T[:3, :3] = R
    return cyl.transform(T)


def build_auto_radius_limiter(cfg: Config, align_dims: dict,
                              rot_axis, rot_angle):
    """Backward-compatible alias for :func:`build_auto_outer_limiter`."""
    return build_auto_outer_limiter(cfg, align_dims, rot_axis, rot_angle)


def build_auto_inner_peel_cylinder(cfg: Config, align_dims: dict,
                                   rot_axis, rot_angle):
    """Solid cylinder R = opened inner bore; subtract to peel inner radial trim."""
    inner_r = cfg.shell_build_inner_radius + cfg.radial_peel
    height = float(cfg.cylinder.height) + 0.002
    m3d.set_circular_segments(_P.circular_segments)
    cyl = m3d.Manifold.cylinder(height, inner_r, center=True)
    R = geo.rodrigues_rotation_matrix(rot_axis, rot_angle)
    T = np.zeros((3, 4), dtype=np.float64)
    T[:3, :3] = R
    return cyl.transform(T)


def peel_inner_skin(result_man, inner_cyl, label, skin_m):
    print(f"    Peeling inner {skin_m * 1000:.2f} mm radial trim on {label} "
          f"(opening bore)...", flush=True)
    t0 = time.perf_counter()
    peeled = result_man - inner_cyl
    print(f"      Done ({time.perf_counter() - t0:.2f} s)")
    return peeled


def warn_wire_radial_mismatch(cfg: Config, wire_stl: str, tol_m: float = 0.0003):
    """Log a warning when measured wire radial extent deviates from analytical."""
    if not os.path.isfile(wire_stl):
        print(f"  WARNING: wire STL not found -- skipping radial check:\n"
              f"    {wire_stl}")
        return
    rot_axis = cfg.cylinder.rot_axis
    rot_angle = cfg.cylinder.rot_angle
    dims = geo.measure_wire_dims(wire_stl, rot_axis, rot_angle)
    exp_outer = cfg.estimated_wire_outer_radius
    exp_inner = cfg.estimated_wire_inner_radius
    exp_center = cfg.estimated_wire_radial_center
    d_outer = abs(dims['outer_r'] - exp_outer)
    d_inner = abs(dims['inner_r'] - exp_inner)
    meas_center = 0.5 * (dims['outer_r'] + dims['inner_r'])
    d_center = abs(meas_center - exp_center)
    print("  Wire radial check:")
    print(f"    Measured outer_r : {dims['outer_r']*1000:.2f} mm  "
          f"(expected {exp_outer*1000:.2f} mm = Rext + peel)")
    print(f"    Measured inner_r : {dims['inner_r']*1000:.2f} mm  "
          f"(expected {exp_inner*1000:.2f} mm = Rint - peel)")
    print(f"    Measured center  : {meas_center*1000:.2f} mm  "
          f"(expected {exp_center*1000:.2f} mm = shell center)")
    print(f"    Shell radii      : outer {cfg.shell_outer_radius*1000:.2f} mm, "
          f"inner {cfg.shell_inner_radius*1000:.2f} mm  "
          f"(center {cfg.shell_radial_center*1000:.2f} mm)")
    if d_outer > tol_m or d_inner > tol_m or d_center > tol_m:
        print(f"  WARNING: wire radial extent differs from analytical model by "
              f"up to {max(d_outer, d_inner, d_center)*1000:.2f} mm (tol {tol_m*1000:.1f} mm). "
              f"Grooves may be misaligned vs the shell.")
    else:
        print("    OK — within tolerance.")


def intersect_with_limiter(result_man, limiter, label, reason):
    print(f"    {reason} on {label}...", flush=True)
    t0 = time.perf_counter()
    trimmed = result_man ^ limiter
    print(f"      Done ({time.perf_counter() - t0:.2f} s)")
    return trimmed


# ---------------------------------------------------------------------------
# Fusion half loading
# ---------------------------------------------------------------------------

def load_fusion_half_mesh(stl_path, fusion_dims, align_axial_center,
                          rot_axis, rot_angle):
    """
    Load a Fusion half-cylinder STL and transform it to pyCoilGen's frame.

    Fusion STLs: millimetres, axis along +Z. pyCoilGen wire: metres, axis
    along X after R_y(pi/2), coil centred on the axial midpoint measured from
    the coil-only wire STL.
    """
    print(f"    Loading: {os.path.basename(stl_path)}")
    tm = trimesh.load(stl_path)
    print(f"      Original: {len(tm.vertices)} verts, {len(tm.faces)} faces")
    print(f"      Bounds (mm): [{tm.bounds[0][0]:.1f}, {tm.bounds[0][1]:.1f}, "
          f"{tm.bounds[0][2]:.1f}] --> [{tm.bounds[1][0]:.1f}, "
          f"{tm.bounds[1][1]:.1f}, {tm.bounds[1][2]:.1f}]")

    if not tm.is_watertight:
        print("      WARNING: Fusion mesh is NOT watertight. Attempting repair...")
        tm.fill_holes()
        tm.fix_normals()
        if not tm.is_watertight:
            print(f"      WARNING: {os.path.basename(stl_path)} is still not "
                  f"watertight after repair -- the boolean subtraction may fail.")

    verts = np.asarray(tm.vertices, dtype=np.float64) * 0.001
    tm_m = trimesh.Trimesh(vertices=verts, faces=tm.faces.copy(), process=False)

    fusion_axial_center = fusion_dims['axial_center_m']
    verts_centred = tm_m.vertices.copy()
    verts_centred[:, 2] -= fusion_axial_center
    tm_centred = trimesh.Trimesh(vertices=verts_centred,
                                 faces=tm_m.faces.copy(), process=False)

    R = geo.rodrigues_rotation_matrix(rot_axis, rot_angle)
    verts_rotated = (R @ tm_centred.vertices.T).T

    axis_hat = R @ np.array([0.0, 0.0, 1.0])
    verts_final = verts_rotated + align_axial_center * axis_hat

    tm_final = trimesh.Trimesh(vertices=verts_final,
                               faces=tm_centred.faces.copy(), process=False)

    print(f"      Transformed (m): [{tm_final.bounds[0][0]:.4f}, "
          f"{tm_final.bounds[0][1]:.4f}, {tm_final.bounds[0][2]:.4f}] --> "
          f"[{tm_final.bounds[1][0]:.4f}, {tm_final.bounds[1][1]:.4f}, "
          f"{tm_final.bounds[1][2]:.4f}]")

    axis = geo.rotated_cylinder_axis(rot_axis, rot_angle)
    v = tm_final.vertices
    axial_coords = v @ axis
    axial_proj = np.outer(axial_coords, axis)
    radial_vecs = v - axial_proj
    radii = np.linalg.norm(radial_vecs, axis=1)
    print(f"      Shell radial range : {radii.min()*1000:.2f} -- "
          f"{radii.max()*1000:.2f} mm")
    print(f"      Shell axial range  : {axial_coords.min()*1000:.1f} -- "
          f"{axial_coords.max()*1000:.1f} mm")

    return tm_final


def shell_output_base(subtract_path):
    """Stable output stem: strip _with_leads so names read ..._shell_g2a.stl."""
    wire_base = os.path.splitext(os.path.basename(subtract_path))[0]
    wire_base = re.sub(r'_with_leads(?:\(\d+\))?$', '', wire_base)
    if 'wire' in wire_base:
        return wire_base.replace('wire', 'shell')
    return wire_base + '_shell'


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _populate_params(cfg: Config):
    s = cfg.shell
    _P.layer = s.layer
    _P.subtract_mode = s.subtract_mode
    _P.groove_expansion = s.groove_expansion
    _P.lead_groove_expansion = s.lead_groove_expansion
    _P.leads_second_subtract = s.leads_second_subtract
    _P.shell_outer_pad = s.shell_outer_pad
    if s.use_custom_stl:
        _P.outer_skin_trim = s.outer_skin_trim
    else:
        _P.outer_skin_trim = cfg.outer_skin_trim
    _P.coil_second_subtract = s.coil_second_subtract
    _P.coil_second_expansion = s.coil_second_expansion
    _P.resolve_self_intersections = s.resolve_self_intersections
    _P.voxel_pitch = s.voxel_pitch
    _P.voxel_slab_length = s.voxel_slab_length
    _P.voxel_slab_overlap = s.voxel_slab_overlap
    _P.smooth_iterations = s.smooth_iterations
    _P.circular_segments = s.circular_segments
    _P.output_in_mm = s.output_in_mm
    _P.output_dir = s.custom_stl or ''   # not used; output_dir passed explicitly


def build_auto_hollow_cylinder(cfg: Config, wire_stl: str) -> "m3d.Manifold":
    """
    Build a single 1-piece hollow cylinder from analytical dimensions.

    ``cylinder.radius`` is the shell outer radius R (GUI).  Inner bore is
    R − (2h + gap − 2×radial_peel). Axial length is exactly ``cylinder.height``
    (GUI). Wires are shorter via ``mesh_length_factor`` in run_gradient.
    """
    s = cfg.shell
    rot_axis = cfg.cylinder.rot_axis
    rot_angle = cfg.cylinder.rot_angle

    dims = geo.measure_wire_dims(wire_stl, rot_axis, rot_angle)
    axial_extent = dims['axial_extent']
    # Centre the shell on the design cylinder (origin), not the wire bbox,
    # so length always matches GUI height for Gx/Gy/Gz.
    axial_center = 0.0

    semi_a = cfg.conductor_semi_a
    cable_height = cfg.cable_height
    peel = cfg.radial_peel
    gap = cfg.layer_crossing_gap

    cyl_outer_r = cfg.shell_outer_radius
    cyl_inner_r = cfg.shell_inner_radius
    cyl_length = float(cfg.cylinder.height)
    if cyl_length <= 0:
        raise RuntimeError(
            f"Auto hollow-cylinder height {cyl_length*1000:.2f} mm <= 0.")

    if cyl_inner_r <= 0:
        raise RuntimeError(
            f"Auto hollow-cylinder inner radius {cyl_inner_r*1000:.2f} mm <= 0. "
            f"Reduce cable height, layer gap, margin, or increase outer radius.")
    if cyl_inner_r >= cyl_outer_r:
        raise RuntimeError(
            f"Auto hollow-cylinder dims degenerate: inner_r={cyl_inner_r*1000:.2f} mm "
            f">= outer_r={cyl_outer_r*1000:.2f} mm.")

    warn_wire_radial_mismatch(cfg, wire_stl)

    print("  Auto hollow-cylinder (1 piece, analytical):")
    print(f"    Wire radial range : {dims['inner_r']*1000:.2f} -- "
          f"{dims['outer_r']*1000:.2f} mm  (reference)")
    print(f"    Wire axial extent : {axial_extent*1000:.1f} mm "
          f"(centre {dims['axial_center']*1000:.2f} mm)")
    print(f"    Cable height (h)  : {cable_height*1000:.3f} mm "
          f"(semi_a {semi_a*1000:.3f} mm)")
    print(f"    Layer gap         : {gap*1000:.3f} mm")
    print(f"    Radial margin     : {peel*1000:.3f} mm per face "
          f"({s.auto_margin_pct*100:.1f}% of cable height)")
    print(f"    Wall thickness    : {cfg.shell_wall_thickness*1000:.2f} mm "
          f"(2h + gap - 2*margin)")
    print(f"    cyl_inner         : {cyl_inner_r*1000:.2f} mm")
    print(f"    cyl_outer         : {cyl_outer_r*1000:.2f} mm")
    print(f"    cyl_length        : {cyl_length*1000:.1f} mm "
          f"(GUI cylinder.height)")

    m3d.set_circular_segments(_P.circular_segments)
    # Slightly taller inner bore so the boolean leaves clean open ends.
    eps = 0.002
    outer = m3d.Manifold.cylinder(cyl_length, cyl_outer_r, center=True)
    inner = m3d.Manifold.cylinder(cyl_length + eps, cyl_inner_r, center=True)
    tube = outer - inner

    R = geo.rodrigues_rotation_matrix(rot_axis, rot_angle)
    axis_hat = R @ np.array([0.0, 0.0, 1.0])
    T = np.zeros((3, 4), dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = axial_center * axis_hat
    return tube.transform(T)


def run_shell(
    cfg: Config,
    wire_with_leads_stl: str | None = None,
    *,
    align_wire_stl: str | None = None,
    leads_wire_stl: str | None = None,
    output_dir: str | None = None,
) -> tuple[str, list[str]]:
    """
    Carve Fusion 360 shell halves using the wire-with-leads STL.

    Returns ``(out_dir, [shell_a_path, shell_b_path])``. Resolves the
    coil-only / coil_open / leads_only STLs from the with-leads path when not
    given explicitly. Raises ``RuntimeError`` (no sys.exit) on missing files.
    """
    _populate_params(cfg)
    rot_axis = cfg.cylinder.rot_axis
    rot_angle = cfg.cylinder.rot_angle

    m3d.set_circular_segments(_P.circular_segments)

    # Resolve the with-leads STL.
    if wire_with_leads_stl is None:
        from .paths import resolve_wire_stl_path, resolve_lead_stl_paths
        wire = resolve_wire_stl_path(
            cfg.output_dir, cfg.gradient_axis,
            cfg.tikhonov_factor, cfg.num_levels,
        )
        wire_with_leads_stl, _, _ = resolve_lead_stl_paths(wire)
    subtract_path = os.path.normpath(wire_with_leads_stl)

    align_path = os.path.normpath(align_wire_stl or derive_align_wire_path(subtract_path))
    stem_base, ext = os.path.splitext(subtract_path)
    stem_base = re.sub(r'_with_leads(?:\(\d+\))?$', '', stem_base)
    leads_path = os.path.normpath(
        leads_wire_stl if leads_wire_stl is not None else stem_base + '_leads_only' + ext)
    coil_open_path = os.path.normpath(stem_base + '_coil_open' + ext)
    if leads_wire_stl is None:
        wire_for_resolve = align_path
        if not os.path.isfile(wire_for_resolve):
            wire_for_resolve = stem_base + ext
        _, resolved_coil_open, resolved_leads = resolve_lead_stl_paths(wire_for_resolve)
        if os.path.isfile(resolved_leads):
            leads_path = os.path.normpath(resolved_leads)
        if os.path.isfile(resolved_coil_open):
            coil_open_path = os.path.normpath(resolved_coil_open)

    mode = _P.subtract_mode
    if mode not in ('with_leads', 'with_leads_by_component', 'two_pass'):
        raise RuntimeError(f"unknown subtract_mode {mode!r}")

    # Custom halves: ensure Rext/Rint/height come from the STL when not already set
    # (GUI/CLI may have applied them; layer-only callers need a safety net).
    if cfg.shell.use_custom_stl and cfg.shell.measured_inner_r is None:
        stl_a0, stl_b0 = cfg.shell_half_paths()
        if os.path.isfile(stl_a0) and os.path.isfile(stl_b0):
            apply_custom_shell_dims(cfg, stl_a0, stl_b0)
            _populate_params(cfg)

    auto_mode = not cfg.shell.use_custom_stl
    layer = _P.layer
    shell_dir = cfg.shell.stl_dir
    out_dir = output_dir or os.path.dirname(subtract_path)
    groove_exp = _P.groove_expansion
    lead_groove_exp = _P.lead_groove_expansion
    skin_trim = _P.outer_skin_trim
    outer_pad = _P.shell_outer_pad
    second_sub = _P.coil_second_subtract
    second_exp = _P.coil_second_expansion
    voxel_remesh = _P.resolve_self_intersections
    export_mm = _P.output_in_mm

    print("=" * 70)
    if cfg.shell.use_custom_stl:
        print("  Split Coil Shell Generator -- custom halves + wire subtraction")
    else:
        print("  Coil Shell Generator -- auto hollow cylinder (1 piece) + wire subtraction")
    print("=" * 70)
    print()
    if cfg.shell.use_custom_stl:
        stl_a_log, stl_b_log = cfg.shell_half_paths()
        print(f"  Shell mode          : custom halves (2 pieces)")
        print(f"  Half A              : {stl_a_log}")
        print(f"  Half B              : {stl_b_log}")
        if cfg.shell.measured_inner_r is not None:
            print(f"  Measured Rext/Rint  : "
                  f"{cfg.shell_outer_radius*1000:.2f} / "
                  f"{cfg.shell_inner_radius*1000:.2f} mm")
            print(f"  Measured height     : {cfg.cylinder.height*1000:.1f} mm")
            print(f"  Wall (fixed)        : {cfg.shell_wall_thickness*1000:.2f} mm")
            print(f"  design_r (pack mid) : {cfg.cylinder_design_radius*1000:.2f} mm")
        elif layer:
            print(f"  Asset layer         : {layer}  (g_{layer}a + g_{layer}b)")
    else:
        print(f"  Shell mode          : auto hollow cylinder (1 piece)")
        print(f"  Groove margin       : {cfg.shell.auto_margin_pct*100:.1f}% of cable height "
              f"({cfg.radial_peel*1000:.2f} mm per face)")
        print(f"  Shell wall          : {cfg.shell_wall_thickness*1000:.2f} mm "
              f"(2h + gap - 2*margin)")
        print(f"  Shell height        : {cfg.cylinder.height*1000:.1f} mm "
              f"(GUI)")
        print(f"  Wire mesh height    : "
              f"{cfg.cylinder.height * cfg.cylinder.mesh_length_factor * 1000:.1f} mm "
              f"(×{cfg.cylinder.mesh_length_factor:g})")
    print(f"  Subtract mode       : {mode}")
    print(f"  Align wire STL      : {align_path}")
    print(f"  With-leads STL      : {subtract_path}")
    if mode == 'with_leads_by_component':
        print(f"  Coil-open STL       : {coil_open_path}")
        print(f"  Leads-only STL      : {leads_path}")
    elif mode == 'two_pass':
        print(f"  Coil subtract STL   : {align_path}  (closed loop -- legacy)")
        print(f"  Leads-only STL      : {leads_path}")
    print(f"  Shell STL dir       : {shell_dir}")
    print(f"  Groove expansion    : {groove_exp * 1000:.2f} mm")
    print(f"  Lead expansion      : {lead_groove_exp * 1000:.2f} mm")
    print(f"  Leads 2nd pass      : {_P.leads_second_subtract and lead_groove_exp > 0}")
    print(f"  Shell outer pad     : {outer_pad * 1000:.2f} mm")
    print(f"  Radial peel per face : {skin_trim * 1000:.2f} mm")
    print(f"  Coil 2nd pass exp   : {second_exp * 1000:.2f} mm" if second_sub else "  Coil 2nd pass       : off")
    print(f"  Voxel remesh        : {voxel_remesh}")
    print(f"  Output units        : {'mm' if export_mm else 'm'}")
    print()

    if not os.path.isfile(align_path):
        raise RuntimeError(f"coil wire STL not found:\n    {align_path}")
    if mode in ('with_leads', 'with_leads_by_component'):
        if not os.path.isfile(subtract_path):
            raise RuntimeError(
                f"with_leads STL not found:\n    {subtract_path}\n  Run leads step first.")
    if mode == 'with_leads_by_component':
        if not os.path.isfile(coil_open_path):
            raise RuntimeError(f"coil_open STL not found:\n    {coil_open_path}")
        if not os.path.isfile(leads_path):
            raise RuntimeError(f"leads_only STL not found:\n    {leads_path}")
    if mode == 'two_pass':
        if not os.path.isfile(leads_path):
            raise RuntimeError(f"leads_only STL not found:\n    {leads_path}")

    align_dims = geo.measure_wire_dims(align_path, rot_axis, rot_angle)
    print("  Wire alignment (coil only, no leads):")
    print_wire_dims(align_dims, "Align")
    subtract_dims = geo.measure_wire_dims(subtract_path, rot_axis, rot_angle)
    print("  With-leads mesh:")
    print_wire_dims(subtract_dims, "Subtract")
    axial_shift = subtract_dims['axial_center'] - align_dims['axial_center']
    if abs(axial_shift) > 1e-4:
        print(f"  NOTE: subtract mesh centre differs from align by "
              f"{axial_shift*1000:.2f} mm -- alignment uses coil-only centre.")
    if mode == 'with_leads_by_component':
        coil_open_dims = geo.measure_wire_dims(coil_open_path, rot_axis, rot_angle)
        print("  Coil-open mesh:")
        print_wire_dims(coil_open_dims, "Coil open")
        leads_dims = geo.measure_wire_dims(leads_path, rot_axis, rot_angle)
        print("  Leads-only mesh:")
        print_wire_dims(leads_dims, "Leads")
    print()

    axis = geo.rotated_cylinder_axis(rot_axis, rot_angle)

    # ----- build the shell manifold(s) -------------------------------------
    coil_wire_fat = None
    if auto_mode:
        total_steps = (4 if mode == 'with_leads' else 5) + (
            1 if second_sub and second_exp > groove_exp else 0)
        step = 1
        print(f"  [{step}/{total_steps}] Building auto hollow cylinder (1 piece)...")
        t0 = time.perf_counter()
        shell_man = build_auto_hollow_cylinder(cfg, align_path)
        print(f"    Ready ({time.perf_counter() - t0:.2f} s)")
        print()
        fusion_dims = None
    else:
        stl_a, stl_b = cfg.shell_half_paths()
        for path in (stl_a, stl_b):
            if not os.path.isfile(path):
                raise RuntimeError(f"shell STL not found:\n    {path}")
        fusion_dims = geo.detect_fusion_cylinder_dims(stl_a, stl_b)
        print("  Custom / Fusion cylinder (from STLs):")
        print(f"    Axial span   : {fusion_dims['z_min_mm']:.1f} -- "
              f"{fusion_dims['z_max_mm']:.1f} mm  "
              f"(length {fusion_dims['axial_length_m']*1000:.1f} mm)")
        print(f"    Radial range : {fusion_dims['inner_r_m']*1000:.2f} -- "
              f"{fusion_dims['outer_r_m']*1000:.2f} mm")
        print()
        total_steps = (6 if mode == 'with_leads' else 7) + (
            1 if second_sub and second_exp > groove_exp else 0)
        step = 1
        print(f"  [{step}/{total_steps}] Loading shell halves...")
        t0 = time.perf_counter()
        mesh_a = load_fusion_half_mesh(stl_a, fusion_dims, align_dims['axial_center'],
                                       rot_axis, rot_angle)
        mesh_b = load_fusion_half_mesh(stl_b, fusion_dims, align_dims['axial_center'],
                                       rot_axis, rot_angle)
        if outer_pad > 0:
            print(f"    Padding outer wall by {outer_pad * 1000:.2f} mm...")
            mesh_a = pad_shell_outer(mesh_a, fusion_dims, axis, outer_pad)
            mesh_b = pad_shell_outer(mesh_b, fusion_dims, axis, outer_pad)
        t1 = time.perf_counter()
        print(f"    Both halves loaded ({t1 - t0:.2f} s)")
        print()

    # ----- prepare wire subtractor(s) (shared) ----------------------------
    step += 1
    wire_subtractors = None
    if second_sub and second_exp > groove_exp:
        print(f"  [{step}/{total_steps}] Preparing coil wire (pass 1b, "
              f"{second_exp * 1000:.2f} mm expansion)...")
        t0 = time.perf_counter()
        coil_wire_fat = prepare_wire_manifold(
            align_path, second_exp, axis, label='coil (2nd pass)')
        print(f"    Ready ({time.perf_counter() - t0:.2f} s)")
        print()
        step += 1

    if mode == 'with_leads':
        print(f"  [{step}/{total_steps}] Preparing with_leads mesh...")
        t0 = time.perf_counter()
        wire_subtractors = [prepare_wire_manifold(
            subtract_path, groove_exp, axis, label='with_leads')]
        print(f"    Ready ({time.perf_counter() - t0:.2f} s)")
        print()
    elif mode == 'with_leads_by_component':
        print(f"  [{step}/{total_steps}] Preparing coil-open mesh...")
        t0 = time.perf_counter()
        coil_open_man = prepare_open_coil_manifold(
            coil_open_path, groove_exp, axis, label='coil_open')
        print(f"    Ready ({time.perf_counter() - t0:.2f} s)")
        print()
        step += 1
        print(f"  [{step}/{total_steps}] Preparing lead components...")
        t0 = time.perf_counter()
        lead_mans = prepare_lead_components(leads_path, lead_groove_exp, label='leads')
        wire_subtractors = [coil_open_man] + lead_mans
        print(f"    {len(lead_mans)} lead component(s) ready "
              f"({time.perf_counter() - t0:.2f} s)")
        print()
    elif mode == 'two_pass':
        print(f"  [{step}/{total_steps}] Preparing closed coil mesh (legacy pass 1)...")
        t0 = time.perf_counter()
        coil_wire = prepare_wire_manifold(align_path, groove_exp, axis, label='coil')
        print(f"    Coil wire ready ({time.perf_counter() - t0:.2f} s)")
        print()
        step += 1
        print(f"  [{step}/{total_steps}] Preparing leads mesh (legacy pass 2)...")
        t0 = time.perf_counter()
        lead_mans = prepare_lead_components(leads_path, lead_groove_exp, label='leads')
        wire_subtractors = [coil_wire] + lead_mans
        print(f"    Leads ready ({time.perf_counter() - t0:.2f} s)")
        print()

    # ----- boolean subtraction ---------------------------------------------
    step += 1
    print(f"  [{step}/{total_steps}] Boolean subtraction...")
    if mode == 'with_leads':
        print("    Single pass -- with_leads mesh...")
    elif mode == 'with_leads_by_component':
        print(f"    By component -- coil_open + {len(wire_subtractors) - 1} lead(s)...")
    else:
        print("    Legacy two_pass -- closed coil + leads...")

    if auto_mode:
        result_man = subtract_wires_from_shell(shell_man, wire_subtractors, 'shell')
        if coil_wire_fat is not None:
            print(f"    Extra coil cut ({second_exp * 1000:.2f} mm)...")
            result_man = subtract_wire_from_shell(result_man, coil_wire_fat, 'shell 2nd')
        if _P.leads_second_subtract and lead_groove_exp > 0 and os.path.isfile(leads_path):
            print(f"    Extra lead cut ({lead_groove_exp * 1000:.2f} mm per side, "
                  f"each tube separately)...")
            lead_mans = prepare_lead_components(leads_path, lead_groove_exp,
                                                label='leads (2nd pass)')
            result_man = subtract_wires_from_shell(result_man, lead_mans, 'shell leads')
        results = [('shell', result_man)]
    else:
        step += 1
        print(f"  [{step}/{total_steps}] Converting shell halves to manifolds...")
        t0 = time.perf_counter()
        shell_a = manifold_from_trimesh(mesh_a)
        shell_b = manifold_from_trimesh(mesh_b)
        print(f"    Both halves ready ({time.perf_counter() - t0:.2f} s)")
        print()
        step += 1
        print(f"  [{step}/{total_steps}] Boolean subtraction (per half)...")
        result_a = subtract_wires_from_shell(shell_a, wire_subtractors, f'A (g_{layer}a)')
        result_b = subtract_wires_from_shell(shell_b, wire_subtractors, f'B (g_{layer}b)')
        if coil_wire_fat is not None:
            print(f"    Extra coil cut ({second_exp * 1000:.2f} mm)...")
            result_a = subtract_wire_from_shell(result_a, coil_wire_fat, f'A 2nd')
            result_b = subtract_wire_from_shell(result_b, coil_wire_fat, f'B 2nd')
        if _P.leads_second_subtract and lead_groove_exp > 0 and os.path.isfile(leads_path):
            print(f"    Extra lead cut ({lead_groove_exp * 1000:.2f} mm per side, "
                  f"each tube separately)...")
            lead_mans = prepare_lead_components(leads_path, lead_groove_exp,
                                                label='leads (2nd pass)')
            result_a = subtract_wires_from_shell(result_a, lead_mans, f'A leads (g_{layer}a)')
            result_b = subtract_wires_from_shell(result_b, lead_mans, f'B leads (g_{layer}b)')
        if outer_pad > 0 or skin_trim > 0:
            limiter_design = build_radius_limiter(
                fusion_dims, align_dims['axial_center'], skin_trim, rot_axis, rot_angle)
            if outer_pad > 0:
                print(f"    Pass 3 -- restore design outer "
                      f"({fusion_dims['outer_r_m']*1000 - skin_trim*1000:.2f} mm)...")
                result_a = intersect_with_limiter(
                    result_a, limiter_design, f'A (g_{layer}a)',
                    f"Restoring design outer ({outer_pad * 1000:.2f} mm pad removed)")
                result_b = intersect_with_limiter(
                    result_b, limiter_design, f'B (g_{layer}b)',
                    f"Restoring design outer ({outer_pad * 1000:.2f} mm pad removed)")
            elif skin_trim > 0:
                print(f"    Pass 3 -- peel outer {skin_trim * 1000:.2f} mm radial trim...")
                result_a = intersect_with_limiter(
                    result_a, limiter_design, f'A (g_{layer}a)',
                    f"Trimming outer {skin_trim * 1000:.2f} mm radial trim")
                result_b = intersect_with_limiter(
                    result_b, limiter_design, f'B (g_{layer}b)',
                    f"Trimming outer {skin_trim * 1000:.2f} mm radial trim")
        results = [('a', result_a), ('b', result_b)]
    print()

    # ----- export ----------------------------------------------------------
    step += 1
    print(f"  [{step}/{total_steps}] Exporting results...")
    os.makedirs(out_dir, exist_ok=True)
    shell_base = shell_output_base(subtract_path)

    out_paths = []
    for label, result_manifold in results:
        result_tm = trimesh_from_manifold(result_manifold)
        if result_tm.bounds is None or len(result_tm.vertices) == 0:
            raise RuntimeError(
                f"empty mesh for {'shell' if auto_mode else 'half ' + label.upper()}"
                f" -- boolean subtract failed")
        if export_mm:
            result_tm.vertices *= 1000.0

        if auto_mode:
            out_name = f"{shell_base}.stl"
        else:
            out_name = f"{shell_base}_g{layer}{label}.stl"
        out_path = unique_path(os.path.join(out_dir, out_name))
        result_tm.export(out_path)
        out_paths.append(out_path)

        unit = 'mm' if export_mm else 'm'
        bb = result_tm.bounds
        print(f"    {'Shell' if auto_mode else 'Half ' + label.upper()} "
              f"({'g_'+layer+label if not auto_mode else '1 piece'}):")
        print(f"      Vertices     : {len(result_tm.vertices)}")
        print(f"      Faces        : {len(result_tm.faces)}")
        print(f"      Watertight   : {result_tm.is_watertight}")
        print(f"      Bounding box : [{bb[0][0]:.2f}, {bb[0][1]:.2f}, "
              f"{bb[0][2]:.2f}] --> [{bb[1][0]:.2f}, {bb[1][1]:.2f}, "
              f"{bb[1][2]:.2f}] ({unit})")
        print(f"      File         : {out_path}")
        print(f"      Size         : {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")
        print()

    print("=" * 70)
    if auto_mode:
        print("  Done! Open the shell STL in your slicer to verify the grooves.")
    else:
        print("  Done! Open both STLs in your slicer to verify the grooves.")
    print("=" * 70)

    return out_dir, out_paths


def main():
    """CLI entry: carve shell halves for a previous run's wire-with-leads STL."""
    import argparse
    parser = argparse.ArgumentParser(description="Carve Fusion shell halves.")
    parser.add_argument('with_leads_stl', nargs='?', default='')
    parser.add_argument('--axis', default='y', choices=('x', 'y', 'z'))
    parser.add_argument('--layer', type=int, default=None)
    parser.add_argument('--output-dir', default='')
    args = parser.parse_args()

    cfg = Config(gradient_axis=args.axis)
    if args.layer is not None:
        cfg.shell.layer = args.layer
        # Layer selects a pair from assets/shells (custom halves workflow).
        cfg.shell.use_custom_stl = True
    if args.output_dir:
        cfg.output_dir = args.output_dir
    run_shell(cfg, wire_with_leads_stl=args.with_leads_stl or None,
              output_dir=args.output_dir or None)


if __name__ == '__main__':
    main()
