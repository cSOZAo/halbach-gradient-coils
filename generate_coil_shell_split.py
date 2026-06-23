"""
Split coil former — carve wire-path grooves into Fusion 360 half-cylinder STLs.

Loads pre-designed half-cylinder STLs exported from Fusion 360 and subtracts
the pyCoilGen wire path from each half.  Fusion dimensions are read from the
STL files (no hard-coded length/radius).

Alignment uses the coil-only wire STL (no leads) so lead extensions do not
shift the gradient axially on the cylinder.  Boolean subtraction uses the
wire STL with leads so inlet/outlet grooves are carved.

Usage:
    python generate_coil_shell_split.py
    python run_coil_mold_pipeline.py

Dependencies:
    numpy, trimesh, manifold3d, scikit-image (for marching cubes)
"""

import os
import sys
import time

import numpy as np
import trimesh
import trimesh.smoothing
import manifold3d as m3d

import coil_mold_common as cfg


# Re-export user-facing knobs from the shared config (edit coil_mold_common.py).
SUBTRACT_WIRE_STL = cfg.SUBTRACT_WIRE_STL
ALIGN_WIRE_STL = cfg.ALIGN_WIRE_STL
LEADS_WIRE_STL = cfg.LEADS_WIRE_STL
GRADIENT_LAYER = cfg.GRADIENT_LAYER
SHELL_STL_DIR = cfg.SHELL_STL_DIR
CYL_ROT_AXIS = cfg.CYL_ROT_AXIS
CYL_ROT_ANGLE = cfg.CYL_ROT_ANGLE
GROOVE_EXPANSION = cfg.GROOVE_EXPANSION
LEAD_GROOVE_EXPANSION = cfg.LEAD_GROOVE_EXPANSION
OUTER_SKIN_TRIM = cfg.OUTER_SKIN_TRIM
COIL_SECOND_SUBTRACT = cfg.COIL_SECOND_SUBTRACT
COIL_SECOND_EXPANSION = cfg.COIL_SECOND_EXPANSION
RESOLVE_SELF_INTERSECTIONS = cfg.RESOLVE_SELF_INTERSECTIONS
VOXEL_PITCH = cfg.VOXEL_PITCH
SMOOTH_ITERATIONS = cfg.SMOOTH_ITERATIONS
VOXEL_SLAB_LENGTH = cfg.VOXEL_SLAB_LENGTH
VOXEL_SLAB_OVERLAP = cfg.VOXEL_SLAB_OVERLAP
CIRCULAR_SEGMENTS = cfg.CIRCULAR_SEGMENTS
OUTPUT_DIR = cfg.OUTPUT_DIR
OUTPUT_IN_MM = cfg.OUTPUT_IN_MM
VOXEL_CROP_PAD = 0.005           # [m] margin around shell bbox for voxel pass


def crop_mesh_to_bounds(wire_tm, bounds):
    """Keep faces with at least one vertex inside *bounds* (shell envelope)."""
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
    """Tight axis-aligned box for one axial slab of *wire_tm*."""
    axis = np.asarray(axis, dtype=float)
    axial = wire_tm.vertices @ axis
    mask = (axial >= a_lo) & (axial <= a_hi)
    if not np.any(mask):
        return None
    pts = wire_tm.vertices[mask]
    return np.array([pts.min(axis=0) - pad, pts.max(axis=0) + pad])


def resolve_self_intersections_slabbed(wire_tm, axis, pitch):
    """
    Voxel-remesh the wire in overlapping axial slabs and union the solids.

    A single voxel pass on the full coil grid is ~700 M voxels and hangs.
    Slabbing along the cylinder axis keeps each pass to ~15–20 M voxels.
    """
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    axial = wire_tm.vertices @ axis
    a_min, a_max = float(axial.min()), float(axial.max())

    combined = None
    pos = a_min
    slab_idx = 0
    print(f"    Slabbed voxel pass: pitch={pitch*1000:.2f} mm, "
          f"slab={VOXEL_SLAB_LENGTH*1000:.0f} mm, overlap={VOXEL_SLAB_OVERLAP*1000:.0f} mm")

    while pos < a_max - 1e-9:
        a_lo = pos - (VOXEL_SLAB_OVERLAP if slab_idx > 0 else 0.0)
        a_hi = min(pos + VOXEL_SLAB_LENGTH, a_max) + VOXEL_SLAB_OVERLAP
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
        pos += VOXEL_SLAB_LENGTH
        slab_idx += 1

    if combined is None:
        raise RuntimeError("Slabbed voxel pass produced no geometry.")
    return combined


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

    if SMOOTH_ITERATIONS > 0:
        print(f"    Taubin smoothing ({SMOOTH_ITERATIONS} iterations)...")
        trimesh.smoothing.filter_taubin(
            mc, lamb=0.5, nu=0.53, iterations=SMOOTH_ITERATIONS)
        print(f"    Smoothed mesh  : {len(mc.vertices)} verts, "
              f"{len(mc.faces)} faces")

    return mc


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

    wire_tm.fix_normals()
    print(f"  {label}: {len(wire_tm.vertices)} verts, {len(wire_tm.faces)} faces")

    if expansion > 0:
        print(f"  Fattening {label} by {expansion * 1000:.2f} mm (per side)...")
        wire_tm = expand_wire_mesh(wire_tm, expansion)

    if RESOLVE_SELF_INTERSECTIONS:
        print(f"  WARNING: voxel remesh enabled for {label} — grooves may look blocky.")
        print("  Resolving self-intersections (slabbed voxel pass)...")
        t_vox = time.perf_counter()
        wire_man = resolve_self_intersections_slabbed(wire_tm, axis, VOXEL_PITCH)
        print(f"  Voxel union ready ({time.perf_counter() - t_vox:.1f} s)", flush=True)
        return wire_man

    return manifold_from_trimesh(wire_tm)


def prepare_leads_manifold(stl_path, expansion, label='leads'):
    """Load leads STL; build manifold as union of connected components."""
    print(f"  Loading {label}: {stl_path}", flush=True)
    wire_tm = trimesh.load(stl_path)
    if expansion > 0:
        print(f"  Fattening {label} by {expansion * 1000:.2f} mm (per side)...")
        wire_tm = expand_wire_mesh(wire_tm, expansion)

    components = wire_tm.split(only_watertight=False)
    print(f"  {label}: {len(components)} component(s), "
          f"{sum(len(c.faces) for c in components)} faces total")
    combined = None
    for i, comp in enumerate(components):
        if not comp.is_watertight:
            print(f"    WARNING: component {i + 1} not watertight — attempting repair")
            comp.fill_holes()
            comp.fix_normals()
        combined = manifold_from_trimesh(comp) if combined is None else combined + manifold_from_trimesh(comp)
    return combined


def subtract_wire_from_shell(shell_man, wire_man, label):
    print(f"    Subtracting from half {label.upper()}...", flush=True)
    t0 = time.perf_counter()
    result = shell_man - wire_man
    print(f"      Done ({time.perf_counter() - t0:.2f} s)")
    return result


def build_radius_limiter(fusion_dims, align_axial_center, trim_m):
    """
    Solid cylinder r = outer_r - trim, aligned with the Fusion shell.

    Intersecting the carved half with this volume peels off the outermost
    *trim* radial band (surface flash from overlapping wire booleans).
    """
    outer_r = fusion_dims['outer_r_m'] - trim_m
    height = fusion_dims['axial_length_m'] + 0.002
    m3d.set_circular_segments(CIRCULAR_SEGMENTS)
    cyl = m3d.Manifold.cylinder(height, outer_r, center=True)
    R = cfg.rodrigues_rotation_matrix(CYL_ROT_AXIS, CYL_ROT_ANGLE)
    axis_hat = R @ np.array([0.0, 0.0, 1.0])
    T = np.zeros((3, 4), dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = align_axial_center * axis_hat
    return cyl.transform(T)


def trim_outer_skin(result_man, limiter, label):
    print(f"    Trimming outer {OUTER_SKIN_TRIM * 1000:.2f} mm skin on {label}...", flush=True)
    t0 = time.perf_counter()
    trimmed = result_man ^ limiter
    print(f"      Done ({time.perf_counter() - t0:.2f} s)")
    return trimmed


def load_fusion_half_mesh(stl_path, fusion_dims, align_axial_center):
    """
    Load a Fusion half-cylinder STL and transform it to pyCoilGen's frame.

    Fusion STLs: millimetres, axis along +Z, spanning z=[z_min, z_max].
    pyCoilGen wire: metres, axis along X after R_y(pi/2), coil centred on the
    axial midpoint measured from the coil-only wire STL.
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

    verts = np.asarray(tm.vertices, dtype=np.float64) * 0.001
    tm_m = trimesh.Trimesh(vertices=verts, faces=tm.faces.copy(), process=False)

    # Centre the full cylinder axially (both halves share the same z span).
    fusion_axial_center = fusion_dims['axial_center_m']
    verts_centred = tm_m.vertices.copy()
    verts_centred[:, 2] -= fusion_axial_center
    tm_centred = trimesh.Trimesh(vertices=verts_centred,
                                 faces=tm_m.faces.copy(), process=False)

    R = cfg.rodrigues_rotation_matrix(CYL_ROT_AXIS, CYL_ROT_ANGLE)
    verts_rotated = (R @ tm_centred.vertices.T).T

    axis_hat = R @ np.array([0.0, 0.0, 1.0])
    verts_final = verts_rotated + align_axial_center * axis_hat

    tm_final = trimesh.Trimesh(vertices=verts_final,
                               faces=tm_centred.faces.copy(), process=False)

    print(f"      Transformed (m): [{tm_final.bounds[0][0]:.4f}, "
          f"{tm_final.bounds[0][1]:.4f}, {tm_final.bounds[0][2]:.4f}] --> "
          f"[{tm_final.bounds[1][0]:.4f}, {tm_final.bounds[1][1]:.4f}, "
          f"{tm_final.bounds[1][2]:.4f}]")

    axis = cfg.rotated_cylinder_axis(CYL_ROT_AXIS, CYL_ROT_ANGLE)
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


def load_fusion_half(stl_path, fusion_dims, align_axial_center):
    return manifold_from_trimesh(
        load_fusion_half_mesh(stl_path, fusion_dims, align_axial_center))


def shell_output_base(subtract_path):
    """Stable output stem: strip _with_leads so names read ..._shell_g2a.stl."""
    wire_base = os.path.splitext(os.path.basename(subtract_path))[0]
    if wire_base.endswith('_with_leads'):
        wire_base = wire_base[:-len('_with_leads')]
    if 'wire' in wire_base:
        return wire_base.replace('wire', 'shell')
    return wire_base + '_shell'


def run_shell_split(
    wire_with_leads_stl,
    *,
    align_wire_stl=None,
    leads_wire_stl=None,
    gradient_layer=None,
    shell_stl_dir=None,
    output_dir=None,
    groove_expansion=None,
    lead_groove_expansion=None,
    outer_skin_trim=None,
    coil_second_subtract=None,
    coil_second_expansion=None,
    resolve_self_intersections=None,
    output_in_mm=None,
):
    """
    Carve Fusion 360 shell halves using wire STLs supplied by the caller.

    Parameters
    ----------
    wire_with_leads_stl : str
        Cable mesh with leads (single-pass subtract fallback, or paired with
        coil-only / leads-only STLs for two-pass mode).
    align_wire_stl : str, optional
        Coil-only wire for axial alignment.  Defaults to the sibling STL without
        ``_with_leads`` in the filename, else *wire_with_leads_stl*.
    leads_wire_stl : str, optional
        Leads-only mesh for pass 2.  Defaults to ``<stem>_leads_only.stl`` next
        to *wire_with_leads_stl*; if missing, subtract uses *wire_with_leads_stl*
        in a single pass.
    gradient_layer, shell_stl_dir, output_dir, groove_expansion, ...
        Override ``coil_mold_common`` defaults when not None.
    """
    m3d.set_circular_segments(CIRCULAR_SEGMENTS)

    subtract_path = os.path.normpath(wire_with_leads_stl)
    align_path = os.path.normpath(
        align_wire_stl or cfg.derive_align_wire_path(subtract_path))
    if leads_wire_stl is not None:
        leads_path = os.path.normpath(leads_wire_stl)
    else:
        base, ext = os.path.splitext(subtract_path)
        if base.endswith('_with_leads'):
            base = base[:-len('_with_leads')]
        leads_path = os.path.normpath(base + '_leads_only' + ext)

    layer = GRADIENT_LAYER if gradient_layer is None else gradient_layer
    shell_dir = SHELL_STL_DIR if shell_stl_dir is None else shell_stl_dir
    out_dir_cfg = OUTPUT_DIR if output_dir is None else output_dir
    groove_exp = GROOVE_EXPANSION if groove_expansion is None else groove_expansion
    lead_groove_exp = (
        LEAD_GROOVE_EXPANSION if lead_groove_expansion is None else lead_groove_expansion)
    skin_trim = OUTER_SKIN_TRIM if outer_skin_trim is None else outer_skin_trim
    second_sub = COIL_SECOND_SUBTRACT if coil_second_subtract is None else coil_second_subtract
    second_exp = COIL_SECOND_EXPANSION if coil_second_expansion is None else coil_second_expansion
    voxel_remesh = (
        RESOLVE_SELF_INTERSECTIONS
        if resolve_self_intersections is None else resolve_self_intersections)
    export_mm = OUTPUT_IN_MM if output_in_mm is None else output_in_mm
    two_pass = os.path.isfile(leads_path)

    print("=" * 70)
    print("  Split Coil Shell Generator -- Fusion 360 halves + wire subtraction")
    print("=" * 70)
    print()
    print(f"  Gradient layer      : {layer}  "
          f"(g_{layer}a + g_{layer}b)")
    print(f"  Align wire STL      : {align_path}")
    print(f"  Coil subtract STL   : {align_path}")
    if two_pass:
        print(f"  Leads subtract STL  : {leads_path}")
    else:
        print(f"  Leads subtract STL  : (missing — run add_coil_leads.py first)")
        print(f"  Fallback wire STL   : {subtract_path}")
    print(f"  Shell STL dir       : {shell_dir}")
    print(f"  Coil expansion      : {groove_exp * 1000:.2f} mm")
    print(f"  Lead expansion      : {lead_groove_exp * 1000:.2f} mm")
    print(f"  Two-pass subtract   : {two_pass}")
    print(f"  Outer skin trim     : {skin_trim * 1000:.2f} mm")
    print(f"  Coil 2nd pass exp   : {second_exp * 1000:.2f} mm" if second_sub else "  Coil 2nd pass       : off")
    print(f"  Voxel remesh        : {voxel_remesh}")
    print(f"  Output units        : {'mm' if export_mm else 'm'}")
    print()

    if not os.path.isfile(align_path):
        print(f"  ERROR: coil wire STL not found:\n    {align_path}")
        sys.exit(1)
    if not two_pass and not os.path.isfile(subtract_path):
        print(f"  ERROR: no leads-only STL and no with_leads fallback at:\n    {subtract_path}")
        sys.exit(1)

    stl_a = os.path.join(shell_dir, f'g_{layer}a.stl')
    stl_b = os.path.join(shell_dir, f'g_{layer}b.stl')
    for path in (stl_a, stl_b):
        if not os.path.isfile(path):
            print(f"  ERROR: shell STL not found:\n    {path}")
            sys.exit(1)

    fusion_dims = cfg.detect_fusion_cylinder_dims(stl_a, stl_b)
    print("  Fusion cylinder (from STLs):")
    print(f"    Axial span   : {fusion_dims['z_min_mm']:.1f} -- "
          f"{fusion_dims['z_max_mm']:.1f} mm  "
          f"(length {fusion_dims['axial_length_m']*1000:.1f} mm)")
    print(f"    Radial range : {fusion_dims['inner_r_m']*1000:.2f} -- "
          f"{fusion_dims['outer_r_m']*1000:.2f} mm")
    print()

    align_dims = cfg.measure_wire_dims(align_path, CYL_ROT_AXIS, CYL_ROT_ANGLE)
    print("  Wire alignment (coil only, no leads):")
    print_wire_dims(align_dims, "Align")
    if two_pass:
        leads_dims = cfg.measure_wire_dims(leads_path, CYL_ROT_AXIS, CYL_ROT_ANGLE)
        print("  Leads-only mesh:")
        print_wire_dims(leads_dims, "Leads")
    else:
        subtract_dims = cfg.measure_wire_dims(subtract_path, CYL_ROT_AXIS, CYL_ROT_ANGLE)
        print("  Wire subtraction (with leads, single pass):")
        print_wire_dims(subtract_dims, "Subtract")
        axial_shift = subtract_dims['axial_center'] - align_dims['axial_center']
        if abs(axial_shift) > 1e-4:
            print(f"  NOTE: subtract mesh centre differs from align by "
                  f"{axial_shift*1000:.2f} mm — alignment uses coil-only centre.")
    print()

    total_steps = (7 if two_pass else 6) + (1 if second_sub else 0)
    step = 1
    print(f"  [{step}/{total_steps}] Loading Fusion 360 shell halves...")
    t0 = time.perf_counter()
    mesh_a = load_fusion_half_mesh(stl_a, fusion_dims, align_dims['axial_center'])
    mesh_b = load_fusion_half_mesh(stl_b, fusion_dims, align_dims['axial_center'])
    t1 = time.perf_counter()
    print(f"    Both halves loaded ({t1 - t0:.2f} s)")
    print()

    step += 1
    axis = cfg.rotated_cylinder_axis(CYL_ROT_AXIS, CYL_ROT_ANGLE)
    coil_wire_fat = None
    if second_sub and second_exp > groove_exp:
        print(f"  [{step}/{total_steps}] Preparing coil wire (pass 1b, "
              f"{second_exp * 1000:.2f} mm expansion)...")
        t0 = time.perf_counter()
        coil_wire_fat = prepare_wire_manifold(
            align_path, second_exp, axis, label='coil (2nd pass)')
        print(f"    Ready ({time.perf_counter() - t0:.2f} s)")
        print()

    leads_wire = None
    if two_pass:
        step += 1
        print(f"  [{step}/{total_steps}] Preparing coil wire mesh (pass 1)...")
        t0 = time.perf_counter()
        coil_wire = prepare_wire_manifold(align_path, groove_exp, axis, label='coil')
        t1 = time.perf_counter()
        print(f"    Coil wire ready ({t1 - t0:.2f} s)")
        print()

        step += 1
        print(f"  [{step}/{total_steps}] Preparing leads mesh (pass 2)...")
        t0 = time.perf_counter()
        leads_wire = prepare_leads_manifold(
            leads_path, lead_groove_exp, label='leads')
        t1 = time.perf_counter()
        print(f"    Leads wire ready ({t1 - t0:.2f} s)")
        print()
    else:
        step += 1
        print(f"  [{step}/{total_steps}] Preparing with_leads mesh (single pass)...")
        t0 = time.perf_counter()
        coil_wire = prepare_wire_manifold(
            subtract_path, groove_exp, axis, label='with_leads')
        t1 = time.perf_counter()
        print(f"    Wire ready ({t1 - t0:.2f} s)")
        print()

    step += 1
    print(f"  [{step}/{total_steps}] Converting shell halves to manifolds...")
    t0 = time.perf_counter()
    shell_a = manifold_from_trimesh(mesh_a)
    shell_b = manifold_from_trimesh(mesh_b)
    t1 = time.perf_counter()
    print(f"    Both halves ready ({t1 - t0:.2f} s)")
    print()

    step += 1
    print(f"  [{step}/{total_steps}] Boolean subtraction...")
    if two_pass:
        print("    Pass 1 — coil grooves...")
        result_a = subtract_wire_from_shell(shell_a, coil_wire, f'A (g_{layer}a)')
        result_b = subtract_wire_from_shell(shell_b, coil_wire, f'B (g_{layer}b)')
        if coil_wire_fat is not None:
            print(f"    Pass 1b — extra coil cut ({second_exp * 1000:.2f} mm)...")
            result_a = subtract_wire_from_shell(result_a, coil_wire_fat, f'A 2nd')
            result_b = subtract_wire_from_shell(result_b, coil_wire_fat, f'B 2nd')
        print("    Pass 2 — lead grooves...")
        result_a = subtract_wire_from_shell(result_a, leads_wire, f'A leads')
        result_b = subtract_wire_from_shell(result_b, leads_wire, f'B leads')
    else:
        print("    Single pass — with_leads mesh...")
        result_a = subtract_wire_from_shell(shell_a, coil_wire, f'A (g_{layer}a)')
        result_b = subtract_wire_from_shell(shell_b, coil_wire, f'B (g_{layer}b)')
    if skin_trim > 0:
        print(f"    Pass 3 — peel outer {skin_trim * 1000:.2f} mm skin...")
        limiter = build_radius_limiter(fusion_dims, align_dims['axial_center'],
                                         skin_trim)
        result_a = trim_outer_skin(result_a, limiter, f'A (g_{layer}a)')
        result_b = trim_outer_skin(result_b, limiter, f'B (g_{layer}b)')
    print()

    step += 1
    print(f"  [{step}/{total_steps}] Exporting results...")
    out_dir = out_dir_cfg or os.path.dirname(subtract_path)
    shell_base = shell_output_base(subtract_path)

    for label, result_manifold in [('a', result_a), ('b', result_b)]:
        result_tm = trimesh_from_manifold(result_manifold)
        if export_mm:
            result_tm.vertices *= 1000.0

        out_name = f"{shell_base}_g{layer}{label}.stl"
        out_path = os.path.join(out_dir, out_name)
        result_tm.export(out_path)

        unit = 'mm' if export_mm else 'm'
        bb = result_tm.bounds
        print(f"    Half {label.upper()} (g_{layer}{label}):")
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
    print("  Done! Open both STLs in your slicer to verify the grooves.")
    print("=" * 70)

    return out_dir, shell_base, layer


def main():
    run_shell_split(SUBTRACT_WIRE_STL)


if __name__ == '__main__':
    main()
