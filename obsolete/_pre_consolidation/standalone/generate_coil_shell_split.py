"""
Split coil former — carve wire-path grooves into Fusion 360 half-cylinder STLs.

Loads pre-designed half-cylinder STLs exported from Fusion 360 and subtracts
the pyCoilGen wire path from each half.  Fusion dimensions are read from the
STL files (no hard-coded length/radius).

Alignment uses the coil-only wire STL (no leads) so lead extensions do not
shift the gradient axially on the cylinder.  Subtraction uses the open cable
with leads (``*_with_leads.stl``) so the cut gap is not carved as a groove.

Usage:
    python generate_coil_shell_split.py

Dependencies:
    numpy, trimesh, manifold3d, scikit-image (marching cubes)
"""

import os
import sys
import time
from typing import Dict, Tuple

import numpy as np
import trimesh
import trimesh.smoothing
import manifold3d as m3d

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from output_utils import (
    resolve_lead_stl_paths,
    resolve_wire_stl_path,
    standalone_design_dir,
    unique_path,
)

# ---- USER PARAMETERS --------------------------------------------------------
GRADIENT_AXIS = 'y'
TIKHONOV_FACTOR = 2500
NUM_LEVELS = 26
GRADIENT_LAYER = 2

OUTPUT_DIR = standalone_design_dir(GRADIENT_AXIS, TIKHONOV_FACTOR, NUM_LEVELS)
SHELL_STL_DIR = os.path.join(PROJECT_ROOT, 'assets', 'cilindros_gradientes_grandes')

ALIGN_WIRE_STL = resolve_wire_stl_path(
    OUTPUT_DIR, GRADIENT_AXIS, TIKHONOV_FACTOR, NUM_LEVELS,
)
SUBTRACT_WIRE_STL, _COIL_OPEN_STL, LEADS_WIRE_STL = resolve_lead_stl_paths(ALIGN_WIRE_STL)

CYL_ROT_AXIS = (0, 1, 0)
CYL_ROT_ANGLE = 3.141592653589793 / 2
GROOVE_EXPANSION = 0.0              # [m] 0 = no fattening; increase for groove clearance
OUTPUT_IN_MM = True

# Solution 1 — union connected wire components, then subtract once.
UNION_BEFORE_SUBTRACT = False

# Solution 2 — voxel-remesh wire (slabbed) into a valid solid before subtract.
# Disable solution 1 when testing this.  Slow at fine pitch.
VOXEL_BEFORE_SUBTRACT = True
VOXEL_PITCH = 0.0002                # [m] 0.2 mm voxel — high-res (very slow)
VOXEL_SLAB_LENGTH = 0.040             # [m] shorter slabs → smaller voxel grids
VOXEL_SLAB_OVERLAP = 0.008            # [m] overlap between axial slabs
VOXEL_CROP_PAD = 0.003                # [m] margin around each slab bbox
VOXEL_SMOOTH_ITERATIONS = 0           # 0 = sharp marching-cubes; >0 softens (blocky look)
# 'ray' avoids subdivide OOM on dense slabs; 'subdivide' is sharper but RAM-heavy.
VOXEL_METHOD = 'ray'
VOXEL_MAX_INPUT_FACES = 60000         # decimate slab mesh before subdivide (if used)

# =============================================================================


def rodrigues_rotation_matrix(axis, angle) -> np.ndarray:
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def rotated_cylinder_axis(rot_axis=CYL_ROT_AXIS, rot_angle=CYL_ROT_ANGLE) -> np.ndarray:
    R = rodrigues_rotation_matrix(rot_axis, rot_angle)
    return R @ np.array([0.0, 0.0, 1.0])


def detect_fusion_cylinder_dims(stl_a: str, stl_b: str) -> Dict[str, float]:
    """Measure Fusion cylinder span from both printable halves [m]."""
    zs_mm = []
    rs_mm = []
    for path in (stl_a, stl_b):
        tm = trimesh.load(path)
        zs_mm.append(tm.vertices[:, 2])
        rs_mm.append(np.linalg.norm(tm.vertices[:, :2], axis=1))

    z_all = np.concatenate(zs_mm)
    r_all = np.concatenate(rs_mm)
    z_min_mm = float(z_all.min())
    z_max_mm = float(z_all.max())

    return {
        'axial_center_m': (z_min_mm + z_max_mm) * 0.0005,
        'axial_length_m': (z_max_mm - z_min_mm) * 0.001,
        'inner_r_m': float(r_all.min()) * 0.001,
        'outer_r_m': float(r_all.max()) * 0.001,
        'z_min_mm': z_min_mm,
        'z_max_mm': z_max_mm,
    }


def measure_wire_dims(stl_path: str,
                      rot_axis=CYL_ROT_AXIS,
                      rot_angle=CYL_ROT_ANGLE) -> Dict[str, float]:
    """Axial and radial extent of a wire STL in the pyCoilGen frame [m]."""
    axis = rotated_cylinder_axis(rot_axis, rot_angle)
    tm = trimesh.load(stl_path)
    v = np.asarray(tm.vertices, dtype=np.float64)
    axial_coords = v @ axis
    radial_vecs = v - np.outer(axial_coords, axis)
    radii = np.linalg.norm(radial_vecs, axis=1)

    return {
        'inner_r': float(radii.min()),
        'outer_r': float(radii.max()),
        'axial_min': float(axial_coords.min()),
        'axial_max': float(axial_coords.max()),
        'axial_center': float((axial_coords.min() + axial_coords.max()) / 2.0),
        'axial_extent': float(axial_coords.max() - axial_coords.min()),
        'path': stl_path,
    }


def derive_align_wire_path(subtract_path: str) -> str:
    """Return the coil-only STL path paired with a *_with_leads* subtract STL."""
    base, ext = os.path.splitext(subtract_path)
    if base.endswith('_with_leads'):
        candidate = base[:-len('_with_leads')] + ext
        if os.path.isfile(candidate):
            return candidate
    return subtract_path


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


def print_wire_dims(dims, label):
    print(f"    {label} radial range    : {dims['inner_r']*1000:.2f} -- "
          f"{dims['outer_r']*1000:.2f} mm")
    print(f"    {label} axial extent    : "
          f"{dims['axial_min']*1000:.1f} -- {dims['axial_max']*1000:.1f} mm "
          f"(length = {dims['axial_extent']*1000:.1f} mm)")
    print(f"    {label} axial centre    : {dims['axial_center']*1000:.2f} mm")


def load_wire_mesh(stl_path, expansion, label='wire'):
    """Load subtractor mesh, optional groove expansion."""
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
    return wire_tm


def wire_mesh_to_manifold(wire_tm):
    return manifold_from_trimesh(wire_tm)


def crop_mesh_to_bounds(wire_tm, bounds):
    """Keep faces with at least one vertex inside *bounds*."""
    bmin, bmax = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    vmask = np.all((wire_tm.vertices >= bmin) & (wire_tm.vertices <= bmax), axis=1)
    fmask = vmask[wire_tm.faces].any(axis=1)
    if not np.any(fmask):
        return wire_tm
    cropped = trimesh.Trimesh(vertices=wire_tm.vertices.copy(),
                              faces=wire_tm.faces[fmask], process=False)
    cropped.remove_unreferenced_vertices()
    return cropped


def slab_axial_bounds(wire_tm, axis, a_lo, a_hi, pad=VOXEL_CROP_PAD):
    """Axis-aligned bbox for one axial slab of *wire_tm*."""
    axis = np.asarray(axis, dtype=float)
    axial = wire_tm.vertices @ axis
    mask = (axial >= a_lo) & (axial <= a_hi)
    if not np.any(mask):
        return None
    pts = wire_tm.vertices[mask]
    return np.array([pts.min(axis=0) - pad, pts.max(axis=0) + pad])


def rotation_align_vector_to_z(source_axis) -> np.ndarray:
    """4x4 matrix: rotate *source_axis* (unit) onto +Z."""
    src = np.asarray(source_axis, dtype=float)
    src /= np.linalg.norm(src)
    dst = np.array([0.0, 0.0, 1.0])
    if np.allclose(src, dst):
        return np.eye(4)
    if np.allclose(src, -dst):
        return trimesh.transformations.rotation_matrix(np.pi, [1.0, 0.0, 0.0])
    cross = np.cross(src, dst)
    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    skew = np.array([[0.0, -cross[2], cross[1]],
                     [cross[2], 0.0, -cross[0]],
                     [-cross[1], cross[0], 0.0]])
    rot3 = np.eye(3) + skew + skew @ skew * (1.0 / (1.0 + dot))
    mat = np.eye(4)
    mat[:3, :3] = rot3
    return mat


def voxel_mesh_from_trimesh(wire_tm, pitch, cylinder_axis):
    """Voxelize, solid-fill, marching-cubes remesh at *pitch* [m]."""
    tm = wire_tm.copy()
    to_z = rotation_align_vector_to_z(cylinder_axis)
    from_z = np.linalg.inv(to_z)
    tm.apply_transform(to_z)

    method = VOXEL_METHOD
    if method == 'subdivide' and len(tm.faces) > VOXEL_MAX_INPUT_FACES:
        print(f"    Decimating {len(tm.faces)} -> {VOXEL_MAX_INPUT_FACES} faces "
              f"before voxel...", flush=True)
        tm = tm.simplify_quadric_decimation(VOXEL_MAX_INPUT_FACES)

    print(f"    Voxelizing at {pitch * 1000:.2f} mm pitch (method={method})...",
          flush=True)
    vg = tm.voxelized(pitch, method=method)
    print(f"    Surface voxels : {vg.filled_count}  "
          f"(grid {vg.shape[0]}x{vg.shape[1]}x{vg.shape[2]})", flush=True)

    vg_filled = vg.fill()
    print(f"    After fill     : {vg_filled.filled_count} voxels", flush=True)

    print("    Running marching cubes...", flush=True)
    mc = vg_filled.marching_cubes
    mc.vertices = mc.vertices * vg_filled.pitch + vg_filled.translation
    mc.apply_transform(from_z)
    print(f"    Remeshed       : {len(mc.vertices)} verts, {len(mc.faces)} faces",
          flush=True)

    if VOXEL_SMOOTH_ITERATIONS > 0:
        print(f"    Taubin smoothing ({VOXEL_SMOOTH_ITERATIONS} iterations)...",
              flush=True)
        trimesh.smoothing.filter_taubin(
            mc, lamb=0.5, nu=0.53, iterations=VOXEL_SMOOTH_ITERATIONS)
    return mc


def voxel_wire_slabbed(wire_tm, axis, pitch):
    """
    Voxel-remesh overlapping axial slabs (solution 2).

    Returns one Manifold per slab.  Subtract from the shell sequentially
    (same net groove volume as union-then-subtract, without a giant union).
    """
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    axial = wire_tm.vertices @ axis
    a_min, a_max = float(axial.min()), float(axial.max())

    slabs = []
    pos = a_min
    slab_idx = 0
    print(f"  Slabbed voxel: pitch={pitch*1000:.2f} mm, "
          f"slab={VOXEL_SLAB_LENGTH*1000:.0f} mm, "
          f"overlap={VOXEL_SLAB_OVERLAP*1000:.0f} mm", flush=True)

    while pos < a_max - 1e-9:
        a_lo = pos - (VOXEL_SLAB_OVERLAP if slab_idx > 0 else 0.0)
        a_hi = min(pos + VOXEL_SLAB_LENGTH, a_max) + VOXEL_SLAB_OVERLAP
        bounds = slab_axial_bounds(wire_tm, axis, a_lo, a_hi)
        if bounds is not None:
            cropped = crop_mesh_to_bounds(wire_tm, bounds)
            if len(cropped.faces) > 0:
                t0 = time.perf_counter()
                print(f"  Slab {slab_idx + 1}: axial [{a_lo*1000:.0f}, {a_hi*1000:.0f}] mm, "
                      f"{len(cropped.faces)} faces...", flush=True)
                mc = voxel_mesh_from_trimesh(cropped, pitch, axis)
                slabs.append(manifold_from_trimesh(mc))
                print(f"    Slab done ({time.perf_counter() - t0:.1f} s)", flush=True)
        pos += VOXEL_SLAB_LENGTH
        slab_idx += 1

    if not slabs:
        raise RuntimeError("Slabbed voxel pass produced no geometry.")
    return slabs


def split_wire_components(wire_tm):
    """Connected components of the wire mesh."""
    components = wire_tm.split(only_watertight=False)
    return [c for c in components if len(c.faces) > 0] or [wire_tm]


def union_wire_manifolds(wire_mans, label):
    """Boolean-union manifold pieces into one subtractor solid (solution 1)."""
    if not wire_mans:
        raise RuntimeError(f"No manifold pieces to union for {label}.")
    if len(wire_mans) == 1:
        return wire_mans[0]

    print(f"  Unioning {len(wire_mans)} {label} component(s)...", flush=True)
    t0 = time.perf_counter()
    combined = wire_mans[0]
    for i, piece_man in enumerate(wire_mans[1:], start=2):
        print(f"    + component {i}/{len(wire_mans)}...", flush=True)
        combined = combined + piece_man
    print(f"  Union ready ({time.perf_counter() - t0:.1f} s)", flush=True)
    return combined


def prepare_wire_subtractor(
    stl_path,
    expansion,
    *,
    union_components=False,
    voxel_remesh=False,
    axis=None,
    label='wire',
):
    """
    Build the wire Manifold subtractor.

    Baseline: direct mesh → Manifold.
    Solution 1 (``union_components``): union connected components.
    Solution 2 (``voxel_remesh``): slabbed voxel remesh; returns a list of
    slab Manifolds for sequential shell subtraction.
    """
    wire_tm = load_wire_mesh(stl_path, expansion, label=label)

    if voxel_remesh:
        if axis is None:
            axis = rotated_cylinder_axis(CYL_ROT_AXIS, CYL_ROT_ANGLE)
        print(f"  Solution 2 — voxel remesh ({VOXEL_PITCH * 1000:.2f} mm pitch)...",
              flush=True)
        t_vox = time.perf_counter()
        slab_mans = voxel_wire_slabbed(wire_tm, axis, VOXEL_PITCH)
        print(f"  {len(slab_mans)} voxel slab(s) ready "
              f"({time.perf_counter() - t_vox:.1f} s)", flush=True)
        return slab_mans

    if not union_components:
        return wire_mesh_to_manifold(wire_tm)

    components = split_wire_components(wire_tm)
    print(f"  {label}: {len(components)} connected component(s) for union")
    mans = []
    for i, comp in enumerate(components, start=1):
        comp = comp.copy()
        comp.fix_normals()
        if not comp.is_watertight:
            print(f"    WARNING: component {i} not watertight — repair")
            comp.fill_holes()
            comp.fix_normals()
        mans.append(wire_mesh_to_manifold(comp))
    return union_wire_manifolds(mans, label)


def subtract_wire_from_shell(shell_man, wire_man, label):
    print(f"    Subtracting from half {label.upper()}...", flush=True)
    t0 = time.perf_counter()
    result = shell_man - wire_man
    print(f"      Done ({time.perf_counter() - t0:.2f} s)")
    return result


def subtract_slabs_from_shell(shell_man, slab_mans, half_label):
    """Sequential subtract — avoids unioning all voxel slabs into one solid."""
    result = shell_man
    n = len(slab_mans)
    for i, slab_man in enumerate(slab_mans, start=1):
        result = subtract_wire_from_shell(
            result, slab_man, f'{half_label} slab {i}/{n}')
    return result


def load_fusion_half_mesh(stl_path, fusion_dims, align_axial_center):
    """
    Load a Fusion half-cylinder STL and transform it to pyCoilGen's frame.

    Fusion STLs: millimetres, axis along +Z.
    pyCoilGen wire: metres, axis along X after R_y(pi/2), centred on the
    axial midpoint of the coil-only wire STL.
    """
    print(f"    Loading: {os.path.basename(stl_path)}")
    tm = trimesh.load(stl_path)
    print(f"      Original: {len(tm.vertices)} verts, {len(tm.faces)} faces")

    if not tm.is_watertight:
        print("      WARNING: Fusion mesh is NOT watertight. Attempting repair...")
        tm.fill_holes()
        tm.fix_normals()

    verts = np.asarray(tm.vertices, dtype=np.float64) * 0.001
    tm_m = trimesh.Trimesh(vertices=verts, faces=tm.faces.copy(), process=False)

    verts_centred = tm_m.vertices.copy()
    verts_centred[:, 2] -= fusion_dims['axial_center_m']
    tm_centred = trimesh.Trimesh(vertices=verts_centred,
                                 faces=tm_m.faces.copy(), process=False)

    R = rodrigues_rotation_matrix(CYL_ROT_AXIS, CYL_ROT_ANGLE)
    verts_rotated = (R @ tm_centred.vertices.T).T
    axis_hat = R @ np.array([0.0, 0.0, 1.0])
    verts_final = verts_rotated + align_axial_center * axis_hat

    tm_final = trimesh.Trimesh(vertices=verts_final,
                               faces=tm_centred.faces.copy(), process=False)

    axis = rotated_cylinder_axis(CYL_ROT_AXIS, CYL_ROT_ANGLE)
    v = tm_final.vertices
    axial_coords = v @ axis
    radii = np.linalg.norm(v - np.outer(axial_coords, axis), axis=1)
    print(f"      Shell radial range : {radii.min()*1000:.2f} -- "
          f"{radii.max()*1000:.2f} mm")
    print(f"      Shell axial range  : {axial_coords.min()*1000:.1f} -- "
          f"{axial_coords.max()*1000:.1f} mm")
    return tm_final


def shell_output_base(subtract_path):
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
    gradient_layer=None,
    shell_stl_dir=None,
    output_dir=None,
    groove_expansion=None,
    union_before_subtract=None,
    voxel_before_subtract=None,
    output_in_mm=None,
):
    """Carve Fusion shell halves: align on coil-only STL, subtract wire with leads."""
    subtract_path = os.path.normpath(wire_with_leads_stl)
    align_path = os.path.normpath(
        align_wire_stl or derive_align_wire_path(subtract_path))

    layer = GRADIENT_LAYER if gradient_layer is None else gradient_layer
    shell_dir = SHELL_STL_DIR if shell_stl_dir is None else shell_stl_dir
    out_dir_cfg = OUTPUT_DIR if output_dir is None else output_dir
    groove_exp = GROOVE_EXPANSION if groove_expansion is None else groove_expansion
    use_union = (
        UNION_BEFORE_SUBTRACT
        if union_before_subtract is None else union_before_subtract)
    use_voxel = (
        VOXEL_BEFORE_SUBTRACT
        if voxel_before_subtract is None else voxel_before_subtract)
    export_mm = OUTPUT_IN_MM if output_in_mm is None else output_in_mm

    if use_voxel and use_union:
        print("  ERROR: enable only one of UNION_BEFORE_SUBTRACT or "
              "VOXEL_BEFORE_SUBTRACT at a time.")
        sys.exit(1)

    print("=" * 70)
    print("  Split Coil Shell Generator -- Fusion 360 halves + wire subtraction")
    print("=" * 70)
    print()
    print(f"  Gradient layer       : {layer}  (g_{layer}a + g_{layer}b)")
    print(f"  Align wire STL       : {align_path}")
    print(f"  Subtract wire STL    : {subtract_path}")
    print(f"  Groove expansion     : {groove_exp * 1000:.2f} mm")
    print(f"  Union before subtract: {use_union}")
    print(f"  Voxel before subtract: {use_voxel}")
    if use_voxel:
        print(f"  Voxel pitch          : {VOXEL_PITCH * 1000:.2f} mm")
        print(f"  Voxel method         : {VOXEL_METHOD}")
        print(f"  Voxel slab length    : {VOXEL_SLAB_LENGTH * 1000:.0f} mm")
        print(f"  Voxel smooth iters   : {VOXEL_SMOOTH_ITERATIONS}")
    print(f"  Output units         : {'mm' if export_mm else 'm'}")
    print()

    if not os.path.isfile(align_path):
        print(f"  ERROR: coil wire STL not found:\n    {align_path}")
        sys.exit(1)
    if not os.path.isfile(subtract_path):
        print(f"  ERROR: with_leads STL not found:\n    {subtract_path}")
        print("  Run add_coil_leads.py first.")
        sys.exit(1)

    stl_a = os.path.join(shell_dir, f'g_{layer}a.stl')
    stl_b = os.path.join(shell_dir, f'g_{layer}b.stl')
    for path in (stl_a, stl_b):
        if not os.path.isfile(path):
            print(f"  ERROR: shell STL not found:\n    {path}")
            sys.exit(1)

    fusion_dims = detect_fusion_cylinder_dims(stl_a, stl_b)
    print("  Fusion cylinder (from STLs):")
    print(f"    Axial span   : {fusion_dims['z_min_mm']:.1f} -- "
          f"{fusion_dims['z_max_mm']:.1f} mm  "
          f"(length {fusion_dims['axial_length_m']*1000:.1f} mm)")
    print(f"    Radial range : {fusion_dims['inner_r_m']*1000:.2f} -- "
          f"{fusion_dims['outer_r_m']*1000:.2f} mm")
    print()

    align_dims = measure_wire_dims(align_path, CYL_ROT_AXIS, CYL_ROT_ANGLE)
    print("  Wire alignment (coil only, no leads):")
    print_wire_dims(align_dims, "Align")
    subtract_dims = measure_wire_dims(subtract_path, CYL_ROT_AXIS, CYL_ROT_ANGLE)
    print("  Subtract mesh (with leads):")
    print_wire_dims(subtract_dims, "Subtract")
    print()

    axis = rotated_cylinder_axis(CYL_ROT_AXIS, CYL_ROT_ANGLE)
    total_steps = 4
    step = 1
    print(f"  [{step}/{total_steps}] Loading Fusion 360 shell halves...")
    t0 = time.perf_counter()
    mesh_a = load_fusion_half_mesh(stl_a, fusion_dims, align_dims['axial_center'])
    mesh_b = load_fusion_half_mesh(stl_b, fusion_dims, align_dims['axial_center'])
    print(f"    Both halves loaded ({time.perf_counter() - t0:.2f} s)")
    print()

    step += 1
    print(f"  [{step}/{total_steps}] Preparing wire subtractor...")
    t0 = time.perf_counter()
    wire_sub = prepare_wire_subtractor(
        subtract_path, groove_exp,
        union_components=use_union,
        voxel_remesh=use_voxel,
        axis=axis,
        label='with_leads')
    print(f"    Ready ({time.perf_counter() - t0:.2f} s)")
    print()

    step += 1
    print(f"  [{step}/{total_steps}] Boolean subtraction...")
    shell_a = manifold_from_trimesh(mesh_a)
    shell_b = manifold_from_trimesh(mesh_b)
    if isinstance(wire_sub, list):
        print(f"    Sequential subtract — {len(wire_sub)} voxel slab(s) per half...")
        result_a = subtract_slabs_from_shell(shell_a, wire_sub, f'A (g_{layer}a)')
        result_b = subtract_slabs_from_shell(shell_b, wire_sub, f'B (g_{layer}b)')
    else:
        result_a = subtract_wire_from_shell(shell_a, wire_sub, f'A (g_{layer}a)')
        result_b = subtract_wire_from_shell(shell_b, wire_sub, f'B (g_{layer}b)')
    print()

    step += 1
    print(f"  [{step}/{total_steps}] Exporting results...")
    out_dir = out_dir_cfg or os.path.dirname(subtract_path)
    shell_base = shell_output_base(subtract_path)

    for label, result_manifold in [('a', result_a), ('b', result_b)]:
        result_tm = trimesh_from_manifold(result_manifold)
        if len(result_tm.vertices) == 0:
            print(f"    ERROR: Half {label.upper()} is empty after boolean.")
            sys.exit(1)
        if export_mm:
            result_tm.vertices *= 1000.0

        out_name = f"{shell_base}_g{layer}{label}.stl"
        out_path = unique_path(os.path.join(out_dir, out_name))
        result_tm.export(out_path)

        bb = result_tm.bounds
        unit = 'mm' if export_mm else 'm'
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
