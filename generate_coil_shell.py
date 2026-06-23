"""
Printable coil former — cylinder shell with wire-path grooves.

Takes the wire-path STL produced by pyCoilGen and subtracts it from a
cylinder shell, producing a 3D-printable former with grooves where the
wire should be wound.

The shell's inner radius, outer radius and axial length are
auto-detected from the wire mesh: the walls sit flush with the
innermost and outermost wire surfaces so that, after the groove is
cut, the wire path is exposed on both the inside and outside of the
former.  Only the cylinder orientation (rotation axis/angle) and an
optional EXTRA_CYL_HEIGHT margin need to be configured.

The boolean subtraction is done with **manifold3d**, which is orders of
magnitude faster than trying to do the same operation in Fusion 360 on
dense triangle meshes.

Self-intersecting wire paths (where tubes overlap at crossings) are
handled via a **voxelization + marching cubes** pass that converts the
wire mesh into a clean, non-self-intersecting volume before the boolean.
Without this, overlapping regions get inverted by manifold3d's even-odd
rule, filling gaps instead of carving them.

Usage:
    1. Set the USER PARAMETERS below (especially WIRE_STL_PATH).
    2. Run:  python generate_coil_shell.py
    3. Open the resulting *_shell.stl in your slicer.

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


# =============================================================================
# USER PARAMETERS — edit these to match your coil design
# =============================================================================

# ---- Input: wire-path STL from pyCoilGen -----------------------------------
# Absolute or relative path to the wire STL file produced by pyCoilGen.
# WIRE_STL_PATH = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)),
#     'resultados', 'resultados_main_z',
#     'Belen_Santi_main_Gz_tk30000_lvl16_wire_0_z.stl',
# )

WIRE_STL_PATH = r"C:\Clemente\VSCode\pyCoilGen-0.2.4\pyCoilGen-0.2.4\pruebas\resultados\resultados_main_x\Gradient_Gx_tk30000_lvl20_wire_0_x.stl"

# ---- Cylinder orientation (must match the coil design) --------------------
# The radial bounds (inner radius, outer radius) and axial length of the
# shell are auto-detected from the wire STL — see ``measure_wire_dims``.
# Only the rotation needs to be set explicitly, since it defines which
# world-space direction is the cylinder's axial direction.
CYL_ROT_AXIS    = (0, 1, 0)     # rotation axis applied by pyCoilGen mesh plugin
CYL_ROT_ANGLE   = np.pi / 2     # [rad] rotation angle (aligns axis perp. to B0)

# ---- Axial margin ----------------------------------------------------------
# Extra axial length added on top of the wire's axial extent.  Set to 0 for
# a shell that is exactly as long as the wire bounding box; increase for
# some breathing room at the endcaps.  Split evenly between both ends.
EXTRA_CYL_HEIGHT = 0.005        # [m] total extra length (2 mm per end)

# ---- Radial offsets --------------------------------------------------------
# Signed offsets applied to the auto-detected inner and outer radii.  Each
# is a distance from the cylinder axis: positive values push the wall
# OUTWARD (increase distance from the axis), negative values pull it
# INWARD (decrease distance).  Use them to leave extra material inside or
# outside the wire tube, or to bury the wire below the surface.
# Examples:
#   INNER_RADIUS_OFFSET = -0.001  -> shell inner wall sits 1 mm closer to
#                                    the axis than the innermost wire
#                                    surface (inner wire path stays buried).
#   OUTER_RADIUS_OFFSET = +0.001  -> shell outer wall sits 1 mm farther
#                                    from the axis than the outermost wire
#                                    surface (outer wire path stays buried).
# Set both to 0 for walls exactly flush with the wire surfaces.
INNER_RADIUS_OFFSET = 0.001       # [m] added to auto-detected inner radius
OUTER_RADIUS_OFFSET = -0.002      # [m] added to auto-detected outer radius

# ---- Groove clearance -------------------------------------------------------
# Extra expansion applied to the wire tube before subtraction, by displacing
# each vertex along its surface *normal*.  This fattens the tube uniformly
# in every direction perpendicular to the wire, which:
#   (1) makes the groove slightly wider than the wire so it fits comfortably
#       when winding;
#   (2) merges closely-spaced adjacent grooves into one, eliminating thin
#       strips of unremoved shell material between neighbouring wire turns.
# Set to 0 for an exact-fit, single-wire groove.
GROOVE_EXPANSION = 0.000       # [m] 0.1 mm clearance on every side of the wire

# ---- Self-intersection resolution ------------------------------------------
# When wire tubes overlap (at path crossings, between go/return layers, etc.)
# the raw wire STL self-intersects.  manifold3d treats self-intersections
# with an even-odd rule, which INVERTS the interior of overlapping regions
# and produces filled patches instead of clean grooves.
#
# Setting RESOLVE_SELF_INTERSECTIONS = True adds a voxelization pass that
# converts the wire mesh into a guaranteed non-self-intersecting volume
# via marching cubes.  This costs ~10-15 seconds extra but fixes all
# collision artifacts.
#
# VOXEL_PITCH controls the resolution.  Smaller = finer detail but slower
# and more memory.  The pitch should be small enough to capture the wire
# tube cross-section (~2 mm wide): 0.3-0.5 mm works well.
RESOLVE_SELF_INTERSECTIONS = False
VOXEL_PITCH = 0.0004            # [m] 0.4 mm voxel size

# Number of Taubin smoothing iterations applied after marching cubes.
# This removes the blocky/staircase look from the voxelized grooves
# without shrinking the mesh (Taubin is a non-shrinking low-pass filter).
# More iterations = smoother grooves.  10-20 is usually enough;
# set to 0 to skip smoothing entirely.
SMOOTH_ITERATIONS = 15 #10

# ---- Mesh resolution -------------------------------------------------------
# Number of segments used to approximate circles in the cylinder shell.
# Higher = smoother cylinder, but more triangles. 256 is a good default.
CIRCULAR_SEGMENTS = 256

# ---- Output -----------------------------------------------------------------
# Where to save the resulting shell STL.  Default: next to the wire STL.
OUTPUT_STL_PATH =  ''  # leave empty to auto-generate from WIRE_STL_PATH

# =============================================================================
# END USER PARAMETERS
# =============================================================================


def rodrigues_rotation_matrix(axis, angle):
    """Build a 3x3 rotation matrix from an axis and angle (Rodrigues)."""
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    K = np.array([[    0, -k[2],  k[1]],
                  [ k[2],     0, -k[0]],
                  [-k[1],  k[0],     0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def rotated_cylinder_axis(rot_axis, rot_angle):
    """Return the unit vector of the cylinder axis after rotation.

    The cylinder plugin builds along +Z, then rotates by (rot_axis,
    rot_angle). This gives the rotated axis direction.
    """
    R = rodrigues_rotation_matrix(rot_axis, rot_angle)
    return R @ np.array([0.0, 0.0, 1.0])


def manifold_from_trimesh(tm):
    """Convert a trimesh.Trimesh to a manifold3d.Manifold."""
    mesh = m3d.Mesh(
        vert_properties=np.asarray(tm.vertices, dtype=np.float32),
        tri_verts=np.asarray(tm.faces, dtype=np.uint32),
    )
    return m3d.Manifold(mesh)


def trimesh_from_manifold(manifold):
    """Convert a manifold3d.Manifold back to a trimesh.Trimesh."""
    mesh = manifold.to_mesh()
    verts = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
    faces = np.asarray(mesh.tri_verts, dtype=np.int64)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def build_cylinder_shell(inner_r, outer_r, height, axial_center=0.0):
    """Build the cylinder shell as a Manifold (outer - inner cylinder),
    already rotated and translated to match pyCoilGen's coordinate frame.

    pyCoilGen's cylinder mesh plugin builds the mesh along +Z, centered
    at the origin, then applies the Rodrigues rotation.  We replicate
    that here, with the additional freedom of sliding the shell along
    its own axis so it can be centered on the wire's axial midpoint
    (the wire is not guaranteed to be symmetric about the origin).

    Args:
        inner_r:      [m] inner radius of the shell (matches innermost
                      wire surface).
        outer_r:      [m] outer radius of the shell (matches outermost
                      wire surface).
        height:       [m] total axial length of the shell.
        axial_center: [m] axial offset of the shell center, measured
                      along the rotated cylinder axis in world frame.
    """
    m3d.set_circular_segments(CIRCULAR_SEGMENTS)

    # Build along Z, centered at origin
    outer = m3d.Manifold.cylinder(height, outer_r, center=True)
    inner = m3d.Manifold.cylinder(height, inner_r, center=True)
    shell = outer - inner

    # Apply the same rotation that pyCoilGen applied to the coil mesh,
    # plus a translation along the rotated axis so the shell is centered
    # on the wire's axial midpoint.
    R = rodrigues_rotation_matrix(CYL_ROT_AXIS, CYL_ROT_ANGLE)
    axis_hat = R @ np.array([0.0, 0.0, 1.0])
    T = np.zeros((3, 4), dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3]  = axial_center * axis_hat
    shell = shell.transform(T)

    return shell


def expand_wire_mesh(wire_tm, expansion):
    """Fatten the wire tube by displacing each vertex along its surface
    normal.

    This widens the tube uniformly in every cross-sectional direction
    (axial, circumferential, radial), which is what actually merges
    adjacent grooves.

    NOTE: an earlier version of this function displaced vertices radially
    outward from the *cylinder axis* (not the wire).  That only shifted
    the tube bodily outward without widening its cross-section, so it
    did **not** help merge adjacent grooves -- thin strips of unremoved
    shell material survived between neighbouring wire turns.  Vertex
    normals fix this: the groove grows by `expansion` on every side.

    Args:
        wire_tm:   trimesh.Trimesh of the wire tube.
        expansion: [m] clearance added on every side of the wire.

    Returns:
        A new trimesh.Trimesh with expanded vertices.
    """
    verts = np.array(wire_tm.vertices, dtype=np.float64)
    # trimesh computes area-weighted vertex normals lazily
    normals = np.array(wire_tm.vertex_normals, dtype=np.float64)
    # Guard against zero-length normals from any degenerate faces
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-12)

    expanded_verts = verts + expansion * normals
    return trimesh.Trimesh(vertices=expanded_verts,
                           faces=wire_tm.faces.copy(),
                           process=False)


def resolve_self_intersections(wire_tm, pitch):
    """Voxelize the wire mesh and remesh via marching cubes to eliminate
    self-intersections.

    When wire tubes overlap (at path crossings, between go/return
    layers connected by normal_shift, etc.) the raw mesh from pyCoilGen
    contains coincident and interpenetrating surfaces.  manifold3d's
    boolean engine treats those overlapping regions with an even-odd
    winding rule, effectively INVERTING the interior of each overlap
    zone.  The visual result is filled patches (instead of clean grooves)
    wherever two wires cross.

    This function fixes the problem by:
      1. Voxelizing the mesh surface (marking every voxel the surface
         passes through).
      2. Filling the interior via scipy's ``binary_fill_holes`` (flood
         fill from the grid boundary -- any cell unreachable from outside
         is interior).
      3. Extracting a new isosurface with scikit-image's marching cubes.

    The result is a clean, non-self-intersecting mesh that represents
    the UNION of all overlapping tube volumes -- exactly what the
    boolean subtraction needs.

    Args:
        wire_tm: trimesh.Trimesh of the (possibly self-intersecting)
            wire tube mesh.
        pitch:   [m] voxel edge length.  Smaller = finer detail but
            slower and more memory.  Recommended: 0.3-0.5 mm for a
            ~2 mm wire tube.

    Returns:
        A new trimesh.Trimesh with no self-intersections.
    """
    print(f"    Voxelizing at {pitch * 1000:.2f} mm pitch...")
    vg = wire_tm.voxelized(pitch)
    print(f"    Surface voxels : {vg.filled_count}  "
          f"(grid {vg.shape[0]}x{vg.shape[1]}x{vg.shape[2]})")

    # Fill interior -- any voxel not reachable from the grid boundary
    # via an empty-voxel path is considered interior.
    vg_filled = vg.fill()
    print(f"    After fill     : {vg_filled.filled_count} voxels")

    # Extract clean isosurface via marching cubes.
    # trimesh's marching_cubes returns vertices in voxel-index space;
    # we must transform them back to world coordinates.
    print(f"    Running marching cubes...")
    mc = vg_filled.marching_cubes

    # Transform from voxel-index space to world coordinates.
    # VoxelGrid stores a 4x4 affine: world = transform @ [ix, iy, iz, 1].
    # The transform encodes pitch (scale) and origin (translation).
    mc.vertices = mc.vertices * vg_filled.pitch + vg_filled.translation

    print(f"    Clean mesh     : {len(mc.vertices)} verts, "
          f"{len(mc.faces)} faces")

    # Taubin smoothing: removes the blocky/staircase artifacts from
    # marching cubes without shrinking the mesh.  The alternating
    # positive/negative weights (lambda/mu) act as a low-pass filter
    # that rounds off voxel edges while preserving volume.
    if SMOOTH_ITERATIONS > 0:
        print(f"    Taubin smoothing ({SMOOTH_ITERATIONS} iterations)...")
        trimesh.smoothing.filter_taubin(
            mc, lamb=0.5, nu=0.53, iterations=SMOOTH_ITERATIONS)
        print(f"    Smoothed mesh  : {len(mc.vertices)} verts, "
              f"{len(mc.faces)} faces")

    return mc


def measure_wire_dims(wire_tm):
    """Measure the wire's radial and axial extent along the cylinder axis.

    This is what drives the auto-detection of shell dimensions.  The
    inner radius of the shell matches the innermost wire surface; the
    outer radius matches the outermost wire surface -- so the groove
    cuts all the way through the wall, exposing the wire on both sides
    of the former.  The axial extent plus ``EXTRA_CYL_HEIGHT`` gives
    the shell's total length.

    Returns:
        dict with keys:
            inner_r       : [m] min distance from the cylinder axis
            outer_r       : [m] max distance from the cylinder axis
            axial_min     : [m] min signed axial coordinate along the axis
            axial_max     : [m] max signed axial coordinate along the axis
            axial_center  : [m] midpoint of the axial extent
            axial_extent  : [m] axial_max - axial_min
    """
    axis = rotated_cylinder_axis(CYL_ROT_AXIS, CYL_ROT_ANGLE)

    v = np.asarray(wire_tm.vertices, dtype=np.float64)
    axial_coords = v @ axis
    axial_proj = np.outer(axial_coords, axis)
    radial_vecs = v - axial_proj
    radii = np.linalg.norm(radial_vecs, axis=1)

    dims = {
        'inner_r':      float(radii.min()),
        'outer_r':      float(radii.max()),
        'axial_min':    float(axial_coords.min()),
        'axial_max':    float(axial_coords.max()),
        'axial_center': float((axial_coords.min() + axial_coords.max()) / 2),
        'axial_extent': float(axial_coords.max() - axial_coords.min()),
    }

    # Diagnostics -- useful for tuning GROOVE_EXPANSION and sanity checks.
    components = wire_tm.split(only_watertight=False)
    print(f"    Connected components : {len(components)}")
    print(f"    Wire radial range    : {dims['inner_r']*1000:.2f} -- "
          f"{dims['outer_r']*1000:.2f} mm")
    tube_r = (dims['outer_r'] - dims['inner_r']) / 2
    print(f"    Est. tube radius     : {tube_r*1000:.2f} mm")
    print(f"    Wire axial extent    : "
          f"{dims['axial_min']*1000:.1f} -- {dims['axial_max']*1000:.1f} mm "
          f"(length = {dims['axial_extent']*1000:.1f} mm)")

    return dims


def load_wire_manifold():
    """Load the wire-path STL and convert to a Manifold.

    Processing pipeline:
      1. Load the STL from disk.
      2. Measure wire dimensions (before any modification) so the shell
         can be built flush with the innermost/outermost wire surfaces.
      3. (Optional) Expand the tube along vertex normals for groove
         clearance.
      4. (Optional) Voxelize + marching cubes to eliminate self-
         intersections at wire crossings.
      5. Convert the clean mesh to a manifold3d Manifold.

    Returns:
        (wire_manifold, dims)
            wire_manifold : manifold3d.Manifold ready for subtraction.
            dims          : dict from ``measure_wire_dims`` describing the
                            raw wire geometry (used to size the shell).
    """
    print(f"  Loading wire STL: {WIRE_STL_PATH}")
    wire_tm = trimesh.load(WIRE_STL_PATH)

    if not wire_tm.is_watertight:
        print("  WARNING: wire mesh is NOT watertight. Boolean may produce artifacts.")
        print("           Attempting to repair...")
        wire_tm.fill_holes()

    # Always ensure consistent outward-pointing normals before vertex expansion
    wire_tm.fix_normals()

    print(f"  Wire mesh: {len(wire_tm.vertices)} verts, {len(wire_tm.faces)} faces")
    # Measure on the RAW wire, before GROOVE_EXPANSION fattens it.
    # This way the shell walls sit exactly on the original wire surfaces,
    # so both inner and outer wire paths are visible once the expanded
    # groove is cut through.
    dims = measure_wire_dims(wire_tm)

    # Step 1: Expand tube for groove clearance (before voxelization so the
    # expanded volume is captured in the voxel pass).
    if GROOVE_EXPANSION > 0:
        bb_before = wire_tm.bounds.copy()
        print(f"  Fattening wire tube by {GROOVE_EXPANSION * 1000:.2f} mm (per side)...")
        wire_tm = expand_wire_mesh(wire_tm, GROOVE_EXPANSION)
        bb_after = wire_tm.bounds
        delta = (bb_after[1] - bb_before[1]) * 1000
        print(f"    Bounding-box growth  : "
              f"dx={delta[0]:.2f}  dy={delta[1]:.2f}  dz={delta[2]:.2f} mm")

    # Step 2: Resolve self-intersections via voxelization + marching cubes.
    # This converts the (possibly self-intersecting) tube mesh into a
    # clean volume whose boolean subtraction works correctly everywhere.
    if RESOLVE_SELF_INTERSECTIONS:
        print(f"  Resolving self-intersections (voxel pitch = "
              f"{VOXEL_PITCH * 1000:.2f} mm)...")
        wire_tm = resolve_self_intersections(wire_tm, VOXEL_PITCH)

    wire_man = manifold_from_trimesh(wire_tm)
    return wire_man, dims


def main():
    print("=" * 70)
    print("  Coil Shell Generator -- boolean subtraction via manifold3d")
    print("=" * 70)
    print()
    print(f"  Groove expansion    : {GROOVE_EXPANSION * 1000:.2f} mm")
    if RESOLVE_SELF_INTERSECTIONS:
        print(f"  Self-intersect fix  : True  "
              f"(pitch = {VOXEL_PITCH * 1000:.2f} mm, "
              f"smooth = {SMOOTH_ITERATIONS} iters)")
    else:
        print(f"  Self-intersect fix  : False")
    print(f"  Extra cyl height    : {EXTRA_CYL_HEIGHT * 1000:.2f} mm "
          f"(margin added to auto-detected length)")
    print(f"  Rotation axis/angle : {CYL_ROT_AXIS}, "
          f"{np.degrees(CYL_ROT_ANGLE):.1f} deg")

    # Compute the cylinder axis in world coords
    axis_hat = rotated_cylinder_axis(CYL_ROT_AXIS, CYL_ROT_ANGLE)
    print(f"  Cylinder axis (world) : [{axis_hat[0]:.4f}, "
          f"{axis_hat[1]:.4f}, {axis_hat[2]:.4f}]")
    print()

    # --- Validate input ---
    if not os.path.isfile(WIRE_STL_PATH):
        print(f"  ERROR: wire STL not found at:\n    {WIRE_STL_PATH}")
        sys.exit(1)

    # --- Load wire FIRST so we can auto-size the shell from it ---
    total_steps = 3
    step = 1
    print(f"  [{step}/{total_steps}] Loading and preparing wire mesh...")
    t0 = time.perf_counter()
    wire, dims = load_wire_manifold()
    t1 = time.perf_counter()
    print(f"    Wire ready ({t1 - t0:.2f} s)")

    # Derive shell dimensions from the wire.
    #   - inner_r / outer_r: flush with the innermost / outermost wire
    #     surface, so once the (expanded) groove is cut the wire tube
    #     is visible from both the inside and outside of the former.
    #   - height: axial extent of the wire plus EXTRA_CYL_HEIGHT margin,
    #     split evenly between both ends.
    # Apply the user-specified radial offsets on top of the auto-detected
    # wire bounds.  Positive = farther from the axis, negative = closer.
    inner_r_auto = dims['inner_r']
    outer_r_auto = dims['outer_r']
    inner_r = inner_r_auto + INNER_RADIUS_OFFSET
    outer_r = outer_r_auto + OUTER_RADIUS_OFFSET
    height  = dims['axial_extent'] + EXTRA_CYL_HEIGHT
    axial_center = dims['axial_center']

    # Sanity check: the walls must not cross or collapse.
    if outer_r <= inner_r:
        print(f"  ERROR: outer radius ({outer_r*1000:.2f} mm) <= inner radius "
              f"({inner_r*1000:.2f} mm) after applying offsets.")
        print(f"         Reduce |INNER_RADIUS_OFFSET| / |OUTER_RADIUS_OFFSET|.")
        sys.exit(1)

    print(f"    Auto-detected shell :")
    print(f"      inner radius      : {inner_r * 1000:.2f} mm "
          f"(wire {inner_r_auto*1000:.2f} + offset "
          f"{INNER_RADIUS_OFFSET*1000:+.2f})")
    print(f"      outer radius      : {outer_r * 1000:.2f} mm "
          f"(wire {outer_r_auto*1000:.2f} + offset "
          f"{OUTER_RADIUS_OFFSET*1000:+.2f})")
    print(f"      wall thickness    : {(outer_r - inner_r) * 1000:.2f} mm")
    print(f"      height            : {height * 1000:.2f} mm "
          f"(wire {dims['axial_extent']*1000:.2f} + margin "
          f"{EXTRA_CYL_HEIGHT*1000:.2f})")
    print(f"      axial center      : {axial_center * 1000:.2f} mm")

    # --- Build cylinder shell using the auto-detected dimensions ---
    step += 1
    print(f"  [{step}/{total_steps}] Building cylinder shell...")
    t0 = time.perf_counter()
    shell = build_cylinder_shell(inner_r=inner_r,
                                 outer_r=outer_r,
                                 height=height,
                                 axial_center=axial_center)
    t1 = time.perf_counter()
    shell_mesh = shell.to_mesh()
    print(f"    Shell: {len(np.asarray(shell_mesh.vert_properties))} verts, "
          f"{len(np.asarray(shell_mesh.tri_verts))} faces  "
          f"({t1 - t0:.2f} s)")

    # --- Boolean subtraction ---
    step += 1
    print(f"  [{step}/{total_steps}] Boolean subtraction (shell - wire)...")
    t0 = time.perf_counter()
    result = shell - wire
    t1 = time.perf_counter()
    print(f"    Subtraction done ({t1 - t0:.2f} s)")

    # --- Convert and export ---
    result_tm = trimesh_from_manifold(result)
    print()
    print(f"  Result mesh:")
    print(f"    Vertices     : {len(result_tm.vertices)}")
    print(f"    Faces        : {len(result_tm.faces)}")
    print(f"    Watertight   : {result_tm.is_watertight}")
    bb = result_tm.bounds
    print(f"    Bounding box : [{bb[0][0]:.4f}, {bb[0][1]:.4f}, {bb[0][2]:.4f}]")
    print(f"                   [{bb[1][0]:.4f}, {bb[1][1]:.4f}, {bb[1][2]:.4f}]")

    # --- Determine output path ---
    out_path = OUTPUT_STL_PATH
    if not out_path:
        base, ext = os.path.splitext(WIRE_STL_PATH)
        # Replace 'wire' with 'shell' in the filename if present
        dir_name = os.path.dirname(base)
        file_name = os.path.basename(base)
        if 'wire' in file_name:
            file_name = file_name.replace('wire', 'shell')
        else:
            file_name = file_name + '_shell'
        out_path = os.path.join(dir_name, file_name + ext)

    result_tm.export(out_path)
    print()
    print(f"  Shell STL written to:")
    print(f"    {out_path}")
    print(f"    File size: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")
    print()
    print("=" * 70)
    print("  Done! Open the STL in your slicer to verify the grooves.")
    print("=" * 70)


if __name__ == '__main__':
    main()
