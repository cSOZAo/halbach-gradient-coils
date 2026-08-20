"""
Automatic lead-in / lead-out (cable start & end) for pyCoilGen coil layouts.

Post-processing helper: takes a finished `CoilSolution`, builds a short
smooth polyline from each free wire end out of the cylinder bore, sweeps
the same oval conductor cross-section along those polylines, and returns
a combined trimesh (coil tubes + leads) ready to export as STL.

Why this module exists
----------------------
pyCoilGen's internal sweep (`create_sweep_along_surface`) needs the
cylinder's surface normals — that only works for points *on* the surface.
Leads go off-surface, so we cannot reuse that function. Instead we use
a parallel-transport (rotation-minimizing) frame along the lead polyline:
same visual result, no surface normals needed.

Conventions
-----------
- All lengths in SI metres.
- `axis_hat` is the UNIT vector of the cylinder's rotated axis in world
  coords. For pyCoilGen's cylinder mesh plugin, the natural axis is +Z
  and then rotated by `CYL_ROT_AXIS` / `CYL_ROT_ANGLE`. Use
  `rotated_cylinder_axis(rot_axis, rot_angle)` to compute it.
- Tangent continuity at the coil/lead junction: we build the lead
  polyline starting at the coil endpoint with its initial tangent equal
  to the outward tangent of the coil wire at that endpoint (i.e.
  `v[0] - v[1]` for the start, `v[-1] - v[-2]` for the end). That way
  the Bezier blend meets the coil with matching tangent.
"""

from __future__ import annotations

import numpy as np
import trimesh


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def rotated_cylinder_axis(rot_axis, rot_angle):
    """Return the unit vector of the cylinder axis after the rotation
    applied by pyCoilGen's cylinder mesh plugin.

    The plugin builds the cylinder with its axis along +Z and then
    applies a Rodrigues rotation (`rot_axis`, `rot_angle`). Multiplying
    that rotation by the +Z unit vector gives the rotated axis.
    """
    k = np.asarray(rot_axis, dtype=float)
    k /= np.linalg.norm(k)
    K = np.array([[    0, -k[2],  k[1]],
                  [ k[2],     0, -k[0]],
                  [-k[1],  k[0],     0]])
    R = np.eye(3) + np.sin(rot_angle) * K + (1 - np.cos(rot_angle)) * (K @ K)
    return R @ np.array([0.0, 0.0, 1.0])


def _parallel_transport_frames(path):
    """Compute a rotation-minimizing frame along a 3D polyline.

    Args:
        path: (N, 3) array of polyline vertices.

    Returns:
        tangents, normals, binormals — each (N, 3), orthonormal per row.
    """
    n = len(path)
    tangents = np.zeros_like(path)
    tangents[:-1] = path[1:] - path[:-1]
    tangents[-1]  = tangents[-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(norms, 1e-12)

    # Initial normal: any unit vector perpendicular to the first tangent
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(seed, tangents[0]))) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    n0 = seed - np.dot(seed, tangents[0]) * tangents[0]
    n0 /= np.linalg.norm(n0)

    normals   = np.zeros_like(path)
    binormals = np.zeros_like(path)
    normals[0]   = n0
    binormals[0] = np.cross(tangents[0], n0)

    # Minimum-rotation propagation: project previous normal onto the
    # plane perpendicular to the new tangent, then re-normalize.
    for i in range(1, n):
        prev = normals[i - 1]
        proj = prev - np.dot(prev, tangents[i]) * tangents[i]
        nrm  = np.linalg.norm(proj)
        if nrm < 1e-9:
            # Degenerate: tangent flipped 180° — fall back to seed method
            seed = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(seed, tangents[i]))) > 0.9:
                seed = np.array([0.0, 1.0, 0.0])
            proj = seed - np.dot(seed, tangents[i]) * tangents[i]
            proj /= np.linalg.norm(proj)
        else:
            proj /= nrm
        normals[i]   = proj
        binormals[i] = np.cross(tangents[i], proj)

    return tangents, normals, binormals


# ---------------------------------------------------------------------------
# Sweep: oval cross-section along a 3D polyline
# ---------------------------------------------------------------------------

def sweep_cross_section(path, cross_section_2d, cap_ends=True):
    """Sweep a closed 2D cross-section along a 3D polyline.

    Args:
        path: (N, 3) polyline vertices (must have N >= 2).
        cross_section_2d: (2, M) closed cross-section in the local frame.
            If the last column duplicates the first (closed loop), it is
            dropped automatically.
        cap_ends: if True, add fan-triangulated end caps (convex sections
            only — our oval is convex).

    Returns:
        A trimesh.Trimesh.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 3:
        raise ValueError(f"path must be (N,3); got {path.shape}")
    if path.shape[0] < 2:
        raise ValueError("path needs at least 2 points")

    cs = np.asarray(cross_section_2d, dtype=float)
    if cs.ndim != 2 or cs.shape[0] != 2:
        raise ValueError(f"cross_section_2d must be (2,M); got {cs.shape}")

    # Drop duplicated closing point if present
    if np.allclose(cs[:, 0], cs[:, -1]):
        cs = cs[:, :-1]
    M = cs.shape[1]
    N = path.shape[0]

    _, normals, binormals = _parallel_transport_frames(path)

    # Place M cross-section points around each path vertex using the PT frame
    vertices = np.zeros((N * M, 3), dtype=float)
    for i in range(N):
        offsets = (cs[0, :, None] * normals[i][None, :] +
                   cs[1, :, None] * binormals[i][None, :])   # (M, 3)
        vertices[i * M:(i + 1) * M, :] = path[i][None, :] + offsets

    # Side faces: each quad between consecutive path vertices → 2 triangles
    faces = []
    for i in range(N - 1):
        for j in range(M):
            a = i * M + j
            b = i * M + ((j + 1) % M)
            c = (i + 1) * M + j
            d = (i + 1) * M + ((j + 1) % M)
            faces.append([a, b, d])
            faces.append([a, d, c])

    # End caps (fan triangulation; winding chosen so outward normals are
    # consistent — reversed at the start cap)
    if cap_ends:
        for j in range(1, M - 1):
            faces.append([0, j + 1, j])          # start cap
        base = (N - 1) * M
        for j in range(1, M - 1):
            faces.append([base, base + j, base + j + 1])  # end cap

    mesh = trimesh.Trimesh(vertices=vertices,
                           faces=np.asarray(faces, dtype=int),
                           process=False)
    return mesh


# ---------------------------------------------------------------------------
# Lead polyline: Bezier blend + straight axial run
# ---------------------------------------------------------------------------

def build_lead_polyline(endpoint, endpoint_tangent, axis_hat,
                        radial_clearance, axial_length, blend_length,
                        exit_direction='nearest', blend_samples=20,
                        axial_samples=10):
    """Build the 3D polyline for one lead wire.

    Routing:
        1. Cubic Bezier blend of length ~`blend_length` that leaves the
           coil endpoint tangent to `endpoint_tangent` and arrives at the
           "exit pad" tangent to the cylinder axis. The radial offset
           relative to the cylinder is `radial_clearance` (measured
           outward from the cylinder surface).
        2. Straight axial run of length `axial_length` past the exit pad.

    Args:
        endpoint:        (3,) coil wire endpoint.
        endpoint_tangent:(3,) tangent at the endpoint pointing AWAY from
                         the coil (into the lead). Does not need to be a
                         unit vector.
        axis_hat:        (3,) unit cylinder axis.
        radial_clearance:[m] outward offset from the cylinder surface at
                         the exit pad.
        axial_length:    [m] length of the straight axial exit run.
        blend_length:    [m] axial distance over which the Bezier blend
                         transitions from radial-out to axial-out.
        exit_direction:  '+axis' | '-axis' | 'nearest'. 'nearest' picks
                         the closer cylinder cap based on the endpoint's
                         axial coordinate.
        blend_samples:   sample count along the Bezier.
        axial_samples:   sample count along the straight run.

    Returns:
        (K, 3) polyline, starting at `endpoint`.
    """
    endpoint      = np.asarray(endpoint, dtype=float)
    axis_hat      = np.asarray(axis_hat, dtype=float)
    axis_hat      = axis_hat / np.linalg.norm(axis_hat)

    # Radial direction from cylinder axis to endpoint (in-plane component)
    radial_vec  = endpoint - np.dot(endpoint, axis_hat) * axis_hat
    radial_norm = np.linalg.norm(radial_vec)
    if radial_norm < 1e-9:
        raise ValueError("Endpoint lies on the cylinder axis — cannot "
                         "define a radial direction.")
    radial_hat = radial_vec / radial_norm

    # Pick axial exit direction
    if exit_direction == '+axis':
        exit_hat = +axis_hat
    elif exit_direction == '-axis':
        exit_hat = -axis_hat
    elif exit_direction == 'nearest':
        axial_coord = float(np.dot(endpoint, axis_hat))
        exit_hat = (+axis_hat) if axial_coord >= 0 else (-axis_hat)
    else:
        raise ValueError(f"Unknown exit_direction: {exit_direction!r}")

    # Exit pad: radial_clearance outward from the surface, blend_length along axis
    p0 = endpoint
    p3 = endpoint + radial_clearance * radial_hat + blend_length * exit_hat

    # Bezier tangents at the two ends
    t0 = np.asarray(endpoint_tangent, dtype=float)
    t0_norm = np.linalg.norm(t0)
    if t0_norm < 1e-12:
        # Degenerate: no incoming tangent info; fall back to pure radial-out
        t0 = radial_hat.copy()
    else:
        t0 = t0 / t0_norm
    t3 = exit_hat

    # Control-point handle length ~ chord/3 gives a visually smooth cubic
    chord     = np.linalg.norm(p3 - p0)
    handle    = chord / 3.0 if chord > 0 else max(blend_length, radial_clearance) / 3.0
    p1 = p0 + handle * t0
    p2 = p3 - handle * t3

    # Sample the Bezier
    ts    = np.linspace(0.0, 1.0, blend_samples)
    one_t = 1.0 - ts
    blend = (one_t[:, None] ** 3) * p0 \
          + 3 * (one_t[:, None] ** 2) * ts[:, None] * p1 \
          + 3 *  one_t[:, None]       * (ts[:, None] ** 2) * p2 \
          + (ts[:, None] ** 3) * p3

    # Straight axial run starting at p3
    straight_ts = np.linspace(0.0, axial_length, axial_samples)
    straight    = p3 + straight_ts[:, None] * exit_hat

    # Drop duplicate p3 in the concatenation
    polyline = np.vstack([blend, straight[1:]])
    return polyline


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def add_lead_wires(solution, axis_hat, *,
                   radial_clearance, axial_length, blend_length,
                   exit_direction='nearest',
                   blend_samples=20, axial_samples=10,
                   cross_section_2d=None, cap_ends=True):
    """Build lead tubes for every coil part and merge them with the
    existing swept layout mesh.

    Args:
        solution:       the CoilSolution returned by pyCoilGen.
        axis_hat:       (3,) unit vector of the (rotated) cylinder axis.
        radial_clearance, axial_length, blend_length, exit_direction,
        blend_samples, axial_samples:
                        forwarded to `build_lead_polyline`.
        cross_section_2d: (2, M) cross-section. If None, reuse
                        `solution.input_args.cross_sectional_points`.
        cap_ends:       cap both ends of each lead tube with fans.

    Returns:
        combined_mesh (trimesh.Trimesh), lead_polylines (list of (K,3)),
        lead_meshes (list of trimesh.Trimesh).

    Notes:
        * We use `trimesh.util.concatenate` — not a boolean union — so
          the result may contain internal walls at the lead/coil
          junction. Slicers handle this fine for 3D printing; CSG-clean
          output would require `trimesh.boolean.union` with a suitable
          backend (manifold3d or blender).
    """
    if cross_section_2d is None:
        cross_section_2d = np.asarray(
            solution.input_args.cross_sectional_points, dtype=float)

    lead_polylines = []
    lead_meshes    = []
    coil_meshes    = []

    for coil_part in solution.coil_parts:
        wire_v = coil_part.wire_path.v          # (3, N)
        if wire_v.shape[1] < 2:
            raise ValueError("wire_path.v has fewer than 2 points; "
                             "cannot compute endpoint tangents.")

        # Tangents pointing AWAY from the coil (into each lead)
        t_start = wire_v[:, 0]  - wire_v[:, 1]
        t_end   = wire_v[:, -1] - wire_v[:, -2]

        poly_start = build_lead_polyline(
            wire_v[:, 0], t_start, axis_hat,
            radial_clearance=radial_clearance,
            axial_length=axial_length,
            blend_length=blend_length,
            exit_direction=exit_direction,
            blend_samples=blend_samples,
            axial_samples=axial_samples,
        )
        poly_end = build_lead_polyline(
            wire_v[:, -1], t_end, axis_hat,
            radial_clearance=radial_clearance,
            axial_length=axial_length,
            blend_length=blend_length,
            exit_direction=exit_direction,
            blend_samples=blend_samples,
            axial_samples=axial_samples,
        )

        mesh_start = sweep_cross_section(poly_start, cross_section_2d,
                                         cap_ends=cap_ends)
        mesh_end   = sweep_cross_section(poly_end,   cross_section_2d,
                                         cap_ends=cap_ends)

        lead_polylines.extend([poly_start, poly_end])
        lead_meshes.extend([mesh_start, mesh_end])

        if coil_part.layout_surface_mesh is not None:
            coil_meshes.append(coil_part.layout_surface_mesh.trimesh_obj)

    combined = trimesh.util.concatenate(coil_meshes + lead_meshes)
    return combined, lead_polylines, lead_meshes


# ---------------------------------------------------------------------------
# Debug plot
# ---------------------------------------------------------------------------

def plot_leads_3d(solution, lead_polylines, axis_hat=None, title=None,
                  ax=None, show=True):
    """Quick 3D sanity plot: coil wire_path + lead polylines."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if ax is None:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')

    for coil_part in solution.coil_parts:
        v = coil_part.wire_path.v
        ax.plot(v[0], v[1], v[2], color='steelblue', linewidth=0.8,
                label='coil wire_path')

    for idx, poly in enumerate(lead_polylines):
        ax.plot(poly[:, 0], poly[:, 1], poly[:, 2],
                color='crimson', linewidth=2.0,
                label='lead polyline' if idx == 0 else None)
        ax.scatter(poly[0, 0], poly[0, 1], poly[0, 2],
                   color='green', s=25,
                   label='lead start (on coil)' if idx == 0 else None)
        ax.scatter(poly[-1, 0], poly[-1, 1], poly[-1, 2],
                   color='black', s=25,
                   label='lead exit' if idx == 0 else None)

    if axis_hat is not None:
        a = np.asarray(axis_hat, dtype=float)
        ax.quiver(0, 0, 0, a[0], a[1], a[2], length=0.05,
                  color='gray', label='cylinder axis')

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    if title:
        ax.set_title(title)
    ax.legend(loc='best', fontsize=8)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    return ax
