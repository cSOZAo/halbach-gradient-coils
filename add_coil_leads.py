"""
add_coil_leads.py
=================
Add two lead wires to a pyCoilGen wire-layout STL for negative-mold workflow.

The pyCoilGen wire is a single closed, water-tight tube (all saddle loops in
one conductor).  To wind physical wire from a 3-D-printed groove mold you need
an inlet and an outlet.  This script:

  1. Locates the outermost wire section (furthest along LEAD_DIRECTION in a
     chosen angular sector).
  2. Estimates the local wire tangent with a small PCA neighbourhood.
  3. Removes a segment of length ``CUT_LOOP_LENGTH`` along the loop at that
     station.  Start separation equals this cut — one parameter, no extra stubs.
  4. Each lead path: short run **toward** the gap (starts face each other),
     then **peels outward** along the loop, then a shell Bezier into the bore.
     Tips fan apart by ``TIP_FAN`` on the shared exit plane.
     Both leads terminate on the **same exit plane** so they are equal length.

Run:
    python add_coil_leads.py

Output: ``<name>_with_leads.stl`` plus verify PNGs.

Dependencies:
    pip install trimesh matplotlib
"""

import os
from collections import defaultdict, deque

import numpy as np
import trimesh


# =============================================================================
#  SETTINGS — defaults from coil_mold_common.py (edit that file for the pipeline)
# =============================================================================

try:
    import coil_mold_common as _cfg
except ImportError:     
    _cfg = None

def _from_cfg(name, fallback):
    return getattr(_cfg, name, fallback) if _cfg is not None else fallback

# When coil_mold_common imports, INPUT_STL defaults to cfg.wire_stl_path() unless
# you set INPUT_STL_OVERRIDE below (the else-branch fallback is only used if
# coil_mold_common is missing).
INPUT_STL_OVERRIDE = ''

if INPUT_STL_OVERRIDE:
    INPUT_STL = INPUT_STL_OVERRIDE
elif _cfg is not None:
    INPUT_STL = _cfg.wire_stl_path(with_leads=False)
else:
    INPUT_STL = (
        r"C:\Clemente\VSCode\pyCoilGen-0.2.4\pyCoilGen-0.2.4\pruebas\resultados\resultados_grande_y\final_2\Gradient_Gy_tk2500_lvl26_wire_0_z.stl"
    )

LEAD_DIRECTION = _from_cfg('LEAD_DIRECTION', np.array([-1.0, 0.0, 0.0]))
SECTOR_MIN_Z = _from_cfg('SECTOR_MIN_Z', 0.10)
SECTOR_MAX_ABS_Y = _from_cfg('SECTOR_MAX_ABS_Y', 0.05)
CYL_AXIS = _from_cfg('CYL_AXIS', None)
SHELL_RADIUS = _from_cfg('SHELL_RADIUS', None)

CONDUCTOR_WIDTH = _from_cfg('CONDUCTOR_WIDTH', 0.0023)
CROSS_SECTION_A_FRAC = _from_cfg('CROSS_SECTION_A_FRAC', 1.0)
CROSS_SECTION_B_FRAC = _from_cfg('CROSS_SECTION_B_FRAC', 1.0)
CROSS_SECTION_N = _from_cfg('CROSS_SECTION_N', 12)
CS_BLEND_RINGS = _from_cfg('CS_BLEND_RINGS', 8)
JUNCTION_RIGID_STEPS = _from_cfg('JUNCTION_RIGID_STEPS', 2)
JUNCTION_PLANE_RINGS = _from_cfg('JUNCTION_PLANE_RINGS', 4)

CUT_LOOP_LENGTH = _from_cfg('CUT_LOOP_LENGTH', 0.040)
GAP_AXIAL_LENGTH = _from_cfg('GAP_AXIAL_LENGTH', 0.012)
WIRE_ISOLATE_HALF = _from_cfg('WIRE_ISOLATE_HALF', 0.008)
TANGENT_RADIUS = _from_cfg('TANGENT_RADIUS', 0.006)

WIRE_TANGENT_RUN = _from_cfg('WIRE_TANGENT_RUN', 0.004)
FACE_TOWARD_GAP = _from_cfg('FACE_TOWARD_GAP', 0.003)
PEEL_OUT = _from_cfg('PEEL_OUT', 0.006)
LEAD_LENGTH = _from_cfg('LEAD_LENGTH', 0.02)
LEAD_BLEND = _from_cfg('LEAD_BLEND', 0.030)
TIP_FAN = _from_cfg('TIP_FAN', 0.015)
LEAD_STEPS = _from_cfg('LEAD_STEPS', 128)
LEAD_0_SPREAD_SIGN = _from_cfg('LEAD_0_SPREAD_SIGN', 1)
LEAD_1_SPREAD_SIGN = _from_cfg('LEAD_1_SPREAD_SIGN', -1)
EXIT_DIRECTION = _from_cfg('EXIT_DIRECTION', None)

# =============================================================================
#  END OF SETTINGS
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
#  GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _infer_axis_hat(lead_dir):
    """Dominant cardinal axis of lead_dir (bore axis)."""
    idx = int(np.argmax(np.abs(lead_dir)))
    axis = np.zeros(3)
    axis[idx] = 1.0
    return axis


def _axial_coord(points, axis_hat):
    """Axial coordinate(s) along *axis_hat* (scalar or 1-D array)."""
    return np.asarray(points) @ axis_hat


def _radial_vec(points, axis_hat):
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        return pts - _axial_coord(pts, axis_hat) * axis_hat
    return pts - np.outer(_axial_coord(pts, axis_hat), axis_hat)


def _radial_hat(point, axis_hat):
    radial = _radial_vec(point, axis_hat)
    n = np.linalg.norm(radial)
    if n < 1e-12:
        seed = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(seed, axis_hat)) > 0.9:
            seed = np.array([0.0, 1.0, 0.0])
        radial = seed - np.dot(seed, axis_hat) * axis_hat
        n = np.linalg.norm(radial)
    return radial / n


def _snap_to_shell(point, axis_hat, shell_radius):
    """Project *point* onto the cylindrical shell at *shell_radius*."""
    pt = np.asarray(point, dtype=float)
    axial = _axial_coord(pt, axis_hat) * axis_hat
    radial = pt - axial
    r = np.linalg.norm(radial)
    if r < 1e-12:
        radial = _radial_hat(pt, axis_hat) * shell_radius
    else:
        radial = radial * (shell_radius / r)
    return axial + radial


def _tangent_offset(vec, at_point, axis_hat):
    """Project *vec* onto the tangent plane at *at_point* (keeps magnitude)."""
    r_hat = _radial_hat(at_point, axis_hat)
    v = np.asarray(vec, dtype=float)
    return v - np.dot(v, r_hat) * r_hat


def _surface_tangent(vec, axis_hat):
    """Unit vector of *vec* projected onto the tangent plane at *vec* (legacy)."""
    radial = (_radial_hat(vec, axis_hat)
              if np.linalg.norm(_radial_vec(vec, axis_hat)) > 1e-12
              else np.array([0., 1., 0.]))
    t = vec - np.dot(vec, radial) * radial
    n = np.linalg.norm(t)
    return t / n if n > 1e-12 else t


def _estimate_shell_radius(vertices, apex, axis_hat, sample_radius=0.015):
    """Median centre-line radius from vertices near the apex."""
    near = vertices[np.linalg.norm(vertices - apex, axis=1) < sample_radius]
    if len(near) < 8:
        near = vertices[np.argsort(np.linalg.norm(vertices - apex, axis=1))[:64]]
    radial = np.linalg.norm(_radial_vec(near, axis_hat), axis=1)
    return float(np.median(radial))


def _pca_tangent(vertices, point, radius):
    near = vertices[np.linalg.norm(vertices - point, axis=1) < radius]
    if len(near) < 3:
        near = vertices[np.argsort(np.linalg.norm(vertices - point, axis=1))[:12]]
    c = near.mean(axis=0)
    _, _, vt = np.linalg.svd(near - c)
    return vt[0] / np.linalg.norm(vt[0])


def _boundary_adjacency(mesh):
    ec = np.bincount(mesh.edges_unique_inverse,
                     minlength=len(mesh.edges_unique))
    be = mesh.edges_unique[ec == 1]
    adj = defaultdict(set)
    for e in be:
        adj[e[0]].add(e[1])
        adj[e[1]].add(e[0])
    return be, adj


def _boundary_loops(mesh):
    be, adj = _boundary_adjacency(mesh)
    if len(be) == 0:
        return [], adj
    seen, loops = set(), []
    for start in np.unique(be):
        if start in seen:
            continue
        comp, q = [], deque([start])
        while q:
            u = q.popleft()
            if u in seen:
                continue
            seen.add(u)
            comp.append(u)
            for w in adj[u]:
                if w not in seen:
                    q.append(w)
        loops.append(comp)
    return loops, adj


def _order_loop(loop, adj):
    if len(loop) < 2:
        return loop
    loop_set = set(loop)
    start = loop[0]
    ordered = [start]
    prev, cur = None, start
    for _ in range(len(loop) - 1):
        nbs = [n for n in adj[cur] if n != prev and n in loop_set]
        if not nbs:
            break
        nxt = nbs[0]
        prev, cur = cur, nxt
        ordered.append(cur)
    return ordered


def _rotate_ordered_ring(ring_idx, ring_3d, center, ax1, ax2):
    """
    Rotate a topologically ordered boundary loop so vertex 0 is at the
    smallest polar angle — keeps mesh edge connectivity intact.
    """
    pts_2d = np.column_stack([(ring_3d - center) @ ax1, (ring_3d - center) @ ax2])
    k = int(np.argmin(np.arctan2(pts_2d[:, 1], pts_2d[:, 0])))
    idx = list(ring_idx[k:]) + list(ring_idx[:k])
    pts = np.vstack([ring_3d[k:], ring_3d[:k]])
    return idx, pts


def _loop_plane(points):
    """Centroid, unit normal (Newell), and two in-plane axes for a ring."""
    pts = np.asarray(points, dtype=float)
    center = pts.mean(axis=0)
    normal = np.zeros(3)
    for i in range(len(pts)):
        p0 = pts[i]
        p1 = pts[(i + 1) % len(pts)]
        normal[0] += (p0[1] - p1[1]) * (p0[2] + p1[2])
        normal[1] += (p0[2] - p1[2]) * (p0[0] + p1[0])
        normal[2] += (p0[0] - p1[0]) * (p0[1] + p1[1])
    nlen = np.linalg.norm(normal)
    if nlen < 1e-12:
        _, _, vt = np.linalg.svd(pts - center)
        normal = vt[-1]
    else:
        normal /= nlen
    ref = (np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9
           else np.array([1.0, 0.0, 0.0]))
    ax1 = np.cross(normal, ref)
    ax1 /= np.linalg.norm(ax1)
    ax2 = np.cross(normal, ax1)
    return center, normal, ax1, ax2


def _rotation_minimizing_frames(path):
    """Twist-free (double-reflection RMF) frames along a polyline."""
    n = len(path)
    T = np.zeros((n, 3))
    N = np.zeros((n, 3))
    B = np.zeros((n, 3))
    for i in range(n - 1):
        d = path[i + 1] - path[i]
        l = np.linalg.norm(d)
        T[i] = d / l if l > 1e-12 else (T[i - 1] if i > 0 else np.array([1., 0., 0.]))
    T[-1] = T[-2]
    ref = np.array([0, 0, 1]) if abs(T[0, 2]) < 0.9 else np.array([1, 0, 0])
    N[0] = np.cross(T[0], ref)
    N[0] /= np.linalg.norm(N[0])
    B[0] = np.cross(T[0], N[0])
    for i in range(1, n):
        v1 = path[i] - path[i - 1]
        c1 = float(np.dot(v1, v1))
        if c1 < 1e-12:
            N[i] = N[i - 1]
            B[i] = B[i - 1]
            continue
        nr = N[i - 1] - (2 / c1) * np.dot(v1, N[i - 1]) * v1
        tr = T[i - 1] - (2 / c1) * np.dot(v1, T[i - 1]) * v1
        v2 = T[i] - tr
        c2 = float(np.dot(v2, v2))
        N[i] = nr if c2 < 1e-12 else nr - (2 / c2) * np.dot(v2, nr) * v2
        nl = np.linalg.norm(N[i])
        N[i] = N[i] / nl if nl > 1e-10 else N[i]
        B[i] = np.cross(T[i], N[i])
        bl = np.linalg.norm(B[i])
        B[i] = B[i] / bl if bl > 1e-10 else B[i]
    return T, N, B


def _shell_bezier(p0, p1, p2, p3, axis_hat, shell_radius, n_steps):
    """Cubic Bezier centre-line; snap only the radial component to the shell."""
    t = np.linspace(0.0, 1.0, n_steps)
    path = (np.outer((1 - t) ** 3, p0)
            + np.outer(3 * (1 - t) ** 2 * t, p1)
            + np.outer(3 * (1 - t) * t ** 2, p2)
            + np.outer(t ** 3, p3))
    for i in range(n_steps):
        path[i] = _snap_to_shell(path[i], axis_hat, shell_radius)
    return path


def _circumferential_unit(wire_hat, at_point, axis_hat):
    """Unit vector along the coil loop (tangent to the cylinder surface)."""
    t = _tangent_offset(wire_hat, at_point, axis_hat)
    n = np.linalg.norm(t)
    if n < 1e-12:
        raise ValueError("Wire tangent has no component along the loop at this point.")
    return t / n


def _pycoilgen_oval_profile():
    """
    Closed ellipse matching ``gradiente_belen_santi_main.py`` / pyCoilGen input.

    Local axes: index 0 = A (sin), index 1 = B (cos) — same as
    ``cross_sectional_points`` passed to pyCoilGen.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, CROSS_SECTION_N, endpoint=True)
    template = np.column_stack([
        CROSS_SECTION_A_FRAC * CONDUCTOR_WIDTH * np.sin(theta),
        CROSS_SECTION_B_FRAC * CONDUCTOR_WIDTH * np.cos(theta),
    ])
    radii = np.linalg.norm(template, axis=1)
    return {
        'template_2d': template,
        'cs_mean': float(radii.mean()),
        'cs_span': float(radii.ptp()),
        'semi_a': CROSS_SECTION_A_FRAC * CONDUCTOR_WIDTH,
        'semi_b': CROSS_SECTION_B_FRAC * CONDUCTOR_WIDTH,
    }


def _ordered_ring_2d(ring_3d, center, ax1, ax2):
    """Cut-face ring projected and ordered by polar angle in the cut plane."""
    pts = np.column_stack([(ring_3d - center) @ ax1, (ring_3d - center) @ ax2])
    ang = np.arctan2(pts[:, 1], pts[:, 0])
    return pts[np.argsort(ang)]


def _align_oval_to_cut(oval_2d, cut_2d):
    """Rotate *oval_2d* so its first vertex aligns with *cut_2d*."""
    oval = _resample_closed_2d(oval_2d, len(cut_2d))
    a0 = np.arctan2(cut_2d[0, 1], cut_2d[0, 0])
    a1 = np.arctan2(oval[0, 1], oval[0, 0])
    rot = a0 - a1
    c_, s_ = np.cos(rot), np.sin(rot)
    return oval @ np.array([[c_, -s_], [s_, c_]])


def _ring_cross_sections(ring_3d, wire_profile):
    """Cut-face 2-D ring plus pyCoilGen oval aligned to the same plane."""
    center, _, ax1, ax2 = _loop_plane(ring_3d)
    cut_2d = _ordered_ring_2d(ring_3d, center, ax1, ax2)
    oval_2d = _align_oval_to_cut(wire_profile['template_2d'], cut_2d)
    prof = wire_profile
    return cut_2d, oval_2d, prof['cs_mean'], prof['cs_span']


def _resample_closed_2d(pts_2d, n_out):
    """Uniformly resample a closed 2-D polyline to *n_out* points."""
    pts = np.asarray(pts_2d, dtype=float)
    if len(pts) < 3:
        return np.resize(pts, (n_out, 2))
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total < 1e-12:
        return np.tile(pts[:1], (n_out, 1))
    targets = np.linspace(0.0, total, n_out, endpoint=False)
    out = np.zeros((n_out, 2))
    for i, st in enumerate(targets):
        j = int(np.searchsorted(s, st, side='right') - 1)
        j = min(max(j, 0), len(pts) - 1)
        t = (st - s[j]) / (s[j + 1] - s[j] + 1e-12)
        out[i] = pts[j] + t * (pts[(j + 1) % len(pts)] - pts[j])
    return out


def _profile_ring_2d(wire_profile, ring_3d, n_pts):
    """Design oval resampled to *n_pts* (legacy helper)."""
    cut_2d, oval_2d, cs_mean, cs_span = _ring_cross_sections(ring_3d, wire_profile)
    if len(oval_2d) != n_pts:
        oval_2d = _resample_closed_2d(oval_2d, n_pts)
        cut_2d = _resample_closed_2d(cut_2d, n_pts)
    return oval_2d, cs_mean, cs_span, cut_2d


def _lead_centerline(p0, wire_tangent, toward_gap, outward, tip, exit_dir,
                     axis_hat, shell_radius, wire_run, face_in, peel_out,
                     blend, tip_fan, n_steps):
    """
    Centre-line: coil tangent → face inward → peel outward → Bezier to tip.

    Lateral Bezier bias scales with *tip_fan* so TIP_FAN=0 yields parallel tips.
    """
    wt = wire_tangent / np.linalg.norm(wire_tangent)
    tg = toward_gap / np.linalg.norm(toward_gap)
    out = outward / np.linalg.norm(outward)
    exit_d = exit_dir / np.linalg.norm(exit_dir)
    fan_scale = min(1.0, float(tip_fan) / 0.005) if tip_fan > 1e-9 else 0.0

    path_parts = []
    p_start = np.asarray(p0, dtype=float)

    if wire_run > 1e-6:
        surf_t = _tangent_offset(wt, p_start, axis_hat)
        sn = np.linalg.norm(surf_t)
        if sn > 1e-12:
            surf_t /= sn
            n_w = max(3, n_steps // 24)
            wrun = p_start + np.outer(np.linspace(0.0, 1.0, n_w), surf_t * wire_run)
            for i in range(n_w):
                wrun[i] = _snap_to_shell(wrun[i], axis_hat, shell_radius)
            path_parts.append(wrun)
            p_start = wrun[-1]

    if face_in > 1e-6:
        n_in = max(3, n_steps // 20)
        inward = p_start + np.outer(np.linspace(0.0, 1.0, n_in), tg * face_in)
        for i in range(n_in):
            inward[i] = _snap_to_shell(inward[i], axis_hat, shell_radius)
        path_parts.append(inward)
        p_start = inward[-1]

    n_out = max(5, n_steps // 8)
    peel = p_start + np.outer(np.linspace(0.0, 1.0, n_out), out * peel_out)
    for i in range(n_out):
        peel[i] = _snap_to_shell(peel[i], axis_hat, shell_radius)
    path_parts.append(peel)

    ps = peel[-1]
    h = blend
    lat = 0.45 * fan_scale
    ax_w = 0.15 + 0.55 * (1.0 - fan_scale)
    p1 = _snap_to_shell(ps + out * h * lat + exit_d * h * ax_w * 0.35,
                        axis_hat, shell_radius)
    p2 = _snap_to_shell(np.asarray(tip, dtype=float) - exit_d * h * ax_w * 0.55,
                        axis_hat, shell_radius)
    p3 = _snap_to_shell(tip, axis_hat, shell_radius)

    n_curve = max(8, n_steps - sum(len(p) for p in path_parts) + 1)
    curve = _shell_bezier(ps, p1, p2, p3, axis_hat, shell_radius, n_curve)
    path_parts.append(curve[1:])
    out_path = path_parts[0]
    for part in path_parts[1:]:
        out_path = np.vstack([out_path, part])
    return out_path


def _tip_fan_direction(loop_dir):
    """
    Tip fan direction along the loop, opposite to peel *loop_dir*.

    Peel follows the spread sign; tips fan the other way so inverting spread
    signs also inverts tip separation and the leads do not cross.
    """
    ld = np.asarray(loop_dir, dtype=float)
    ld /= np.linalg.norm(ld)
    return -ld


def _common_tip(exit_axial, anchor, axis_hat, shell_radius, fan_offset):
    """
    Tip on the shared exit plane.

    *anchor* is the common bore-side point (mid-gap); *fan_offset* adds
    ``TIP_FAN`` separation only — peel-out does not shift tips when fan=0.
    """
    radial = _radial_vec(anchor, axis_hat)
    fan = (_tangent_offset(fan_offset, anchor, axis_hat)
           if np.linalg.norm(fan_offset) > 1e-12 else np.zeros(3))
    tip = float(exit_axial) * axis_hat + radial + fan
    return _snap_to_shell(tip, axis_hat, shell_radius)


def _orient_face_normal(normal, center, axis_hat, gap_center_axial):
    """Face normal points out of the gap (away from bridge mid-plane)."""
    n = normal / np.linalg.norm(normal)
    center_ax = _axial_coord(center, axis_hat)
    if center_ax >= gap_center_axial:
        if np.dot(n, axis_hat) < 0:
            n = -n
    else:
        if np.dot(n, axis_hat) > 0:
            n = -n
    return n


def _loop_metrics(points):
    """Return centroid, mean cross-section radius, and in-plane span."""
    pts = np.asarray(points, dtype=float)
    center = pts.mean(axis=0)
    radii = np.linalg.norm(pts - center, axis=1)
    return center, float(radii.mean()), float(radii.max() - radii.min())


def _find_apex(vertices, lead_dir):
    proj = vertices @ lead_dir
    sector = ((vertices[:, 2] > SECTOR_MIN_Z) &
              (np.abs(vertices[:, 1]) < SECTOR_MAX_ABS_Y))
    idx = np.where(sector)[0]
    if len(idx) == 0:
        idx = np.arange(len(vertices))
    apex_v = idx[np.argmax(proj[idx])]
    return apex_v, vertices[apex_v]


def _perp_distance_to_loop(points, apex, circ_unit):
    """Distance from each point to the line through *apex* along *circ_unit*."""
    pts = np.asarray(points, dtype=float)
    d = pts - apex
    along = np.outer(d @ circ_unit, circ_unit)
    return np.linalg.norm(d - along, axis=1)


def _flood_bridge(mesh, apex_v, apex, axis_hat, circ_unit, gap_axial_length,
                  cut_loop_length, perp_half):
    """
    Remove a short conductor segment at the apex station on one loop.

    *cut_loop_length* — fixed arc removed along the loop (small, for clean faces).
    *perp_half* — half-width perpendicular to the loop (isolates one conductor).
    """
    half_loop = cut_loop_length / 2.0

    fcen = mesh.triangles_center
    apex_ax = _axial_coord(apex, axis_hat)
    in_axial = (np.abs(_axial_coord(fcen, axis_hat) - apex_ax)
                < gap_axial_length / 2.0)
    along_loop = (fcen - apex) @ circ_unit
    in_loop = np.abs(along_loop) < half_loop
    in_perp = _perp_distance_to_loop(fcen, apex, circ_unit) < perp_half
    keep = in_axial & in_loop & in_perp

    nbr = defaultdict(list)
    for a, b in mesh.face_adjacency:
        nbr[a].append(b)
        nbr[b].append(a)

    faces_with_apex = np.where((mesh.faces == apex_v).any(axis=1))[0]
    faces_with_apex = [f for f in faces_with_apex if keep[f]]
    if not faces_with_apex:
        raise RuntimeError(
            "Apex face not inside cut slab — increase WIRE_ISOLATE_HALF, "
            "GAP_AXIAL_LENGTH, or CUT_LOOP_LENGTH."
        )

    removed = set()
    q = deque(faces_with_apex)
    while q:
        f = q.popleft()
        if f in removed or not keep[f]:
            continue
        removed.add(f)
        for g in nbr[f]:
            if g not in removed and keep[g]:
                q.append(g)

    mask = np.zeros(len(mesh.faces), dtype=bool)
    mask[np.array(sorted(removed))] = True
    return mask


def _build_attached_lead(ring_indices, vertices, wire_profile, wire_tangent,
                         toward_gap, outward, tip, exit_dir, axis_hat,
                         shell_radius, wire_run, face_in, peel_out, lead_blend,
                         tip_fan, n_steps):
    """RMF sweep: cut-face profile blends into pyCoilGen oval along the path."""
    ring_3d = vertices[ring_indices]
    n_pts = len(ring_indices)
    center, _, ax1, ax2 = _loop_plane(ring_3d)
    ring_indices, ring_3d = _rotate_ordered_ring(ring_indices, ring_3d, center, ax1, ax2)
    center = _snap_to_shell(center, axis_hat, shell_radius)
    oval_2d, cs_mean, cs_span, cut_2d = _profile_ring_2d(wire_profile, ring_3d, n_pts)
    path0 = center.copy()

    path = _lead_centerline(
        p0           = center,
        wire_tangent = wire_tangent,
        toward_gap   = toward_gap,
        outward      = outward,
        tip          = tip,
        exit_dir     = exit_dir,
        axis_hat     = axis_hat,
        shell_radius = shell_radius,
        wire_run     = wire_run,
        face_in      = face_in,
        peel_out     = peel_out,
        blend        = lead_blend,
        tip_fan      = tip_fan,
        n_steps      = n_steps,
    )
    path[-1] = np.asarray(tip, dtype=float)
    n_path = len(path)
    n_w = max(3, n_steps // 24) if wire_run > 1e-6 else 0
    n_in = max(3, n_steps // 20) if face_in > 1e-6 else 0
    n_peel = max(5, n_steps // 8)
    peel_end = min(n_w + n_in + n_peel - 1, n_path - 1)
    departure = path[peel_end]

    _, N, B = _rotation_minimizing_frames(path)
    cos_a = float(np.dot(N[0], ax1))
    sin_a = float(np.dot(B[0], ax1))
    angle = np.arctan2(sin_a, cos_a)
    c_, s_ = np.cos(angle), np.sin(angle)
    for i in range(n_path):
        Nn = c_ * N[i] + s_ * B[i]
        Bn = -s_ * N[i] + c_ * B[i]
        N[i], B[i] = Nn, Bn

    extra_rings = []
    blend_n = max(1, CS_BLEND_RINGS)
    n_rigid = min(JUNCTION_RIGID_STEPS, n_path - 1)
    n_plane = min(JUNCTION_PLANE_RINGS, n_path - 1 - n_rigid)

    for i in range(1, n_path):
        if i <= n_rigid:
            extra_rings.append(ring_3d + (path[i] - path0))
            continue

        blend_i = i - 1 - n_rigid
        t = min(1.0, blend_i / blend_n)
        r2d = (1.0 - t) * cut_2d + t * oval_2d

        if i <= n_rigid + n_plane:
            Ni, Bi = ax1, ax2
        else:
            Ni, Bi = N[i], B[i]

        extra_rings.append(
            path[i]
            + np.outer(r2d[:, 0], Ni)
            + np.outer(r2d[:, 1], Bi)
        )

    cap_center = extra_rings[-1].mean(axis=0)
    extra_vertices = np.vstack(extra_rings + [cap_center[None, :]])

    faces = []
    cap_local = n_path * n_pts
    for i in range(n_path - 1):
        row_a = np.arange(i * n_pts, i * n_pts + n_pts)
        row_b = np.arange((i + 1) * n_pts, (i + 1) * n_pts + n_pts)
        for j in range(n_pts):
            jn = (j + 1) % n_pts
            a, b = row_a[j], row_a[jn]
            c, d = row_b[j], row_b[jn]
            faces += [[a, b, d], [a, d, c]]
    last_ring = np.arange((n_path - 1) * n_pts, n_path * n_pts)
    for j in range(n_pts):
        faces.append([last_ring[j], cap_local, last_ring[(j + 1) % n_pts]])

    return extra_vertices, np.array(faces, dtype=int), np.asarray(ring_indices), {
        'center': center,
        'departure': np.asarray(departure, dtype=float),
        'tip': np.asarray(path[-1], dtype=float),
        'cs_mean': cs_mean,
        'cs_span': cs_span,
        'n_path': n_path,
        'approach_run': 0.0,
    }


def _attach_leads(open_mesh, lead_parts):
    vertices = open_mesh.vertices.copy()
    faces = [open_mesh.faces.copy()]
    for extra_vertices, lead_faces, ring_indices, n_path in lead_parts:
        n_pts = len(ring_indices)
        offset = len(vertices)
        remap = np.empty(n_path * n_pts + 1, dtype=np.int64)
        remap[:n_pts] = ring_indices
        remap[n_pts:] = offset + np.arange(len(extra_vertices))
        faces.append(remap[lead_faces])
        vertices = np.vstack([vertices, extra_vertices])
    return trimesh.Trimesh(vertices=vertices, faces=np.vstack(faces))


def _cap_ring(ring_3d):
    """Triangulate a closed ring to seal the weld end of a standalone lead tube."""
    n_pts = len(ring_3d)
    center = ring_3d.mean(axis=0)
    verts = np.vstack([ring_3d, center[None, :]])
    cap_i = n_pts
    faces = [[cap_i, i, (i + 1) % n_pts] for i in range(n_pts)]
    return verts, np.array(faces, dtype=int)


def _standalone_leads_mesh(open_mesh, lead_parts):
    """Watertight lead tubes only — local indices, weld end capped."""
    parts = []
    for extra_vertices, lead_faces, ring_indices, _ in lead_parts:
        ring_3d = open_mesh.vertices[ring_indices]
        body = trimesh.Trimesh(
            vertices=np.vstack([ring_3d, extra_vertices]),
            faces=lead_faces,
            process=False,
        )
        cap_verts, cap_faces = _cap_ring(ring_3d)
        # cap_verts duplicates ring + center; remap cap faces to body vertex indices
        cap_i = len(body.vertices)
        body.vertices = np.vstack([body.vertices, cap_verts[len(ring_3d):]])
        cap_faces = np.array([[cap_i, i, (i + 1) % len(ring_3d)]
                              for i in range(len(ring_3d))], dtype=int)
        body.faces = np.vstack([body.faces, cap_faces])
        body.update_faces(body.unique_faces())
        body.fix_normals()
        parts.append(body)
    return trimesh.util.concatenate(parts)


def _shell_radius_of_points(points, axis_hat):
    return np.linalg.norm(_radial_vec(np.asarray(points), axis_hat), axis=1)


def _verify_result(final_mesh, n_coil_vertices, apex, axis_hat, shell_radius,
                   ring_info, circ_unit, output_stl):
    ec = np.bincount(final_mesh.edges_unique_inverse,
                     minlength=len(final_mesh.edges_unique))
    n_boundary = int((ec == 1).sum())

    coil_r = _shell_radius_of_points(final_mesh.vertices[:n_coil_vertices], axis_hat)
    lead_r = _shell_radius_of_points(final_mesh.vertices[n_coil_vertices:], axis_hat)
    coil_near = coil_r[np.linalg.norm(final_mesh.vertices[:n_coil_vertices] - apex, axis=1) < 0.025]
    print(f"  Shell radius target : {shell_radius * 1e3:.2f} mm")
    print(f"  Coil  radius (cut)  : {np.median(coil_near) * 1e3:.2f} mm  "
          f"(spread {np.ptp(coil_near) * 1e3:.2f} mm)")
    print(f"  Lead  radius        : {np.median(lead_r) * 1e3:.2f} mm  "
          f"(spread {np.ptp(lead_r) * 1e3:.2f} mm)")

    c0, c1 = ring_info[0]['center'], ring_info[1]['center']
    gap = float(np.linalg.norm(c1 - c0))
    loop_gap = abs(float(np.dot(c1 - c0, circ_unit)))
    yz_gap = float(np.hypot(c1[1] - c0[1], c1[2] - c0[2]))
    ax_gap = abs(_axial_coord(c0, axis_hat) - _axial_coord(c1, axis_hat))
    print(f"  Cut gap (loop)    : {loop_gap * 1e3:.1f} mm  "
          f"(CUT_LOOP_LENGTH {CUT_LOOP_LENGTH * 1e3:.1f} mm)")
    tip_sep = abs(float(np.dot(ring_info[1]['tip'] - ring_info[0]['tip'], circ_unit)))
    print(f"  Tip separation    : {tip_sep * 1e3:.1f} mm  "
          f"(TIP_FAN {TIP_FAN * 1e3:.1f} mm)")
    print(f"  Cut face geometry : total {gap * 1e3:.1f} mm  "
          f"(axial {ax_gap * 1e3:.1f} mm, YZ {yz_gap * 1e3:.1f} mm)")
    t0_ax = _axial_coord(ring_info[0]['tip'], axis_hat)
    t1_ax = _axial_coord(ring_info[1]['tip'], axis_hat)
    print(f"  Tip axial coords    : {t0_ax * 1e3:.2f} mm, {t1_ax * 1e3:.2f} mm  "
          f"(delta {abs(t0_ax - t1_ax) * 1e6:.1f} um)")
    for i, info in enumerate(ring_info):
        print(f"  Lead {i} cross-sect  : mean radius {info['cs_mean'] * 1e3:.2f} mm, "
              f"span {info['cs_span'] * 1e3:.2f} mm")

    rad = max(LEAD_LENGTH * 1.4, 0.04)
    dist = np.linalg.norm(final_mesh.vertices - apex, axis=1)
    crop_faces = final_mesh.faces[np.all(dist[final_mesh.faces] < rad, axis=1)]

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        lead_v = final_mesh.vertices[n_coil_vertices:]
        for name, (elev, azim) in {'iso': (24, -60),
                                    'sideY': (0, 90),
                                    'sideZ': (12, 0)}.items():
            fig = plt.figure(figsize=(9, 7))
            ax = fig.add_subplot(111, projection='3d')
            vs = final_mesh.vertices
            step = max(1, len(crop_faces) // 4000)
            ax.plot_trisurf(vs[:, 0], vs[:, 1], vs[:, 2],
                            triangles=crop_faces[::step],
                            color='steelblue', alpha=0.45, linewidth=0)
            ax.scatter(lead_v[:, 0], lead_v[:, 1], lead_v[:, 2],
                       s=3, c='lime', alpha=0.85, label='leads')
            ax.set_xlim(apex[0] - rad, apex[0] + rad)
            ax.set_ylim(apex[1] - rad, apex[1] + rad)
            ax.set_zlim(apex[2] - rad, apex[2] + rad)
            try:
                ax.set_box_aspect((1, 1, 1))
            except Exception:
                pass
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f'Lead junction ({name})')
            ax.legend(loc='upper right', fontsize=8)
            png = os.path.splitext(output_stl)[0] + f'_verify_{name}.png'
            fig.savefig(png, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Verify image   : {png}")
    except Exception as exc:
        print(f"  (Could not write verify image: {exc})")

    status = 'OK - watertight' if n_boundary == 0 else f'{n_boundary} open edges'
    print(f"  Boundary edges : {n_boundary}  ({status})")
    return n_boundary == 0


# =============================================================================
#  MAIN PIPELINE
# =============================================================================

def main():
    input_stl = INPUT_STL
    if not input_stl:
        input_stl = input("Path to the wire STL file: ").strip().strip('"').strip("'")
    input_stl = os.path.normpath(os.path.expandvars(os.path.expanduser(input_stl)))
    if not os.path.isfile(input_stl):
        raise FileNotFoundError(f"STL file not found: {input_stl}")

    base, ext = os.path.splitext(input_stl)
    output_stl = base + "_with_leads" + ext
    leads_only_stl = base + "_leads_only" + ext
    print(f"\nInput  : {input_stl}") 
    print(f"Output : {output_stl}\n")

    print("Loading mesh...")
    mesh = trimesh.load(input_stl, force='mesh')
    print(f"  {len(mesh.vertices)} vertices | {len(mesh.faces)} faces | "
          f"watertight={mesh.is_watertight}")

    lead_dir = LEAD_DIRECTION / np.linalg.norm(LEAD_DIRECTION)
    axis_hat = (np.asarray(CYL_AXIS, dtype=float) if CYL_AXIS is not None
                else _infer_axis_hat(lead_dir))
    axis_hat = axis_hat / np.linalg.norm(axis_hat)

    print("Locating outermost wire section...")
    apex_v, apex = _find_apex(mesh.vertices, lead_dir)
    wire_tangent = _pca_tangent(mesh.vertices, apex, TANGENT_RADIUS)
    shell_radius = (float(SHELL_RADIUS) if SHELL_RADIUS is not None
                    else _estimate_shell_radius(mesh.vertices, apex, axis_hat))
    print(f"  Apex point      : {np.round(apex, 4)}")
    print(f"  Wire tangent    : {np.round(wire_tangent, 3)}")
    print(f"  Cylinder axis   : {np.round(axis_hat, 3)}")
    print(f"  Shell radius    : {shell_radius * 1e3:.2f} mm")

    wire_hat = wire_tangent / np.linalg.norm(wire_tangent)
    circ_unit = _circumferential_unit(wire_hat, apex, axis_hat)
    wire_profile = _pycoilgen_oval_profile()
    print(f"  Loop direction  : {np.round(circ_unit, 3)}")
    print(f"  Wire oval       : A={wire_profile['semi_a'] * 1e3:.2f} mm  "
          f"B={wire_profile['semi_b'] * 1e3:.2f} mm  "
          f"({CROSS_SECTION_N} pts)")
    print(f"  Cut / start gap : {CUT_LOOP_LENGTH * 1e3:.1f} mm")
    if CUT_LOOP_LENGTH > 0.025:
        print(f"  WARNING           : CUT_LOOP_LENGTH > 25 mm - coil groove is "
              f"heavily opened; leads keep wire profile via template.")

    print("Cutting open gap...")
    del_mask = _flood_bridge(mesh, apex_v, apex, axis_hat, circ_unit,
                             GAP_AXIAL_LENGTH, CUT_LOOP_LENGTH,
                             WIRE_ISOLATE_HALF)
    print(f"  Removed {int(del_mask.sum())} faces from one conductor")
    open_mesh = trimesh.Trimesh(vertices=mesh.vertices.copy(),
                                faces=mesh.faces[~del_mask], process=False)
    open_mesh.remove_unreferenced_vertices()

    print("Detecting cut-face rings...")
    loops, adj = _boundary_loops(open_mesh)
    print(f"  Found {len(loops)} boundary loop(s)")
    if len(loops) != 2:
        raise RuntimeError(
            f"Expected 2 cut-face rings, got {len(loops)}.\n"
            "  Adjust CUT_LOOP_LENGTH / GAP_AXIAL_LENGTH / WIRE_ISOLATE_HALF."
        )

    ordered_loops = [_order_loop(l, adj) for l in loops]
    ring_data = []
    for ring_idx in ordered_loops:
        ring_pts = open_mesh.vertices[ring_idx]
        center = ring_pts.mean(axis=0)
        ring_data.append((ring_idx, center))

    if EXIT_DIRECTION is not None:
        exit_dir = np.asarray(EXIT_DIRECTION, dtype=float)
    else:
        axial_comp = float(np.dot(lead_dir, axis_hat))
        exit_dir = np.sign(axial_comp) * axis_hat if abs(axial_comp) > 1e-9 else axis_hat
    exit_dir = exit_dir / np.linalg.norm(exit_dir)
    print(f"  Axial exit direction: {np.round(exit_dir, 3)}")

    ref_center = np.mean([c for _, c in ring_data], axis=0)
    ring_data.sort(key=lambda rd: float(np.dot(rd[1] - ref_center, circ_unit)))
    cut_loop_gap = abs(float(np.dot(ring_data[1][1] - ring_data[0][1], circ_unit)))
    print(f"  Measured gap    : {cut_loop_gap * 1e3:.1f} mm at weld points")

    exit_axial = float(_axial_coord(ref_center, axis_hat)
                       + np.dot(exit_dir, axis_hat) * LEAD_LENGTH)
    tip_anchor = _snap_to_shell(ref_center, axis_hat, shell_radius)
    print(f"  Shared exit plane : X = {exit_axial * 1e3:.2f} mm")

    spread_signs = (LEAD_0_SPREAD_SIGN, LEAD_1_SPREAD_SIGN)
    print(f"  Spread signs      : lead0={spread_signs[0]:+d}, lead1={spread_signs[1]:+d}")

    print("Building leads (tangent -> face-in -> peel-out -> bore)...")
    lead_parts = []
    ring_info = []
    for i, (ring_idx, center) in enumerate(ring_data):
        sign = float(spread_signs[i])
        loop_dir = circ_unit * sign
        toward_gap = -loop_dir
        fan_offset = np.zeros(3)
        if TIP_FAN > 1e-9:
            tip_fan_dir = _tip_fan_direction(loop_dir)
            fan_offset = tip_fan_dir * (TIP_FAN / 2.0)
        tip = _common_tip(exit_axial, tip_anchor, axis_hat, shell_radius, fan_offset)
        extra, faces, ridx, info = _build_attached_lead(
            ring_indices  = ring_idx,
            vertices      = open_mesh.vertices,
            wire_profile  = wire_profile,
            wire_tangent  = wire_tangent,
            toward_gap    = toward_gap,
            outward       = loop_dir,
            tip           = tip,
            exit_dir      = exit_dir,
            axis_hat      = axis_hat,
            shell_radius  = shell_radius,
            wire_run      = WIRE_TANGENT_RUN,
            face_in       = FACE_TOWARD_GAP,
            peel_out      = PEEL_OUT,
            lead_blend    = LEAD_BLEND,
            tip_fan       = TIP_FAN,
            n_steps       = LEAD_STEPS,
        )
        lead_parts.append((extra, faces, ridx, info['n_path']))
        ring_info.append(info)
        print(f"  Lead {i}: ring n={len(ring_idx)}  "
              f"weld={np.round(info['center'], 4)}  "
              f"depart={np.round(info['departure'], 4)}  "
              f"cs={info['cs_mean'] * 1e3:.2f}mm")

    print("Welding leads to coil...")
    n_coil_vertices = len(open_mesh.vertices)
    leads_only = _standalone_leads_mesh(open_mesh, lead_parts)
    leads_only.fix_normals()
    final = _attach_leads(open_mesh, lead_parts)
    final.fix_normals()

    os.makedirs(os.path.dirname(os.path.abspath(output_stl)), exist_ok=True)
    leads_only.export(leads_only_stl)
    final.export(output_stl)
    print(f"  Leads-only STL : {leads_only_stl}")

    print("Verifying...")
    _verify_result(final, n_coil_vertices, apex, axis_hat, shell_radius,
                   ring_info, circ_unit, output_stl)

    print(f"\n{'=' * 60}")
    print("  DONE")
    print(f"{'=' * 60}")
    print(f"  Cut / start gap: {CUT_LOOP_LENGTH * 1e3:.1f} mm")
    print(f"  Wire tangent   : {WIRE_TANGENT_RUN * 1e3:.1f} mm")
    print(f"  Face-in run    : {FACE_TOWARD_GAP * 1e3:.1f} mm")
    print(f"  Peel-out run   : {PEEL_OUT * 1e3:.1f} mm")
    print(f"  Tip fan        : {TIP_FAN * 1e3:.1f} mm")
    print(f"  Conductor oval : {CONDUCTOR_WIDTH * 1e3:.2f} mm  "
          f"A={CROSS_SECTION_A_FRAC} B={CROSS_SECTION_B_FRAC}")
    print(f"  Spread signs   : lead0={LEAD_0_SPREAD_SIGN:+d}, lead1={LEAD_1_SPREAD_SIGN:+d}")
    print(f"  Final mesh     : {len(final.vertices)} verts / {len(final.faces)} faces")
    print(f"  Saved to       : {output_stl}")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
