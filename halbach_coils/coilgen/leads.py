"""
Add two lead wires to a pyCoilGen wire-layout STL for the negative-mold
workflow.

The pyCoilGen wire is a single closed, water-tight tube (all saddle loops in
one conductor). To wind physical wire from a 3-D-printed groove mold you need
an inlet and an outlet. This module:

  1. Locates the outermost wire section (furthest along ``lead_direction`` in
     a chosen angular sector of the cylinder).
  2. Estimates the local wire tangent with a small PCA neighbourhood.
  3. Removes a segment of length ``cut_loop_length`` along the loop at that
     station, leaving two cut faces on opposite sides of the gap.
  4. Each lead is a single C1-smooth cubic Bezier from its cut face to the
     shared axial exit plane. The initial tangent is the wire's running
     direction *away from the gap* (so the lead emerges tangentially from the
     main wire), and the final tangent is the bore axis (so the lead exits
     axially). Tips fan apart by ``tip_fan`` on the shared exit plane. Both
     leads terminate on the same exit plane so they are equal length. The
     cross-section blends from the real cut-face shape into the pyCoilGen oval
     over ``cs_blend_rings`` on a twist-free RMF basis.

Axis awareness (fix for the non-Gy bug)
---------------------------------------
The former script picked the apex with a world-frame filter (Z > 0.10 and
|Y| < 0.05) tuned for Gy. Here the apex is selected in *cylindrical*
coordinates relative to the bore axis: an angular wedge in the radial plane
around ``preset.sector_ref_dir`` plus the axial projection onto
``preset.lead_direction``. The bore axis defaults to
``cfg.rotated_cylinder_axis`` rather than being inferred from the lead
direction, and the spread signs / exit direction come from the per-axis
preset in :mod:`coilgen.config`.

Output: ``<name>_with_leads.stl``, ``<name>_coil_open.stl``,
``<name>_leads_only.stl``.
"""

from __future__ import annotations

import os
from collections import defaultdict, deque
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
import trimesh

from . import geometry as geo
from .config import Config
from .paths import unique_lead_output_paths


# Module-level scalar params, populated by run_leads(). Helpers read from here
# to stay 1:1 with the original add_coil_leads.py without threading every
# constant through every signature.
_P = SimpleNamespace()


# ---------------------------------------------------------------------------
# Geometry helpers (axis-agnostic; all use axis_hat)
# ---------------------------------------------------------------------------

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


def _unit(vec, fallback=None):
    """Return a normalized vector, or a normalized fallback if degenerate."""
    v = np.asarray(vec, dtype=float)
    n = np.linalg.norm(v)
    if n > 1e-12:
        return v / n
    if fallback is None:
        raise ValueError("Cannot normalize a degenerate vector without fallback.")
    fb = np.asarray(fallback, dtype=float)
    fn = np.linalg.norm(fb)
    if fn < 1e-12:
        raise ValueError("Fallback vector is also degenerate.")
    return fb / fn


def _surface_frame(center, ref_center, vertices, axis_hat, tangent_radius,
                   fallback_tangent=None):
    """
    Local lead frame at a cut face.

    n: surface normal (radial direction)
    t: wire tangent projected to the shell and signed away from the cut gap
    b: RMF-compatible binormal (t x n)
    """
    n_hat = _radial_hat(center, axis_hat)
    raw_t = _pca_tangent(vertices, center, tangent_radius)
    t_hat = _tangent_offset(raw_t, center, axis_hat)

    if np.linalg.norm(t_hat) < 1e-12 and fallback_tangent is not None:
        t_hat = _tangent_offset(fallback_tangent, center, axis_hat)
    t_hat = _unit(t_hat, fallback_tangent if fallback_tangent is not None else axis_hat)

    away = _tangent_offset(np.asarray(center) - np.asarray(ref_center), center, axis_hat)
    if np.linalg.norm(away) > 1e-9:
        if np.dot(t_hat, away) < 0.0:
            t_hat = -t_hat
    elif fallback_tangent is not None and np.dot(t_hat, fallback_tangent) < 0.0:
        t_hat = -t_hat

    b_hat = _unit(np.cross(t_hat, n_hat), np.cross(axis_hat, n_hat))
    return n_hat, t_hat, b_hat


def _project_exit_tangent(exit_dir, at_point, axis_hat):
    """Exit direction projected into the local shell tangent plane."""
    tangent = _tangent_offset(exit_dir, at_point, axis_hat)
    if np.linalg.norm(tangent) < 1e-12:
        tangent = _tangent_offset(axis_hat, at_point, axis_hat)
    return _unit(tangent, axis_hat)


def _route_frame(center, ref_center, exit_dir, axis_hat, wire_tangent):
    """Axis-independent shell routing frame for a lead cut face."""
    route_dir = _project_exit_tangent(exit_dir, center, axis_hat)
    side_dir = _tangent_offset(np.asarray(center) - np.asarray(ref_center),
                               center, axis_hat)
    # Tip separation should be lateral to the route, not another axial offset.
    side_dir = side_dir - np.dot(side_dir, route_dir) * route_dir
    if np.linalg.norm(side_dir) < 1e-9:
        side_dir = _tangent_offset(wire_tangent, center, axis_hat)
        side_dir = side_dir - np.dot(side_dir, route_dir) * route_dir
    side_dir = _unit(side_dir, wire_tangent)
    return route_dir, side_dir


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
    """Rotate a topologically ordered boundary loop so vertex 0 is at the
    smallest polar angle -- keeps mesh edge connectivity intact."""
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
    """Closed ellipse matching the pyCoilGen conductor cross-section."""
    theta = np.linspace(0.0, 2.0 * np.pi, _P.cross_section_n, endpoint=True)
    template = np.column_stack([
        _P.conductor_semi_a * np.sin(theta),
        _P.conductor_semi_b * np.cos(theta),
    ])
    radii = np.linalg.norm(template, axis=1)
    return {
        'template_2d': template,
        'cs_mean': float(radii.mean()),
        'cs_span': float(radii.ptp()),
        'semi_a': _P.conductor_semi_a,
        'semi_b': _P.conductor_semi_b,
    }


def _ordered_ring_2d(ring_3d, center, ax1, ax2):
    pts = np.column_stack([(ring_3d - center) @ ax1, (ring_3d - center) @ ax2])
    ang = np.arctan2(pts[:, 1], pts[:, 0])
    return pts[np.argsort(ang)]


def _resample_closed_2d(pts_2d, n_out):
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


def _align_oval_to_cut(oval_2d, cut_2d):
    oval = _resample_closed_2d(oval_2d, len(cut_2d))
    a0 = np.arctan2(cut_2d[0, 1], cut_2d[0, 0])
    a1 = np.arctan2(oval[0, 1], oval[0, 0])
    rot = a0 - a1
    c_, s_ = np.cos(rot), np.sin(rot)
    return oval @ np.array([[c_, -s_], [s_, c_]])


def _ring_cross_sections(ring_3d, wire_profile):
    center, _, ax1, ax2 = _loop_plane(ring_3d)
    cut_2d = _ordered_ring_2d(ring_3d, center, ax1, ax2)
    oval_2d = _align_oval_to_cut(wire_profile['template_2d'], cut_2d)
    prof = wire_profile
    return cut_2d, oval_2d, prof['cs_mean'], prof['cs_span']


def _profile_ring_2d(wire_profile, ring_3d, n_pts):
    cut_2d, oval_2d, cs_mean, cs_span = _ring_cross_sections(ring_3d, wire_profile)
    if len(oval_2d) != n_pts:
        oval_2d = _resample_closed_2d(oval_2d, n_pts)
        cut_2d = _resample_closed_2d(cut_2d, n_pts)
    return oval_2d, cs_mean, cs_span, cut_2d


def _profile_ring_2d_in_frame(wire_profile, ring_3d, normal, binormal, n_pts):
    """Cut-face and oval profiles expressed in the local surface frame."""
    center = np.asarray(ring_3d, dtype=float).mean(axis=0)
    cut_2d = np.column_stack([
        (ring_3d - center) @ normal,
        (ring_3d - center) @ binormal,
    ])
    oval_2d = _resample_closed_2d(wire_profile['template_2d'], n_pts)
    if len(cut_2d) != n_pts:
        cut_2d = _resample_closed_2d(cut_2d, n_pts)
    radii = np.linalg.norm(oval_2d, axis=1)
    return oval_2d, float(radii.mean()), float(radii.ptp()), cut_2d


def _lead_centerline(p0, tangent, route_dir, tip, exit_tangent, axis_hat,
                     shell_radius, blend, n_steps):
    """Two-stage shell path: toward-gap fillet, then route-aligned exit run."""
    start_t = _unit(tangent)
    route_t = _unit(route_dir)
    end_t = _unit(exit_tangent)
    p0 = _snap_to_shell(np.asarray(p0, dtype=float), axis_hat, shell_radius)
    tip = _snap_to_shell(np.asarray(tip, dtype=float), axis_hat, shell_radius)

    chord = float(np.linalg.norm(tip - p0))
    if chord < 1e-12:
        return np.tile(p0, (max(2, n_steps), 1))

    fillet_len = min(float(blend), 0.30 * chord, 0.006)
    fillet_len = max(fillet_len, min(1.5 * _P.conductor_width, 0.20 * chord))
    route_forward = max(float(np.dot(tip - p0, route_t)), 0.0)
    join_route = min(max(0.35 * route_forward, fillet_len), 0.65 * chord)
    join = _snap_to_shell(p0 + start_t * fillet_len + route_t * join_route,
                          axis_hat, shell_radius)

    h1 = min(0.60 * fillet_len, 0.35 * np.linalg.norm(join - p0))
    c1 = _snap_to_shell(p0 + start_t * h1, axis_hat, shell_radius)
    c2 = _snap_to_shell(join - route_t * h1, axis_hat, shell_radius)
    n_fillet = max(8, min(n_steps // 3, 32))
    fillet = _shell_bezier(p0, c1, c2, join, axis_hat, shell_radius, n_fillet)

    remaining = float(np.linalg.norm(tip - join))
    if remaining < 1e-6:
        return fillet
    h2 = min(float(blend), 0.45 * remaining)
    h2 = max(h2, min(1.5 * _P.conductor_width, 0.30 * remaining))
    c3 = _snap_to_shell(join + route_t * h2, axis_hat, shell_radius)
    c4 = _snap_to_shell(tip - end_t * h2, axis_hat, shell_radius)
    n_run = max(8, n_steps - len(fillet) + 1)
    run = _shell_bezier(join, c3, c4, tip, axis_hat, shell_radius, n_run)
    return np.vstack([fillet, run[1:]])


def _common_tip(exit_axial, anchor, axis_hat, shell_radius, fan_offset):
    """Tip on the shared exit plane."""
    radial = _radial_vec(anchor, axis_hat)
    fan = (_tangent_offset(fan_offset, anchor, axis_hat)
           if np.linalg.norm(fan_offset) > 1e-12 else np.zeros(3))
    tip = float(exit_axial) * axis_hat + radial + fan
    return _snap_to_shell(tip, axis_hat, shell_radius)


def _perp_distance_to_loop(points, apex, circ_unit):
    pts = np.asarray(points, dtype=float)
    d = pts - apex
    along = np.outer(d @ circ_unit, circ_unit)
    return np.linalg.norm(d - along, axis=1)


# ---------------------------------------------------------------------------
# Axis-aware apex selection (the fix)
# ---------------------------------------------------------------------------

def _find_apex(vertices, lead_dir, axis_hat, sector_ref_dir, sector_angular_half):
    """
    Locate the outermost wire station in an angular wedge of the cylinder.

    The wedge is defined in the radial plane (perpendicular to ``axis_hat``)
    around ``sector_ref_dir`` with half-width ``sector_angular_half`` [rad].
    The apex is the vertex in that wedge with the largest projection onto
    ``lead_dir`` (axial exit direction). Falls back to all vertices if the
    wedge is empty, but logs a warning so the misplacement is not silent.
    """
    proj = vertices @ lead_dir
    radial = _radial_vec(vertices, axis_hat)
    radial_norm = np.linalg.norm(radial, axis=1)
    safe = radial_norm > 1e-12
    ref = np.asarray(sector_ref_dir, dtype=float)
    ref = ref - np.dot(ref, axis_hat) * axis_hat      # project into radial plane
    ref = ref / (np.linalg.norm(ref) + 1e-12)
    cos_angle = np.zeros(radial_norm.shape)
    cos_angle[safe] = (radial[safe] @ ref) / radial_norm[safe]
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    sector = angle < sector_angular_half
    idx = np.where(sector)[0]
    if len(idx) == 0:
        print(f"  WARNING: apex sector empty (ref={np.round(ref,3)}, "
              f"half={sector_angular_half:.3f} rad); falling back to all vertices.")
        idx = np.arange(len(vertices))
    apex_v = idx[np.argmax(proj[idx])]
    return apex_v, vertices[apex_v]


# ---------------------------------------------------------------------------
# Cut / flood / build
# ---------------------------------------------------------------------------

def _flood_bridge(mesh, apex_v, apex, axis_hat, circ_unit, gap_axial_length,
                  cut_loop_length, perp_half):
    """Remove a short conductor segment at the apex station on one loop."""
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
            "Apex face not inside cut slab -- increase wire_isolate_half, "
            "gap_axial_length, or cut_loop_length."
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


def _build_attached_lead(ring_indices, vertices, wire_profile, normal,
                         junction_tangent, binormal, route_dir, tip,
                         exit_tangent, axis_hat, shell_radius, lead_blend,
                         n_steps):
    """RMF sweep: cut-face profile blends into pyCoilGen oval along the path."""
    ring_3d = vertices[ring_indices]
    n_pts = len(ring_indices)
    center, _, ax1, ax2 = _loop_plane(ring_3d)
    ring_indices, ring_3d = _rotate_ordered_ring(ring_indices, ring_3d, center, ax1, ax2)
    center = _snap_to_shell(center, axis_hat, shell_radius)
    oval_2d, cs_mean, cs_span, cut_2d = _profile_ring_2d_in_frame(
        wire_profile, ring_3d, normal, binormal, n_pts,
    )

    path = _lead_centerline(
        p0=center, tangent=junction_tangent, route_dir=route_dir, tip=tip,
        exit_tangent=exit_tangent,
        axis_hat=axis_hat, shell_radius=shell_radius, blend=lead_blend,
        n_steps=n_steps,
    )
    path[-1] = np.asarray(tip, dtype=float)
    n_path = len(path)
    departure = path[min(n_path - 1, max(1, n_path // 2))]
    mid_i = min(max(1, n_path // 2), n_path - 2)
    mid_tangent = _unit(path[mid_i + 1] - path[mid_i], route_dir)

    _, N, B = _rotation_minimizing_frames(path)
    first_tangent = _unit(path[1] - path[0], junction_tangent)
    target_normal = normal - np.dot(normal, first_tangent) * first_tangent
    target_normal = _unit(target_normal, normal)
    cos_a = float(np.dot(N[0], target_normal))
    sin_a = float(np.dot(B[0], target_normal))
    angle = np.arctan2(sin_a, cos_a)
    c_, s_ = np.cos(angle), np.sin(angle)
    for i in range(n_path):
        Nn = c_ * N[i] + s_ * B[i]
        Bn = -s_ * N[i] + c_ * B[i]
        N[i], B[i] = Nn, Bn
    section_dot_normal = abs(float(np.dot(N[0], normal)))

    extra_rings = []
    blend_n = max(1, _P.cs_blend_rings)
    n_rigid = min(_P.junction_rigid_steps, n_path - 1)

    # RMF-aligned sweep for every path ring. N[i], B[i] are already aligned to
    # the cut-face basis (ax1, ax2) at i=0 by the rotation above, so using them
    # directly keeps the lead tangent-continuous with the wire. The rigid
    # junction rings carry the cut-face profile (cut_2d); after n_rigid the
    # profile blends cut_2d -> oval_2d on the same RMF basis for a smooth
    # section transition (no frame jump).
    for i in range(1, n_path):
        Ni, Bi = N[i], B[i]
        if i <= n_rigid:
            ring = path[i] + np.outer(cut_2d[:, 0], Ni) + np.outer(cut_2d[:, 1], Bi)
            ring = np.array([
                _snap_to_shell(p, axis_hat, shell_radius) for p in ring
            ])
            extra_rings.append(ring)
            continue
        blend_i = i - 1 - n_rigid
        t = min(1.0, blend_i / blend_n)
        r2d = (1.0 - t) * cut_2d + t * oval_2d
        extra_rings.append(
            path[i]
            + np.outer(r2d[:, 0], Ni)
            + np.outer(r2d[:, 1], Bi)
        )

    n_ring = len(extra_rings)
    cap_center = extra_rings[-1].mean(axis=0)
    extra_vertices = np.vstack(extra_rings + [cap_center[None, :]])

    faces = []
    cap_local = n_pts + n_ring * n_pts
    for i in range(n_ring):
        row_a = np.arange(i * n_pts, (i + 1) * n_pts)
        row_b = np.arange((i + 1) * n_pts, (i + 2) * n_pts)
        for j in range(n_pts):
            jn = (j + 1) % n_pts
            a, b = row_a[j], row_a[jn]
            c, d = row_b[j], row_b[jn]
            faces += [[a, b, d], [a, d, c]]
    last_ring = np.arange(n_pts + (n_ring - 1) * n_pts, n_pts + n_ring * n_pts)
    for j in range(n_pts):
        faces.append([last_ring[j], cap_local, last_ring[(j + 1) % n_pts]])

    return extra_vertices, np.array(faces, dtype=int), np.asarray(ring_indices), {
        'center': center,
        'departure': np.asarray(departure, dtype=float),
        'tip': np.asarray(path[-1], dtype=float),
        'cs_mean': cs_mean,
        'cs_span': cs_span,
        'n_path': n_ring + 1,
        'approach_run': 0.0,
        'normal': np.asarray(normal, dtype=float),
        'junction_tangent': np.asarray(junction_tangent, dtype=float),
        'binormal': np.asarray(binormal, dtype=float),
        'route_dir': np.asarray(route_dir, dtype=float),
        'first_tangent': first_tangent,
        'mid_tangent': mid_tangent,
        'first_tangent_dot': float(np.dot(first_tangent, junction_tangent)),
        'route_tangent_dot': float(np.dot(mid_tangent, route_dir)),
        'section_normal_dot': section_dot_normal,
        'tip_forward': float(np.dot(np.asarray(path[-1]) - center, route_dir)),
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
    n_pts = len(ring_3d)
    center = ring_3d.mean(axis=0)
    verts = np.vstack([ring_3d, center[None, :]])
    cap_i = n_pts
    faces = [[cap_i, i, (i + 1) % n_pts] for i in range(n_pts)]
    return verts, np.array(faces, dtype=int)


def _standalone_leads_mesh(open_mesh, lead_parts):
    parts = []
    for extra_vertices, lead_faces, ring_indices, _ in lead_parts:
        ring_3d = open_mesh.vertices[ring_indices]
        body = trimesh.Trimesh(
            vertices=np.vstack([ring_3d, extra_vertices]),
            faces=lead_faces,
            process=False,
        )
        cap_i = len(body.vertices)
        body.vertices = np.vstack([body.vertices, ring_3d.mean(axis=0)[None, :]])
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
                   ring_info, circ_unit, axis_name):
    ec = np.bincount(final_mesh.edges_unique_inverse,
                     minlength=len(final_mesh.edges_unique))
    n_boundary = int((ec == 1).sum())
    n_nonmanifold = int((ec > 2).sum())

    coil_r = _shell_radius_of_points(final_mesh.vertices[:n_coil_vertices], axis_hat)
    lead_r = _shell_radius_of_points(final_mesh.vertices[n_coil_vertices:], axis_hat)
    coil_near = coil_r[np.linalg.norm(
        final_mesh.vertices[:n_coil_vertices] - apex, axis=1) < 0.025]
    print(f"  Shell radius target : {shell_radius * 1e3:.2f} mm")
    print(f"  Coil  radius (cut)  : {np.median(coil_near) * 1e3:.2f} mm  "
          f"(spread {np.ptp(coil_near) * 1e3:.2f} mm)")
    print(f"  Lead  radius        : {np.median(lead_r) * 1e3:.2f} mm  "
          f"(spread {np.ptp(lead_r) * 1e3:.2f} mm)")

    c0, c1 = ring_info[0]['center'], ring_info[1]['center']
    gap = float(np.linalg.norm(c1 - c0))
    loop_gap = abs(float(np.dot(c1 - c0, circ_unit)))
    rad_gap = float(np.linalg.norm(_radial_vec(c1, axis_hat) - _radial_vec(c0, axis_hat)))
    ax_gap = abs(_axial_coord(c0, axis_hat) - _axial_coord(c1, axis_hat))
    print(f"  Cut gap (loop)    : {loop_gap * 1e3:.1f} mm  "
          f"(cut_loop_length {_P.cut_loop_length * 1e3:.1f} mm)")
    tip_sep = abs(float(np.dot(ring_info[1]['tip'] - ring_info[0]['tip'], circ_unit)))
    print(f"  Tip separation    : {tip_sep * 1e3:.1f} mm  "
          f"(tip_fan {_P.tip_fan * 1e3:.1f} mm)")
    print(f"  Cut face geometry : total {gap * 1e3:.1f} mm  "
          f"(axial {ax_gap * 1e3:.1f} mm, radial {rad_gap * 1e3:.1f} mm)")
    t0_ax = _axial_coord(ring_info[0]['tip'], axis_hat)
    t1_ax = _axial_coord(ring_info[1]['tip'], axis_hat)
    print(f"  Tip axial coords    : {t0_ax * 1e3:.2f} mm, {t1_ax * 1e3:.2f} mm  "
          f"(delta {abs(t0_ax - t1_ax) * 1e6:.1f} um)")
    for i, info in enumerate(ring_info):
        print(f"  Lead {i} cross-sect  : mean radius {info['cs_mean'] * 1e3:.2f} mm, "
              f"span {info['cs_span'] * 1e3:.2f} mm")
        print(f"  Lead {i} frame check : toward-gap {info['first_tangent_dot']:.3f}, "
              f"route {info['route_tangent_dot']:.3f}, "
              f"normal {info['section_normal_dot']:.3f}, "
              f"route-forward {info['tip_forward'] * 1e3:.1f} mm")

    cut_vec = c1 - c0
    tip_vec = ring_info[1]['tip'] - ring_info[0]['tip']
    denom = np.linalg.norm(cut_vec) * np.linalg.norm(tip_vec)
    tip_order_dot = float(np.dot(cut_vec, tip_vec) / denom) if denom > 1e-12 else 0.0
    frame_ok = all(
        info['first_tangent_dot'] > 0.75
        and info['route_tangent_dot'] > 0.80
        and info['section_normal_dot'] > 0.90
        and info['tip_forward'] >= -1e-4
        for info in ring_info
    )
    tip_order_ok = tip_order_dot > 0.0
    print(f"  Tip order check  : dot {tip_order_dot:.3f}  "
          f"({'OK' if tip_order_ok else 'CROSSED'})")

    status = 'OK - watertight' if n_boundary == 0 and n_nonmanifold == 0 else (
        f'{n_boundary} open edges, {n_nonmanifold} non-manifold edges')
    print(f"  Boundary edges : {n_boundary}  ({status})")
    print(f"  Local frame checks: {'OK' if frame_ok and tip_order_ok else 'CHECK'}")
    if n_boundary > 0:
        print("  TIP: open edges at the lead/coil junction -- try increasing "
              "cs_blend_rings / junction_rigid_steps in "
              "LeadsConfig for a smoother tangent-continuous transition.")
    return n_boundary == 0 and n_nonmanifold == 0 and frame_ok and tip_order_ok


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _populate_params(cfg: Config):
    w = cfg.wire
    lc = cfg.leads
    _P.conductor_width = w.conductor_width
    _P.conductor_semi_a = cfg.conductor_semi_a
    _P.conductor_semi_b = cfg.conductor_semi_b
    _P.cross_section_a_frac = w.cross_section_a_frac
    _P.cross_section_b_frac = w.cross_section_b_frac
    _P.cross_section_n = w.cross_section_n
    _P.cs_blend_rings = lc.cs_blend_rings
    _P.junction_rigid_steps = lc.junction_rigid_steps
    _P.junction_plane_rings = lc.junction_plane_rings
    _P.cut_loop_length = lc.cut_loop_length
    _P.gap_axial_length = lc.gap_axial_length
    _P.wire_isolate_half = lc.wire_isolate_half
    _P.tangent_radius = lc.tangent_radius
    _P.wire_tangent_run = lc.wire_tangent_run
    _P.face_toward_gap = lc.face_toward_gap
    _P.peel_out = lc.peel_out
    _P.lead_junction_coil_backset = lc.lead_junction_coil_backset
    _P.lead_junction_gap_backset = lc.lead_junction_gap_backset
    _P.lead_length = lc.lead_length
    _P.lead_blend = lc.lead_blend
    _P.tip_fan = lc.tip_fan
    _P.lead_steps = lc.lead_steps


def run_leads(
    cfg: Config,
    input_stl: Optional[str] = None,
) -> Tuple[str, str, str]:
    """
    Add leads to ``input_stl`` (a pyCoilGen wire STL).

    Returns ``(with_leads, coil_open, leads_only)`` output STL paths. Uses the
    per-axis lead preset from ``cfg`` for the apex sector, lead direction and
    spread signs, and the cylinder bore axis from ``cfg.rotated_cylinder_axis``.
    """
    _populate_params(cfg)
    preset = cfg.lead_preset()

    if input_stl is None:
        from .paths import resolve_wire_stl_path
        input_stl = resolve_wire_stl_path(
            cfg.output_dir, cfg.gradient_axis,
            cfg.tikhonov_factor, cfg.num_levels,
        )
    if not input_stl or not os.path.isfile(input_stl):
        raise FileNotFoundError(f"Wire STL file not found: {input_stl!r}")

    output_stl, coil_open_stl, leads_only_stl = unique_lead_output_paths(input_stl)
    print(f"\nInput  : {input_stl}")
    print(f"Output : {output_stl}\n")

    print("Loading mesh...")
    mesh = trimesh.load(input_stl, force='mesh')
    print(f"  {len(mesh.vertices)} vertices | {len(mesh.faces)} faces | "
          f"watertight={mesh.is_watertight}")

    lead_dir = np.asarray(preset.lead_direction, dtype=float)
    lead_dir = lead_dir / np.linalg.norm(lead_dir)
    # Bore axis defaults to the rotated cylinder axis (fix: was inferred from
    # lead_dir, which only worked for Gy).
    axis_hat = (np.asarray(cfg.leads.cyl_axis, dtype=float)
                if cfg.leads.cyl_axis is not None
                else cfg.rotated_cylinder_axis)
    axis_hat = axis_hat / np.linalg.norm(axis_hat)
    axis_name = {0: 'X', 1: 'Y', 2: 'Z'}[int(np.argmax(np.abs(axis_hat)))]

    print("Locating outermost wire section...")
    apex_v, apex = _find_apex(
        mesh.vertices, lead_dir, axis_hat,
        preset.sector_ref_dir, preset.sector_angular_half,
    )
    wire_tangent = _pca_tangent(mesh.vertices, apex, _P.tangent_radius)
    shell_radius = (float(cfg.leads.shell_radius) if cfg.leads.shell_radius is not None
                    else _estimate_shell_radius(mesh.vertices, apex, axis_hat))
    print(f"  Apex point      : {np.round(apex, 4)}")
    print(f"  Wire tangent    : {np.round(wire_tangent, 3)}")
    print(f"  Cylinder axis   : {np.round(axis_hat, 3)} ({axis_name})")
    print(f"  Shell radius    : {shell_radius * 1e3:.2f} mm")

    wire_hat = wire_tangent / np.linalg.norm(wire_tangent)
    circ_unit = _circumferential_unit(wire_hat, apex, axis_hat)
    wire_profile = _pycoilgen_oval_profile()
    print(f"  Loop direction  : {np.round(circ_unit, 3)}")
    print(f"  Wire oval       : A={wire_profile['semi_a'] * 1e3:.2f} mm  "
          f"B={wire_profile['semi_b'] * 1e3:.2f} mm  "
          f"({_P.cross_section_n} pts)")
    print(f"  Cut / start gap : {_P.cut_loop_length * 1e3:.1f} mm")
    if _P.cut_loop_length > 0.080:
        print(f"  WARNING           : cut_loop_length > 80 mm - coil groove is "
              f"very heavily opened; consider reducing lead_length/lead_blend.")

    print("Cutting open gap...")
    del_mask = _flood_bridge(mesh, apex_v, apex, axis_hat, circ_unit,
                             _P.gap_axial_length, _P.cut_loop_length,
                             _P.wire_isolate_half)
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
            "  Adjust cut_loop_length / gap_axial_length / wire_isolate_half "
            "(or the lead preset sector for this axis)."
        )

    ordered_loops = [_order_loop(l, adj) for l in loops]
    ring_data = []
    for ring_idx in ordered_loops:
        ring_pts = open_mesh.vertices[ring_idx]
        center = ring_pts.mean(axis=0)
        ring_data.append((ring_idx, center))

    exit_direction = preset.exit_direction
    if exit_direction is not None:
        exit_dir = np.asarray(exit_direction, dtype=float)
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
                       + np.dot(exit_dir, axis_hat) * _P.lead_length)
    tip_anchor = _snap_to_shell(ref_center, axis_hat, shell_radius)
    print(f"  Shared exit plane : {axis_name} = {exit_axial * 1e3:.2f} mm")

    spread_signs = preset.spread_signs
    print(f"  Spread signs (fallback): lead0={spread_signs[0]:+d}, lead1={spread_signs[1]:+d}")

    print("Building leads (local frame -> smooth shell Bezier to bore)...")
    lead_parts = []
    ring_info = []
    for i, (ring_idx, center) in enumerate(ring_data):
        fallback = circ_unit * float(spread_signs[i])
        normal, tangent, binormal = _surface_frame(
            center, ref_center, open_mesh.vertices, axis_hat,
            _P.tangent_radius, fallback_tangent=fallback,
        )
        toward_gap = _tangent_offset(ref_center - center, center, axis_hat)
        junction_tangent = _unit(toward_gap, -tangent)
        if np.dot(junction_tangent, -tangent) < 0.0:
            junction_tangent = -junction_tangent
        route_dir, side_dir = _route_frame(center, ref_center, exit_dir,
                                           axis_hat, tangent)
        fan_offset = np.zeros(3)
        if _P.tip_fan > 1e-9:
            fan_offset = side_dir * (_P.tip_fan / 2.0)
        tip = _common_tip(exit_axial, tip_anchor, axis_hat, shell_radius, fan_offset)
        min_forward = max(0.35 * _P.lead_length, 2.0 * _P.conductor_width)
        forward = float(np.dot(tip - center, route_dir))
        if forward < min_forward:
            tip = _snap_to_shell(tip + route_dir * (min_forward - forward),
                                 axis_hat, shell_radius)
        exit_tangent = _project_exit_tangent(exit_dir, tip, axis_hat)
        extra, faces, ridx, info = _build_attached_lead(
            ring_indices=ring_idx,
            vertices=open_mesh.vertices,
            wire_profile=wire_profile,
            normal=normal,
            junction_tangent=junction_tangent,
            binormal=binormal,
            route_dir=route_dir,
            tip=tip,
            exit_tangent=exit_tangent,
            axis_hat=axis_hat,
            shell_radius=shell_radius,
            lead_blend=_P.lead_blend,
            n_steps=_P.lead_steps,
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
    open_mesh.export(coil_open_stl)
    leads_only.export(leads_only_stl)
    final.export(output_stl)
    print(f"  Coil-open STL  : {coil_open_stl}")
    print(f"  Leads-only STL : {leads_only_stl}")

    print("Verifying...")
    _verify_result(final, n_coil_vertices, apex, axis_hat, shell_radius,
                   ring_info, circ_unit, axis_name)

    print(f"\n{'=' * 60}")
    print("  DONE")
    print(f"{'=' * 60}")
    print(f"  Gradient axis  : {cfg.axis_label}")
    print(f"  Cut / start gap: {_P.cut_loop_length * 1e3:.1f} mm")
    print(f"  Lead length    : {_P.lead_length * 1e3:.1f} mm")
    print(f"  Lead blend     : {_P.lead_blend * 1e3:.1f} mm")
    print(f"  Tip fan        : {_P.tip_fan * 1e3:.1f} mm")
    print(f"  Conductor diam : {_P.conductor_width * 1e3:.2f} mm  "
          f"A={_P.cross_section_a_frac} B={_P.cross_section_b_frac}")
    print(f"  Spread signs   : lead0={spread_signs[0]:+d}, lead1={spread_signs[1]:+d}")
    print(f"  Final mesh     : {len(final.vertices)} verts / {len(final.faces)} faces")
    print(f"  Saved to       : {output_stl}")
    print(f"{'=' * 60}\n")

    return output_stl, coil_open_stl, leads_only_stl


def main():
    """CLI entry: add leads to a wire STL using a default Config."""
    import argparse
    parser = argparse.ArgumentParser(description="Add lead wires to a wire STL.")
    parser.add_argument('input_stl', nargs='?', default='')
    parser.add_argument('--axis', default='y', choices=('x', 'y', 'z'))
    parser.add_argument('--output-dir', default='')
    args = parser.parse_args()

    cfg = Config(gradient_axis=args.axis)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    run_leads(cfg, input_stl=args.input_stl or None)


if __name__ == '__main__':
    main()
