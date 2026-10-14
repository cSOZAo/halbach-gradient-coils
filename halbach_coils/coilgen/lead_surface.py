"""Cylindrical lead geometry adapted from the preserved surface-lead fix.

Internal units are metres. Input is the exact ordered pyCoilGen sweep; the
coil outside the cut is retained. No GUI, shell carving or electrical model.

The profile-preserving routing was integrated in the preserved checkout by
3c6bb0f9d56ccd4e937b766c9c64ef8df7ae9bb4. This module keeps its geometry
checks and removes the former experimental UI/export workflow.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json

import manifold3d as m3d
import numpy as np
import trimesh
from scipy.spatial import cKDTree


class GeometryRejected(ValueError):
    """No validated geometry may be exported for this candidate."""


def unit(v):
    v = np.asarray(v, dtype=float)
    lengths = np.linalg.norm(v, axis=-1, keepdims=True)
    if not np.all(np.isfinite(v)) or np.any(lengths < 1e-12):
        raise GeometryRejected("Degenerate or non-finite direction")
    return v / lengths


@dataclass
class Options:
    cut_lengths: tuple = (.020, .026, .032, .040)
    tip_separations: tuple = (.015, .020, .026)
    max_step: float = .0003
    min_bend_radius: float = .006
    clearance: float = .0003
    exit_margin: float = .005
    candidates: int = 16
    section_tolerance: float = .02
    radius_tolerance: float = .00015
    sector_angular_half: float = .35

    def validate(self):
        for name in ("max_step", "min_bend_radius", "exit_margin", "radius_tolerance"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if (not np.isfinite(self.clearance) or self.clearance < 0
                or not isinstance(self.candidates, (int, np.integer))
                or isinstance(self.candidates, (bool, np.bool_)) or self.candidates < 1):
            raise ValueError("Invalid clearance/candidate count")
        if not 0 < self.section_tolerance < 1:
            raise ValueError("Invalid section tolerance")
        if not 0 < self.sector_angular_half <= np.pi:
            raise ValueError("Invalid sector angular half-width")
        if (not len(self.cut_lengths) or not len(self.tip_separations)
                or any(not np.isfinite(x) or x <= 0
                       for x in (*self.cut_lengths, *self.tip_separations))):
            raise ValueError("Cut lengths and tip separations must be positive")


@dataclass
class Reference:
    path: np.ndarray
    rings: np.ndarray
    faces: np.ndarray
    axis: np.ndarray
    origin: np.ndarray
    semi_a: float
    semi_b: float
    shell_inner: float
    shell_outer: float
    shell_height: float

    @classmethod
    def load(cls, filename):
        with np.load(filename, allow_pickle=False) as data:
            kwargs = {key: data[key].copy() for key in cls.__dataclass_fields__}
        for key in ("semi_a", "semi_b", "shell_inner", "shell_outer", "shell_height"):
            kwargs[key] = float(kwargs[key])
        ref = cls(**kwargs)
        ref.validate()
        return ref

    def validate(self):
        for name in ('axis', 'origin', 'path', 'rings'):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))
        self.faces = np.asarray(self.faces)
        if (self.axis.shape != (3,) or self.origin.shape != (3,)
                or not np.all(np.isfinite(self.origin))):
            raise GeometryRejected('Axis and origin must be finite three-dimensional vectors')
        self.axis = unit(self.axis)
        if self.rings.ndim != 3 or self.rings.shape[2] != 3 or self.rings.shape[1] < 8:
            raise GeometryRejected("Expected ordered sections with at least eight corners")
        if self.path.shape != (len(self.rings), 3) or len(self.path) < 12:
            raise GeometryRejected("Invalid path/ring correspondence")
        if not np.all(np.isfinite(self.rings)) or not np.all(np.isfinite(self.path)):
            raise GeometryRejected("Non-finite reference geometry")
        dimensions = (self.semi_a, self.semi_b, self.shell_inner,
                      self.shell_outer, self.shell_height)
        if (not np.all(np.isfinite(dimensions)) or self.shell_height <= 0
                or not (0 < self.semi_b and 0 < self.semi_a
                        and 0 < self.shell_inner < self.shell_outer)):
            raise GeometryRejected("Invalid profile or shell dimensions")
        n, corners, _ = self.rings.shape
        if (self.faces.ndim != 2 or self.faces.shape[1] != 3 or not len(self.faces)
                or not np.issubdtype(self.faces.dtype, np.integer)):
            raise GeometryRejected('Expected triangular faces with integer indices')
        station = self.faces // corners
        if np.any(self.faces < 0) or np.any(self.faces >= n*corners):
            raise GeometryRejected("Face index outside the ring sweep")
        delta = (station[:, :, None] - station[:, None, :]) % n
        if np.any(~np.isin(delta, (0, 1, n-1))):
            raise GeometryRejected("Faces do not follow adjacent sweep stations")
        mesh = self.mesh()
        if not mesh.is_watertight:
            raise GeometryRejected("Reference sweep is not closed")

    def mesh(self):
        return trimesh.Trimesh(self.rings.reshape(-1, 3), self.faces.copy(), process=False)


@dataclass
class Lead:
    path: np.ndarray
    tangent: np.ndarray
    rings: np.ndarray
    min_radius: float
    metrics: dict


def radial(points, axis, origin):
    p = np.asarray(points) - origin
    return p - np.outer(p @ axis, axis)


def vector_areas(rings):
    q = rings - rings.mean(axis=1, keepdims=True)
    return np.linalg.norm(np.cross(q, np.roll(q, -1, axis=1)).sum(axis=1), axis=1)/2


def tube_mesh(rings):
    """Cap a sequence of corresponding planar convex sections."""
    nr, nc, _ = rings.shape
    a = (np.arange(nr-1)[:, None]*nc + np.arange(nc)).ravel()
    b = (np.arange(nr-1)[:, None]*nc + (np.arange(nc)+1) % nc).ravel()
    sides = np.vstack([np.column_stack([a, b, b+nc]),
                       np.column_stack([a, b+nc, a+nc])])
    verts = np.vstack([rings.reshape(-1, 3), rings[0].mean(0), rings[-1].mean(0)])
    j = np.arange(nc)
    caps = np.vstack([np.column_stack([np.full(nc, nr*nc), (j+1) % nc, j]),
                      np.column_stack([np.full(nc, nr*nc+1), (nr-1)*nc+j,
                                       (nr-1)*nc+(j+1) % nc])])
    mesh = trimesh.Trimesh(verts, np.vstack([sides, caps]), process=False)
    mesh.fix_normals()
    return mesh


def quintic(p0, p1, d0, d1, dd0=None):
    """Hermite polynomial with prescribed endpoint derivatives; final dd=0."""
    c = np.zeros((6, 2))
    c[0], c[1] = p0, d0
    c[2] = np.zeros(2) if dd0 is None else np.asarray(dd0)/2
    rhs = np.vstack([p1-c[:3].sum(0), d1-c[1]-2*c[2], -2*c[2]])
    c[3:] = np.linalg.solve(np.array([[1,1,1], [3,4,5], [6,12,20]]), rhs)
    return c


def eval_surface(c, t, R, e1, e2, axis, origin):
    uv = np.polynomial.polynomial.polyval(t, c).T
    d = np.polynomial.polynomial.polyval(t, np.arange(1,6)[:, None]*c[1:]).T
    dd = np.polynomial.polynomial.polyval(t, (np.arange(2,6)*np.arange(1,5))[:, None]*c[2:]).T
    th = uv[:, 0]/R
    n = np.cos(th)[:, None]*e1 + np.sin(th)[:, None]*e2
    circum = -np.sin(th)[:, None]*e1 + np.cos(th)[:, None]*e2
    p = origin + R*n + uv[:, 1, None]*axis
    dp = d[:, 0, None]*circum + d[:, 1, None]*axis
    ddp = dd[:, 0, None]*circum - (d[:, 0]**2/R)[:, None]*n + dd[:, 1, None]*axis
    speed = np.linalg.norm(dp, axis=1)
    if np.any(speed < 1e-8):
        raise GeometryRejected("Stationary point on lead")
    curvature = np.linalg.norm(np.cross(dp, ddp), axis=1)/speed**3
    return p, unit(dp), n, curvature


def surface_lead(ring, outward, axis, origin, exit_axial, angular_offset,
                 handle_scale, options, initial_curvature=0.):
    """Preserve the input polygon, transported with a radial section axis."""
    center = ring.mean(0)
    e1 = unit(radial(center[None], axis, origin)[0])
    e2 = unit(np.cross(axis, e1))
    R = np.linalg.norm(radial(center[None], axis, origin)[0])
    t0 = unit(outward - np.dot(outward, e1)*e1)
    junction_angle = float(np.degrees(np.arccos(np.clip(unit(outward) @ t0, -1, 1))))
    if junction_angle > 1.0:
        raise GeometryRejected("Junction tangent differs by more than one degree")
    start = np.array([0., np.dot(center-origin, axis)])
    end = np.array([angular_offset*R, exit_axial])
    distance = np.linalg.norm(end-start)
    speed = distance*handle_scale
    d0 = np.array([np.dot(t0, e2), np.dot(t0, axis)])*speed
    d1 = np.array([0., np.sign(exit_axial-start[1])*distance])
    dd0 = initial_curvature*speed**2*np.array([-d0[1], d0[0]])/speed
    coeff = quintic(start, end, d0, d1, dd0)
    dense_t = np.linspace(0, 1, max(513, int(distance/options.max_step)*5))
    dense, _, _, curvature = eval_surface(coeff, dense_t, R, e1, e2, axis, origin)
    min_radius = 1/max(float(curvature.max()), 1e-12)
    if min_radius < options.min_bend_radius:
        raise GeometryRejected("Bend radius below minimum")
    arclen = np.r_[0., np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    sample_s = np.linspace(0, arclen[-1], int(np.ceil(arclen[-1]/options.max_step))+1)
    sample_t = np.interp(sample_s, arclen, dense_t)
    path, T, N, _ = eval_surface(coeff, sample_t, R, e1, e2, axis, origin)
    B = unit(np.cross(T, N))
    offsets = ring-center
    coords = np.column_stack([offsets @ N[0], offsets @ B[0]])
    residual = float(np.max(np.abs(offsets @ T[0])))
    if residual > 2e-5:
        raise GeometryRejected("Input ring not normal to the outward tangent")
    rings = path[:, None] + coords[None, :, 0, None]*N[:, None] + coords[None, :, 1, None]*B[:, None]
    rings[0] = ring  # Exact shared boundary; the residual is independently bounded.
    area = vector_areas(rings)
    error = float(np.max(np.abs(area/area[0]-1)))
    if error > options.section_tolerance or area.min() <= 0:
        raise GeometryRejected("Section area changed")
    metrics = {
        "length_mm": float(arclen[-1]*1000), "min_bend_radius_mm": min_radius*1000,
        "section_area_min_mm2": float(area.min()*1e6),
        "section_area_max_mm2": float(area.max()*1e6),
        "section_area_relative_error": error,
        "height_in_frame_mm": float(np.ptp(coords[:, 0])*1000),
        "width_in_frame_mm": float(np.ptp(coords[:, 1])*1000),
        "initial_planarity_residual_mm": residual*1000,
        "junction_tangent_error_deg": junction_angle,
        "max_step_mm": float(np.linalg.norm(np.diff(path, axis=0), axis=1).max()*1000),
        "exit_tangent_error_deg": float(np.degrees(np.arccos(np.clip(abs(T[-1] @ axis), -1, 1)))),
    }
    return Lead(path, T, rings, min_radius, metrics)


def segment_closest(a, b, c, d):
    """Exact pair distances and vectors from the closest point on A to B."""
    u, v, w = b-a, d-c, a-c
    A, B, C = (u*u).sum(1), (u*v).sum(1), (v*v).sum(1)
    D, E = (u*w).sum(1), (v*w).sum(1)
    denom = A*C-B*B
    safe = denom > 1e-24
    s = np.divide(B*E-C*D, denom, out=np.zeros_like(A), where=safe)
    t = np.divide(A*E-B*D, denom, out=np.zeros_like(A), where=safe)
    interior = safe & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
    distances = np.full(len(a), np.inf)
    vectors = -w-s[:,None]*u+t[:,None]*v
    distances[interior] = np.linalg.norm(vectors[interior], axis=1)
    for point, start, edge, norm2, sign in ((a,c,v,C,1), (b,c,v,C,1), (c,a,u,A,-1), (d,a,u,A,-1)):
        f = np.clip(((point-start)*edge).sum(1)/np.maximum(norm2, 1e-30), 0, 1)
        candidate = sign*(start+f[:,None]*edge-point)
        new_distance = np.linalg.norm(candidate,axis=1)
        better = new_distance < distances
        vectors[better] = candidate[better]
        distances[better] = new_distance[better]
    return distances, vectors


def segment_distances(a, b, c, d):
    return segment_closest(a,b,c,d)[0]


def near_segment_pairs(path_a, path_b, threshold):
    a, b = path_a[:-1], path_a[1:]
    c, d = path_b[:-1], path_b[1:]
    if not len(a) or not len(c):
        return np.empty(0, int), np.empty(0, int), np.empty(0)
    tree = cKDTree((c+d)/2)
    radius = threshold + np.linalg.norm(b-a, axis=1)/2 + np.linalg.norm(d-c, axis=1).max()/2
    lists = tree.query_ball_point((a+b)/2, radius)
    ia = np.repeat(np.arange(len(a)), [len(x) for x in lists])
    ib = np.array([j for group in lists for j in group], dtype=int)
    dist = segment_distances(a[ia], b[ia], c[ib], d[ib])
    return ia, ib, dist


def support_gaps(path_a, rings_a, path_b, rings_b, i, j):
    """Separating-plane gap of the actual two swept polygon segments.

    The convex hull of two rings contains every triangle between those rings.
    A positive gap on any separating axis therefore bounds the mesh clearance.
    Failure to separate is conservative (it need not mean an intersection).
    """
    if not len(i):
        return np.empty(0)
    distances, vectors = segment_closest(path_a[i],path_a[i+1],path_b[j],path_b[j+1])
    direction = vectors / np.maximum(distances[:,None],1e-30)
    # Subtract a shared origin to reduce cancellation after large translations.
    base = path_a[i,None,:]
    a = np.concatenate([rings_a[i]-base,rings_a[i+1]-base],axis=1)
    b = np.concatenate([rings_b[j]-base,rings_b[j+1]-base],axis=1)
    gap = (b*direction[:,None]).sum(2).min(1) - (a*direction[:,None]).sum(2).max(1)
    gap[distances < 1e-12] = -np.inf
    return gap


def clearance_check(leads, coil_centers, envelope_radius, clearance, coil_rings=None):
    """Conservative swept-sphere bound using segment distances, not samples.

    Adjacent segments within 4 envelope radii in arc length belong to the
    intended junction/local sweep; nonlocal pairs are never excluded.
    """
    threshold = 2*envelope_radius+clearance
    i, j, pair = near_segment_pairs(leads[0].path, leads[1].path, threshold)
    gap = support_gaps(leads[0].path,leads[0].rings,leads[1].path,leads[1].rings,i,j)
    if len(pair) and gap.min() < clearance:
        raise GeometryRejected("Lead-lead clearance")
    lower_bounds = []
    coil_s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(coil_centers, axis=0), axis=1))]
    for k, lead in enumerate(leads):
        # coil_centers starts at lead 1's junction and ends at lead 0's.
        ls = np.r_[0., np.cumsum(np.linalg.norm(np.diff(lead.path, axis=0), axis=1))]
        i, j, dist = near_segment_pairs(lead.path, coil_centers, threshold)
        distance_from_join = coil_s[j] if k == 1 else coil_s[-1]-coil_s[j+1]
        local = (ls[i] + distance_from_join) <= 4*envelope_radius
        coil_gap = (dist-2*envelope_radius if coil_rings is None else
                    support_gaps(lead.path,lead.rings,coil_centers,coil_rings,i,j))
        if np.any(coil_gap[~local] < clearance):
            raise GeometryRejected("Lead-coil clearance")
        i, j, dist_self = near_segment_pairs(lead.path, lead.path, threshold)
        nonlocal_pair = np.abs(ls[i]-ls[j]) > 4*envelope_radius
        self_gap = support_gaps(lead.path,lead.rings,lead.path,lead.rings,i[nonlocal_pair],j[nonlocal_pair])
        if np.any(self_gap < clearance):
            raise GeometryRejected("Lead self-clearance")
        lower_bounds.append(min(clearance, float(coil_gap[~local].min()) if np.any(~local) else clearance)*1000)
    return {"nonlocal_surface_clearance_lower_bound_mm": lower_bounds,
            "method": "Separating planes of actual swept polygons after a bounding-sphere broad phase",
            "envelope_radius_mm": envelope_radius*1000,
            "required_surface_clearance_mm": clearance*1000,
            "local_arc_exclusion_mm": 4*envelope_radius*1000}


def cut_indices(ref, station, length):
    centers = ref.rings.mean(1)
    circuit_length = np.linalg.norm(np.roll(centers, -1, axis=0)-centers, axis=1).sum()
    if not np.isfinite(length) or length <= 0 or length >= circuit_length:
        raise GeometryRejected('Cut exceeds circuit length or has invalid length')
    n = len(centers)
    result = []
    for sign in (-1, 1):
        i, travelled = station, 0.
        while travelled < length/2:
            j = (i+sign) % n
            travelled += np.linalg.norm(centers[j]-centers[i])
            i = j
            if i == station:
                raise GeometryRejected("Cut exceeds circuit length")
        result.append(i)
    return result


def open_reference(ref, left, right):
    n, nc, _ = ref.rings.shape
    order = np.arange(right, right+(left-right) % n+1) % n
    if len(order) < 5:
        raise GeometryRejected("Cut removes too much of the circuit")
    ids = (order[:, None]*nc + np.arange(nc)).ravel()
    remap = np.full(n*nc, -1, dtype=int)
    remap[ids] = np.arange(len(ids))
    faces = remap[ref.faces]
    faces = faces[(faces >= 0).all(1)]
    mesh = trimesh.Trimesh(ref.rings[order].reshape(-1,3), faces, process=False)
    return mesh, order


def attach(open_mesh, leads):
    nc = leads[0].rings.shape[1]
    vertices = [np.asarray(open_mesh.vertices)]
    faces = [np.asarray(open_mesh.faces)]
    offset = len(open_mesh.vertices)
    for k, lead in enumerate(leads):
        body = tube_mesh(lead.rings)
        nbody = len(lead.rings)*nc
        # Drop the start cap; reuse the actual coil boundary vertex indices.
        selected = body.faces[~(body.faces == nbody).any(1)]
        new_vertices = np.vstack([lead.rings[1:].reshape(-1,3), lead.rings[-1].mean(0)])
        mapping = np.empty(nbody+2, dtype=int)
        mapping[:nc] = np.arange(len(open_mesh.vertices)-nc, len(open_mesh.vertices)) if k == 0 else np.arange(nc)
        mapping[nc:nbody] = offset+np.arange(nbody-nc)
        mapping[nbody] = -1
        mapping[nbody+1] = offset+len(new_vertices)-1
        faces.append(mapping[selected])
        vertices.append(new_vertices)
        offset += len(new_vertices)
    result = trimesh.Trimesh(np.vstack(vertices), np.vstack(faces), process=False)
    result.fix_normals()
    return result


def manifold(mesh):
    """Perform booleans in mm, preserving precision for small grooves."""
    result = m3d.Manifold(m3d.Mesh64(
        vert_properties=np.asarray(mesh.vertices*1000, dtype=np.float64),
        tri_verts=np.asarray(mesh.faces, dtype=np.uint64)))
    if result.status() != m3d.Error.NoError or result.is_empty():
        raise GeometryRejected(f"Boolean input rejected: {result.status()}")
    return result


def mesh_metrics(mesh):
    counts = np.bincount(mesh.edges_unique_inverse)
    areas = mesh.area_faces
    return {"vertices": len(mesh.vertices), "faces": len(mesh.faces),
            "watertight": bool(mesh.is_watertight),
            "winding_consistent": bool(mesh.is_winding_consistent),
            "boundary_edges": int((counts == 1).sum()),
            "nonmanifold_edges": int((counts > 2).sum()),
            "degenerate_faces": int((areas < 1e-20).sum()),
            "small_faces_below_1e_minus_8_mm2": int((areas < 1e-14).sum()),
            "volume_mm3": float(mesh.volume*1e9)}


def validate_mesh(mesh):
    metrics = mesh_metrics(mesh)
    if (not metrics["watertight"] or not metrics["winding_consistent"]
            or metrics["degenerate_faces"] or metrics["volume_mm3"] <= 0):
        raise GeometryRejected(f"Invalid mesh: {metrics}")
    manifold(mesh)
    return metrics


def plan(ref, sector_ref, exit_sign=-1, options=None):
    options = options or Options()
    options.validate()
    ref.validate()
    if exit_sign not in (-1, 1):
        raise GeometryRejected('Exit sign must be -1 or 1')
    centers = ref.rings.mean(1)
    radial_vectors = radial(centers, ref.axis, ref.origin)
    radii = np.linalg.norm(radial_vectors, axis=1)
    sector = unit(np.asarray(sector_ref)-np.dot(sector_ref, ref.axis)*ref.axis)
    eligible = np.where(unit(radial_vectors) @ sector > np.cos(options.sector_angular_half))[0]
    if not len(eligible):
        raise GeometryRejected("No stations in requested exit sector")
    axial = (centers-ref.origin) @ ref.axis
    ranked = eligible[np.argsort(-exit_sign*axial[eligible])]
    # Spatially separate candidate stations to avoid trying near duplicates.
    chosen = []
    for i in ranked:
        if all(np.linalg.norm(centers[i]-centers[j]) > .002 for j in chosen):
            chosen.append(int(i))
        if len(chosen) >= options.candidates:
            break
    rejected = {}
    envelope = float(np.linalg.norm(ref.rings-centers[:,None], axis=2).max())
    exit_axial = exit_sign*(ref.shell_height/2+options.exit_margin)
    for station in chosen:
        for gap in options.cut_lengths:
            try:
                left, right = cut_indices(ref, station, gap)
                opened, order = open_reference(ref, left, right)
            except GeometryRejected as exc:
                key = str(exc).split(':')[0]
                rejected[key] = rejected.get(key, 0) + 1
                continue
            end_ids = (left, right)
            # A constant-radius route must not flatten a layer transition.
            if any(np.ptp(radii[(np.arange(i-2,i+3)) % len(radii)]) > options.radius_tolerance for i in end_ids):
                rejected["Layer transition at cut"] = rejected.get("Layer transition at cut",0)+1
                continue
            anchor = unit(radial_vectors[station])
            side = unit(np.cross(ref.axis, anchor))
            angles = np.arctan2(radial_vectors[list(end_ids)] @ side,
                                radial_vectors[list(end_ids)] @ anchor)
            order_sign = np.sign(angles[0]-angles[1]) or 1
            for separation in options.tip_separations:
                target_angles = np.array([order_sign, -order_sign])*separation/(2*radii[station])
                offsets = np.arctan2(np.sin(target_angles-angles), np.cos(target_angles-angles))
                # The feasible curvature window can be narrow for tall cable
                # profiles. Keep an intermediate handle between 1.0 and 1.4;
                # both shorter and longer transitions can exceed the same
                # minimum bend-radius constraint.
                for handle in (.65, 1., 1.2, 1.4, 2.):
                    try:
                        leads = []
                        for k, idx in enumerate(end_ids):
                            outward = centers[(idx+1) % len(centers)]-centers[(idx-1) % len(centers)]
                            outward *= 1 if k == 0 else -1
                            # Derive the precise plane normal from the ordered ring.
                            _, _, vt = np.linalg.svd(ref.rings[idx]-centers[idx])
                            plane_t = vt[-1]
                            if plane_t @ outward < 0:
                                plane_t = -plane_t
                            lead = surface_lead(ref.rings[idx], plane_t, ref.axis, ref.origin,
                                                exit_axial, offsets[k], handle, options)
                            leads.append(lead)
                        clear = clearance_check(leads, centers[order], envelope, options.clearance, ref.rings[order])
                        result = attach(opened, leads)
                        metrics = validate_mesh(result)
                        for lead in leads:
                            validate_mesh(tube_mesh(lead.rings))
                        # Exact lead-to-lead solid intersection complements bounds.
                        lead_solids = [manifold(tube_mesh(x.rings)) for x in leads]
                        overlap = lead_solids[0] ^ lead_solids[1]
                        if overlap.volume() > 1e-6:
                            raise GeometryRejected("Lead solid intersection")
                        retained_solid = manifold(tube_mesh(ref.rings[order]))
                        junction_volumes = [float((retained_solid ^ solid).volume())
                                            for solid in lead_solids]
                        if any(abs(v) > 1e-6 for v in junction_volumes):
                            raise GeometryRejected("Lead-coil solid intersection at junction")
                        removed = np.arange(left,left+(right-left) % len(centers)+1) % len(centers)
                        removed_arc = float(np.linalg.norm(np.diff(centers[removed],axis=0),axis=1).sum()*1000)
                        report = {"status": "validated", "options": asdict(options),
                                  "cut_station": station, "cut_indices": [int(left),int(right)],
                                  "cut_requested_mm": gap*1000, "tip_separation_requested_mm": separation*1000,
                                  "cut_removed_arc_mm": removed_arc,
                                  "handle_scale": handle, "mesh": metrics,
                                  "leads": [x.metrics for x in leads], "clearance": clear,
                                  "retained_vertices_exact": bool(np.array_equal(result.vertices[:len(opened.vertices)], opened.vertices)),
                                  "lead_coil_intersection_volumes_mm3": junction_volumes,
                                  "rejected_candidates": rejected,
                                  "scope": "Lead geometry only; pre-existing coil crossings are retained; no electromagnetic certification."}
                        return result, leads, opened, report
                    except GeometryRejected as exc:
                        key = str(exc).split(":")[0]
                        rejected[key] = rejected.get(key,0)+1
    raise GeometryRejected("No feasible lead pair: " + json.dumps(rejected))
