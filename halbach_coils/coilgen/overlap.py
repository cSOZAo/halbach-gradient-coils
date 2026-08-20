"""
Wire overlap / collision detection.

After pyCoilGen produces a wire layout (before leads are attached), this
module checks whether non-adjacent wire segments come closer than the
conductor width (i.e. the winding would short or overlap when swept into a
solid). The GUI surfaces the result as an accept/reject prompt.

Strategy
--------
1. Collect every segment (start, end) from all coil parts' ``wire_path.v``.
2. Build a KD-tree on segment midpoints to prefilter candidate pairs within
   ``2 * clearance_radius`` of each other (clearance_radius is a multiple of
   the conductor width).
3. For each candidate pair that is NOT adjacent (segments sharing a vertex),
   compute the exact 3D segment-segment distance.
4. Report pairs below the threshold plus the global minimum distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .config import Config


@dataclass
class OverlapReport:
    n_collisions: int
    min_distance_m: float
    threshold_m: float
    pairs: List[Tuple[int, int, float]]   # (seg_i, seg_j, distance)


def _collect_segments(solution) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """
    Return ``(starts, ends, adjacency)``.

    ``starts`` / ``ends`` are (M, 3) arrays of segment endpoints. ``adjacency``
    is a list of (i, j) index pairs whose segments share a vertex within the
    same part and must be excluded from the overlap check.
    """
    starts_list = []
    ends_list = []
    adjacency: List[Tuple[int, int]] = []

    seg_offset = 0
    for part in solution.coil_parts:
        wp = getattr(part, 'wire_path', None)
        if wp is None or wp.v is None:
            continue
        v = np.asarray(wp.v, dtype=np.float64)        # (3, N)
        if v.shape[1] < 2:
            continue
        seg_start = v[:, :-1].T                        # (N-1, 3)
        seg_end = v[:, 1:].T                           # (N-1, 3)
        n_seg = seg_start.shape[0]

        starts_list.append(seg_start)
        ends_list.append(seg_end)

        # Adjacent segments within this part share a vertex: (k, k+1).
        for k in range(n_seg - 1):
            adjacency.append((seg_offset + k, seg_offset + k + 1))
        seg_offset += n_seg

    if not starts_list:
        return (np.zeros((0, 3)), np.zeros((0, 3)), adjacency)

    starts = np.vstack(starts_list)
    ends = np.vstack(ends_list)
    return starts, ends, adjacency


def _segment_segment_distance(p0: np.ndarray, p1: np.ndarray,
                              q0: np.ndarray, q1: np.ndarray) -> float:
    """Shortest distance between segments [p0,p1] and [q0,q1]."""
    d1 = p1 - p0
    d2 = q1 - q0
    r = p0 - q0
    a = float(d1 @ d1)
    e = float(d2 @ d2)
    f = float(d2 @ r)
    eps = 1e-18

    if a <= eps and e <= eps:
        return float(np.linalg.norm(p0 - q0))
    if a <= eps:
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = float(d1 @ r)
        if e <= eps:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            b = float(d1 @ d2)
            denom = a * e - b * b
            if denom != 0.0:
                s = np.clip((b * f - c * e) / denom, 0.0, 1.0)
            else:
                s = 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)

    closest_p = p0 + s * d1
    closest_q = q0 + t * d2
    return float(np.linalg.norm(closest_p - closest_q))


def detect_collisions(solution, cfg: Config) -> OverlapReport:
    """
    Detect non-adjacent wire segments closer than the clearance threshold.

    Threshold = ``cfg.overlap_clearance * cfg.wire.conductor_width``.
    """
    threshold = cfg.overlap_clearance * cfg.wire.conductor_width

    starts, ends, adjacency = _collect_segments(solution)
    if starts.shape[0] < 2:
        return OverlapReport(0, float('inf'), threshold, [])

    adj_set = set(adjacency)
    adj_set |= {(j, i) for (i, j) in adjacency}

    midpoints = 0.5 * (starts + ends)
    # Prefilter radius: two segments can be close only if their midpoints are
    # within ~2x (segment_half_len + threshold). Use a generous 3x threshold
    # plus the longest segment half-length as the query radius.
    seg_lens = np.linalg.norm(ends - starts, axis=1)
    max_half = float(seg_lens.max()) if seg_lens.size else 0.0
    query_radius = max(2.0 * threshold + max_half, 4.0 * threshold)

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None

    pairs: List[Tuple[int, int, float]] = []
    min_dist = float('inf')

    if cKDTree is not None:
        tree = cKDTree(midpoints)
        candidate_pairs = tree.query_pairs(query_radius, output_type='ndarray')
        for i, j in candidate_pairs:
            if (int(i), int(j)) in adj_set:
                continue
            d = _segment_segment_distance(
                starts[i], ends[i], starts[j], ends[j],
            )
            if d < min_dist:
                min_dist = d
            if d < threshold:
                pairs.append((int(i), int(j), d))
    else:
        # O(n^2) fallback (no scipy). Subsample to keep it bounded.
        n = starts.shape[0]
        step = max(1, n // 2000)
        idx = np.arange(0, n, step)
        for ii in range(len(idx)):
            for jj in range(ii + 1, len(idx)):
                i, j = int(idx[ii]), int(idx[jj])
                if (i, j) in adj_set:
                    continue
                d = _segment_segment_distance(
                    starts[i], ends[i], starts[j], ends[j],
                )
                if d < min_dist:
                    min_dist = d
                if d < threshold:
                    pairs.append((i, j, d))

    pairs.sort(key=lambda p: p[2])
    return OverlapReport(
        n_collisions=len(pairs),
        min_distance_m=min_dist if np.isfinite(min_dist) else float('inf'),
        threshold_m=threshold,
        pairs=pairs,
    )
