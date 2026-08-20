"""Detection of multi-wire congestion at pyCoilGen return-path crossings.

Ordinary pyCoilGen crossings contain two branches and are resolved by moving
one branch to the inner radial layer. The problematic case is a crossing whose
two branches receive the same radial-layer displacement. In the known
three-wire failure mode, one cable remains outside while the two inside form
the colliding X.

The old detector compared every pair of nearby 3-D segments. It consequently
reported thousands of false collisions between neighbouring samples of the
same continuous curve. This module instead consumes the exact 2-D crossing
metadata captured by :func:`pyCoilGen.sub_functions.shift_return_paths` and
checks pyCoilGen's actual layer assignment before grouping collision sites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from .config import Config


@dataclass(frozen=True)
class CollisionSite:
    """One surface location occupied by three or more wire-path branches."""

    part_index: int
    location_uv: Tuple[float, float]
    cable_count: int
    crossing_count: int
    segment_indices: Tuple[int, ...]

    @property
    def inner_layer_count(self) -> int:
        return self.cable_count - 1


@dataclass
class OverlapReport:
    """Multi-wire congestion sites found in a completed pyCoilGen solution."""

    sites: List[CollisionSite] = field(default_factory=list)
    cluster_tolerance_m: float = 0.0

    @property
    def n_collisions(self) -> int:
        """Compatibility name: now counts congestion sites, not segment pairs."""
        return len(self.sites)

    @property
    def max_cables(self) -> int:
        return max((site.cable_count for site in self.sites), default=0)

    def user_question(self) -> str:
        """Spanish warning shown between gradient generation and leads/shell."""
        if not self.sites:
            return ""
        if len(self.sites) == 1:
            site = self.sites[0]
            description = (
                f"Hay {site.cable_count} cables intentando pasar por el mismo lugar; "
                f"entonces 1 quedará en la capa externa y "
                f"{site.inner_layer_count} quedarán en la interna."
            )
        else:
            lines = [
                f"Se detectaron {len(self.sites)} lugares con tres o más cables:"
            ]
            for index, site in enumerate(self.sites, start=1):
                lines.append(
                    f"• Lugar {index}: hay {site.cable_count} cables; 1 quedará "
                    f"en la capa externa y {site.inner_layer_count} quedarán en la interna."
                )
            description = "\n".join(lines)
        return (
            f"{description}\n\n"
            "¿Desea continuar o descartar el resultado?\n\n"
            "Sí: continuar con leads y shell.\n"
            "No: descartar este resultado y detener el pipeline."
        )


@dataclass(frozen=True)
class _CrossingRecord:
    point_uv: np.ndarray
    segment_a: int
    segment_b: int
    path_position_a: float
    path_position_b: float
    layer_factor_a: float
    layer_factor_b: float


def _metadata_for_part(part) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read stored crossing metadata, with a fallback for older solutions."""
    points = getattr(part, 'crossing_points_uv', None)
    segments = getattr(part, 'crossing_segments', None)
    positions = getattr(part, 'crossing_path_positions', None)
    layers = getattr(part, 'crossing_layer_factors', None)
    if points is not None and segments is not None and positions is not None:
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        segments = np.asarray(segments, dtype=int).reshape(-1, 2)
        positions = np.asarray(positions, dtype=float).reshape(-1, 2)
        if layers is not None:
            layers = np.asarray(layers, dtype=float).reshape(-1, 2)
        else:
            layers = _layer_factors_from_part(part, points, segments)
        return points, segments, positions, layers

    # Compatibility path for solutions produced before crossing metadata was
    # persisted. New GUI runs never need this comparatively expensive pass.
    wire_path = getattr(part, 'wire_path', None)
    if wire_path is None or wire_path.uv is None or wire_path.v is None:
        empty_f = np.empty((0, 2), dtype=float)
        return empty_f, np.empty((0, 2), dtype=int), empty_f, empty_f

    from pyCoilGen.sub_functions.shift_return_paths import (
        InterX, intersection_points_for_pairs,
    )

    _unused_points, segments = InterX(np.asarray(wire_path.uv, dtype=float))
    segments = np.asarray(segments, dtype=int)
    if segments.size == 0:
        empty_f = np.empty((0, 2), dtype=float)
        return empty_f, np.empty((0, 2), dtype=int), empty_f, empty_f
    segments = segments.reshape(-1, 2)
    points = intersection_points_for_pairs(wire_path.uv, segments)
    lengths = np.linalg.norm(np.diff(wire_path.v, axis=1), axis=0)
    point_positions = np.concatenate(([0.0], np.cumsum(lengths)))
    segment_positions = 0.5 * (point_positions[:-1] + point_positions[1:])
    layers = _layer_factors_from_part(part, points, segments)
    return points, segments, segment_positions[segments], layers


def _layer_factors_from_part(part, points: np.ndarray,
                             segments: np.ndarray) -> np.ndarray:
    """Recover layer factors for solutions saved before they were persisted."""
    shift_array = getattr(part, 'shift_array', None)
    wire_path = getattr(part, 'wire_path', None)
    if shift_array is None or wire_path is None or wire_path.uv is None:
        # Unknown is deliberately treated as separated. Guessing from crossing
        # proximity was the source of the false positives this detector replaces.
        return np.column_stack((np.zeros(len(points)), np.ones(len(points))))

    from pyCoilGen.sub_functions.shift_return_paths import layer_factors_for_crossings

    shifts = np.asarray(shift_array, dtype=float)
    uv = np.asarray(wire_path.uv, dtype=float)
    if (not len(segments)
            or np.max(segments) + 1 >= uv.shape[1]
            or np.max(segments) + 1 >= len(shifts)):
        return np.column_stack((np.zeros(len(points)), np.ones(len(points))))
    return layer_factors_for_crossings(uv, segments, points, shifts)


def _cluster_records(records: list[_CrossingRecord], tolerance: float) -> list[list[int]]:
    """Connected components of crossing points separated by at most tolerance."""
    count = len(records)
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left in range(count):
        for right in range(left + 1, count):
            if np.linalg.norm(records[left].point_uv - records[right].point_uv) <= tolerance:
                union(left, right)

    components: dict[int, list[int]] = {}
    for index in range(count):
        components.setdefault(find(index), []).append(index)
    return list(components.values())


def _count_path_branches(path_positions: list[float], merge_distance: float) -> int:
    """Merge repeated intersections belonging to the same nearby path branch."""
    if not path_positions:
        return 0
    ordered = sorted(float(value) for value in path_positions)
    branches = 1
    previous = ordered[0]
    for position in ordered[1:]:
        if position - previous > merge_distance:
            branches += 1
        previous = position
    return branches


def detect_collisions(solution, cfg: Config) -> OverlapReport:
    """Return crossings whose branches are physically assigned to the same layer."""
    # A layer transition occupies space on both sides of its exact geometric
    # crossing. Treat crossings within 2.5 cable heights/widths as the same
    # congestion zone; this catches neighbouring crossings whose smoothed
    # ramps overlap while continuing to scale with the selected conductor.
    cable_footprint = max(
        float(cfg.cable_height),
        float(2.0 * cfg.conductor_semi_b),
        float(cfg.winding.cut_width),
    )
    tolerance = 2.5 * max(float(cfg.overlap_clearance), 0.0) * cable_footprint
    merge_distance = 1.5 * tolerance
    sites: list[CollisionSite] = []
    layer_shift = abs(float(cfg.normal_shift_length))
    required_separation = float(cfg.cable_height)

    for part_index, part in enumerate(getattr(solution, 'coil_parts', [])):
        points, segments, positions, layers = _metadata_for_part(part)
        records = [
            _CrossingRecord(points[index], int(segments[index, 0]),
                            int(segments[index, 1]), float(positions[index, 0]),
                            float(positions[index, 1]), float(layers[index, 0]),
                            float(layers[index, 1]))
            for index in range(len(points))
            if np.all(np.isfinite(points[index]))
            and np.all(np.isfinite(layers[index]))
            # Different layer factors create radial clearance. Equal factors
            # identify the actual X left by pyCoilGen, regardless of how many
            # harmless projected crossings happen to be nearby.
            and (abs(float(layers[index, 0]) - float(layers[index, 1]))
                 * layer_shift + 1e-12 < required_separation)
        ]
        if not records:
            continue

        for component in _cluster_records(records, tolerance):
            component_records = [records[index] for index in component]
            path_positions: list[float] = []
            segment_indices: set[int] = set()
            for record in component_records:
                path_positions.extend((record.path_position_a, record.path_position_b))
                segment_indices.update((record.segment_a, record.segment_b))
            colliding_inner_branches = max(
                2, _count_path_branches(path_positions, merge_distance))
            # The same-layer X is formed by the n-1 branches that pyCoilGen
            # sent inward after leaving one competing cable on the outer layer.
            cable_count = colliding_inner_branches + 1
            location = np.mean([record.point_uv for record in component_records], axis=0)
            sites.append(CollisionSite(
                part_index=part_index,
                location_uv=(float(location[0]), float(location[1])),
                cable_count=cable_count,
                crossing_count=len(component_records),
                segment_indices=tuple(sorted(segment_indices)),
            ))

    sites.sort(key=lambda site: (-site.cable_count, site.part_index, site.location_uv))
    return OverlapReport(sites=sites, cluster_tolerance_m=tolerance)
