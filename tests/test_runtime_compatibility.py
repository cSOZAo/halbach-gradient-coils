"""Smoke tests for the supported scientific-Python runtime stack."""

import numpy as np
import trimesh

from halbach_coils.coilgen.shell import manifold_from_trimesh, trimesh_from_manifold
from pyCoilGen.sub_functions.data_structures import Mesh


def test_manifold_round_trip_preserves_box_geometry():
    source = trimesh.creation.box(extents=(1.0, 2.0, 3.0))

    restored = trimesh_from_manifold(manifold_from_trimesh(source))

    assert restored.is_watertight
    assert np.isclose(abs(restored.volume), 6.0)


def test_mesh_cleanup_uses_current_trimesh_face_api():
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    mesh = Mesh(vertices=vertices, faces=np.array([[0, 1, 2], [0, 1, 2]]))

    mesh.cleanup()

    assert len(mesh.get_faces()) == 1
    assert len(mesh.get_vertices()) == 3
