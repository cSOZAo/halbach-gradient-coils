import numpy as np
import pytest
import trimesh

from halbach_coils.coilgen.shell import export_validated_shell_stl


def test_float32_collapsed_edge_exports_as_a_closed_shell(tmp_path):
    base = trimesh.creation.box(extents=[100, 100, 100])
    a, b = base.faces[0][:2]
    extra = len(base.vertices)
    vertices = np.vstack([base.vertices, base.vertices[a] +
                          (base.vertices[b] - base.vertices[a]) * 1e-10])
    faces = []
    for face in base.faces:
        if a in face and b in face:
            for index in range(3):
                rotated = np.roll(face, -index)
                if set(rotated[:2]) == {a, b}:
                    x, y, z = rotated
                    faces.extend([[x, extra, z], [extra, y, z]])
                    break
        else:
            faces.append(face)
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    assert mesh.is_watertight
    target = tmp_path / 'shell.stl'
    exported = export_validated_shell_stl(mesh, target)
    reloaded = trimesh.load_mesh(target)
    assert reloaded.is_watertight and reloaded.is_winding_consistent
    assert len(reloaded.faces) == len(base.faces) < len(mesh.faces)
    assert np.all(reloaded.nondegenerate_faces())
    assert reloaded.volume == pytest.approx(base.volume)
    assert exported.is_watertight


def test_an_open_shell_is_not_published(tmp_path):
    mesh = trimesh.creation.box()
    mesh.update_faces(np.arange(len(mesh.faces)) != 0)
    target = tmp_path / 'invalid.stl'
    with pytest.raises(RuntimeError, match='failed validation'):
        export_validated_shell_stl(mesh, target)
    assert not target.exists()
