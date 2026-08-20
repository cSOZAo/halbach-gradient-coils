from pathlib import Path
from types import SimpleNamespace

from halbach_coils.coilgen import paths
from pyCoilGen.export_factory.export_cad_file import export_CAD_file


STEM = 'Gradient_Gx_tk2500_lvl12'


def test_wire_name_does_not_repeat_internal_axis():
    assert paths.wire_stl_name(STEM, 'x') == f'{STEM}_wire_0.stl'
    assert paths.WIRE_CAD_FILENAME == '{project}_wire_{part_index}.stl'


def test_physical_axis_is_inferred_from_generated_and_derived_names():
    assert paths.infer_gradient_axis(
        r'C:\results\Gradient_Gx_tk2500_lvl12_wire_0.stl') == 'x'
    assert paths.infer_gradient_axis(
        '/results/Gradient_GZ_tk10_lvl5_wire_0_with_leads.stl') == 'z'
    assert paths.infer_gradient_axis('/results/my_wire.stl') is None


def test_resolve_wire_prefers_canonical_name(tmp_path: Path):
    canonical = tmp_path / f'{STEM}_wire_0.stl'
    legacy = tmp_path / f'{STEM}_wire_0_y.stl'
    canonical.touch()
    legacy.touch()

    resolved = paths.resolve_wire_stl_path(str(tmp_path), 'x', 2500, 12)

    assert resolved == str(canonical)


def test_resolve_wire_accepts_legacy_internal_axis_name(tmp_path: Path):
    legacy = tmp_path / f'{STEM}_wire_0_y.stl'
    legacy.touch()

    resolved = paths.resolve_wire_stl_path(str(tmp_path), 'x', 2500, 12)

    assert resolved == str(legacy)


def test_unique_stem_treats_canonical_wire_as_occupied(tmp_path: Path):
    (tmp_path / f'{STEM}_wire_0.stl').touch()

    assert paths.unique_stem(str(tmp_path), STEM, gradient_axis='x') == f'{STEM}(2)'


def test_pycoilgen_export_uses_canonical_filename(tmp_path: Path):
    exported = []
    wire_mesh = SimpleNamespace(
        export=lambda filename, file_type=None: exported.append(filename))
    solution = SimpleNamespace(
        input_args=SimpleNamespace(
            save_stl_flag=True,
            CAD_filename=paths.WIRE_CAD_FILENAME,
            output_directory=str(tmp_path),
            project_name=STEM,
            field_shape_function='y',
        ),
        coil_parts=[SimpleNamespace(
            layout_surface_mesh=wire_mesh,
            coil_mesh=SimpleNamespace(),
        )],
    )

    export_CAD_file(solution)

    assert exported == [str(tmp_path / f'{STEM}_wire_0.stl')]
