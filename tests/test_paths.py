"""Unit tests for :mod:`coilgen.paths`."""

import os

import pytest

from coilgen import paths


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('x')
    return path


def _set_mtime(path, when):
    os.utime(path, (when, when))


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------

def test_output_bases_live_under_the_project_root():
    assert paths.STANDALONE_OUTPUT_BASE == os.path.join(
        paths.PROJECT_ROOT, 'resultados', 'standalone')
    assert paths.PIPELINE_OUTPUT_BASE == os.path.join(
        paths.PROJECT_ROOT, 'resultados', 'pipeline')


def test_design_folder_name_truncates_tikhonov_to_int():
    assert paths.design_folder_name('y', 2500.0, 26) == 'Gy_tk2500_lvl26'
    assert paths.design_folder_name('z', 10000.7, 30) == 'Gz_tk10000_lvl30'


def test_gradient_project_stem():
    assert paths.gradient_project_stem('y', 2500, 26) == 'Gradient_Gy_tk2500_lvl26'


def test_standalone_design_dir():
    expected = os.path.join(paths.STANDALONE_OUTPUT_BASE, 'Gy_tk2500_lvl26')
    assert paths.standalone_design_dir('y', 2500, 26) == expected


@pytest.mark.parametrize('axis,internal', [('x', 'y'), ('y', 'z'), ('z', 'x')])
def test_wire_stl_name_uses_the_internal_axis(axis, internal):
    assert paths.wire_stl_name('Stem', axis) == f'Stem_wire_0_{internal}.stl'


def test_wire_stl_name_defaults_to_gy():
    assert paths.wire_stl_name('Stem') == 'Stem_wire_0_z.stl'


# ---------------------------------------------------------------------------
# dir_has_outputs
# ---------------------------------------------------------------------------

def test_dir_has_outputs_false_for_missing_or_file(tmp_path):
    a_file = _touch(str(tmp_path / 'f.txt'))

    assert paths.dir_has_outputs(str(tmp_path / 'nope')) is False
    assert paths.dir_has_outputs(a_file) is False


def test_dir_has_outputs_false_for_empty_tree(tmp_path):
    (tmp_path / 'empty' / 'nested').mkdir(parents=True)

    assert paths.dir_has_outputs(str(tmp_path / 'empty')) is False


def test_dir_has_outputs_true_for_nested_file(tmp_path):
    _touch(str(tmp_path / 'run' / 'nested' / 'out.stl'))

    assert paths.dir_has_outputs(str(tmp_path / 'run')) is True


# ---------------------------------------------------------------------------
# unique_path / unique_stem / unique_run_dir
# ---------------------------------------------------------------------------

def test_unique_path_returns_input_when_free(tmp_path):
    target = str(tmp_path / 'wire.stl')

    assert paths.unique_path(target) == target


def test_unique_path_increments_until_free(tmp_path):
    _touch(str(tmp_path / 'wire.stl'))
    _touch(str(tmp_path / 'wire(2).stl'))

    assert paths.unique_path(str(tmp_path / 'wire.stl')) == str(tmp_path / 'wire(3).stl')


def test_unique_path_strips_an_existing_suffix_before_incrementing(tmp_path):
    _touch(str(tmp_path / 'wire(2).stl'))

    # 'wire(2).stl' exists -> base 'wire' -> next free is 'wire(3).stl'
    assert paths.unique_path(str(tmp_path / 'wire(2).stl')) == str(tmp_path / 'wire(3).stl')


def test_unique_stem_uses_default_markers_for_the_axis(tmp_path):
    stem = 'Gradient_Gy'
    assert paths.unique_stem(str(tmp_path), stem, gradient_axis='y') == stem

    _touch(str(tmp_path / f'{stem}_wire_0_z.stl'))          # Gy -> internal z
    assert paths.unique_stem(str(tmp_path), stem, gradient_axis='y') == f'{stem}(2)'

    _touch(str(tmp_path / f'{stem}(2)_metrics.txt'))
    assert paths.unique_stem(str(tmp_path), stem, gradient_axis='y') == f'{stem}(3)'


def test_unique_stem_ignores_other_axis_markers(tmp_path):
    stem = 'Gradient_Gy'
    _touch(str(tmp_path / f'{stem}_wire_0_x.stl'))          # Gz marker, not Gy

    assert paths.unique_stem(str(tmp_path), stem, gradient_axis='y') == stem


def test_unique_stem_honours_custom_markers_and_creates_the_dir(tmp_path):
    directory = str(tmp_path / 'made' / 'here')

    assert paths.unique_stem(directory, 'run', ('{stem}.log',)) == 'run'
    assert os.path.isdir(directory)

    _touch(os.path.join(directory, 'run.log'))
    assert paths.unique_stem(directory, 'run', ('{stem}.log',)) == 'run(2)'


def test_unique_run_dir_creates_a_fresh_folder_each_call(tmp_path):
    base = str(tmp_path / 'pipeline')

    first = paths.unique_run_dir(base, 'Gy_tk2500_lvl26')
    second = paths.unique_run_dir(base, 'Gy_tk2500_lvl26')
    third = paths.unique_run_dir(base, 'Gy_tk2500_lvl26')

    assert [os.path.basename(p) for p in (first, second, third)] == [
        'Gy_tk2500_lvl26', 'Gy_tk2500_lvl26(2)', 'Gy_tk2500_lvl26(3)']
    assert all(os.path.isdir(p) for p in (first, second, third))


# ---------------------------------------------------------------------------
# Active stem file
# ---------------------------------------------------------------------------

def test_write_then_read_active_stem_roundtrip(tmp_path):
    directory = str(tmp_path / 'run')

    paths.write_active_stem(directory, 'Gradient_Gy_tk2500_lvl26(2)')

    assert os.path.isfile(os.path.join(directory, paths.ACTIVE_STEM_FILE))
    assert paths.read_active_stem(directory) == 'Gradient_Gy_tk2500_lvl26(2)'


def test_read_active_stem_strips_whitespace(tmp_path):
    _touch(str(tmp_path / paths.ACTIVE_STEM_FILE))
    (tmp_path / paths.ACTIVE_STEM_FILE).write_text('  stem \n')

    assert paths.read_active_stem(str(tmp_path)) == 'stem'


def test_read_active_stem_returns_default_when_absent(tmp_path):
    assert paths.read_active_stem(str(tmp_path)) == ''
    assert paths.read_active_stem(str(tmp_path), 'fallback') == 'fallback'


# ---------------------------------------------------------------------------
# Lead / wire STL resolution
# ---------------------------------------------------------------------------

def test_resolve_lead_stl_paths_falls_back_to_conventional_names(tmp_path):
    wire = str(tmp_path / 'Gradient_Gy_wire_0_z.stl')

    with_leads, coil_open, leads_only = paths.resolve_lead_stl_paths(wire)

    assert with_leads == str(tmp_path / 'Gradient_Gy_wire_0_z_with_leads.stl')
    assert coil_open == str(tmp_path / 'Gradient_Gy_wire_0_z_coil_open.stl')
    assert leads_only == str(tmp_path / 'Gradient_Gy_wire_0_z_leads_only.stl')


def test_resolve_lead_stl_paths_picks_the_newest_suffixed_file(tmp_path):
    wire = str(tmp_path / 'wire.stl')
    older = _touch(str(tmp_path / 'wire_with_leads.stl'))
    newer = _touch(str(tmp_path / 'wire_with_leads(2).stl'))
    _set_mtime(older, 1_000_000)
    _set_mtime(newer, 2_000_000)

    with_leads, coil_open, _ = paths.resolve_lead_stl_paths(wire)

    assert with_leads == newer
    # coil_open has no match on disk -> conventional name
    assert coil_open == str(tmp_path / 'wire_coil_open.stl')


def test_resolve_wire_stl_path_prefers_the_active_stem(tmp_path):
    directory = str(tmp_path)
    paths.write_active_stem(directory, 'Gradient_Gy_tk2500_lvl26(2)')
    active = _touch(str(tmp_path / 'Gradient_Gy_tk2500_lvl26(2)_wire_0_z.stl'))
    _touch(str(tmp_path / 'Gradient_Gy_tk2500_lvl26_wire_0_z.stl'))

    assert paths.resolve_wire_stl_path(directory, 'y', 2500, 26) == active


def test_resolve_wire_stl_path_falls_back_to_the_default_stem(tmp_path):
    default = _touch(str(tmp_path / 'Gradient_Gy_tk2500_lvl26_wire_0_z.stl'))

    assert paths.resolve_wire_stl_path(str(tmp_path), 'y', 2500, 26) == default


def test_resolve_wire_stl_path_globs_and_skips_derived_stls(tmp_path):
    suffixed = _touch(str(tmp_path / 'Gradient_Gy_tk2500_lvl26(3)_wire_0_z.stl'))
    derived = _touch(str(tmp_path / 'Gradient_Gy_tk2500_lvl26(4)_wire_0_z_with_leads.stl'))
    _set_mtime(suffixed, 1_000_000)
    _set_mtime(derived, 3_000_000)          # newer, but must be ignored

    assert paths.resolve_wire_stl_path(str(tmp_path), 'y', 2500, 26) == suffixed


def test_resolve_wire_stl_path_returns_conventional_name_when_nothing_exists(tmp_path):
    expected = str(tmp_path / 'Gradient_Gy_tk2500_lvl26_wire_0_z.stl')

    assert paths.resolve_wire_stl_path(str(tmp_path), 'y', 2500, 26) == expected


# ---------------------------------------------------------------------------
# derive_align_wire_path / unique_lead_output_paths
# ---------------------------------------------------------------------------

def test_derive_align_wire_path_finds_the_coil_only_stl(tmp_path):
    coil = _touch(str(tmp_path / 'wire.stl'))
    subtract = str(tmp_path / 'wire_with_leads.stl')

    assert paths.derive_align_wire_path(subtract) == coil


def test_derive_align_wire_path_handles_a_numbered_subtract_stl(tmp_path):
    coil = _touch(str(tmp_path / 'wire.stl'))
    subtract = str(tmp_path / 'wire_with_leads(2).stl')

    assert paths.derive_align_wire_path(subtract) == coil


def test_derive_align_wire_path_returns_input_when_no_coil_stl(tmp_path):
    subtract = str(tmp_path / 'wire_with_leads.stl')

    assert paths.derive_align_wire_path(subtract) == subtract


def test_unique_lead_output_paths_are_untouched_when_free(tmp_path):
    wire = str(tmp_path / 'wire.stl')

    with_leads, coil_open, leads_only = paths.unique_lead_output_paths(wire)

    assert with_leads == str(tmp_path / 'wire_with_leads.stl')
    assert coil_open == str(tmp_path / 'wire_coil_open.stl')
    assert leads_only == str(tmp_path / 'wire_leads_only.stl')


def test_unique_lead_output_paths_share_one_suffix(tmp_path):
    wire = str(tmp_path / 'wire.stl')
    _touch(str(tmp_path / 'wire_with_leads.stl'))

    with_leads, coil_open, leads_only = paths.unique_lead_output_paths(wire)

    assert with_leads == str(tmp_path / 'wire_with_leads(2).stl')
    assert coil_open == str(tmp_path / 'wire_coil_open(2).stl')
    assert leads_only == str(tmp_path / 'wire_leads_only(2).stl')


def test_unique_lead_output_paths_when_only_a_sibling_is_taken(tmp_path):
    """
    Documents current behaviour: the ``(n)`` suffix is derived from the
    *_with_leads* name only. When that name is free but a sibling already
    exists, the sibling is reused (and would be overwritten) — the retry in
    ``unique_lead_output_paths`` cannot bump a name that is not taken.
    """
    wire = str(tmp_path / 'wire.stl')
    existing_sibling = _touch(str(tmp_path / 'wire_coil_open.stl'))

    with_leads, coil_open, leads_only = paths.unique_lead_output_paths(wire)

    assert with_leads == str(tmp_path / 'wire_with_leads.stl')
    assert coil_open == existing_sibling
    assert leads_only == str(tmp_path / 'wire_leads_only.stl')
