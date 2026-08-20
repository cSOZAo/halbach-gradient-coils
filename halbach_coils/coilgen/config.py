"""
Single source of truth for the coil-mold workflow.

Replaces the three former ``coil_mold_common.py`` copies (pipeline / pipeline_gz
/ standalone) with one dataclass-driven configuration. Per-axis lead presets
live here so leads no longer hardcode Gy-only world-frame filters.

Workflow
--------
    gradiente (run_gradient) -> wire STL
    leads     (run_leads)    -> wire STL with leads + coil_open + leads_only
    shell     (run_shell)    -> printable shell halves

Coordinate frame
----------------
B0 is parallel to +Y. pyCoilGen optimizes Bz, so the cylinder mesh is rotated
R_y(pi/2) (bore axis -> +X). ``Config.internal_axis`` maps a physical gradient
axis to the pyCoilGen internal axis (x->y, y->z, z->x).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from . import geometry as geo
from . import paths as _paths


# Electrical resistivity at approximately 20 °C [ohm.m].  pyCoilGen's
# historical argument is named ``specific_conductivity_conductor``, but its
# resistance calculations use the value as resistivity (R = rho L / A).
CONDUCTOR_MATERIAL_RESISTIVITY = {
    'Cu': 1.68e-8,
    'Al': 2.82e-8,
    'Ag': 1.59e-8,
}


# ---------------------------------------------------------------------------
# Per-axis lead presets (resolves the non-Gy leads bug).
# ---------------------------------------------------------------------------
# The former add_coil_leads.py picked the apex with a world-frame filter
# (Z > 0.10 and |Y| < 0.05) tuned for Gy. Here the apex is selected in
# *cylindrical* coordinates relative to the bore axis: an angular wedge in
# the radial plane around ``sector_ref_dir`` plus the axial projection onto
# ``lead_direction``. Leads always exit along the bore axis (±X after the
# R_y(pi/2) rotation); only the angular station changes per gradient axis.
#
# These are starting points — the exact Gx/Gz station may need empirical
# tuning, exposed through the GUI. spread_signs control which way each lead
# peels away from the cut.

@dataclass
class LeadAxisPreset:
    lead_direction: np.ndarray        # axial exit direction (along bore)
    sector_ref_dir: np.ndarray        # reference direction in the radial plane
    sector_angular_half: float        # [rad] half-width of the apex wedge
    spread_signs: tuple = (1, -1)
    exit_direction: Optional[np.ndarray] = None


_LEAD_PRESETS = {
    'x': LeadAxisPreset(
        lead_direction=np.array([-1.0, 0.0, 0.0]),
        sector_ref_dir=np.array([0.0, 1.0, 0.0]),
        sector_angular_half=0.35,
        spread_signs=(1, -1),
    ),
    'y': LeadAxisPreset(
        lead_direction=np.array([-1.0, 0.0, 0.0]),
        sector_ref_dir=np.array([0.0, 0.0, 1.0]),
        sector_angular_half=0.35,
        spread_signs=(1, -1),
    ),
    'z': LeadAxisPreset(
        lead_direction=np.array([-1.0, 0.0, 0.0]),
        sector_ref_dir=np.array([0.0, 1.0, 0.0]),
        sector_angular_half=0.35,
        spread_signs=(1, -1),
    ),
}


# ---------------------------------------------------------------------------
# Nested configuration blocks
# ---------------------------------------------------------------------------

@dataclass
class WireConfig:
    conductor_width: float = 0.002            # [m] nominal diameter when A=B=1
    cross_section_n: int = 16
    cross_section_a_frac: float = 2.0
    cross_section_b_frac: float = 1.0


@dataclass
class TargetConfig:
    rx: float = 0.125                          # [m]
    ry: float = 0.125                          # [m]
    rz: float = 0.125                          # [m]
    resol_radial: int = 8
    resol_angular: int = 28


@dataclass
class CylinderConfig:
    height: float = 0.430                      # [m] requested shell / cylinder length
    radius: float = 0.039                      # [m] physical outer radius
    n_circ: int = 200
    n_long: int = 10
    rot_axis: tuple = (0, 1, 0)
    rot_angle: float = np.pi / 2
    # pyCoilGen mesh axial length = height * this (< 1 keeps windings short of
    # the shell ends so grooves are not flush with the cylinder rim).
    mesh_length_factor: float = 0.95


@dataclass
class WindingConfig:
    cut_width: float = 0.001                   # [m]
    pot_offset_factor: float = 0.5
    min_loop_signif: int = 5
    # None -> derive from cable_height + layer_gap_mm. A negative shift moves
    # the selected crossover branch inward because pyCoilGen normals point outward.
    normal_shift: Optional[float] = None        # [m] direct override
    layer_gap_mm: float = 0.0                   # [m] radial gap at layer crossing
    normal_shift_height_factor: float = 1.40    # legacy; ignored when layer_gap_mm set
    normal_shift_smooth: list = field(default_factory=lambda: [7, 7, 7])
    smooth_factor: int = 3


@dataclass
class FastHenryConfig:
    enabled: bool = True
    bin_path: str = ''                         # empty -> resolve via PATH
    material: str = 'Cu'
    specific_conductivity: float = CONDUCTOR_MATERIAL_RESISTIVITY['Cu']  # resistivity [ohm.m]


@dataclass
class LeadsConfig:
    # Apex / cut geometry (axis-aware preset overrides lead_direction, sector,
    # spread_signs, exit_direction when ``use_preset`` is True).
    use_preset: bool = True
    preset: Optional[LeadAxisPreset] = None
    cyl_axis: Optional[np.ndarray] = None      # None -> rotated_cylinder_axis
    shell_radius: Optional[float] = None       # None -> infer from mesh

    cut_loop_length: float = 0.020             # [m] gap opened along the loop so the
                                               # shared-tip lead arc has room without crossing
    gap_axial_length: float = 0.012            # [m]
    wire_isolate_half: float = 0.008           # [m]
    tangent_radius: float = 0.006              # [m]
    # Legacy S-curve params (no longer wired into the centreline; kept for GUI
    # / back-compat). The lead now uses a toward-gap elbow plus a route-aligned
    # exit run, sized by lead_length / lead_blend.
    wire_tangent_run: float = 0.020            # [m] legacy
    face_toward_gap: float = 0.003             # [m] legacy
    peel_out: float = 0.006                    # [m] legacy
    lead_junction_coil_backset: float = 0.0    # [m] legacy (was 0.003; lip removed)
    lead_junction_gap_backset: float = 0.0     # [m] legacy (was 0.002; lip removed)
    lead_length: float = 0.035                 # [m] longer axial run keeps fan from curling sideways
    lead_blend: float = 0.015                  # [m] cubic Bezier handle floor
    tip_fan: float = 0.015                     # [m]
    lead_steps: int = 128
    cs_blend_rings: int = 20                   # smoother wire->lead section transition
    junction_rigid_steps: int = 4              # RMF-aligned rings at the junction
    junction_plane_rings: int = 4


@dataclass
class ShellConfig:
    layer: int = 2                             # 2 -> g_2a/b, 3 -> g_3a/b (assets/shells, CLI legacy)
    stl_dir: str = ''                          # empty -> assets/shells
    use_custom_stl: bool = False
    # Custom shell: two halves (one STL each). When use_custom_stl and both
    # exist, Config.shell_half_paths() returns them; otherwise falls back to
    # the Fusion assets by layer.
    custom_stl_a: Optional[str] = None
    custom_stl_b: Optional[str] = None
    # Single custom STL kept for backward compat (maps to half A only).
    custom_stl: Optional[str] = None
    # Fixed bore radius from measured custom STL halves [m]. When set with
    # use_custom_stl, shell_inner_radius uses this instead of the auto wall formula.
    measured_inner_r: Optional[float] = None

    # Autogenerated hollow-cylinder shell (use_custom_stl = False).
    # Axial length = Config.cylinder.height (GUI). Wires use mesh_length_factor.
    auto_length_factor: float = 1.0            # legacy; shell length uses cylinder.height
    auto_margin_pct: float = 0.50              # radial peel per face = pct * cable_height

    subtract_mode: str = 'with_leads'          # 'with_leads' | 'with_leads_by_component' | 'two_pass'
    groove_expansion: float = 0.0             # [m] coil pass (was 0 with stale "0.35 mm" comment)
    lead_groove_expansion: float = 0.00025    # [m] leads-only 2nd pass
    shell_outer_pad: float = 0.0              # auto-set in __post_init__
    leads_second_subtract: bool = True
    coil_second_subtract: bool = False
    coil_second_expansion: float = 0.00050    # [m]
    outer_skin_trim: float = 0.0              # [m]

    resolve_self_intersections: bool = False
    voxel_pitch: float = 0.0004
    voxel_slab_length: float = 0.090
    voxel_slab_overlap: float = 0.012
    smooth_iterations: int = 8
    circular_segments: int = 256
    output_in_mm: bool = True


def default_shell_stl_dir() -> str:
    return os.path.join(_paths.PROJECT_ROOT, 'assets', 'shells')


def list_shell_pairs(stl_dir: Optional[str] = None) -> list[tuple[str, str, str]]:
    """
    Scan ``stl_dir`` for matched half pairs ``g_{N}a.stl`` + ``g_{N}b.stl``.

    Returns ``[(label, path_a, path_b), ...]`` sorted by layer index N.
    Labels look like ``"Capa 1"``, ``"Capa 2"``, ...
    """
    directory = stl_dir or default_shell_stl_dir()
    if not os.path.isdir(directory):
        return []

    pairs: list[tuple[int, str, str, str]] = []
    for name in os.listdir(directory):
        lower = name.lower()
        if not (lower.startswith('g_') and lower.endswith('a.stl')):
            continue
        # g_12a.stl -> stem g_12a, index "12"
        stem = name[:-4]  # drop .stl
        if len(stem) < 2 or stem[-1].lower() != 'a':
            continue
        index_str = stem[2:-1]  # between g_ and trailing a
        if not index_str.isdigit():
            continue
        path_a = os.path.join(directory, name)
        # Prefer exact sibling with same casing pattern for b.
        path_b = os.path.join(directory, stem[:-1] + 'b.stl')
        if not os.path.isfile(path_b):
            path_b = os.path.join(directory, stem[:-1] + 'B.stl')
        if not os.path.isfile(path_b):
            continue
        n = int(index_str)
        pairs.append((n, f'Capa {n}', path_a, path_b))

    pairs.sort(key=lambda item: item[0])
    return [(label, a, b) for _, label, a, b in pairs]


def apply_custom_shell_dims(cfg: 'Config', stl_a: str, stl_b: str) -> dict:
    """
    Measure custom half STLs and write Rext / Rint / height into ``cfg``.

    Sets ``use_custom_stl``, custom paths, ``cylinder.radius`` / ``height``,
    and ``shell.measured_inner_r``. Returns the dims dict from
    :func:`geometry.detect_fusion_cylinder_dims`.
    """
    dims = geo.detect_fusion_cylinder_dims(stl_a, stl_b)
    cfg.shell.use_custom_stl = True
    cfg.shell.custom_stl_a = stl_a
    cfg.shell.custom_stl_b = stl_b
    cfg.cylinder.radius = float(dims['outer_r_m'])
    cfg.cylinder.height = float(dims['axial_length_m'])
    cfg.shell.measured_inner_r = float(dims['inner_r_m'])
    return dims


@dataclass
class SweepConfig:
    tk_min: float = 1.0
    tk_max: float = 100000.0
    n_coarse: int = 10
    fine: bool = True
    n_fine: int = 7


@dataclass
class PipelineControl:
    run_gradient: bool = True
    run_leads: bool = True
    run_shell: bool = True


# ---------------------------------------------------------------------------
# Top-level Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    gradient_axis: str = 'x'                   # 'x' | 'y' | 'z'
    tikhonov_factor: float = 2500
    num_levels: int = 26

    target: TargetConfig = field(default_factory=TargetConfig)
    cylinder: CylinderConfig = field(default_factory=CylinderConfig)
    winding: WindingConfig = field(default_factory=WindingConfig)
    wire: WireConfig = field(default_factory=WireConfig)
    fasthenry: FastHenryConfig = field(default_factory=FastHenryConfig)
    leads: LeadsConfig = field(default_factory=LeadsConfig)
    shell: ShellConfig = field(default_factory=ShellConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    control: PipelineControl = field(default_factory=PipelineControl)

    output_dir: str = ''                       # run output folder
    show_plots: bool = True
    overlap_warn: bool = True                  # warn about 3+ cables sharing a zone
    overlap_clearance: float = 1.0             # multiplier for multi-wire zone footprint

    def __post_init__(self) -> None:
        if self.gradient_axis.lower() not in ('x', 'y', 'z'):
            raise ValueError(f"gradient_axis must be 'x', 'y' or 'z', got {self.gradient_axis!r}")
        if self.leads.preset is None:
            self.leads.preset = _LEAD_PRESETS[self.gradient_axis.lower()]
        if self.shell.stl_dir == '':
            self.shell.stl_dir = default_shell_stl_dir()
        if self.shell.shell_outer_pad == 0.0:
            self.shell.shell_outer_pad = max(0.0005, 0.65 * self.conductor_semi_a)

    # ----- derived geometry ------------------------------------------------

    @property
    def internal_axis(self) -> str:
        return geo.internal_field_axis(self.gradient_axis)

    @property
    def axis_label(self) -> str:
        return f'G{self.gradient_axis.upper()}'

    @property
    def rotated_cylinder_axis(self) -> np.ndarray:
        return geo.rotated_cylinder_axis(
            self.cylinder.rot_axis, self.cylinder.rot_angle,
        )

    @property
    def conductor_semi_a(self) -> float:
        return 0.5 * self.wire.cross_section_a_frac * self.wire.conductor_width

    @property
    def conductor_semi_b(self) -> float:
        return 0.5 * self.wire.cross_section_b_frac * self.wire.conductor_width

    @property
    def cable_height(self) -> float:
        return 2.0 * self.conductor_semi_a

    @property
    def _custom_shell_fixed(self) -> bool:
        return bool(
            self.shell.use_custom_stl
            and self.shell.measured_inner_r is not None
        )

    @property
    def cylinder_design_radius(self) -> float:
        """
        Radius of the pyCoilGen winding surface (outer-layer path centerline).

        Chosen so the two-layer wire pack is centered in the shell wall:

            design_r = (Rext + Rint) / 2 + (h + gap) / 2

        Auto mode (wall from cable + peel) reduces to::

            design_r = Rext - semi_a + radial_peel

        Custom STL mode uses measured Rext/Rint (fixed wall) with the
        midplane form above.
        """
        if self._custom_shell_fixed:
            pack_half = 0.5 * (self.cable_height + self.layer_crossing_gap)
            return self.shell_radial_center + pack_half
        return (self.cylinder.radius
                - self.conductor_semi_a
                + self.radial_peel)

    @property
    def normal_shift_length(self) -> float:
        if self.winding.normal_shift is not None:
            return self.winding.normal_shift
        return -(self.cable_height + self.winding.layer_gap_mm)

    @property
    def shell_radial_center(self) -> float:
        """Midpoint between shell inner and outer radii."""
        return 0.5 * (self.shell_inner_radius + self.shell_outer_radius)

    @property
    def estimated_wire_radial_center(self) -> float:
        """Midpoint between innermost and outermost wire faces (analytical)."""
        return self.cylinder_design_radius + 0.5 * self.normal_shift_length

    @property
    def layer_crossing_gap(self) -> float:
        """Radial gap between outer-layer bottom and inner-layer top at a crossing."""
        return self.winding.layer_gap_mm

    @property
    def estimated_wire_inner_radius(self) -> float:
        center_r = self.cylinder_design_radius
        base_inner = center_r - self.conductor_semi_a
        shifted_inner = center_r + self.normal_shift_length - self.conductor_semi_a
        return min(base_inner, shifted_inner)

    @property
    def estimated_wire_outer_radius(self) -> float:
        center_r = self.cylinder_design_radius
        base_outer = center_r + self.conductor_semi_a
        shifted_outer = center_r + self.normal_shift_length + self.conductor_semi_a
        return max(base_outer, shifted_outer)

    @property
    def radial_peel(self) -> float:
        """Per-face groove margin = auto_margin_pct × cable_height."""
        return self.shell.auto_margin_pct * self.cable_height

    @property
    def outer_skin_trim(self) -> float:
        """Alias for :attr:`radial_peel` (backward compatibility)."""
        return self.radial_peel

    @property
    def groove_margin(self) -> float:
        """Alias for :attr:`radial_peel` (backward compatibility)."""
        return self.radial_peel

    @property
    def shell_wall_thickness(self) -> float:
        """Radial wall thickness (custom: Rext−Rint; auto: 2h + gap − 2×peel)."""
        if self._custom_shell_fixed:
            return self.shell_outer_radius - float(self.shell.measured_inner_r)
        return (2.0 * self.cable_height
                + self.layer_crossing_gap
                - 2.0 * self.radial_peel)

    @property
    def shell_outer_radius(self) -> float:
        """Outer shell radius (= GUI / measured ``cylinder.radius``)."""
        return self.cylinder.radius

    @property
    def shell_inner_radius(self) -> float:
        """Inner bore radius (custom: measured STL; auto: from wall formula)."""
        if self._custom_shell_fixed:
            return float(self.shell.measured_inner_r)
        return self.shell_outer_radius - self.shell_wall_thickness

    @property
    def shell_build_outer_radius(self) -> float:
        """Alias for :attr:`shell_outer_radius` (backward compatibility)."""
        return self.shell_outer_radius

    @property
    def shell_build_inner_radius(self) -> float:
        """Alias for :attr:`shell_inner_radius` (backward compatibility)."""
        return self.shell_inner_radius

    @property
    def shell_final_inner_radius(self) -> float:
        """Alias for :attr:`shell_inner_radius` (backward compatibility)."""
        return self.shell_inner_radius

    @property
    def estimated_shell_inner_radius(self) -> float:
        return self.shell_inner_radius

    @property
    def estimated_shell_outer_radius(self) -> float:
        return self.shell_outer_radius

    @property
    def estimated_shell_thickness(self) -> float:
        return self.shell_wall_thickness

    @property
    def fasthenry_conductor_width(self) -> float:
        return 2.0 * self.conductor_semi_b

    @property
    def fasthenry_conductor_height(self) -> float:
        return (np.pi / 2.0) * self.conductor_semi_a

    @property
    def cross_sectional_points(self) -> np.ndarray:
        theta = np.linspace(0, 2 * np.pi, self.wire.cross_section_n, endpoint=True)
        return np.vstack([
            self.conductor_semi_a * np.sin(theta),
            self.conductor_semi_b * np.cos(theta),
        ])

    @property
    def design_folder(self) -> str:
        return _paths.design_folder_name(
            self.gradient_axis, self.tikhonov_factor, self.num_levels,
        )

    @property
    def project_stem_base(self) -> str:
        return _paths.gradient_project_stem(
            self.gradient_axis, self.tikhonov_factor, self.num_levels,
        )

    # ----- lead preset -----------------------------------------------------

    def lead_preset(self) -> LeadAxisPreset:
        if self.leads.preset is None:
            self.leads.preset = _LEAD_PRESETS[self.gradient_axis.lower()]
        return self.leads.preset

    # ----- shell helpers ---------------------------------------------------

    def shell_half_paths(self) -> tuple[str, str]:
        """
        Return ``(stl_a, stl_b)`` for the shell subtraction.

        When ``use_custom_stl`` and the two custom half STLs are provided and
        exist, return those. Otherwise fall back to the Fusion assets by
        ``shell.layer`` (CLI legacy / assets workflow).
        """
        if self.shell.use_custom_stl:
            a = self.shell.custom_stl_a
            b = self.shell.custom_stl_b
            # Backward compat: a single custom_stl maps to half A only.
            if a is None and self.shell.custom_stl is not None:
                a = self.shell.custom_stl
            if a and b and os.path.isfile(a) and os.path.isfile(b):
                return a, b
        layer = self.shell.layer
        return (
            os.path.join(self.shell.stl_dir, f'g_{layer}a.stl'),
            os.path.join(self.shell.stl_dir, f'g_{layer}b.stl'),
        )

    # ----- serialization ---------------------------------------------------

    def to_params_dict(self) -> dict:
        """Flat dict of user parameters for the metrics txt header."""
        return {
            'gradient_axis': self.gradient_axis,
            'tikhonov_factor': self.tikhonov_factor,
            'num_levels': self.num_levels,
            'target_rx_m': self.target.rx,
            'target_ry_m': self.target.ry,
            'target_rz_m': self.target.rz,
            'resol_radial': self.target.resol_radial,
            'resol_angular': self.target.resol_angular,
            'cyl_height_m': self.cylinder.height,
            'cyl_mesh_length_factor': self.cylinder.mesh_length_factor,
            'cyl_mesh_height_m': self.cylinder.height * self.cylinder.mesh_length_factor,
            'cyl_radius_m': self.cylinder.radius,
            'cyl_outer_radius_m': self.cylinder.radius,
            'cyl_design_radius_m': self.cylinder_design_radius,
            'cyl_n_circ': self.cylinder.n_circ,
            'cyl_n_long': self.cylinder.n_long,
            'cyl_rot_axis': self.cylinder.rot_axis,
            'cyl_rot_angle': self.cylinder.rot_angle,
            'cut_width_m': self.winding.cut_width,
            'pot_offset_factor': self.winding.pot_offset_factor,
            'min_loop_significance_pct': self.winding.min_loop_signif,
            'normal_shift_m': self.normal_shift_length,
            'layer_gap_mm': self.winding.layer_gap_mm,
            'layer_crossing_gap_m': self.layer_crossing_gap,
            'radial_peel_m': self.radial_peel,
            'outer_skin_trim_m': self.radial_peel,
            'shell_wall_thickness_m': self.shell_wall_thickness,
            'shell_outer_radius_m': self.shell_outer_radius,
            'shell_inner_radius_m': self.shell_inner_radius,
            'shell_build_outer_radius_m': self.shell_outer_radius,
            'shell_build_inner_radius_m': self.shell_inner_radius,
            'shell_final_inner_radius_m': self.shell_inner_radius,
            'normal_shift_height_factor': self.winding.normal_shift_height_factor,
            'normal_shift_smooth': self.winding.normal_shift_smooth,
            'auto_margin_pct': self.shell.auto_margin_pct,
            'smooth_factor': self.winding.smooth_factor,
            'conductor_width_m': self.wire.conductor_width,
            'conductor_semi_a_m': self.conductor_semi_a,
            'conductor_semi_b_m': self.conductor_semi_b,
            'cable_height_m': self.cable_height,
            'cross_section_n': self.wire.cross_section_n,
            'cross_section_a_frac': self.wire.cross_section_a_frac,
            'cross_section_b_frac': self.wire.cross_section_b_frac,
            'enable_fasthenry': self.fasthenry.enabled,
            'conductor_material': self.fasthenry.material,
            'fasthenry_conductor_width_m': self.fasthenry_conductor_width,
            'fasthenry_conductor_height_m': self.fasthenry_conductor_height,
            'specific_conductivity_conductor_ohm_m': self.fasthenry.specific_conductivity,
        }


def preset_for(axis: str) -> LeadAxisPreset:
    """Return the default lead preset for ``axis`` (used by the GUI)."""
    return _LEAD_PRESETS[axis.lower()]
