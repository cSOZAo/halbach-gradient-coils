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
    conductor_width: float = 0.00225          # [m]
    cross_section_n: int = 16
    cross_section_a_frac: float = 1.6
    cross_section_b_frac: float = 0.7


@dataclass
class TargetConfig:
    rx: float = 0.125                          # [m]
    ry: float = 0.125                          # [m]
    rz: float = 0.125                          # [m]
    resol_radial: int = 8
    resol_angular: int = 28


@dataclass
class CylinderConfig:
    height: float = 0.430                      # [m]
    radius: float = 0.150                      # [m]
    n_circ: int = 200
    n_long: int = 10
    rot_axis: tuple = (0, 1, 0)
    rot_angle: float = np.pi / 2


@dataclass
class WindingConfig:
    cut_width: float = 0.001                   # [m]
    pot_offset_factor: float = 0.5
    min_loop_signif: int = 5
    normal_shift: float = -0.005               # [m]
    normal_shift_smooth: list = field(default_factory=lambda: [7, 7, 7])
    smooth_factor: int = 3


@dataclass
class FastHenryConfig:
    enabled: bool = True
    bin_path: str = ''                         # empty -> resolve via PATH
    specific_conductivity: float = 1.8e-8      # [ohm.m]


@dataclass
class LeadsConfig:
    # Apex / cut geometry (axis-aware preset overrides lead_direction, sector,
    # spread_signs, exit_direction when ``use_preset`` is True).
    use_preset: bool = True
    preset: Optional[LeadAxisPreset] = None
    cyl_axis: Optional[np.ndarray] = None      # None -> rotated_cylinder_axis
    shell_radius: Optional[float] = None       # None -> infer from mesh

    cut_loop_length: float = 0.050             # [m]
    gap_axial_length: float = 0.012            # [m]
    wire_isolate_half: float = 0.008           # [m]
    tangent_radius: float = 0.006              # [m]
    wire_tangent_run: float = 0.020            # [m] longer tangent follow -> smoother junction
    face_toward_gap: float = 0.003             # [m]
    peel_out: float = 0.006                    # [m]
    lead_junction_coil_backset: float = 0.003  # [m]
    lead_junction_gap_backset: float = 0.002   # [m]
    lead_length: float = 0.02                  # [m]
    lead_blend: float = 0.030                  # [m]
    tip_fan: float = 0.015                     # [m]
    lead_steps: int = 128
    cs_blend_rings: int = 20                   # smoother wire->lead section transition
    junction_rigid_steps: int = 4              # RMF-aligned rings at the junction
    junction_plane_rings: int = 4


@dataclass
class ShellConfig:
    layer: int = 2                             # 2 -> g_2a/b, 3 -> g_3a/b (Fusion assets, CLI legacy)
    stl_dir: str = ''                          # empty -> assets/cilindros_gradientes_grandes
    use_custom_stl: bool = False
    # Custom shell: two halves (one STL each). When use_custom_stl and both
    # exist, Config.shell_half_paths() returns them; otherwise falls back to
    # the Fusion assets by layer.
    custom_stl_a: Optional[str] = None
    custom_stl_b: Optional[str] = None
    # Single custom STL kept for backward compat (maps to half A only).
    custom_stl: Optional[str] = None

    # Autogenerated hollow-cylinder shell (use_custom_stl = False). Dimensions
    # are computed from the coil-only wire mesh; see build_auto_hollow_cylinder.
    auto_length_factor: float = 1.05           # cylinder length = wire axial extent * this
    auto_margin_pct: float = 0.15              # margin per side = pct * cable_height (2*semi_a)

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
    gradient_axis: str = 'y'                   # 'x' | 'y' | 'z'
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
    overlap_warn: bool = True
    overlap_clearance: float = 1.0             # min gap = clearance * conductor_width

    def __post_init__(self) -> None:
        if self.gradient_axis.lower() not in ('x', 'y', 'z'):
            raise ValueError(f"gradient_axis must be 'x', 'y' or 'z', got {self.gradient_axis!r}")
        if self.leads.preset is None:
            self.leads.preset = _LEAD_PRESETS[self.gradient_axis.lower()]
        if self.shell.stl_dir == '':
            self.shell.stl_dir = os.path.join(
                _paths.PROJECT_ROOT, 'assets', 'cilindros_gradientes_grandes',
            )
        if self.shell.shell_outer_pad == 0.0:
            semi_a = self.wire.cross_section_a_frac * self.wire.conductor_width
            self.shell.shell_outer_pad = max(0.0005, 0.65 * semi_a)

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
    def fasthenry_conductor_width(self) -> float:
        return 2.0 * self.wire.cross_section_b_frac * self.wire.conductor_width

    @property
    def fasthenry_conductor_height(self) -> float:
        return (np.pi / 2.0) * self.wire.cross_section_a_frac * self.wire.conductor_width

    @property
    def cross_sectional_points(self) -> np.ndarray:
        theta = np.linspace(0, 2 * np.pi, self.wire.cross_section_n, endpoint=True)
        return np.vstack([
            self.wire.cross_section_a_frac * self.wire.conductor_width * np.sin(theta),
            self.wire.cross_section_b_frac * self.wire.conductor_width * np.cos(theta),
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
            'cyl_radius_m': self.cylinder.radius,
            'cyl_n_circ': self.cylinder.n_circ,
            'cyl_n_long': self.cylinder.n_long,
            'cyl_rot_axis': self.cylinder.rot_axis,
            'cyl_rot_angle': self.cylinder.rot_angle,
            'cut_width_m': self.winding.cut_width,
            'pot_offset_factor': self.winding.pot_offset_factor,
            'min_loop_significance_pct': self.winding.min_loop_signif,
            'normal_shift_m': self.winding.normal_shift,
            'normal_shift_smooth': self.winding.normal_shift_smooth,
            'smooth_factor': self.winding.smooth_factor,
            'conductor_width_m': self.wire.conductor_width,
            'cross_section_n': self.wire.cross_section_n,
            'cross_section_a_frac': self.wire.cross_section_a_frac,
            'cross_section_b_frac': self.wire.cross_section_b_frac,
            'enable_fasthenry': self.fasthenry.enabled,
            'fasthenry_conductor_width_m': self.fasthenry_conductor_width,
            'fasthenry_conductor_height_m': self.fasthenry_conductor_height,
            'specific_conductivity_conductor_ohm_m': self.fasthenry.specific_conductivity,
        }


def preset_for(axis: str) -> LeadAxisPreset:
    """Return the default lead preset for ``axis`` (used by the GUI)."""
    return _LEAD_PRESETS[axis.lower()]
