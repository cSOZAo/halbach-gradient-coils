"""Persist and plot the final wire field without opening a GUI from a worker.

The NPZ contains only numeric arrays and strings, so results can be reopened
without unpickling a pyCoilGen solution. Coordinates and field vectors use the
physical STL frame. Fields are projected on the *signed* B0 direction and are
shown for one ampere, including the target's original amplitude.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


PLOT_KINDS = ("comparison", "linearity", "spatial")
PLOT_LABELS = {
    "comparison": "Objective and generated field",
    "linearity": "Linearity and residual",
    "spatial": "Spatial field and target error",
}

# Registered by gui.i18n; kept with these figures so all captions stay together.
FIELD_PLOT_TRANSLATIONS_ES = {
    "Field results": "Resultados del campo",
    "Objective and generated field": "Campo objetivo y generado",
    "Linearity and residual": "Linealidad y residuo",
    "Spatial field and target error": "Campo espacial y error respecto al objetivo",
    "Objective at 1 A": "Objetivo a 1 A",
    "Generated wire field at 1 A": "Campo generado por el cable a 1 A",
    "Linear fit": "Ajuste lineal",
    "Position {axis} [mm]": "Posición {axis} [mm]",
    "Field parallel to B0 [mT/A]": "Campo paralelo a B0 [mT/A]",
    "Residual to linear fit [mT/A]": "Residuo del ajuste lineal [mT/A]",
    "Generated minus objective [mT/A]": "Generado menos objetivo [mT/A]",
    "Generated field": "Campo generado",
    "Error relative to objective": "Error respecto al objetivo",
    "Gradient G{axis} | B0 {direction}": "Gradiente G{axis} | B0 {direction}",
    "Slope: {slope:.4g} mT/(m A) | Residual RMS: {rms:.4g} mT/A":
        "Pendiente: {slope:.4g} mT/(m A) | RMS del residuo: {rms:.4g} mT/A",
    "Sampled ROI points; no interpolation. Physical axes match the STL viewer.":
        "Puntos muestreados de la ROI; sin interpolación. Ejes físicos iguales a los del visor STL.",
    "Both curves use 1 A; the objective amplitude is not rescaled to the generated field.":
        "Ambas curvas usan 1 A; la amplitud objetivo se conserva sin ajustarla al campo generado.",
    "Field plots:": "Gráficos del campo:",
    "Open saved field results...": "Abrir resultados de campo guardados...",
    "Select saved field results": "Seleccionar resultados de campo guardados",
    "Field result data": "Datos de resultados del campo",
    "Cannot open field results": "No se pueden abrir los resultados del campo",
    "View:": "Vista:",
}


def _translate(tr: Callable | None, source: str, **values) -> str:
    return tr(source, **values) if tr else source.format(**values)


def _b0_vector(direction: str) -> np.ndarray:
    direction = direction.strip().lower()
    if len(direction) == 1:
        direction = "+" + direction
    if direction not in ("+x", "-x", "+y", "-y", "+z", "-z"):
        raise ValueError("B0 direction must be one of +/-x, +/-y, +/-z")
    vector = np.zeros(3)
    vector["xyz".index(direction[1])] = -1.0 if direction[0] == "-" else 1.0
    return vector


@dataclass
class FieldPlotData:
    """Physical ROI samples and signed B-parallel values, in m and mT/A."""

    coords_m: np.ndarray
    generated_mT_per_A: np.ndarray
    target_mT_per_A: np.ndarray
    gradient_axis: str
    b0_direction: str
    project_name: str = "gradient"

    def __post_init__(self):
        self.coords_m = np.asarray(self.coords_m, dtype=float)
        self.generated_mT_per_A = np.asarray(self.generated_mT_per_A, dtype=float)
        self.target_mT_per_A = np.asarray(self.target_mT_per_A, dtype=float)
        self.gradient_axis = str(self.gradient_axis).lower()
        self.b0_direction = str(self.b0_direction).lower()
        if len(self.b0_direction) == 1:
            self.b0_direction = "+" + self.b0_direction
        _b0_vector(self.b0_direction)
        if self.gradient_axis not in ("x", "y", "z"):
            raise ValueError("Gradient axis must be x, y or z")
        if self.coords_m.ndim != 2 or self.coords_m.shape[0] != 3:
            raise ValueError("Field coordinates must have shape (3, N)")
        n = self.coords_m.shape[1]
        if n < 2 or self.generated_mT_per_A.shape != (n,) or self.target_mT_per_A.shape != (n,):
            raise ValueError("At least two matching coordinate, target and field samples are required")
        if not all(np.all(np.isfinite(a)) for a in (
            self.coords_m, self.generated_mT_per_A, self.target_mT_per_A,
        )):
            raise ValueError("Field results contain non-finite samples")
        if np.ptp(self.gradient_coords_m) == 0:
            raise ValueError("The ROI must span the gradient axis")

    @property
    def gradient_coords_m(self) -> np.ndarray:
        return self.coords_m["xyz".index(self.gradient_axis)]

    def linearity(self) -> tuple[float, np.ndarray, np.ndarray, float]:
        """Return slope [mT/(m A)], fit, residual [mT/A], and residual RMS."""
        coords = self.gradient_coords_m
        # Center first to avoid loss of precision for an off-center ROI.
        centered = coords - np.mean(coords)
        field = self.generated_mT_per_A
        slope = float(np.dot(centered, field - field.mean()) / np.dot(centered, centered))
        fit = slope * centered + field.mean()
        residual = field - fit
        rms = float(np.sqrt(np.mean(residual ** 2)))
        return slope, fit, residual, rms


def field_data_from_solution(solution, gradient_axis: str,
                             b0_direction: str = "+y",
                             project_name: str = "gradient") -> FieldPlotData:
    """Extract the final connected-wire field and the target at one ampere."""
    coords = np.asarray(solution.target_field.coords, dtype=float)
    generated = np.asarray(solution.solution_errors.combined_field_layout_per1Amp, dtype=float)
    target = np.asarray(solution.solution_errors.target_field_1A.b, dtype=float)
    if generated.shape != coords.shape or target.shape != coords.shape:
        raise ValueError("Solution field vectors must match physical coordinates (3, N)")
    b0 = _b0_vector(b0_direction)
    return FieldPlotData(coords, b0 @ generated * 1000.0, b0 @ target * 1000.0,
                         gradient_axis, b0_direction, project_name)


def load_field_data(path: str | Path) -> FieldPlotData:
    """Load a saved result with pickle disabled and validate its sample arrays."""
    with np.load(path, allow_pickle=False) as archive:
        if int(archive["format_version"]) != 1:
            raise ValueError("Unsupported field result format")
        return FieldPlotData(
            coords_m=archive["coords_m"],
            generated_mT_per_A=archive["generated_mT_per_A"],
            target_mT_per_A=archive["target_mT_per_A"],
            gradient_axis=str(archive["gradient_axis"].item()),
            b0_direction=str(archive["b0_direction"].item()),
            project_name=str(archive["project_name"].item()),
        )


def draw_field_figure(figure, data: FieldPlotData, kind: str,
                      tr: Callable | None = None):
    """Draw into a Figure shared by the PNG exporter and the Tk viewer."""
    if kind not in PLOT_KINDS:
        raise ValueError(f"Unknown field plot: {kind}")
    figure.clear()
    text = lambda source, **values: _translate(tr, source, **values)
    heading = text("Gradient G{axis} | B0 {direction}",
                   axis=data.gradient_axis.upper(), direction=data.b0_direction.upper())
    figure.suptitle(f"{text(PLOT_LABELS[kind])}\n{heading}", fontsize=14)
    coord_mm = data.gradient_coords_m * 1000.0
    order = np.argsort(coord_mm)
    field = data.generated_mT_per_A
    target = data.target_mT_per_A

    if kind == "comparison":
        ax = figure.add_subplot(111)
        ax.plot(coord_mm[order], target[order], color="#c24b3b", linewidth=2,
                label=text("Objective at 1 A"))
        ax.scatter(coord_mm, field, s=12, alpha=0.55, color="#2378a6",
                   label=text("Generated wire field at 1 A"), rasterized=True)
        ax.set_xlabel(text("Position {axis} [mm]", axis=data.gradient_axis.upper()))
        ax.set_ylabel(text("Field parallel to B0 [mT/A]"))
        ax.legend()
        ax.grid(alpha=0.3, linestyle=":")
        figure.text(0.5, 0.025,
                    text("Both curves use 1 A; the objective amplitude is not rescaled to the generated field."),
                    ha="center", fontsize=9)
        figure.subplots_adjust(left=0.12, right=0.97, bottom=0.16, top=0.85)
    elif kind == "linearity":
        slope, fit, residual, rms = data.linearity()
        ax = figure.add_subplot(211)
        residual_ax = figure.add_subplot(212, sharex=ax)
        ax.scatter(coord_mm, field, s=12, alpha=0.55, color="#2378a6",
                   label=text("Generated wire field at 1 A"), rasterized=True)
        ax.plot(coord_mm[order], fit[order], color="#c24b3b", linewidth=2,
                label=text("Linear fit"))
        ax.set_ylabel(text("Field parallel to B0 [mT/A]"))
        ax.legend(loc="best")
        residual_ax.scatter(coord_mm, residual, s=12, alpha=0.55, color="#2378a6",
                            rasterized=True)
        residual_ax.axhline(0, color="#555555", linewidth=1)
        residual_ax.set_xlabel(text("Position {axis} [mm]", axis=data.gradient_axis.upper()))
        residual_ax.set_ylabel(text("Residual to linear fit [mT/A]"))
        ax.grid(alpha=0.3, linestyle=":")
        residual_ax.grid(alpha=0.3, linestyle=":")
        figure.text(0.5, 0.025, text(
            "Slope: {slope:.4g} mT/(m A) | Residual RMS: {rms:.4g} mT/A",
            slope=slope, rms=rms), ha="center", fontsize=10)
        figure.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.85, hspace=0.3)
    else:
        from matplotlib.colors import Normalize

        coords_mm = data.coords_m * 1000.0
        b0 = _b0_vector(data.b0_direction)
        spans = np.ptp(coords_mm, axis=1)
        arrow_length = float(max(spans.max() * 0.25, 1.0))
        center = coords_mm.mean(axis=1)
        for index, (values, caption, colorbar_label) in enumerate((
            (field, "Generated field", "Field parallel to B0 [mT/A]"),
            (field - target, "Error relative to objective", "Generated minus objective [mT/A]"),
        ), 1):
            ax = figure.add_subplot(1, 2, index, projection="3d")
            limit = float(max(np.max(np.abs(values)), 1e-12))
            cloud = ax.scatter(*coords_mm, c=values, s=9, cmap="coolwarm",
                               norm=Normalize(-limit, limit), alpha=0.8,
                               rasterized=True)
            ax.quiver(*center, *b0, length=arrow_length, color="#343434",
                      arrow_length_ratio=0.25, linewidth=2)
            arrow_tip = center + b0 * arrow_length * 1.1
            ax.text(*arrow_tip, "B0", color="#222222")
            ax.set_title(text(caption), fontsize=11)
            ax.set_xlabel("X [mm]")
            ax.set_ylabel("Y [mm]")
            ax.set_zlabel("Z [mm]")
            ax.set_box_aspect(np.maximum(spans, max(spans.max() * 0.05, 1.0)))
            colorbar = figure.colorbar(cloud, ax=ax, fraction=0.035, pad=0.12,
                                      shrink=0.7)
            colorbar.set_label(text(colorbar_label), fontsize=9)
            colorbar.ax.tick_params(labelsize=8)
        figure.text(0.5, 0.025,
                    text("Sampled ROI points; no interpolation. Physical axes match the STL viewer."),
                    ha="center", fontsize=9)
        figure.subplots_adjust(left=0.02, right=0.88, bottom=0.1, top=0.83, wspace=0.42)
    return figure


def create_field_figure(data: FieldPlotData, kind: str = "comparison",
                        tr: Callable | None = None):
    """Create an OO matplotlib Figure; never import pyplot or change backends."""
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 7), dpi=100)
    return draw_field_figure(figure, data, kind, tr)


def save_field_plots(solution, output_dir: str | Path, gradient_axis: str,
                     b0_direction: str = "+y", project_name: str = "gradient",
                     tr: Callable | None = None) -> dict:
    """Save reusable data and all PNGs, including when GUI/show_plots is false.

    Safe in a worker: FigureCanvasAgg renders directly without a Tk event loop
    or pyplot.show(). The returned dict is suitable for storing in metrics.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    data = field_data_from_solution(solution, gradient_axis, b0_direction, project_name)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Stems normally come from paths.unique_stem; disallow path components here.
    if not project_name or "/" in project_name or "\\" in project_name or project_name in (".", ".."):
        raise ValueError("The field result project name must be a filename stem")
    data_path = output / f"{project_name}_field_data.npz"
    np.savez_compressed(
        data_path,
        format_version=np.asarray(1, dtype=np.int64),
        coords_m=data.coords_m,
        generated_mT_per_A=data.generated_mT_per_A,
        target_mT_per_A=data.target_mT_per_A,
        gradient_axis=np.asarray(data.gradient_axis),
        b0_direction=np.asarray(data.b0_direction),
        project_name=np.asarray(data.project_name),
    )
    png_paths = {}
    for kind in PLOT_KINDS:
        figure = create_field_figure(data, kind, tr)
        canvas = FigureCanvasAgg(figure)
        path = output / f"{project_name}_field_{kind}.png"
        canvas.print_figure(str(path), dpi=150, facecolor="white")
        png_paths[kind] = str(path)
        figure.clear()
    return {"data_path": str(data_path), "png_paths": png_paths}
