"""Tk field-result viewer. Open only from the GUI's main-thread on_done hook."""

from __future__ import annotations

from pathlib import Path
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from .plot_controls import enable_3d_zoom

from halbach_coils.coilgen.field_plots import (
    PLOT_KINDS, PLOT_LABELS, create_field_figure, draw_field_figure, load_field_data,
)


def electrical_result_text(metrics_path, tr=None):
    """Present the final solver result in the completion dialog and field viewer."""
    tr = tr or (lambda text, **values: text.format(**values))
    if not metrics_path or not Path(metrics_path).is_file():
        return ''
    result = json.loads(Path(metrics_path).read_text(encoding='utf-8'))
    fh = result.get('fasthenry', {})
    if fh.get('status') == 'success':
        return tr('Final conductor — FastHenry R: {resistance} Ω | L: {inductance} µH',
                  resistance=f"{fh['resistance_ohm']:.7g}",
                  inductance=f"{fh['inductance_H'] * 1e6:.7g}")
    return tr('Final conductor — FastHenry: {status}. {reason}',
              status=tr(fh.get('status', 'unavailable')), reason=fh.get('reason', ''))


class FieldViewer(tk.Toplevel):
    """Saved result with a plot selector, matplotlib toolbar and rotatable ROI."""

    def __init__(self, parent, data_path: str | Path):
        # Load before creating a window so invalid files cannot leave a blank one.
        self.data = load_field_data(data_path)
        self.data_path = str(Path(data_path).resolve())
        super().__init__(parent)
        self.root = parent.winfo_toplevel()
        self.tr = getattr(self.root, "tr", lambda text, **values: text.format(**values))
        self.geometry("1100x760")
        self.minsize(800, 560)
        self.title(self.tr("Field results") + " — " + self.data.project_name)

        control = ttk.Frame(self, padding=8)
        control.pack(fill="x")
        self.view_label = ttk.Label(control, text=self.tr("View:"))
        self.view_label.pack(side="left", padx=(0, 6))
        self.selection = tk.StringVar()
        self.selector = ttk.Combobox(control, textvariable=self.selection,
                                     state="readonly", width=45)
        self.selector.pack(side="left")
        self.selector.bind("<<ComboboxSelected>>", self._on_selected)
        self.path_label = ttk.Label(self, text=self.data_path, padding=(10, 0))
        self.path_label.pack(fill="x")
        self.metrics_path = str(Path(self.data_path).with_name(
            Path(self.data_path).name.removesuffix('_field_data.npz') + '_final_metrics.json'))
        self.metrics_label = ttk.Label(self, padding=(10, 4), wraplength=1000)
        self.metrics_label.pack(fill='x')

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        self.figure = create_field_figure(self.data, tr=self.tr)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        enable_3d_zoom(self.canvas)
        toolbar_frame = ttk.Frame(self)
        toolbar_frame.pack(side="bottom", fill="x")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame, pack_toolbar=False)
        self.toolbar.pack(fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._current_kind = "comparison"
        self.retranslate()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def retranslate(self):
        """Called by the application after changing its language, if open."""
        self.title(self.tr("Field results") + " — " + self.data.project_name)
        self.view_label.configure(text=self.tr("View:"))
        self.selector.configure(values=[self.tr(PLOT_LABELS[k]) for k in PLOT_KINDS])
        self.selector.current(PLOT_KINDS.index(self._current_kind))
        self.metrics_label.configure(text=electrical_result_text(self.metrics_path, self.tr))
        self._draw()

    def _on_selected(self, _event=None):
        index = self.selector.current()
        if index >= 0:
            self._current_kind = PLOT_KINDS[index]
            self._draw()

    def _draw(self):
        draw_field_figure(self.figure, self.data, self._current_kind, tr=self.tr)
        self.toolbar.update()
        self.canvas.draw_idle()

    def _on_close(self):
        self.figure.clear()
        self.destroy()


def open_field_viewer(parent, source: str | Path | dict | None = None) -> FieldViewer | None:
    """Open an artifact returned by save_field_plots, a data path, or a picker."""
    root = parent.winfo_toplevel()
    tr = getattr(root, "tr", lambda text, **values: text.format(**values))
    if isinstance(source, dict):
        source = source.get("data_path")
    if source is None:
        source = filedialog.askopenfilename(
            parent=parent, title=tr("Select saved field results"),
            filetypes=[(tr("Field result data"), "*_field_data.npz"), ("NPZ", "*.npz")],
        )
    if not source:
        return None
    try:
        return FieldViewer(parent, source)
    except (OSError, ValueError, KeyError) as exc:
        messagebox.showerror(tr("Cannot open field results"), str(exc), parent=parent)
        return None
