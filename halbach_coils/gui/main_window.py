"""
Main GUI window — mode selector (Pipeline / Standalone / Tikhonov sweep),
shared output-directory field, and application language menu.

From the repository root, run with
``.\\.venv\\Scripts\\python.exe halbach_coils\\run_gui.py``.
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, ttk

from .pipeline_panel import PipelinePanel
from .standalone_panel import StandalonePanel
from .sweep_panel import SweepPanel
from .i18n import DEFAULT_LANGUAGE, LANGUAGES, Localizer


class CoilGenApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.localizer = Localizer(DEFAULT_LANGUAGE)
        self.language_var = tk.StringVar(value=DEFAULT_LANGUAGE)
        self.b0_var = tk.StringVar(value='+y')
        self._active_job_runner = None
        self._closing = False
        self.title("pyCoilGen - MRI gradient coil design")
        self.geometry("1280x820")
        self.minsize(960, 640)

        self._build_menu()

        # Shared output directory
        top = ttk.Frame(self, padding=8)
        top.pack(fill='x')
        ttk.Label(top, text="Output directory:").pack(side='left', padx=4)
        self.output_dir_var = tk.StringVar(value='')
        ttk.Entry(top, textvariable=self.output_dir_var, width=50).pack(side='left', fill='x', expand=True, padx=4)
        ttk.Button(top, text="Browse...", command=self._pick_output_dir).pack(side='left', padx=4)

        # Mode notebook
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=8, pady=6)

        self.pipeline_panel = PipelinePanel(
            self.notebook, self._get_output_dir, self._set_output_dir, self)
        self.notebook.add(self.pipeline_panel, text="Pipeline")

        self.standalone_panel = StandalonePanel(
            self.notebook, self._get_output_dir, self._set_output_dir, self)
        self.notebook.add(self.standalone_panel, text="Standalone")

        self.sweep_panel = SweepPanel(
            self.notebook, self._get_output_dir, self._set_output_dir, self)
        self.notebook.add(self.sweep_panel, text="Tikhonov sweep")

        # Capture every widget's English source text after all panels exist.
        self._refresh_language()
        self.b0_var.trace_add('write', self._on_b0_changed)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def tr(self, source: str, **values) -> str:
        """Translate an English GUI source string in the active language."""
        return self.localizer.translate(source, **values)

    def _build_menu(self):
        self.menu_bar = tk.Menu(self, tearoff=False)
        self.settings_menu = tk.Menu(self.menu_bar, tearoff=False)
        self.language_menu = tk.Menu(self.settings_menu, tearoff=False)
        for code, label in LANGUAGES.items():
            self.language_menu.add_radiobutton(
                label=label,
                value=code,
                variable=self.language_var,
                command=self._on_language_changed,
            )
        self.settings_menu.add_cascade(label="Language", menu=self.language_menu)
        self.menu_bar.add_cascade(label="Settings", menu=self.settings_menu)
        self.b0_menu = tk.Menu(self.menu_bar, tearoff=False)
        for direction in ('+x', '-x', '+y', '-y', '+z', '-z'):
            self.b0_menu.add_radiobutton(
                label=direction.upper(), value=direction, variable=self.b0_var)
        self.menu_bar.add_cascade(label="B₀: +Y", menu=self.b0_menu)
        self.results_menu = tk.Menu(self.menu_bar, tearoff=False)
        self.results_menu.add_command(label="Open field plots...", command=self._open_field_plots)
        self.menu_bar.add_cascade(label="Results", menu=self.results_menu)
        self.configure(menu=self.menu_bar)

    def _on_b0_changed(self, *_):
        direction = self.b0_var.get()
        self.menu_bar.entryconfigure(1, label=f"B₀: {direction.upper()}")
        if hasattr(self, 'pipeline_panel'):
            self.pipeline_panel._redraw_previews()

    def _open_field_plots(self):
        path = filedialog.askopenfilename(
            parent=self, title=self.tr("Open field plots..."),
            initialdir=self._get_output_dir() or None,
            filetypes=[(self.tr("Field data"), '*_field_data.npz')])
        if path:
            self.show_field_plots(path)

    def show_field_plots(self, data):
        from .field_viewer import open_field_viewer
        open_field_viewer(self, data)

    def _on_language_changed(self):
        self.localizer.set_language(self.language_var.get())
        self._refresh_language()

    def _refresh_language(self):
        self.title(self.tr("pyCoilGen - MRI gradient coil design"))
        self.menu_bar.entryconfigure(0, label=self.tr("Settings"))
        self.settings_menu.entryconfigure(0, label=self.tr("Language"))
        self.menu_bar.entryconfigure(2, label=self.tr("Results"))
        self.results_menu.entryconfigure(0, label=self.tr("Open field plots..."))
        self.notebook.tab(self.pipeline_panel, text="Pipeline")
        self.notebook.tab(self.standalone_panel, text="Standalone")
        self.notebook.tab(self.sweep_panel, text=self.tr("Tikhonov sweep"))
        self.localizer.refresh_widget_tree(self)
        for panel in (self.pipeline_panel, self.standalone_panel, self.sweep_panel):
            retranslate = getattr(panel, "retranslate", None)
            if retranslate is not None:
                retranslate()
        from .field_viewer import FieldViewer
        for child in self.winfo_children():
            if isinstance(child, FieldViewer):
                child.retranslate()

    def _get_output_dir(self) -> str:
        return self.output_dir_var.get().strip()

    def _set_output_dir(self, path: str):
        self.output_dir_var.set(path)

    def _pick_output_dir(self):
        path = filedialog.askdirectory(title=self.tr("Select output directory"))
        if path:
            self.output_dir_var.set(os.path.abspath(path))

    def _on_close(self):
        self._closing = True
        runner = self._active_job_runner
        if runner is not None and runner.is_running:
            runner.request_stop()
            self.after(60, self._finish_close)
        else:
            self._finish_close()

    def _finish_close(self):
        runner = self._active_job_runner
        if runner is not None and runner.is_running:
            self.after(60, self._finish_close)
            return
        for panel in (self.pipeline_panel, self.standalone_panel, self.sweep_panel):
            panel.runner.stop_polling()
        self.destroy()

    def on_job_state_changed(self):
        panels = (self.pipeline_panel, self.standalone_panel, self.sweep_panel)
        busy = any(panel.runner is not None and panel.runner.is_running for panel in panels)
        for panel in panels:
            panel._set_running(panel.runner is not None and panel.runner.is_running)
            panel.run_btn.configure(state='disabled' if busy or self._closing else 'normal')


def main():
    app = CoilGenApp()
    app.mainloop()


if __name__ == '__main__':
    main()
