"""
Main GUI window — mode selector (Pipeline / Standalone / Barrido Tikhonov)
plus a shared output-directory field that every panel uses.

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


class CoilGenApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("pyCoilGen - Diseno de bobinas de gradiente MRI")
        self.geometry("1280x820")
        self.minsize(960, 640)

        # Shared output directory
        top = ttk.Frame(self, padding=8)
        top.pack(fill='x')
        ttk.Label(top, text="Directorio de salida:").pack(side='left', padx=4)
        self.output_dir_var = tk.StringVar(value='')
        ttk.Entry(top, textvariable=self.output_dir_var, width=50).pack(side='left', fill='x', expand=True, padx=4)
        ttk.Button(top, text="Examinar...", command=self._pick_output_dir).pack(side='left', padx=4)

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
        self.notebook.add(self.sweep_panel, text="Barrido Tikhonov")

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _get_output_dir(self) -> str:
        return self.output_dir_var.get().strip()

    def _set_output_dir(self, path: str):
        self.output_dir_var.set(path)

    def _pick_output_dir(self):
        path = filedialog.askdirectory(title="Seleccionar directorio de salida")
        if path:
            self.output_dir_var.set(os.path.abspath(path))

    def _on_close(self):
        self.destroy()


def main():
    app = CoilGenApp()
    app.mainloop()


if __name__ == '__main__':
    main()
