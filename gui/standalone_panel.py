"""
Standalone panel — run a single step (wire / leads / shell) from existing
files.

Action selector chooses which step to run; the required inputs change per
action. Each step runs in a worker thread with its output piped to the log.
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from coilgen.config import Config, apply_custom_shell_dims
from .runner import WorkerRunner
from .units import mm_to_m
from .widgets import (
    PanelBase, build_log_box, build_run_bar, require_output_dir,
    shell_pair_choices, show_failure, show_invalid_params,
)


class StandalonePanel(PanelBase):
    def _build(self):
        top = ttk.LabelFrame(self, text="Standalone (un paso)", padding=10)
        top.pack(fill='x', padx=8, pady=6)

        ttk.Label(top, text="Accion:").grid(row=0, column=0, sticky='w', padx=4, pady=4)
        self.action_var = tk.StringVar(value='wire')
        self.action_combo = ttk.Combobox(
            top, textvariable=self.action_var,
            values=['wire', 'leads', 'shell'], state='readonly', width=10,
        )
        self.action_combo.grid(row=0, column=1, sticky='w')
        self.action_combo.bind('<<ComboboxSelected>>', lambda _e: self._refresh())

        # Common axis selector (wire needs it; leads/shell infer from file)
        ttk.Label(top, text="Eje (para wire):").grid(row=1, column=0, sticky='w', padx=4, pady=4)
        self.axis_var = tk.StringVar(value='y')
        ttk.Combobox(top, textvariable=self.axis_var, values=['x', 'y', 'z'],
                     state='readonly', width=6).grid(row=1, column=1, sticky='w')

        ttk.Label(top, text="Tikhonov (wire):").grid(row=2, column=0, sticky='w', padx=4, pady=4)
        self.tikhonov_var = tk.StringVar(value='2500')
        ttk.Entry(top, textvariable=self.tikhonov_var, width=10).grid(row=2, column=1, sticky='w')
        ttk.Label(top, text="Niveles (wire):").grid(row=2, column=2, sticky='w', padx=4)
        self.levels_var = tk.StringVar(value='26')
        ttk.Entry(top, textvariable=self.levels_var, width=8).grid(row=2, column=3, sticky='w')
        ttk.Label(top, text="Radio externo [mm]:").grid(row=3, column=0, sticky='w', padx=4, pady=4)
        self.radius_var = tk.StringVar(value='150')
        ttk.Entry(top, textvariable=self.radius_var, width=10).grid(row=3, column=1, sticky='w')

        # File inputs (per action)
        self.file_frame = ttk.LabelFrame(self, text="Archivos", padding=8)
        self.file_frame.pack(fill='x', padx=8, pady=4)

        self.input_stl_var = tk.StringVar(value='')
        ttk.Label(self.file_frame, text="STL entrada:").grid(row=0, column=0, sticky='w', padx=4, pady=4)
        ttk.Entry(self.file_frame, textvariable=self.input_stl_var, width=50).grid(row=0, column=1, sticky='we', padx=4)
        ttk.Button(self.file_frame, text="Examinar...", command=lambda: self._pick('input')).grid(row=0, column=2, padx=4)

        self._shell_pair_by_label, pair_labels, default_pair = shell_pair_choices()
        self.shell_pair_var = tk.StringVar(value=default_pair)
        ttk.Label(self.file_frame, text="Par STL (assets/shells):").grid(
            row=1, column=0, sticky='w', padx=4, pady=4)
        self.shell_pair_combo = ttk.Combobox(
            self.file_frame, textvariable=self.shell_pair_var,
            values=pair_labels, state='readonly', width=16,
        )
        self.shell_pair_combo.grid(row=1, column=1, sticky='w')
        self.file_frame.columnconfigure(1, weight=1)

        self._refresh()

        # Run + log
        self.progress = build_run_bar(self, "Correr paso", self._on_run)
        self.log_text = build_log_box(self, height=16)

        self.runner = WorkerRunner(self.log_text, self.progress, self.root)

    def _refresh(self):
        action = self.action_var.get()
        # wire: needs axis/tikhonov/levels/radius, no input STL
        # leads: needs input wire STL
        # shell: needs input with-leads STL + layer
        if action == 'wire':
            self.input_stl_var.set('')
        # labels handled by context; user fills what is relevant

    def _pick(self, which):
        path = filedialog.askopenfilename(
            title="Seleccionar STL", filetypes=[("STL", "*.stl"), ("All", "*.*")])
        if path:
            self.input_stl_var.set(path)

    def _on_run(self):
        out_dir = require_output_dir(self.get_output_dir)
        if not out_dir:
            return
        action = self.action_var.get()
        try:
            cfg = Config(gradient_axis=self.axis_var.get(), show_plots=False)
            cfg.tikhonov_factor = float(self.tikhonov_var.get())
            cfg.num_levels = int(self.levels_var.get())
            cfg.cylinder.radius = mm_to_m(self.radius_var.get())
            cfg.output_dir = out_dir
            if action == 'shell':
                label = self.shell_pair_var.get().strip()
                pair = self._shell_pair_by_label.get(label)
                if pair is None:
                    raise ValueError(
                        "No hay un par STL seleccionado en assets/shells.")
                apply_custom_shell_dims(cfg, pair[0], pair[1])
        except (ValueError, KeyError) as exc:
            show_invalid_params(exc)
            return

        def _target():
            if action == 'wire':
                from coilgen.gradient import run_gradient
                sol, metrics, overlap = run_gradient(cfg, output_dir=out_dir)
                return out_dir, overlap
            elif action == 'leads':
                from coilgen.leads import run_leads
                in_stl = self.input_stl_var.get().strip()
                if not in_stl or not os.path.isfile(in_stl):
                    raise FileNotFoundError("Seleccione un STL de wire valido.")
                paths = run_leads(cfg, input_stl=in_stl)
                return paths, None
            else:  # shell
                from coilgen.shell import run_shell
                in_stl = self.input_stl_var.get().strip()
                if not in_stl or not os.path.isfile(in_stl):
                    raise FileNotFoundError("Seleccione un STL with-leads valido.")
                out_dir2, shell_paths = run_shell(cfg, wire_with_leads_stl=in_stl,
                                                  output_dir=out_dir)
                return shell_paths, None

        def _on_done(result, err):
            if err is not None:
                show_failure("Standalone", err)
            else:
                payload, overlap = result
                msg = f"Listo.\n{payload}"
                if overlap is not None and overlap.n_collisions > 0:
                    msg += (f"\n\nColisiones detectadas: {overlap.n_collisions} "
                            f"(min {overlap.min_distance_m*1000:.3f} mm).")
                messagebox.showinfo("Standalone", msg)

        self.runner.run(_target, on_done=_on_done)
