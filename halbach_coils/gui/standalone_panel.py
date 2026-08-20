"""
Standalone panel — run a single step (wire / leads / shell) from existing
files.

Action selector chooses which step to run; the required inputs change per
action. Each step runs in a worker thread with its output piped to the log.
"""

from __future__ import annotations

import math
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from coilgen.config import (
    CONDUCTOR_MATERIAL_RESISTIVITY,
    Config,
    apply_custom_shell_dims,
    list_shell_pairs,
)
from coilgen.paths import infer_gradient_axis
from .runner import WorkerRunner
from .units import mm_to_m


class StandalonePanel(ttk.Frame):
    def __init__(self, master, get_output_dir, set_output_dir, root):
        super().__init__(master)
        self.get_output_dir = get_output_dir
        self.set_output_dir = set_output_dir
        self.root = root
        self.runner: Optional[WorkerRunner] = None
        self._build()

    def _build(self):
        top = ttk.LabelFrame(self, text="Standalone (one step)", padding=10)
        top.pack(fill='x', padx=8, pady=6)

        ttk.Label(top, text="Action:").grid(row=0, column=0, sticky='w', padx=4, pady=4)
        self.action_var = tk.StringVar(value='wire')
        self.action_combo = ttk.Combobox(
            top, textvariable=self.action_var,
            values=['wire', 'leads', 'shell'], state='readonly', width=10,
        )
        self.action_combo.grid(row=0, column=1, sticky='w')
        self.action_combo.bind('<<ComboboxSelected>>', lambda _e: self._refresh())

        # For downstream steps this is automatically inferred from generated
        # filenames, while remaining editable for arbitrary user-named STLs.
        ttk.Label(top, text="Gradient axis:").grid(row=1, column=0, sticky='w', padx=4, pady=4)
        self.axis_var = tk.StringVar(value='x')
        ttk.Combobox(top, textvariable=self.axis_var, values=['x', 'y', 'z'],
                     state='readonly', width=6).grid(row=1, column=1, sticky='w')

        ttk.Label(top, text="Tikhonov (wire):").grid(row=2, column=0, sticky='w', padx=4, pady=4)
        self.tikhonov_var = tk.StringVar(value='2500')
        ttk.Entry(top, textvariable=self.tikhonov_var, width=10).grid(row=2, column=1, sticky='w')
        ttk.Label(top, text="Levels (wire):").grid(row=2, column=2, sticky='w', padx=4)
        self.levels_var = tk.StringVar(value='12')
        ttk.Entry(top, textvariable=self.levels_var, width=8).grid(row=2, column=3, sticky='w')
        ttk.Label(top, text="Outer radius [mm]:").grid(row=3, column=0, sticky='w', padx=4, pady=4)
        self.radius_var = tk.StringVar(value='39')
        ttk.Entry(top, textvariable=self.radius_var, width=10).grid(row=3, column=1, sticky='w')
        ttk.Label(top, text="Height [mm]:").grid(row=3, column=2, sticky='w', padx=4)
        self.height_var = tk.StringVar(value='170')
        ttk.Entry(top, textvariable=self.height_var, width=10).grid(row=3, column=3, sticky='w')
        ttk.Label(top, text="ROI radius [mm]:").grid(row=4, column=0, sticky='w', padx=4, pady=4)
        self.roi_var = tk.StringVar(value='20')
        ttk.Entry(top, textvariable=self.roi_var, width=10).grid(row=4, column=1, sticky='w')
        ttk.Label(top, text="Conductor material:").grid(
            row=5, column=0, sticky='w', padx=4, pady=4)
        self.material_var = tk.StringVar(value='Cu')
        material_combo = ttk.Combobox(
            top, textvariable=self.material_var,
            values=[*CONDUCTOR_MATERIAL_RESISTIVITY, 'Custom'],
            state='readonly', width=8,
        )
        material_combo.grid(row=5, column=1, sticky='w')
        material_combo.bind(
            '<<ComboboxSelected>>', lambda _event: self._on_material_changed())
        ttk.Label(top, text="Resistivity [ohm.m]:").grid(
            row=5, column=2, sticky='w', padx=4)
        self.resistivity_var = tk.StringVar(
            value=f"{CONDUCTOR_MATERIAL_RESISTIVITY['Cu']:.3g}")
        ttk.Entry(top, textvariable=self.resistivity_var, width=10).grid(
            row=5, column=3, sticky='w')

        # File inputs (per action)
        self.file_frame = ttk.LabelFrame(self, text="Files", padding=8)
        self.file_frame.pack(fill='x', padx=8, pady=4)

        self.input_stl_var = tk.StringVar(value='')
        ttk.Label(self.file_frame, text="Input STL:").grid(row=0, column=0, sticky='w', padx=4, pady=4)
        ttk.Entry(self.file_frame, textvariable=self.input_stl_var, width=50).grid(row=0, column=1, sticky='we', padx=4)
        ttk.Button(self.file_frame, text="Browse...", command=lambda: self._pick('input')).grid(row=0, column=2, padx=4)

        self._shell_pairs = list_shell_pairs()
        self._shell_pair_by_label = {label: (a, b) for label, a, b in self._shell_pairs}
        pair_labels = [label for label, _, _ in self._shell_pairs]
        default_pair = pair_labels[1] if len(pair_labels) > 1 else (
            pair_labels[0] if pair_labels else '')
        self.shell_pair_var = tk.StringVar(value=default_pair)
        ttk.Label(self.file_frame, text="STL pair (assets/shells):").grid(
            row=1, column=0, sticky='w', padx=4, pady=4)
        self.shell_pair_combo = ttk.Combobox(
            self.file_frame, textvariable=self.shell_pair_var,
            values=pair_labels, state='readonly', width=16,
        )
        self.shell_pair_combo.grid(row=1, column=1, sticky='w')
        self.file_frame.columnconfigure(1, weight=1)

        self._refresh()

        # Run + log
        bar = ttk.Frame(self)
        bar.pack(fill='x', padx=8, pady=4)
        ttk.Button(bar, text="Run step", command=self._on_run).pack(side='left')
        self.progress = ttk.Progressbar(bar, mode='determinate', maximum=100, length=300)
        self.progress.pack(side='left', padx=10)

        log_frame = ttk.LabelFrame(self, text="Log", padding=4)
        log_frame.pack(fill='both', expand=True, padx=8, pady=6)
        self.log_text = tk.Text(log_frame, height=16, wrap='word')
        self.log_text.pack(side='left', fill='both', expand=True)
        sb = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        sb.pack(side='right', fill='y')
        self.log_text.configure(yscrollcommand=sb.set)

        self.runner = WorkerRunner(self.log_text, self.progress, self.root)

    def _refresh(self):
        action = self.action_var.get()
        # wire: needs axis/tikhonov/levels/radius, no input STL
        # leads: needs input wire STL
        # shell: needs input with-leads STL + layer
        if action == 'wire':
            self.input_stl_var.set('')
        # labels handled by context; user fills what is relevant

    def _on_material_changed(self):
        resistivity = CONDUCTOR_MATERIAL_RESISTIVITY.get(self.material_var.get())
        if resistivity is not None:
            self.resistivity_var.set(f'{resistivity:.3g}')

    def _pick(self, which):
        path = filedialog.askopenfilename(
            title=self.root.tr("Select STL"),
            filetypes=[("STL", "*.stl"), (self.root.tr("All files"), "*.*")])
        if path:
            self.input_stl_var.set(path)
            inferred_axis = infer_gradient_axis(path)
            if inferred_axis is not None:
                self.axis_var.set(inferred_axis)

    def _on_run(self):
        out_dir = self.get_output_dir()
        if not out_dir:
            messagebox.showwarning(self.root.tr("Missing directory"),
                                   self.root.tr("Select an output directory first."))
            return
        action = self.action_var.get()
        input_stl = self.input_stl_var.get().strip()
        inferred_axis = infer_gradient_axis(input_stl) if input_stl else None
        if inferred_axis is not None:
            self.axis_var.set(inferred_axis)
        try:
            cfg = Config(gradient_axis=self.axis_var.get(), show_plots=False)
            cfg.tikhonov_factor = float(self.tikhonov_var.get())
            cfg.num_levels = int(self.levels_var.get())
            cfg.cylinder.radius = mm_to_m(self.radius_var.get())
            cfg.cylinder.height = mm_to_m(self.height_var.get())
            roi_radius = mm_to_m(self.roi_var.get())
            cfg.target.rx = cfg.target.ry = cfg.target.rz = roi_radius
            resistivity = float(self.resistivity_var.get())
            if not math.isfinite(resistivity) or resistivity <= 0:
                raise ValueError(self.root.tr(
                    "Resistivity must be a positive number."))
            cfg.fasthenry.material = self.material_var.get()
            cfg.fasthenry.specific_conductivity = resistivity
            cfg.output_dir = out_dir
            if action == 'shell':
                label = self.shell_pair_var.get().strip()
                pair = self._shell_pair_by_label.get(label)
                if pair is None:
                    raise ValueError(self.root.tr(
                        "No STL pair is selected in assets/shells."))
                apply_custom_shell_dims(cfg, pair[0], pair[1])
        except (ValueError, KeyError) as exc:
            messagebox.showerror(self.root.tr("Invalid parameters"), str(exc))
            return

        def _target():
            if action == 'wire':
                from coilgen.gradient import run_gradient
                sol, metrics, overlap = run_gradient(cfg, output_dir=out_dir)
                return out_dir, overlap
            elif action == 'leads':
                from coilgen.leads import run_leads
                if not input_stl or not os.path.isfile(input_stl):
                    raise FileNotFoundError(self.root.tr("Select a valid wire STL."))
                paths = run_leads(cfg, input_stl=input_stl)
                return paths, None
            else:  # shell
                from coilgen.shell import run_shell
                if not input_stl or not os.path.isfile(input_stl):
                    raise FileNotFoundError(self.root.tr("Select a valid with-leads STL."))
                out_dir2, shell_paths = run_shell(cfg, wire_with_leads_stl=input_stl,
                                                  output_dir=out_dir)
                return shell_paths, None

        def _on_done(result, err):
            if err is not None:
                messagebox.showerror(
                    "Standalone", self.root.tr("Failed: {error}", error=err))
            else:
                payload, overlap = result
                msg = self.root.tr("Done.\n{payload}", payload=payload)
                if overlap is not None and overlap.n_collisions > 0:
                    msg += self.root.tr(
                        "\n\nAreas with 3 or more cables: {count}. Maximum at one location: {maximum} cables.",
                        count=overlap.n_collisions, maximum=overlap.max_cables)
                messagebox.showinfo("Standalone", msg)

        self.runner.run(_target, on_done=_on_done)

    def retranslate(self):
        """Static widget text is refreshed by the application localizer."""
