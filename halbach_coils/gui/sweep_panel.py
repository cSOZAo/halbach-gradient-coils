"""
Tikhonov sweep panel.

Select dimensions and Tikhonov sweep limits, toggle fine adjustment, and run
the headless sweep. Results are shown in a table and saved as CSV/TXT in the
output directory. Suggested widened range for Gz is pre-filled.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional

from coilgen.config import Config
from coilgen import sweep as sweep_mod
from .runner import WorkerRunner
from .units import mm_to_m


class SweepPanel(ttk.Frame):
    def __init__(self, master, get_output_dir, set_output_dir, root):
        super().__init__(master)
        self.get_output_dir = get_output_dir
        self.set_output_dir = set_output_dir
        self.root = root
        self.runner: Optional[WorkerRunner] = None
        self._build()

    def _build(self):
        form = ttk.LabelFrame(self, text="Tikhonov sweep", padding=10)
        form.pack(fill='x', padx=8, pady=6)

        ttk.Label(form, text="Axis:").grid(row=0, column=0, sticky='w', padx=4, pady=4)
        self.axis_var = tk.StringVar(value='z')
        axis_combo = ttk.Combobox(form, textvariable=self.axis_var,
                                  values=['x', 'y', 'z'], state='readonly', width=6)
        axis_combo.grid(row=0, column=1, sticky='w')
        axis_combo.bind('<<ComboboxSelected>>', lambda _e: self._fill_defaults())

        ttk.Label(form, text="Outer radius [mm]:").grid(row=1, column=0, sticky='w', padx=4, pady=4)
        self.radius_var = tk.StringVar(value='150')
        ttk.Entry(form, textvariable=self.radius_var, width=10).grid(row=1, column=1, sticky='w')
        ttk.Label(form, text="Height [mm]:").grid(row=1, column=2, sticky='w', padx=4)
        self.height_var = tk.StringVar(value='430')
        ttk.Entry(form, textvariable=self.height_var, width=10).grid(row=1, column=3, sticky='w')
        ttk.Label(form, text="Levels:").grid(row=2, column=0, sticky='w', padx=4, pady=4)
        self.levels_var = tk.StringVar(value='26')
        ttk.Entry(form, textvariable=self.levels_var, width=8).grid(row=2, column=1, sticky='w')

        ttk.Label(form, text="Tikhonov min:").grid(row=3, column=0, sticky='w', padx=4, pady=4)
        self.tk_min_var = tk.StringVar(value='1')
        ttk.Entry(form, textvariable=self.tk_min_var, width=10).grid(row=3, column=1, sticky='w')
        ttk.Label(form, text="Tikhonov max:").grid(row=3, column=2, sticky='w', padx=4)
        self.tk_max_var = tk.StringVar(value='1000000')
        ttk.Entry(form, textvariable=self.tk_max_var, width=12).grid(row=3, column=3, sticky='w')
        ttk.Label(form, text="Coarse points:").grid(row=4, column=0, sticky='w', padx=4, pady=4)
        self.n_coarse_var = tk.StringVar(value='12')
        ttk.Entry(form, textvariable=self.n_coarse_var, width=8).grid(row=4, column=1, sticky='w')
        self.fine_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(form, text="Fine adjustment", variable=self.fine_var).grid(row=4, column=2, columnspan=2, sticky='w')

        self._fill_defaults()

        bar = ttk.Frame(self)
        bar.pack(fill='x', padx=8, pady=4)
        ttk.Button(bar, text="Run sweep", command=self._on_run).pack(side='left')
        self.progress = ttk.Progressbar(bar, mode='determinate', maximum=100, length=300)
        self.progress.pack(side='left', padx=10)

        # Results table
        res = ttk.LabelFrame(self, text="Results", padding=4)
        res.pack(fill='both', expand=True, padx=8, pady=4)
        cols = ('phase', 'tikhonov', 'slope', 'mean_error', 'rmse')
        self.tree = ttk.Treeview(res, columns=cols, show='headings', height=10)
        self._column_labels = {
            'phase': 'Phase', 'tikhonov': 'Tikhonov', 'slope': 'Slope',
            'mean_error': 'Mean error', 'rmse': 'RMSE',
        }
        for c, w in zip(cols, (60, 90, 110, 90, 110)):
            self.tree.heading(c, text=self.root.tr(self._column_labels[c]))
            self.tree.column(c, width=w, anchor='e')
        self.tree.pack(side='left', fill='both', expand=True)
        sb = ttk.Scrollbar(res, command=self.tree.yview)
        sb.pack(side='right', fill='y')
        self.tree.configure(yscrollcommand=sb.set)

        log_frame = ttk.LabelFrame(self, text="Log", padding=4)
        log_frame.pack(fill='both', expand=True, padx=8, pady=4)
        self.log_text = tk.Text(log_frame, height=8, wrap='word')
        self.log_text.pack(side='left', fill='both', expand=True)
        lsb = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        lsb.pack(side='right', fill='y')
        self.log_text.configure(yscrollcommand=lsb.set)

        self.runner = WorkerRunner(self.log_text, self.progress, self.root)

    def _fill_defaults(self):
        axis = self.axis_var.get()
        d_min, d_max, d_n = sweep_mod.DEFAULT_RANGES.get(axis, (1.0, 100000.0, 10))
        self.tk_min_var.set(str(int(d_min)))
        self.tk_max_var.set(str(int(d_max)))
        self.n_coarse_var.set(str(d_n))

    def _on_run(self):
        out_dir = self.get_output_dir()
        if not out_dir:
            messagebox.showwarning(self.root.tr("Missing directory"),
                                   self.root.tr("Select an output directory first."))
            return
        try:
            cfg = Config(gradient_axis=self.axis_var.get(),
                         num_levels=int(self.levels_var.get()),
                         show_plots=False, overlap_warn=False)
            cfg.cylinder.radius = mm_to_m(self.radius_var.get())
            cfg.cylinder.height = mm_to_m(self.height_var.get())
            cfg.sweep.fine = self.fine_var.get()
        except (ValueError, KeyError) as exc:
            messagebox.showerror(self.root.tr("Invalid parameters"), str(exc))
            return

        tk_min = float(self.tk_min_var.get())
        tk_max = float(self.tk_max_var.get())
        n_coarse = int(self.n_coarse_var.get())

        # Clear table
        for item in self.tree.get_children():
            self.tree.delete(item)

        def _on_progress(phase, tk, event, row=None):
            # called from worker thread -> schedule UI update on main thread
            if event == 'done' and row is not None:
                self.root.after(0, lambda r=row: self.tree.insert(
                    '', 'end',
                    values=(r['Fase'], r['Tikhonov'],
                            f"{r['Pendiente_mT_per_m_per_A']:.4g}",
                            f"{r['Error_Medio_pct']:.4g}",
                            f"{r['RMSE_per_range_mT_per_m_per_A']:.4g}")))

        def _target():
            return sweep_mod.run_tikhonov_sweep(
                cfg, tk_min=tk_min, tk_max=tk_max, n_coarse=n_coarse,
                fine=self.fine_var.get(), output_base_dir=out_dir,
                on_progress=_on_progress,
            )

        def _on_done(result, err):
            if err is not None:
                messagebox.showerror(
                    self.root.tr("Sweep"), self.root.tr("Failed: {error}", error=err))
            elif result is not None:
                bs = result.best_slope
                be = result.best_error
                messagebox.showinfo(
                    self.root.tr("Sweep complete"),
                    self.root.tr(
                        "Best slope: tk={tk} ({value:.4g} mT/(m.A))\nLowest error: tk={error_tk} ({error:.4g} %)\nCSV: {path}",
                        tk=bs['Tikhonov'],
                        value=bs['Pendiente_mT_per_m_per_A'],
                        error_tk=be['Tikhonov'],
                        error=be['Error_Medio_pct'],
                        path=result.csv_path))

        self.runner.run(_target, on_done=_on_done)

    def retranslate(self):
        for column, source in self._column_labels.items():
            self.tree.heading(column, text=self.root.tr(source))
