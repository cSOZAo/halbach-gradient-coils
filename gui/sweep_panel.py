"""
Tikhonov sweep panel.

Select dimensions and Tikhonov sweep limits, toggle fine adjustment, and run
the headless sweep. Results are shown in a table and saved as CSV/TXT in the
output directory. Suggested widened range for Gz is pre-filled.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from coilgen.config import Config
from coilgen import sweep as sweep_mod
from .runner import WorkerRunner
from .units import mm_to_m
from .widgets import (
    PanelBase, attach_vscrollbar, build_log_box, build_run_bar,
    require_output_dir, show_failure, show_invalid_params,
)


class SweepPanel(PanelBase):
    def _build(self):
        form = ttk.LabelFrame(self, text="Barrido Tikhonov", padding=10)
        form.pack(fill='x', padx=8, pady=6)

        ttk.Label(form, text="Eje:").grid(row=0, column=0, sticky='w', padx=4, pady=4)
        self.axis_var = tk.StringVar(value='z')
        axis_combo = ttk.Combobox(form, textvariable=self.axis_var,
                                  values=['x', 'y', 'z'], state='readonly', width=6)
        axis_combo.grid(row=0, column=1, sticky='w')
        axis_combo.bind('<<ComboboxSelected>>', lambda _e: self._fill_defaults())

        ttk.Label(form, text="Radio externo [mm]:").grid(row=1, column=0, sticky='w', padx=4, pady=4)
        self.radius_var = tk.StringVar(value='150')
        ttk.Entry(form, textvariable=self.radius_var, width=10).grid(row=1, column=1, sticky='w')
        ttk.Label(form, text="Altura [mm]:").grid(row=1, column=2, sticky='w', padx=4)
        self.height_var = tk.StringVar(value='430')
        ttk.Entry(form, textvariable=self.height_var, width=10).grid(row=1, column=3, sticky='w')
        ttk.Label(form, text="Niveles:").grid(row=2, column=0, sticky='w', padx=4, pady=4)
        self.levels_var = tk.StringVar(value='26')
        ttk.Entry(form, textvariable=self.levels_var, width=8).grid(row=2, column=1, sticky='w')

        ttk.Label(form, text="Tikhonov min:").grid(row=3, column=0, sticky='w', padx=4, pady=4)
        self.tk_min_var = tk.StringVar(value='1')
        ttk.Entry(form, textvariable=self.tk_min_var, width=10).grid(row=3, column=1, sticky='w')
        ttk.Label(form, text="Tikhonov max:").grid(row=3, column=2, sticky='w', padx=4)
        self.tk_max_var = tk.StringVar(value='1000000')
        ttk.Entry(form, textvariable=self.tk_max_var, width=12).grid(row=3, column=3, sticky='w')
        ttk.Label(form, text="N puntos grueso:").grid(row=4, column=0, sticky='w', padx=4, pady=4)
        self.n_coarse_var = tk.StringVar(value='12')
        ttk.Entry(form, textvariable=self.n_coarse_var, width=8).grid(row=4, column=1, sticky='w')
        self.fine_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(form, text="Ajuste fino", variable=self.fine_var).grid(row=4, column=2, columnspan=2, sticky='w')

        self._fill_defaults()

        self.progress = build_run_bar(self, "Correr barrido", self._on_run)

        # Results table
        res = ttk.LabelFrame(self, text="Resultados", padding=4)
        res.pack(fill='both', expand=True, padx=8, pady=4)
        cols = ('Fase', 'Tikhonov', 'Pendiente', 'ErrorMedio', 'RMSE')
        self.tree = ttk.Treeview(res, columns=cols, show='headings', height=10)
        for c, w in zip(cols, (60, 90, 110, 90, 110)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor='e')
        attach_vscrollbar(res, self.tree)

        self.log_text = build_log_box(self, height=8)

        self.runner = WorkerRunner(self.log_text, self.progress, self.root)

    def _fill_defaults(self):
        axis = self.axis_var.get()
        d_min, d_max, d_n = sweep_mod.DEFAULT_RANGES.get(axis, (1.0, 100000.0, 10))
        self.tk_min_var.set(str(int(d_min)))
        self.tk_max_var.set(str(int(d_max)))
        self.n_coarse_var.set(str(d_n))

    def _on_run(self):
        out_dir = require_output_dir(self.get_output_dir)
        if not out_dir:
            return
        try:
            cfg = Config(gradient_axis=self.axis_var.get(),
                         num_levels=int(self.levels_var.get()),
                         show_plots=False, overlap_warn=False)
            cfg.cylinder.radius = mm_to_m(self.radius_var.get())
            cfg.cylinder.height = mm_to_m(self.height_var.get())
            cfg.sweep.fine = self.fine_var.get()
        except (ValueError, KeyError) as exc:
            show_invalid_params(exc)
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
                show_failure("Barrido", err)
            elif result is not None:
                bs = result.best_slope
                be = result.best_error
                messagebox.showinfo(
                    "Barrido completo",
                    f"Mejor pendiente: tk={bs['Tikhonov']} "
                    f"({bs['Pendiente_mT_per_m_per_A']:.4g} mT/(m.A))\n"
                    f"Menor error: tk={be['Tikhonov']} "
                    f"({be['Error_Medio_pct']:.4g} %)\n"
                    f"CSV: {result.csv_path}")

        self.runner.run(_target, on_done=_on_done)
