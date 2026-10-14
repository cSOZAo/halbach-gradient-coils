"""Compact advanced parameter editor with translated hover explanations."""

from __future__ import annotations

import copy
import tkinter as tk
from tkinter import ttk

from coilgen.advanced import PARAMETERS, apply_advanced_values, config_values, density_summary


_TEXT_ES = {
    'Advanced CoilGen parameters': 'Parámetros avanzados de CoilGen',
    'Show more parameters': 'Mostrar más parámetros',
    'Restore defaults': 'Volver a los valores por defecto',
    'Mesh': 'Malla',
    'Target ROI': 'ROI objetivo',
    'Winding': 'Trazado',
    'Solver': 'Cálculo',
    'Geometry preview updates with valid values. Field and winding changes require a new design run.':
        'La vista geométrica se actualiza con valores válidos. Los cambios del campo y del bobinado requieren ejecutar un nuevo diseño.',
    'Mesh: {vertices:,} vertices / {triangles:,} triangles · ROI: {points:,} stored points':
        'Malla: {vertices:,} vértices / {triangles:,} triángulos · ROI: {points:,} puntos almacenados',
    'Check the highlighted parameter values.': 'Revise los valores de los parámetros.',
}


class _HoverHelp:
    def __init__(self, widget, get_text):
        self.widget = widget
        self.get_text = get_text
        self.pending = None
        self.window = None
        widget.bind('<Enter>', self._enter, add='+')
        widget.bind('<Leave>', self._hide, add='+')
        widget.bind('<ButtonPress>', self._hide, add='+')
        widget.bind('<Destroy>', self._hide, add='+')

    def _enter(self, event):
        self._hide()
        self.position = (event.x_root + 14, event.y_root + 20)
        self.pending = self.widget.after(450, self._show)

    def _show(self):
        self.pending = None
        if not self.widget.winfo_exists():
            return
        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)
        x, y = self.position
        self.window.wm_geometry(f'+{x}+{y}')
        ttk.Label(self.window, text=self.get_text(), padding=8, wraplength=430,
                  relief='solid', borderwidth=1).pack()

    def _hide(self, _event=None):
        if self.pending is not None:
            self.widget.after_cancel(self.pending)
            self.pending = None
        if self.window is not None:
            self.window.destroy()
            self.window = None


class AdvancedControls(ttk.LabelFrame):
    """Edit advanced values without changing the baseline when simply opened.

    on_change() runs after 250 ms of inactivity and only for valid values.
    apply_to(cfg) raises ValueError for invalid values, including collapsed
    controls; load_config(cfg) keeps per-panel presets reproducible.
    """

    def __init__(self, master, cfg=None, *, root=None, on_change=None):
        super().__init__(master, padding=6)
        self.root = root or self.winfo_toplevel()
        self.on_change = on_change
        self._loading = True
        self._pending = None
        self.variables = {}
        self._labels = {}
        self._inputs = {}
        self._tooltips = []
        self._valid = True

        self.expanded = tk.BooleanVar(self, value=False)
        header = ttk.Frame(self)
        header.pack(fill='x')
        self.toggle = ttk.Checkbutton(header, variable=self.expanded, command=self._toggle)
        self.toggle.pack(side='left')
        self.reset_button = ttk.Button(header, command=self.reset_defaults)
        self.reset_button.pack(side='right', padx=(6, 0))
        self.body = ttk.Frame(self)
        self.tabs = ttk.Notebook(self.body)
        self.tabs.pack(fill='x')
        self.groups = {}
        for key, title in (('mesh', 'Mesh'), ('roi', 'Target ROI'),
                           ('winding', 'Winding'), ('solver', 'Solver')):
            group = ttk.Frame(self.tabs, padding=6)
            group.columnconfigure(0, weight=1)
            self.tabs.add(group, text=title)
            self.groups[key] = (group, title)

        rows = {key: 0 for key in self.groups}
        for spec in PARAMETERS:
            group = self.groups[spec.group][0]
            row = rows[spec.group]
            rows[spec.group] += 1
            label = ttk.Label(group)
            label.grid(row=row, column=0, sticky='w', padx=(0, 6), pady=2)
            var = tk.StringVar(self)
            self.variables[spec.path] = var
            if spec.kind == 'choice':
                entry = ttk.Combobox(group, textvariable=var, values=spec.choices,
                                     state='readonly', width=15)
            else:
                entry = ttk.Entry(group, textvariable=var, width=17)
            entry.grid(row=row, column=1, sticky='ew', pady=2)
            self._labels[spec.path] = label
            self._inputs[spec.path] = entry
            for widget in (label, entry):
                self._tooltips.append(_HoverHelp(widget, lambda s=spec: s.help(self._language())))
            var.trace_add('write', self._changed)

        self.summary = ttk.Label(self.body, wraplength=390, justify='left')
        self.summary.pack(fill='x', pady=(6, 2))
        self.notice = ttk.Label(self.body, wraplength=390, justify='left')
        self.notice.pack(fill='x', pady=(2, 0))
        if cfg is None:
            from coilgen.config import Config
            cfg = Config()
        self.load_config(cfg)
        self.bind('<Destroy>', self._destroyed, add='+')

    def _language(self):
        return getattr(getattr(self.root, 'localizer', None), 'language', 'en')

    def _tr(self, text, **values):
        translated = _TEXT_ES.get(text, text) if self._language() == 'es' else text
        return translated.format(**values)

    def _toggle(self):
        if self.expanded.get():
            self.body.pack(fill='x', pady=(4, 0))
        else:
            self.body.pack_forget()
        self._changed()

    def load_config(self, cfg):
        self._loading = True
        self._base_cfg = copy.deepcopy(cfg)
        values = config_values(cfg)
        for spec in PARAMETERS:
            value = values[spec.path]
            if spec.path == 'winding.cut_width':
                text = f'{value * 1000:g}'
            elif spec.kind == 'triple':
                text = ', '.join(str(item) for item in value)
            elif isinstance(value, float):
                text = f'{value:g}'
            else:
                text = str(value)
            self.variables[spec.path].set(text)
        self._loading = False
        self.retranslate()

    def reset_defaults(self):
        """Reset only this catalog; retain the main design and B0 selections."""
        cfg = copy.deepcopy(self._base_cfg)
        apply_advanced_values(cfg, {spec.path: copy.deepcopy(spec.default)
                                    for spec in PARAMETERS})
        self.load_config(cfg)
        self._changed()

    def apply_to(self, cfg):
        values = {path: variable.get() for path, variable in self.variables.items()}
        try:
            values['winding.cut_width'] = float(values['winding.cut_width']) / 1000.0
        except ValueError:
            # The backend supplies the same descriptive validation as other fields.
            pass
        apply_advanced_values(cfg, values, language=self._language())

    def _changed(self, *_args):
        if self._loading:
            return
        if self._pending is not None:
            self.after_cancel(self._pending)
        self._pending = self.after(250, self._notify)

    def _notify(self):
        self._pending = None
        self._update_summary()
        if self._valid and self.on_change is not None:
            self.on_change()

    def _update_summary(self):
        iterative = self.variables['advanced.sf_opt_method'].get() == 'minimize'
        for path in ('advanced.minimize_method', 'advanced.maxiter', 'advanced.tol'):
            widget = self._inputs[path]
            widget.configure(state=('readonly' if path.endswith('minimize_method') else 'normal')
                             if iterative else 'disabled')
        try:
            cfg = copy.deepcopy(self._base_cfg)
            self.apply_to(cfg)
            counts = density_summary(cfg)
            text = self._tr('Mesh: {vertices:,} vertices / {triangles:,} triangles · ROI: {points:,} stored points',
                            vertices=counts['vertices'], triangles=counts['triangles'],
                            points=counts['target_points'])
        except (ValueError, TypeError) as exc:
            self._valid = False
            self.summary.configure(text=str(exc), foreground='#a32323')
        else:
            self._valid = True
            self.summary.configure(text=text, foreground='')

    def retranslate(self):
        self.configure(text=self._tr('Advanced CoilGen parameters'))
        self.toggle.configure(text=self._tr('Show more parameters'))
        self.reset_button.configure(text=self._tr('Restore defaults'))
        for spec in PARAMETERS:
            self._labels[spec.path].configure(text=spec.label(self._language()))
        for group, title in self.groups.values():
            self.tabs.tab(group, text=self._tr(title))
        self.notice.configure(text=self._tr(
            'Geometry preview updates with valid values. Field and winding changes require a new design run.'))
        self._update_summary()

    def _destroyed(self, event):
        if event.widget is self and self._pending is not None:
            self.after_cancel(self._pending)
            self._pending = None
