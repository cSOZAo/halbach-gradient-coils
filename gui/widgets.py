"""
Shared scaffolding for the GUI panels.

- ``PanelBase``: the frame wiring every panel repeats (output-dir accessors,
  root reference, worker runner slot).
- Widget builders for the log box, the run bar, and the shell-pair combo.
- The message boxes the panels show for the same conditions.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable, Dict, List, Optional, Tuple

from coilgen.config import list_shell_pairs
from .runner import WorkerRunner


class PanelBase(ttk.Frame):
    """Notebook page that runs steps through a :class:`WorkerRunner`."""

    def __init__(self, master, get_output_dir, set_output_dir, root):
        super().__init__(master)
        self.get_output_dir = get_output_dir
        self.set_output_dir = set_output_dir
        self.root = root
        self.runner: Optional[WorkerRunner] = None
        self._build()

    def _build(self):  # pragma: no cover - implemented by each panel
        raise NotImplementedError


def attach_vscrollbar(parent, widget) -> ttk.Scrollbar:
    """Pack *widget* with a vertical scrollbar on its right."""
    widget.pack(side='left', fill='both', expand=True)
    sb = ttk.Scrollbar(parent, command=widget.yview)
    sb.pack(side='right', fill='y')
    widget.configure(yscrollcommand=sb.set)
    return sb


def build_log_box(parent, height: int = 16, title: str = "Log") -> tk.Text:
    """Labelled, scrollable text widget for the worker log."""
    frame = ttk.LabelFrame(parent, text=title, padding=4)
    frame.pack(fill='both', expand=True, padx=8, pady=6)
    log_text = tk.Text(frame, height=height, wrap='word')
    attach_vscrollbar(frame, log_text)
    return log_text


def build_run_bar(parent, text: str, command: Callable[[], None],
                  length: int = 300) -> ttk.Progressbar:
    """Run button plus determinate progress bar."""
    bar = ttk.Frame(parent)
    bar.pack(fill='x', padx=8, pady=4)
    ttk.Button(bar, text=text, command=command).pack(side='left')
    progress = ttk.Progressbar(bar, mode='determinate', maximum=100,
                               length=length)
    progress.pack(side='left', padx=10)
    return progress


def shell_pair_choices() -> Tuple[Dict[str, Tuple[str, str]], List[str], str]:
    """
    Discover ``assets/shells`` half pairs for a combobox.

    Returns ``(pair_by_label, labels, default_label)``; the default is the
    second pair when available (the usual working layer).
    """
    pairs = list_shell_pairs()
    pair_by_label = {label: (a, b) for label, a, b in pairs}
    labels = [label for label, _, _ in pairs]
    if len(labels) > 1:
        default = labels[1]
    elif labels:
        default = labels[0]
    else:
        default = ''
    return pair_by_label, labels, default


def require_output_dir(get_output_dir: Callable[[], str]) -> str:
    """Return the shared output directory, warning the user when unset."""
    out_dir = get_output_dir()
    if not out_dir:
        messagebox.showwarning("Falta directorio",
                               "Seleccione un directorio de salida primero.")
        return ''
    return out_dir


def show_invalid_params(exc: Exception) -> None:
    messagebox.showerror("Parametros invalidos", str(exc))


def show_failure(title: str, exc: object) -> None:
    messagebox.showerror(title, f"Fallo: {exc}")
