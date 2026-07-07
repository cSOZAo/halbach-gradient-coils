"""
Shared background-runner infrastructure for the GUI.

Runs a blocking callable in a worker thread while piping its stdout/stderr to
a queue that the Tk main loop drains into a ``tk.Text`` log widget. Keeps the
UI responsive (pyCoilGen runs take minutes) and lets panels request an
accept/reject decision (e.g. wire overlap) via a thread-safe dialog hook.
"""

from __future__ import annotations

import io
import queue
import sys
import threading
import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional


class _StreamRedirect(io.TextIOBase):
    """File-like object that enqueues lines for the GUI to drain."""

    def __init__(self, q: queue.Queue):
        self.q = q
        self._buf = ''

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buf += s
        while '\n' in self._buf:
            line, self._buf = self._buf.split('\n', 1)
            self.q.put(line + '\n')
        return len(s)

    def flush(self):  # noqa: D401
        if self._buf:
            self.q.put(self._buf)
            self._buf = ''


class WorkerRunner:
    """
    Owns the log queue and worker thread for a panel.

    Usage in a panel:
        self.runner = WorkerRunner(self.log_text, self.progress, self.root)
        self.runner.run(target_fn, args=(...), on_done=self._on_done,
                        ask_user=self._ask_user)
    """

    def __init__(self, log_widget: tk.Text, progress: Optional[ttk.Progressbar],
                 root: tk.Tk):
        self.log_widget = log_widget
        self.progress = progress
        self.root = root
        self.log_q: queue.Queue = queue.Queue()
        self.user_q: queue.Queue = queue.Queue()
        self.worker: Optional[threading.Thread] = None
        self._poll_id = None
        self.stop_requested = threading.Event()

    # ----- log draining ---------------------------------------------------

    def start_polling(self):
        if self._poll_id is None:
            self._drain_log()

    def _drain_log(self):
        try:
            while True:
                line = self.log_q.get_nowait()
                self.log_widget.insert('end', line)
                self.log_widget.see('end')
        except queue.Empty:
            pass
        self._poll_id = self.root.after(120, self._drain_log)

    def stop_polling(self):
        if self._poll_id is not None:
            self.root.after_cancel(self._poll_id)
            self._poll_id = None

    # ----- user decision hook (overlap accept/reject) --------------------

    def ask_user(self, question: str) -> bool:
        """
        Called from the worker thread. Puts a request on user_q, then blocks
        until the main thread answers. The main thread must call
        ``answer_user()`` from a dialog handler.
        """
        self.user_q.put(('ask', question))
        answer = self.user_q.get()  # blocks until answered
        return bool(answer)

    def answer_user(self, value: bool):
        self.user_q.put(bool(value))

    # ----- run ------------------------------------------------------------

    def run(self, target: Callable, args: tuple = (), kwargs: dict | None = None,
            on_done: Optional[Callable] = None, ask_user: Optional[Callable] = None):
        if self.worker is not None and self.worker.is_alive():
            self.log(">> a job is already running\n")
            return
        self.stop_requested.clear()
        self.log_widget.delete('1.0', 'end')
        if self.progress is not None:
            self.progress['value'] = 0
        self.start_polling()

        kwargs = kwargs or {}

        def _worker():
            old_stdout, old_stderr = sys.stdout, sys.stderr
            redirect = _StreamRedirect(self.log_q)
            sys.stdout = redirect
            sys.stderr = redirect
            # If the panel passed an ask_user callback, wire it through.
            if ask_user is not None:
                kwargs.setdefault('ask_user', ask_user)
            try:
                result = target(*args, **kwargs)
                self.log_q.put('\n>> DONE\n')
                if on_done is not None:
                    self.root.after(0, lambda: on_done(result, None))
            except Exception as exc:  # surface to the log + on_done
                self.log_q.put(f'\n>> ERROR: {exc}\n')
                if on_done is not None:
                    err = exc
                    self.root.after(0, lambda: on_done(None, err))
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr
                redirect.flush()
                if self.progress is not None:
                    self.root.after(0, lambda: self.progress.configure(value=100))

        self.worker = threading.Thread(target=_worker, daemon=True)
        self.worker.start()

    def log(self, msg: str):
        self.log_q.put(msg)

    def request_stop(self):
        """Ask the running job to stop at its next cooperative checkpoint."""
        self.stop_requested.set()
        self.log(">> stop requested; waiting for the current step to finish\n")

    def should_stop(self) -> bool:
        return self.stop_requested.is_set()
