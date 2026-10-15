"""One terminable GUI calculation per application, isolated in a process."""

from __future__ import annotations

import multiprocessing
import os
import pickle
import queue
import signal
import subprocess
import threading
from typing import Callable, Optional

from .jobs import JOB_KINDS, execute_job, job_output_dir, write_job_status
from halbach_coils.coilgen.process_control import WindowsJob as _WindowsJob


class JobCancelled(Exception):
    """The user stopped the calculation; any output is partial."""



def _terminate_tree(process):
    """Fallback when Windows job containment is unavailable; never blocks Tk."""
    if not process.is_alive():
        return
    try:
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                           capture_output=True, timeout=4,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except (OSError, subprocess.TimeoutExpired):
        pass
    if process.is_alive():
        process.kill()


class WorkerRunner:
    """GUI callbacks stay on Tk's main thread; calculations use spawn.

    run_job(kind, cfg, extra=..., on_done=..., on_progress=..., ask_user=...)
    is the GUI API. run() is retained for importable callables; GUI closures
    must use run_job(). A single root owns the application-wide job lock.
    """

    def __init__(self, log_widget, progress, root):
        self.log_widget, self.progress, self.root = log_widget, progress, root
        self.log_q = queue.Queue()
        self.worker = None
        self._poll_id = None
        self._events = self._answers = None
        self._windows_job = None
        self._completion = None
        self._startup_error = None
        self._on_done = self._on_progress = self._ask_user_callback = None
        self.stop_requested = threading.Event()
        self.job_kind = None
        self.output_dir = ''

    @property
    def is_running(self):
        # Keep the lock until the main-thread poll reports process exit.
        return self.worker is not None

    def _state_changed(self):
        callback = getattr(self.root, 'on_job_state_changed', None)
        if callback is not None:
            callback()

    def run_job(self, kind, cfg, *, extra=None, on_done=None, ask_user=None,
                on_progress=None):
        if kind not in JOB_KINDS:
            raise ValueError(f'Unknown GUI job: {kind}')
        return self._start(kind, cfg, dict(extra or {}), on_done, ask_user, on_progress)

    def run(self, target: Callable, args: tuple = (), kwargs: dict | None = None,
            on_done: Optional[Callable] = None, ask_user: Optional[Callable] = None):
        """Compatibility API; target must be importable/picklable for spawn."""
        return self._start('_callable', None, {
            'target': target, 'args': args, 'kwargs': kwargs or {},
            'ask_user_enabled': ask_user is not None,
        }, on_done, ask_user, None)

    def _start(self, kind, cfg, extra, on_done, ask_user, on_progress):
        active = getattr(self.root, '_active_job_runner', None)
        if active is not None and active.is_running:
            self.log('>> A calculation is already running. Stop it before starting another.\n')
            self.start_polling()
            return False
        self.root._active_job_runner = self
        self.stop_requested.clear()
        self._completion = None
        self._startup_error = None
        self.job_kind, self.output_dir = kind, job_output_dir(kind, cfg, extra)
        self._on_done, self._on_progress, self._ask_user_callback = on_done, on_progress, ask_user
        self.log_widget.delete('1.0', 'end')
        self.log_q = queue.Queue()
        if self.progress is not None:
            self.progress['value'] = 0
        try:
            pickle.dumps((kind, cfg, extra))
            context = multiprocessing.get_context('spawn')
            self._events, self._answers = context.Queue(), context.Queue()
            gate = context.Event()
            self.worker = context.Process(
                target=execute_job,
                args=(kind, cfg, extra, self._events, self._answers, gate, self.output_dir),
                name=f'CoilGen-{kind}',
            )
            self.worker.start()
            if os.name == 'nt':
                try:
                    self._windows_job = _WindowsJob(self.worker.pid)
                except OSError:
                    self._windows_job = None  # taskkill /T fallback
            write_job_status(self.output_dir, 'running', kind=kind, pid=self.worker.pid)
            gate.set()  # child cannot launch tools before containment is ready
        except Exception as exc:
            if self.worker is not None and self.worker.pid is not None:
                self._startup_error = exc
                self.request_stop()
                self._completion = (None, exc)
                self.start_polling()
            else:
                self._finish(None, exc)
            return False
        self._state_changed()
        self.start_polling()
        return True

    def start_polling(self):
        if self._poll_id is None:
            self._poll_id = self.root.after(60, self._drain_log)

    def _drain_log(self):
        self._poll_id = None
        try:
            while True:
                line = self.log_q.get_nowait()
                self.log_widget.insert('end', line)
                self.log_widget.see('end')
        except queue.Empty:
            pass
        if self.worker is not None:
            # Killed queues may contain partial frames. Discard them on Stop
            # so cancellation never waits for a half-written result/log.
            if not self.stop_requested.is_set():
                self._drain_events()
            if not self.worker.is_alive():
                self.worker.join(timeout=0)
                if self.stop_requested.is_set():
                    self._finish(None, self._startup_error or JobCancelled(
                        'Calculation stopped; output is partial.'))
                else:
                    self._drain_events()
                    result, error = self._completion or (
                        None, RuntimeError(f'Calculation process exited with code {self.worker.exitcode}.'))
                    self._finish(result, error)
        if self.worker is not None or not self.log_q.empty():
            self.start_polling()

    def _drain_events(self):
        if self._events is None:
            return
        for _ in range(200):
            try:
                event = self._events.get_nowait()
            except (queue.Empty, EOFError, OSError):
                break
            kind = event[0]
            if kind == 'log':
                self.log_widget.insert('end', event[1])
                self.log_widget.see('end')
            elif kind == 'question':
                self._answers.put(self.ask_user(event[1]))
            elif kind == 'progress' and self._on_progress is not None:
                self._on_progress(*event[1:])
            elif kind == 'done':
                self._completion = (event[1], None)
            elif kind == 'error':
                self.log(event[2])
                self._completion = (None, RuntimeError(event[1]))

    def ask_user(self, question):
        """Only the Tk poll calls this; a child requests decisions via its queue."""
        callback = self._ask_user_callback
        return True if callback is None else bool(callback(question))

    def _finish(self, result, error):
        callback = self._on_done
        status = 'cancelled' if isinstance(error, JobCancelled) else 'error' if error else 'completed'
        pid = self.worker.pid if self.worker is not None else None
        try:
            write_job_status(self.output_dir, status, kind=self.job_kind, pid=pid,
                             error=error)
        except OSError as exc:
            self.log(f'>> Could not write job status: {exc}\n')
        if self._windows_job is not None:
            self._windows_job.close()
            self._windows_job = None
        for channel in (self._events, self._answers):
            if channel is not None:
                channel.cancel_join_thread()
                channel.close()
        self._events = self._answers = None
        self.worker = None
        self._on_done = self._on_progress = self._ask_user_callback = None
        if getattr(self.root, '_active_job_runner', None) is self:
            self.root._active_job_runner = None
        if self.progress is not None:
            self.progress.configure(value=100 if error is None else 0)
        self.log('\n>> CANCELLED\n' if status == 'cancelled' else
                 f'\n>> ERROR: {error}\n' if error else '\n>> DONE\n')
        self._state_changed()
        if callback is not None:
            callback(result, error)

    def stop_polling(self):
        if self._poll_id is not None:
            self.root.after_cancel(self._poll_id)
            self._poll_id = None

    def log(self, message):
        self.log_q.put(message)

    def request_stop(self):
        if self.worker is None or self.stop_requested.is_set():
            return
        self.stop_requested.set()
        self.log('>> Stopping calculation and child processes now.\n')
        if self._windows_job is not None:
            self._windows_job.close()
            self._windows_job = None
        else:
            threading.Thread(target=_terminate_tree, args=(self.worker,), daemon=True).start()

    def should_stop(self):
        return self.stop_requested.is_set()
