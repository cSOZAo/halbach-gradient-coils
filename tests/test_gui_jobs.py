"""Exercise real spawn workers and cancellation, without running a coil solve."""

import ctypes
import heapq
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from types import SimpleNamespace

import pytest

from halbach_coils.gui.jobs import JOB_STATUS_FILENAME, job_output_dir, write_job_status
from halbach_coils.gui.runner import JobCancelled, WorkerRunner
from halbach_coils.gui import runner as runner_module


class _Root:
    def __init__(self):
        self.events = []
        self.serial = 0
        self.cancelled = set()
        self.state_changes = 0

    def after(self, delay, callback):
        self.serial += 1
        heapq.heappush(self.events, (time.monotonic() + delay / 1000, self.serial, callback))
        return self.serial

    def after_cancel(self, token):
        self.cancelled.add(token)

    def on_job_state_changed(self):
        self.state_changes += 1

    def tick(self):
        while self.events and self.events[0][0] <= time.monotonic():
            _when, token, callback = heapq.heappop(self.events)
            if token not in self.cancelled:
                callback()


class _Text:
    def __init__(self):
        self.lines = []

    def delete(self, *_args):
        self.lines.clear()

    def insert(self, _position, text):
        self.lines.append(text)

    def see(self, *_args):
        pass


def _spin(root, condition, timeout=12):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        root.tick()
        time.sleep(0.01)
    root.tick()
    assert condition(), 'GUI worker did not reach the expected state in time'


def _result(new_cwd=None):
    if new_cwd is not None:
        os.chdir(new_cwd)
    print('worker output')
    return {'pid': os.getpid(), 'cwd': os.getcwd()}


def _decision(*, ask_user):
    return ask_user('continuar')


def _failure():
    raise RuntimeError('expected calculation failure')


def _busy():
    while True:
        sum(index * index for index in range(10000))


def _busy_with_child(pid_path):
    options = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], **options)
    Path(pid_path).write_text(json.dumps({'child': child.pid, 'worker': os.getpid()}))
    print('numerical worker and external tool started', flush=True)
    _busy()


def _windows_pid_alive(pid):
    from ctypes import wintypes
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    api.OpenProcess.restype = wintypes.HANDLE
    api.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    api.WaitForSingleObject.restype = wintypes.DWORD
    api.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = api.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE, read-only
    if not handle:
        return False
    try:
        return api.WaitForSingleObject(handle, 0) == 0x102
    finally:
        api.CloseHandle(handle)


def test_spawn_keeps_stdout_in_worker_and_delivers_completion_on_gui_thread(tmp_path):
    root, text, done = _Root(), _Text(), []
    runner = WorkerRunner(text, None, root)
    parent_stdout = sys.stdout
    parent_cwd = os.getcwd()
    parent_thread = threading.get_ident()
    cfg = SimpleNamespace(output_dir=str(tmp_path))
    assert runner._start('_callable', cfg, {'target': _result, 'args': (str(tmp_path),)},
                         lambda result, error: done.append((result, error, threading.get_ident())),
                         None, None)
    _spin(root, lambda: bool(done))
    assert done[0][0]['pid'] != os.getpid()
    assert done[0][1] is None
    assert done[0][2] == parent_thread
    assert sys.stdout is parent_stdout
    assert os.getcwd() == parent_cwd
    assert done[0][0]['cwd'] == str(tmp_path)
    root.tick()
    assert 'worker output' in ''.join(text.lines)
    assert root._active_job_runner is None
    assert json.loads((tmp_path / JOB_STATUS_FILENAME).read_text())['status'] == 'completed'


def test_overlap_decision_crosses_process_queue_and_runs_on_gui_thread():
    root, done, questions = _Root(), [], []
    runner = WorkerRunner(_Text(), None, root)
    caller_thread = threading.get_ident()

    def answer(question):
        questions.append((question, threading.get_ident()))
        return False

    assert runner.run(_decision, ask_user=answer,
                      on_done=lambda result, error: done.append((result, error)))
    _spin(root, lambda: bool(done))
    assert done == [(False, None)]
    assert questions == [('continuar', caller_thread)]


def test_application_lock_blocks_other_tab_until_cancelled_worker_exits():
    root, cancelled, done = _Root(), [], []
    first = WorkerRunner(_Text(), None, root)
    other = WorkerRunner(_Text(), None, root)
    try:
        assert first.run(_busy, on_done=lambda result, error: cancelled.append(error))
        assert not other.run(_result)
        first.request_stop()
        assert not other.run(_result)
        _spin(root, lambda: bool(cancelled), timeout=6)
        assert isinstance(cancelled[0], JobCancelled)
        assert root._active_job_runner is None
        assert other.run(_result, on_done=lambda result, error: done.append((result, error)))
        _spin(root, lambda: bool(done))
        assert done[0][1] is None
    finally:
        for runner in (first, other):
            if runner.is_running:
                runner.request_stop()
                _spin(root, lambda: not runner.is_running, timeout=6)


@pytest.mark.skipif(os.name != 'nt', reason='Windows descendant containment regression')
def test_stop_kills_cpu_worker_and_external_child_within_seconds(tmp_path):
    root, done = _Root(), []
    runner = WorkerRunner(_Text(), None, root)
    pid_path = tmp_path / 'pids.json'
    cfg = SimpleNamespace(output_dir=str(tmp_path))
    try:
        assert runner._start('_callable', cfg, {'target': _busy_with_child, 'args': (str(pid_path),)},
                             lambda result, error: done.append(error), None, None)
        _spin(root, pid_path.exists)
        pids = json.loads(pid_path.read_text())
        assert _windows_pid_alive(pids['child'])
        started = time.monotonic()
        runner.request_stop()
        _spin(root, lambda: bool(done), timeout=5)
        assert time.monotonic() - started < 5
        assert isinstance(done[0], JobCancelled)
        assert not _windows_pid_alive(pids['worker'])
        assert not _windows_pid_alive(pids['child'])
        status = json.loads((tmp_path / JOB_STATUS_FILENAME).read_text())
        assert status['status'] == 'cancelled'
        assert status['partial'] is True
    finally:
        if runner.is_running:
            runner.request_stop()
            _spin(root, lambda: not runner.is_running, timeout=6)


def test_spawn_serialization_error_releases_lock_and_reports_error():
    root, done = _Root(), []
    runner = WorkerRunner(_Text(), None, root)
    assert not runner.run(lambda: None, on_done=lambda result, error: done.append(error))
    assert done and isinstance(done[0], Exception)
    assert root._active_job_runner is None
    assert not runner.is_running


def test_os_spawn_error_releases_application_lock(monkeypatch):
    root, done = _Root(), []
    runner = WorkerRunner(_Text(), None, root)
    context = runner_module.multiprocessing.get_context('spawn')

    class CannotStart:
        pid = None

        def __init__(self, **_kwargs):
            pass

        def start(self):
            raise PermissionError('spawn denied')

    monkeypatch.setattr(runner_module.multiprocessing, 'get_context', lambda _mode: SimpleNamespace(
        Queue=context.Queue, Event=context.Event, Process=CannotStart))
    assert not runner.run(_result, on_done=lambda result, error: done.append(error))
    assert isinstance(done[0], PermissionError)
    assert root._active_job_runner is None
    assert not runner.is_running


def test_worker_exception_reports_error_and_marks_partial_output(tmp_path):
    root, done = _Root(), []
    runner = WorkerRunner(_Text(), None, root)
    cfg = SimpleNamespace(output_dir=str(tmp_path))
    assert runner._start('_callable', cfg, {'target': _failure},
                         lambda result, error: done.append(error), None, None)
    _spin(root, lambda: bool(done))
    assert 'expected calculation failure' in str(done[0])
    status = json.loads((tmp_path / JOB_STATUS_FILENAME).read_text())
    assert status['status'] == 'error'
    assert status['partial'] is True


def test_sweep_progress_is_delivered_by_main_thread_poll():
    root, received = _Root(), []
    runner = WorkerRunner(_Text(), None, root)
    runner._events = queue.Queue()
    runner._on_progress = lambda *args: received.append((args, threading.get_ident()))
    row = {'Tikhonov': 100, 'Fase': 'Grueso'}
    runner._events.put(('progress', 'Grueso', 100, 'done', row))
    runner._drain_events()
    assert received == [(('Grueso', 100, 'done', row), threading.get_ident())]


def test_sweep_marker_path_matches_actual_axis_directory(tmp_path):
    cfg = SimpleNamespace(output_dir=str(tmp_path), axis_label='GZ')
    assert job_output_dir('sweep', cfg, {'output_base_dir': str(tmp_path)}) == str(tmp_path / 'Eje_GZ')
    write_job_status(str(tmp_path), 'cancelled', kind='pipeline', pid=123)
    marker = json.loads((tmp_path / JOB_STATUS_FILENAME).read_text())
    assert marker['partial'] is True
    assert marker['pid'] == 123


def test_real_app_stop_unlocks_all_tabs_and_close_terminates_active_worker():
    # This exercises actual buttons/main-window callbacks, not only the runner.
    package_dir = str(Path(__file__).resolve().parents[1] / 'halbach_coils')
    if package_dir not in sys.path:
        sys.path.insert(0, package_dir)
    from halbach_coils.gui.main_window import CoilGenApp

    try:
        app = CoilGenApp()
    except tk.TclError as exc:
        pytest.skip(f'Tk display unavailable: {exc}')
    app.withdraw()
    completed, destroyed = [], []
    panels = (app.pipeline_panel, app.standalone_panel, app.sweep_panel)
    runner = app.pipeline_panel.runner
    app.bind('<Destroy>', lambda event: destroyed.append(True) if event.widget is app else None)

    def pump_until(condition):
        deadline = time.monotonic() + 8
        while not condition() and time.monotonic() < deadline:
            app.update()
            time.sleep(0.01)
        assert condition()

    try:
        assert runner.run(_busy, on_done=lambda result, error: completed.append(error))
        assert all(str(panel.run_btn.cget('state')) == 'disabled' for panel in panels)
        app.pipeline_panel._on_stop()
        pump_until(lambda: bool(completed))
        assert isinstance(completed[0], JobCancelled)
        assert all(str(panel.run_btn.cget('state')) == 'normal' for panel in panels)
        assert runner.run(_busy, on_done=lambda result, error: completed.append(error))
        app._on_close()
        pump_until(lambda: bool(destroyed))
        assert not runner.is_running
        assert isinstance(completed[-1], JobCancelled)
    finally:
        if not destroyed:
            if runner.is_running:
                runner.request_stop()
                pump_until(lambda: not runner.is_running)
            app.destroy()
