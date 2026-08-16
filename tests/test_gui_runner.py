"""
Unit tests for :mod:`gui.runner`.

``WorkerRunner`` only duck-types the Tk widgets it talks to (``insert``/``see``/
``delete`` on the log, ``after``/``after_cancel`` on the root), so the whole
worker/queue/redirect machinery is testable headlessly with fake widgets — no
display and no Tk main loop.
"""

import queue
import sys
import threading

import pytest

tk = pytest.importorskip('tkinter')

from gui.runner import WorkerRunner, _StreamRedirect                # noqa: E402


class FakeText:
    def __init__(self):
        self.lines = []
        self.deleted = 0
        self.seen = 0

    def insert(self, where, text):
        self.lines.append(text)

    def see(self, where):
        self.seen += 1

    def delete(self, start, end):
        self.lines.clear()
        self.deleted += 1

    @property
    def text(self):
        return ''.join(self.lines)


class FakeProgress:
    def __init__(self):
        self.values = {}
        self.configured = []

    def __setitem__(self, key, value):
        self.values[key] = value

    def configure(self, **kwargs):
        self.configured.append(kwargs)


class FakeRoot:
    """Records ``after`` callbacks so tests can run them deterministically."""

    def __init__(self):
        self.callbacks = []
        self.cancelled = []
        self._next = 0

    def after(self, delay, fn=None):
        self._next += 1
        if fn is not None:
            self.callbacks.append((delay, fn))
        return f'id{self._next}'

    def after_cancel(self, handle):
        self.cancelled.append(handle)

    def run_pending(self):
        """Run the queued zero-delay callbacks (the poll loop is skipped)."""
        pending = [c for c in self.callbacks if c[0] == 0]
        self.callbacks = [c for c in self.callbacks if c[0] != 0]
        for _, fn in pending:
            fn()


@pytest.fixture
def runner():
    log, progress, root = FakeText(), FakeProgress(), FakeRoot()
    r = WorkerRunner(log, progress, root)
    return r, log, progress, root


def _join(runner_obj, timeout=5.0):
    runner_obj.worker.join(timeout)
    assert not runner_obj.worker.is_alive()


# ---------------------------------------------------------------------------
# _StreamRedirect
# ---------------------------------------------------------------------------

def test_stream_redirect_enqueues_complete_lines_only():
    q = queue.Queue()
    r = _StreamRedirect(q)

    assert r.write('hello ') == 6
    assert q.empty()

    r.write('world\nsecond\n')

    assert q.get_nowait() == 'hello world\n'
    assert q.get_nowait() == 'second\n'
    assert q.empty()


def test_stream_redirect_flush_emits_the_partial_tail():
    q = queue.Queue()
    r = _StreamRedirect(q)
    r.write('no newline')

    r.flush()

    assert q.get_nowait() == 'no newline'
    r.flush()                                   # nothing buffered any more
    assert q.empty()


def test_stream_redirect_ignores_empty_writes():
    q = queue.Queue()

    assert _StreamRedirect(q).write('') == 0
    assert q.empty()


def test_stream_redirect_is_usable_as_stdout():
    q = queue.Queue()
    r = _StreamRedirect(q)
    old = sys.stdout
    sys.stdout = r
    try:
        print('from a worker')
    finally:
        sys.stdout = old

    assert q.get_nowait() == 'from a worker\n'


# ---------------------------------------------------------------------------
# log draining / polling
# ---------------------------------------------------------------------------

def test_drain_log_moves_queued_lines_into_the_widget(runner):
    r, log, _, root = runner
    r.log('a\n')
    r.log('b\n')

    r.start_polling()

    assert log.text == 'a\nb\n'
    assert log.seen == 2
    assert r._poll_id is not None
    assert root.callbacks[-1][0] == 120          # reschedules itself


def test_start_polling_is_idempotent(runner):
    r, _, _, root = runner
    r.start_polling()
    first = r._poll_id

    r.start_polling()

    assert r._poll_id == first


def test_stop_polling_cancels_the_scheduled_callback(runner):
    r, _, _, root = runner
    r.start_polling()
    handle = r._poll_id

    r.stop_polling()

    assert root.cancelled == [handle]
    assert r._poll_id is None


def test_stop_polling_without_polling_is_a_no_op(runner):
    r, _, _, root = runner

    r.stop_polling()

    assert root.cancelled == []


# ---------------------------------------------------------------------------
# ask_user / answer_user
# ---------------------------------------------------------------------------

def test_answer_user_enqueues_a_boolean_decision(runner):
    r, _, _, _ = runner

    r.answer_user(1)
    r.answer_user(0)

    assert r.user_q.get_nowait() is True
    assert r.user_q.get_nowait() is False


def test_ask_user_enqueues_the_question_for_the_main_thread(runner):
    r, _, _, _ = runner
    # ``ask_user`` posts its request and then blocks on the same queue, so it
    # is only usable when the answer is already pending; drop one in first.
    r.answer_user(True)

    accepted = r.ask_user('overlap detected -- keep it?')

    assert accepted is True
    assert ('ask', 'overlap detected -- keep it?') in list(r.user_q.queue)


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

def test_run_executes_the_target_and_reports_done(runner):
    r, log, progress, root = runner
    results = []

    r.run(lambda a, b=0: a + b, args=(2,), kwargs={'b': 3},
          on_done=lambda res, err: results.append((res, err)))
    _join(r)
    root.run_pending()
    r._drain_log()

    assert results == [(5, None)]
    assert '>> DONE' in log.text
    assert progress.values['value'] == 0
    assert progress.configured == [{'value': 100}]
    assert log.deleted == 1


def test_run_captures_worker_stdout_into_the_log(runner):
    r, log, _, root = runner

    r.run(lambda: print('pyCoilGen says hi'))
    _join(r)
    r._drain_log()

    assert 'pyCoilGen says hi\n' in log.text
    assert sys.stdout is not None


def test_run_reports_exceptions_to_the_log_and_on_done(runner):
    r, log, _, root = runner
    seen = []

    r.run(lambda: (_ for _ in ()).throw(RuntimeError('pyCoilGen exploded')),
          on_done=lambda res, err: seen.append((res, err)))
    _join(r)
    root.run_pending()
    r._drain_log()

    assert '>> ERROR: pyCoilGen exploded' in log.text
    assert seen[0][0] is None
    assert isinstance(seen[0][1], RuntimeError)


def test_run_restores_stdout_after_a_failure(runner):
    r, _, _, _ = runner
    before = sys.stdout

    r.run(lambda: (_ for _ in ()).throw(ValueError('boom')))
    _join(r)

    assert sys.stdout is before


def test_run_wires_the_ask_user_callback_into_the_target(runner):
    r, _, _, _ = runner
    captured = {}

    def target(ask_user=None):
        captured['ask_user'] = ask_user
        return 'ok'

    hook = lambda question: True                              # noqa: E731
    r.run(target, ask_user=hook)
    _join(r)

    assert captured['ask_user'] is hook


def test_run_does_not_override_an_explicit_ask_user_kwarg(runner):
    r, _, _, _ = runner
    captured = {}
    explicit = lambda q: False                                # noqa: E731

    r.run(lambda ask_user=None: captured.setdefault('ask_user', ask_user),
          kwargs={'ask_user': explicit}, ask_user=lambda q: True)
    _join(r)

    assert captured['ask_user'] is explicit


def test_run_refuses_to_start_a_second_job(runner):
    r, log, _, _ = runner
    gate = threading.Event()

    r.run(lambda: gate.wait(5))
    r.run(lambda: 'second')
    r._drain_log()
    assert 'a job is already running' in log.text

    gate.set()
    _join(r)


def test_run_without_a_progressbar_still_completes():
    r = WorkerRunner(FakeText(), None, FakeRoot())

    r.run(lambda: 1)
    _join(r)

    assert r.worker is not None


# ---------------------------------------------------------------------------
# cooperative stop
# ---------------------------------------------------------------------------

def test_request_stop_sets_the_flag_and_logs(runner):
    r, log, _, _ = runner
    assert r.should_stop() is False

    r.request_stop()
    r.start_polling()

    assert r.should_stop() is True
    assert 'stop requested' in log.text


def test_run_clears_a_previous_stop_request(runner):
    r, _, _, _ = runner
    r.request_stop()

    r.run(lambda: 1)
    _join(r)

    assert r.should_stop() is False
