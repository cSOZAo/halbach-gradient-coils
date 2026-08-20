from halbach_coils.gui.runner import WorkerRunner


class _ImmediateRoot:
    def after(self, _delay, callback):
        callback()


def test_worker_decision_is_executed_through_main_thread_callback():
    runner = object.__new__(WorkerRunner)
    runner.root = _ImmediateRoot()
    runner._ask_user_callback = lambda question: question == 'continuar'

    assert runner.ask_user('continuar') is True
    assert runner.ask_user('descartar') is False
