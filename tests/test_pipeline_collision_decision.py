from halbach_coils.coilgen.config import Config
from halbach_coils.coilgen.overlap import CollisionSite, OverlapReport
from halbach_coils import run_pipeline as pipeline_module


def _collision_report():
    return OverlapReport(sites=[CollisionSite(
        part_index=0,
        location_uv=(0.0, 0.0),
        cable_count=3,
        crossing_count=3,
        segment_indices=(10, 100, 200),
    )])


def test_rejecting_collision_result_stops_before_leads_and_shell(tmp_path, monkeypatch):
    cfg = Config(show_plots=False)
    cfg.output_dir = str(tmp_path)
    calls = []
    questions = []

    monkeypatch.setattr(
        pipeline_module, 'run_gradient',
        lambda *args, **kwargs: (object(), {}, _collision_report()),
    )
    monkeypatch.setattr(pipeline_module, 'run_leads',
                        lambda *args, **kwargs: calls.append('leads'))
    monkeypatch.setattr(pipeline_module, 'run_shell',
                        lambda *args, **kwargs: calls.append('shell'))

    def reject(question):
        questions.append(question)
        return False

    result = pipeline_module.run_pipeline(cfg, ask_user=reject)

    assert result.discarded is True
    assert calls == []
    assert len(questions) == 1
    assert "Hay 3 cables intentando pasar por el mismo lugar" in questions[0]


def test_accepting_collision_result_continues_with_leads_and_shell(tmp_path, monkeypatch):
    cfg = Config(show_plots=False)
    cfg.output_dir = str(tmp_path)
    calls = []

    monkeypatch.setattr(
        pipeline_module, 'run_gradient',
        lambda *args, **kwargs: (object(), {}, _collision_report()),
    )
    monkeypatch.setattr(pipeline_module, '_ensure_wire_exists',
                        lambda *args, **kwargs: str(tmp_path / 'wire.stl'))
    monkeypatch.setattr(pipeline_module, 'run_leads',
                        lambda *args, **kwargs: calls.append('leads'))
    monkeypatch.setattr(pipeline_module, 'run_shell',
                        lambda *args, **kwargs: calls.append('shell'))

    result = pipeline_module.run_pipeline(cfg, ask_user=lambda _question: True)

    assert result.discarded is False
    assert calls == ['leads', 'shell']
