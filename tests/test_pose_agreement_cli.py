"""Agreement CLI publication controls with fake admission/evaluation, never real scores."""
from types import SimpleNamespace

import pytest

from backend.swingsync.movement import pose_artifacts as pa
from backend.swingsync.research import pose_validation as pv
from scripts.evaluate_pose_reference import evaluate


@pytest.fixture
def agreement_request(tmp_path, monkeypatch):
    reference = tmp_path / 'reference.json'
    reference.write_text('{"fixture":"software only"}')
    artifact = SimpleNamespace(result={}, run_sha256='1'*64, result_sha256='2'*64)
    monkeypatch.setattr(pa, 'load_pose_artifact', lambda *args, **kwargs: artifact)
    monkeypatch.setattr(pa, 'assert_artifact_unchanged', lambda _: None)
    monkeypatch.setattr(pv, 'evaluate_reference', lambda *args, **kwargs: {
        'scope': 'synthetic CLI test', 'scored_count': 1, 'model_abstention_count': 0,
        'validated_for_coaching': False, 'app_integration_permitted': False, 'advice': []})
    return dict(run=tmp_path/'run.json', run_sha256='1'*64, video=tmp_path/'fixture.mp4',
                reference=reference, output=tmp_path/'agreement.json')


def test_exclusive_output_contains_input_identity_without_admission(agreement_request):
    report = evaluate(**agreement_request)
    assert report['reference_file_sha256'] == pa.sha256(agreement_request['reference'])
    assert report['empirical_accuracy_study_completed'] is False
    assert report['validated_for_coaching'] is False
    before = agreement_request['output'].read_bytes()
    with pytest.raises(ValueError, match='refusing to overwrite'):
        evaluate(**agreement_request)
    assert agreement_request['output'].read_bytes() == before


def test_unreviewed_reference_failure_publishes_nothing(agreement_request, monkeypatch):
    def reject(*args, **kwargs):
        raise ValueError('Scoring requires manual independent/model-blinded attestations')
    monkeypatch.setattr(pv, 'evaluate_reference', reject)
    with pytest.raises(ValueError, match='manual independent'):
        evaluate(**agreement_request)
    assert not agreement_request['output'].exists()


def test_reference_changed_during_scoring_never_publishes(agreement_request, monkeypatch):
    def mutate(*args, **kwargs):
        agreement_request['reference'].write_text('{"changed":true}')
        return {'scored_count': 1}
    monkeypatch.setattr(pv, 'evaluate_reference', mutate)
    with pytest.raises(ValueError, match='Reference/protocol/evaluation source drift'):
        evaluate(**agreement_request)
    assert not agreement_request['output'].exists()
