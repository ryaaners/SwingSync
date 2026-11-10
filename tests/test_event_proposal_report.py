"""Synthetic artifact/publication attacks; no trained predictions or accuracy evidence."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from backend.swingsync.research import event_proposal_report as report
from backend.swingsync.research.event_input_contract import validate_metadata

ACTUAL_PROVENANCE = report._provenance


@pytest.fixture
def saved(tmp_path, monkeypatch):
    video=tmp_path/'fixture.mp4'
    video.write_bytes(b'Synthetic software fixture; no model or actual video')
    frames=[{'frame_index':i,'pts':tick,'time_base_numerator':1,'time_base_denominator':1000,
             'pts_seconds':tick/1000} for i,tick in enumerate((5000,5034))]
    probs=np.full((2,9),.05,dtype=np.float32)
    probs[:,8]=.6
    probs[0,0]=.1
    probs[1,1]=.1
    probs[:,8]=.55
    heads={}
    for name in ('frame','temporal'):
        peaks=np.argsort(probs[:,:8],axis=0)[-1].tolist()
        events=[{'name':report.EVENT_NAMES[k],'class_index':k,'frame_index':i,
                 **{key:frames[i][key] for key in ('pts_seconds','pts','time_base_numerator','time_base_denominator')},
                 'score':float(probs[i,k]),'background_probability':float(probs[i,8])} for k,i in enumerate(peaks)]
        heads[name]={'events':events,'strict_order':False,'duplicate_peak_count':6}
    context=validate_metadata('face-on','right','fixture </pre><img src=x onerror=bad>',True,True,
                              'Synthetic software fixture','source')
    geometry={'width':160,'height':160,'frame_count':2}
    inspection={'frames':frames,**geometry}
    monkeypatch.setattr(report,'_provenance',lambda _:None)
    monkeypatch.setattr(report,'_inspect',lambda _:inspection)
    run={'schema_version':1,'state':'completed','job_id':'synthetic','context':context,'identity':{},
         'video_sha256':report.sha(video),'video':geometry,'completed_at':'fixture',
         'normal_model_available':False,'validated_for_serving':False,'accuracy_metrics_computed':False,'advice':[]}
    result={**run,'primary_head':'temporal','temporal_context':'reference-64-reset',
            'order_constrained_decoding':False,'scores_are_calibrated':False,'domain_compatibility_validated':False,
            'outcome_integration_permitted':False,'physical_timing_validated':False,'frames':frames,'heads':heads,'arrays_file':'arrays.npz'}
    arrays={'features':np.zeros((2,1280),dtype=np.float32),'frame_probabilities':probs.copy(),
            'temporal_probabilities':probs.copy(),'pts_seconds':np.array([5.,5.034],dtype=np.float64)}
    path=tmp_path/'run.json'
    def publish():
        np.savez_compressed(tmp_path/'arrays.npz',**arrays)
        run['arrays_sha256']=result['arrays_sha256']=report.sha(tmp_path/'arrays.npz')
        (tmp_path/'result.json').write_text(json.dumps(result))
        run['result_sha256']=report.sha(tmp_path/'result.json')
        path.write_text(json.dumps(run))
        return path,report.sha(path)
    publish()
    return SimpleNamespace(run=run,result=result,arrays=arrays,video=video,path=path,publish=publish)


def load(saved):
    path,digest=saved.publish()
    return report.load_event_evidence(path,expected_run_sha256=digest,video=saved.video)


def test_synthetic_completed_peaks_admit_without_model_import(saved):
    evidence=load(saved)
    assert evidence.result['heads']['temporal']['events'][0]['pts_seconds']==5.
    assert evidence.result['heads']['temporal']['duplicate_peak_count']==6
    assert evidence.result['heads']['temporal']['events'][0]['background_probability']>.1
    report.assert_unchanged(evidence)


def test_wrong_known_hash_rejected_before_native_decoder(saved,monkeypatch):
    monkeypatch.setattr(report,'_inspect',lambda _:pytest.fail('Decoder must not start'))
    with pytest.raises(ValueError,match='SHA256 mismatch'):
        report.load_event_evidence(saved.path,expected_run_sha256='0'*64,video=saved.video)


@pytest.mark.parametrize('field,value',[('frame_index',1),('pts_seconds',0.),('score',.99),
                                        ('name','Invented event'),('class_index',True)])
def test_rehashed_proposal_tampering_refused(saved,field,value):
    saved.result['heads']['temporal']['events'][0][field]=value
    with pytest.raises(ValueError,match='proposal/peak'):
        load(saved)


@pytest.mark.parametrize('field,value',[('scores_are_calibrated',True),('normal_model_available',True),
                                        ('accuracy_metrics_computed',True),('advice',['invented feedback']),
                                        ('domain_compatibility_validated',True),('physical_timing_validated',True)])
def test_rehashed_claim_promotion_refused(saved,field,value):
    saved.result[field]=value
    with pytest.raises(ValueError):
        load(saved)


def test_false_order_claim_or_changed_probabilities_refused(saved):
    saved.result['heads']['temporal']['strict_order']=True
    with pytest.raises(ValueError,match='order diagnostics'):
        load(saved)
    saved.result['heads']['temporal']['strict_order']=False
    saved.arrays['temporal_probabilities'][0,0]=.9
    with pytest.raises(ValueError,match='probabilities malformed'):
        load(saved)


def test_snapshot_and_video_drift_refuse_publication(saved):
    evidence=load(saved)
    evidence.result['heads']['temporal']['events'][0]['score']=.99
    with pytest.raises(ValueError,match='snapshot/source drift'):
        report.assert_unchanged(evidence)
    saved.result['heads']['temporal']['events'][0]['score']=float(saved.arrays['temporal_probabilities'][0,0])
    evidence=load(saved)
    saved.video.write_bytes(b'changed')
    with pytest.raises(ValueError,match='snapshot/source drift'):
        report.assert_unchanged(evidence)


def test_actual_source_frame_correspondence_mismatch_refused(saved,monkeypatch):
    expected=copy.deepcopy({'frames':saved.result['frames'],**saved.result['video']})
    expected['frames'][1]['pts']=5035
    monkeypatch.setattr(report,'_inspect',lambda _:expected)
    with pytest.raises(ValueError,match='PTS/raster'):
        load(saved)


def test_report_canonical_cards_escaping_and_no_overwrite(saved,monkeypatch):
    evidence=load(saved)
    monkeypatch.setattr(report,'extract_frames_bounded',lambda video,info,indices:{str(i):'fixture' for i in indices})
    out=saved.path.parent/'review'
    manifest=report.build_report(evidence,out)
    page=(out/'review.html').read_text()
    assert manifest['event_count']==8 and manifest['canonical_frame_indices']==[0,1]
    assert manifest['normal_model_available'] is False and not manifest['browser_time_alignment_validated']
    assert 'approximate' in page and 'data-time="0.0"' in page
    assert 'Canonical Address proposal frame' in page
    assert '</pre><img src=x onerror=bad>' not in page
    assert 'eight forced independent class peaks' in page
    before=(out/'manifest.json').read_bytes()
    with pytest.raises(FileExistsError):
        report.build_report(evidence,out)
    assert (out/'manifest.json').read_bytes()==before


def test_report_derivation_drift_removes_incomplete_output(saved,monkeypatch):
    evidence=load(saved)
    monkeypatch.setattr(report,'extract_frames_bounded',lambda *args:{'0':'fixture','1':'fixture'})
    original=report.sha
    calls={}
    def changed(path):
        text=str(path)
        if text.endswith('scripts/report_event_proposals.py'):
            calls[text]=calls.get(text,0)+1
            if calls[text]>1:
                return 'f'*64
        return original(path)
    monkeypatch.setattr(report,'sha',changed)
    out=saved.path.parent/'failed-review'
    with pytest.raises(ValueError,match='source drift'):
        report.build_report(evidence,out)
    assert not out.exists()


def registered_provenance(saved):
    c=json.loads((report.ROOT/'experiments/heldout-v1.json').read_text())
    training={name:report.sha(report.ROOT/'backend/swingsync/research'/name)
              for name in ('experiment.py','models.py','training.py','evaluation.py','features.py','golfdb.py','timeline.py')}
    paths=('backend/swingsync/research/event_input.py','backend/swingsync/research/event_input_contract.py',
           'scripts/propose_events.py','experiments/event-input-v1.json')
    run=copy.deepcopy(saved.run)
    run['identity']={'heldout_contract_sha256':report.CONTRACT_SHA,'feature_pipeline':c['feature_pipeline'],
      'backbone_weight_sha256':'7ebf99e03e254b273379b23edca7ec0da9f48273b23a332b93c1c99d49e86e8f',
      'context':'reference-64-reset','batch_size':16,'source_sha256':{p:report.sha(report.ROOT/p) for p in paths},
      'protocol_sha256':report.sha(report.ROOT/'experiments/event-input-v1.json'),
      'selected_heads':c['heads'],'video_sha256':run['video_sha256'],
      'manifest_sha256':c['manifest_sha256'],'annotations_sha256':c['dataset_sha256'],
      'selected_head_admission':{'contract_sha256':report.CONTRACT_SHA,'dependency_lock_sha256':c['dependency_lock_sha256'],
       'evaluator_sha256':report.sha(report.ROOT/'backend/swingsync/research/heldout.py'),
       'evaluation_cli_sha256':report.sha(report.ROOT/'scripts/evaluate_event_heads.py'),
       'training_implementation_sha256':training,'manifest_sha256':c['manifest_sha256'],'dataset_sha256':c['dataset_sha256']}}
    return run


@pytest.mark.parametrize('attack',['none','missing_training','wrong_dataset','wrong_protocol'])
def test_complete_registered_provenance_coverage(saved,attack):
    run=registered_provenance(saved)
    if attack=='none':
        ACTUAL_PROVENANCE(run)
        return
    if attack=='missing_training':
        run['identity']['selected_head_admission']['training_implementation_sha256']={}
    elif attack=='wrong_dataset':
        run['identity']['selected_head_admission']['dataset_sha256']='0'*64
    else:
        run['identity']['protocol_sha256']='0'*64
    with pytest.raises(ValueError):
        ACTUAL_PROVENANCE(run)


def test_drift_during_final_manifest_write_cleans_completed_bundle(saved,monkeypatch):
    evidence=load(saved)
    monkeypatch.setattr(report,'extract_frames_bounded',lambda *args:{'0':'fixture','1':'fixture'})
    original=report.json.dump
    def change_after_dump(value,stream,**kwargs):
        original(value,stream,**kwargs)
        if value.get('event_count')==8:
            saved.video.write_bytes(b'changed while publishing manifest')
    monkeypatch.setattr(report.json,'dump',change_after_dump)
    out=saved.path.parent/'failed-final-publication'
    with pytest.raises(ValueError,match='snapshot/source drift'):
        report.build_report(evidence,out)
    assert not out.exists()
