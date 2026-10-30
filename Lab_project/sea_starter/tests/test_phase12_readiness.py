from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
import numpy as np
import pytest
import torch
from persian_sea.model.ablation_backends import sinusoidal_positions, TemporalPositionSignal

ROOT=Path(__file__).resolve().parents[1]
def script(name):
    location=ROOT/'scripts'/name
    spec=importlib.util.spec_from_file_location(name,location)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod


def test_positional_signal_changes_under_permutation():
    x=torch.zeros(2,5,8)
    signal=TemporalPositionSignal(0.1)
    y=signal(x)
    assert y.shape == x.shape and not torch.allclose(y[:,0],y[:,1])
    assert torch.isfinite(y).all()


def test_sinusoidal_odd_dimensions():
    for d in (1,2,5,8):
        p=sinusoidal_positions(4,d,device=torch.device('cpu'),dtype=torch.float32)
        assert p.shape==(4,d) and p.isfinite().all()


def test_new_backends_and_existing_registry():
    pytest.importorskip("persian_sea.core.registry")
    from persian_sea.model.builders import build_visual_encoder

    for backend in ('rgb_only','pose_rgb_positional','pose_rgb','pose_only','pose_rgb_motion'):
        obj=build_visual_encoder(backend,pose_input_dim=4,rgb_input_dim=8,
                                 num_views=2,model_dim=384,output_dim=16)
        assert isinstance(obj,torch.nn.Module)


def test_rgb_only_uses_no_pose():
    pytest.importorskip("persian_sea.core.registry")
    from persian_sea.model.builders import build_visual_encoder
    m=build_visual_encoder('rgb_only', rgb_input_dim=8,num_views=2,model_dim=384,output_dim=16)
    m.eval()
    pose=torch.randn(1,2,75,4)
    rgb=torch.randn(1,2,2,8)
    with torch.no_grad():
        a=m(pose,rgb)['fused']
        b=m(pose*0,rgb)['fused']
    assert torch.allclose(a,b) and a.shape==(1,16)


def test_exact_window_overlap_merge_and_dedupe():
    mod=script('eval_continuous.py')
    rows=[{'text':'من به خانه'}, {'text':'به خانه می روم'}, {'text':'می روم'}]
    assert mod.merge_windows(rows)=='من به خانه می روم'
    assert mod.merge_windows([{'text':'سلام'}, {'text':'سلام'}])=='سلام'


def test_reject_vtt_guided_prediction_json():
    mod=script('eval_continuous.py')
    with pytest.raises(ValueError,match='subtitle-free'):
        mod.check_prediction({'video_id':'a','windows':[{'start':1,'end':2,'text':'ok'}]},'a')


def test_audit_prevents_split_leakage(tmp_path):
    mod=script('audit_phase12.py')
    ids=[]
    for name, contents in [('train','v1\n'),('val','v1\n'),('test','v3\n')]:
        path=tmp_path/f'{name}.txt'; path.write_text(contents);ids.append(path)
    report=mod.audit(*ids,tmp_path/'visual',tmp_path/'raw',tmp_path/'segments_gt',tmp_path/'subtitles_gt')
    assert not report['ok']
    assert any('overlapping' in x for x in report['errors'])
