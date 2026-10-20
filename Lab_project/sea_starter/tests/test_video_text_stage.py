
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from persian_sea.video_text.config import VideoTextConfig,read_config
from persian_sea.video_text.dataset import (
    VideoTextDataset,collate_video_text,VideoDiverseBatchSampler,check_disjoint_splits
)
from persian_sea.video_text.model import VideoTextModel,mil_similarity_matrix,multi_scale_spans
from persian_sea.video_text.losses import contrastive_loss,retrieval_diagnostic


class ToyVisual(nn.Module):
    def __init__(self,output_dim=8,rgb_input_dim=8,**kwargs):
        super().__init__()
        self.fc=nn.Linear(rgb_input_dim+4,output_dim)
    def forward(self,pose,rgb):
        a=pose.mean(dim=(1,2));b=rgb.mean(dim=(1,2))
        return {"fused":F.normalize(self.fc(torch.cat([a,b],dim=-1)),dim=-1)}


def mock_files(tmp_path:Path,train_n=4,val_n=2,test_n=2):
    for folder in ('subs','visual','text','approved'):
        (tmp_path/folder).mkdir()
    nums={'train':train_n,'val':val_n,'test':test_n}
    count=0
    for spl,n in nums.items():
        vids=[]
        for k in range(n):
            v=f"v_{spl}_{k}";vids.append(v)
            cues=[(2.,4.,f"جمله متفاوت {v} یک"),(6.,8.,f"جمله متفاوت {v} دو")]
            body='WEBVTT\n\n'+''.join(f"00:00:{int(a):02d}.000 --> 00:00:{int(b):02d}.000\n{t}\n\n" for a,b,t in cues)
            (tmp_path/'subs'/f'{v}.vtt').write_text(body,encoding='utf-8')
            (tmp_path/'approved'/f'{v}.vtt').write_text(body,encoding='utf-8')
            rg=np.random.default_rng(100+count);count+=1
            np.savez_compressed(tmp_path/'visual'/f'{v}.npz',
                pose=rg.normal(size=(8,2,75,4)).astype('float16'),
                rgb=rg.normal(size=(8,2,4,8)).astype('float16'),
                starts=np.arange(8,dtype='float32')*1.5,
                ends=np.arange(8,dtype='float32')*1.5+0.7,
                views=np.array(['signer','left_hand','right_hand','face']),rgb_hidden=np.array([8]))
            np.savez_compressed(tmp_path/'text'/f'{v}.npz',
                embeddings=rg.normal(size=(2,8)).astype('float16'),
                texts=np.array([c[2] for c in cues]),
                starts=np.array([2,6],dtype='float32'),ends=np.array([4,8],dtype='float32'))
        (tmp_path/f'{spl}.txt').write_text('\n'.join(vids)+'\n',encoding='utf-8')
    return VideoTextConfig(
        train_ids=str(tmp_path/'train.txt'),val_ids=str(tmp_path/'val.txt'),
        test_ids=str(tmp_path/'test.txt'),subtitles_dir=str(tmp_path/'subs'),
        approved_subtitles_dir=str(tmp_path/'approved'),
        feature_cache_dir=str(tmp_path/'visual'),text_cache_dir=str(tmp_path/'text'),
        output_dir=str(tmp_path/'out'),embedding_dim=8,model_dim=8,
        temporal_layers=0,temporal_heads=2,visual_backend='toy',
        candidate_margin_seconds=2,min_candidate_segments=2,max_candidate_segments=5,
        span_sizes=(1,2,4),batch_size=2,epochs=1,max_train_steps=2,
        val_max_samples=4,visual_microbatch=4,num_workers=0,amp=False)


def test_stage_config_validation():
    with pytest.raises(ValueError):VideoTextConfig(pair_source='proposed').validate()
    with pytest.raises(ValueError):VideoTextConfig(batch_size=1).validate()
    with pytest.raises(ValueError):VideoTextConfig(temporal_heads=7).validate()


def test_dataset_reads_real_npz_vtt_and_preserves_time(tmp_path):
    cfg=mock_files(tmp_path)
    ds=VideoTextDataset(cfg,cfg.train_ids)
    assert len(ds)==8
    b=collate_video_text([ds[0],ds[2]])
    assert b['pose'].shape[-2:]==(75,4)
    assert b['rgb'].shape[-1]==8
    assert b['mask'].any(dim=1).all()
    assert float(b['rough_start'][0])==2.0
    ds_approved=VideoTextDataset(replace(cfg,pair_source='approved'),cfg.train_ids)
    assert len(ds_approved)==len(ds)


def test_source_split_overlap_rejected(tmp_path):
    cfg=mock_files(tmp_path)
    check_disjoint_splits(cfg.train_ids,cfg.val_ids,cfg.test_ids)
    Path(cfg.test_ids).write_text('v_train_0\n',encoding='utf-8')
    with pytest.raises(ValueError,match='leakage'):
        check_disjoint_splits(cfg.train_ids,cfg.val_ids,cfg.test_ids)


def test_batch_sampler_uses_distinct_video_ids(tmp_path):
    cfg=mock_files(tmp_path)
    ds=VideoTextDataset(cfg,cfg.train_ids)
    batches=list(VideoDiverseBatchSampler(ds.refs,2,seed=3))
    assert batches
    for batch in batches:
        assert len({ds.refs[i].video_id for i in batch})==len(batch)


def test_time_padding_invariance_and_gradients():
    cfg=VideoTextConfig(embedding_dim=8,temporal_layers=0,temporal_heads=2,
                        span_sizes=(1,2,4),visual_microbatch=2)
    model=VideoTextModel(cfg,text_dim=8,rgb_dim=8,num_views=4,visual=ToyVisual())
    pose=torch.randn(2,5,2,75,4)
    rgb=torch.randn(2,5,2,4,8)
    s=torch.tensor([[1.,2.,3.,4.,0.],[1.,2.,0.,0.,0.]])
    e=s+0.6
    mask=torch.tensor([[1,1,1,1,0],[1,1,0,0,0]],dtype=torch.bool)
    b={'pose':pose,'rgb':rgb,'starts':s,'ends':e,'mask':mask,
       'text_embedding':torch.randn(2,8)}
    model.eval()
    with torch.no_grad():
        x=model(b)['video_tokens']
        v0=model({k:(v[:1,:4] if k in ('pose','rgb','starts','ends','mask') else v[:1])
                  for k,v in b.items()})['video_tokens']
    assert torch.allclose(x[0,:4],v0[0],atol=1e-5)
    model.train()
    output=model(b)
    mat=mil_similarity_matrix(output['text_vectors'],output['span_vectors'],output['span_mask'],0.1)
    loss,info=contrastive_loss(mat,['a','b'],['الف','ب'],0.07)
    assert torch.isfinite(loss) and info['active_queries']==2
    loss.backward()
    assert model.text.network[1].weight.grad is not None
    assert model.video.visual.fc.weight.grad is not None


def test_duplicate_text_is_never_negative():
    x=torch.tensor([[2.,3.,0.],[4.,5.,0.],[0.,0.,2.]],requires_grad=True)
    loss,info=contrastive_loss(x,['a','b','c'],['مشترک','مشترک','غیرمشترک'],0.07)
    assert info['masked_pairs']==2
    assert torch.isfinite(loss)
    loss.backward();assert x.grad is not None


def test_mil_candidate_count_normalisation():
    a=torch.tensor([[1.,0.]])
    one=torch.tensor([[[1.,0.]]])
    many=torch.tensor([[[1.,0.],[1.,0.]]])
    s1=mil_similarity_matrix(a,one,torch.tensor([[True]]),.1)
    s2=mil_similarity_matrix(a,many,torch.tensor([[True,True]]),.1)
    assert torch.allclose(s1,s2,atol=1e-6)


def test_validation_checkpoint_smoke(tmp_path):
    from persian_sea.model.builders import visual_backends
    from persian_sea.video_text.train import run_training,evaluate_checkpoint
    try:
        visual_backends.register('toy',lambda **kwargs:ToyVisual(output_dim=kwargs['output_dim'],
                                                                 rgb_input_dim=kwargs['rgb_input_dim']))
    except ValueError:pass
    cfg=mock_files(tmp_path)
    ckpt=run_training(cfg)
    assert ckpt.is_file()
    result=evaluate_checkpoint(cfg,ckpt)
    assert result['num_pairs']==4
    assert result['queries']>=2
    assert 0<=result['r1']<=1
    assert (Path(cfg.output_dir)/'history.jsonl').is_file()


def test_signer_source_recording_audit(tmp_path):
    from persian_sea.video_text.split_audit import audit_signer_splits
    cfg=mock_files(tmp_path)
    file=tmp_path/'metadata.csv'
    videos={}
    for spl,path in (('train',cfg.train_ids),('val',cfg.val_ids),('test',cfg.test_ids)):
        for name in Path(path).read_text().splitlines():
            videos[name]=(spl,name)
    file.write_text('video_id,signer_id,source_recording_id\n'+
                    ''.join(f'{k},{s},{rec}\n' for k,(s,rec) in videos.items()),encoding='utf-8')
    info=audit_signer_splits(file,cfg.train_ids,cfg.val_ids,cfg.test_ids,True)
    assert info['unique_signers']==3
    # Change a recording ID to one that already exists in another split.
    text=file.read_text(encoding='utf-8').replace('v_val_0,val,v_val_0','v_val_0,val,v_train_0')
    file.write_text(text,encoding='utf-8')
    with pytest.raises(ValueError,match='source recording'):
        audit_signer_splits(file,cfg.train_ids,cfg.val_ids,cfg.test_ids)
