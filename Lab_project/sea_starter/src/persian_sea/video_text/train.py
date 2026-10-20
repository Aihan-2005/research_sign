from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
import random
import time
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset

from .config import VideoTextConfig
from .dataset import VideoTextDataset,VideoDiverseBatchSampler,collate_video_text,check_disjoint_splits
from .model import VideoTextModel,mil_similarity_matrix
from .losses import contrastive_loss,retrieval_diagnostic


MODEL_KEYS = ('visual_backend', 'embedding_dim', 'model_dim',
              'temporal_layers', 'temporal_heads', 'span_sizes', 'span_pooling')


def architecture_config(cfg: VideoTextConfig) -> dict:
    return {key: asdict(cfg)[key] for key in MODEL_KEYS}


def checkpoint_architecture(raw: dict) -> dict:
    result = dict(raw)
    result.setdefault('span_pooling', 'mean')
    return {key: result.get(key) for key in MODEL_KEYS}


def verify_checkpoint_text_source(stored_cfg: dict, cfg: VideoTextConfig) -> None:

    for key in ("text_backend", "text_model", "text_instruction",
                "require_text_cache_metadata", "pair_source"):
        default = False if key == "require_text_cache_metadata" else ("raw" if key == "pair_source" else "")
        if stored_cfg.get(key, default) != getattr(cfg, key):
            raise ValueError(f"Checkpoint text/source provenance differs: {key}")


def seed_all(seed:int):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def move(batch,device):
    return {k:(v.to(device,non_blocking=True) if torch.is_tensor(v) else v)
            for k,v in batch.items()}


def dims(dataset:VideoTextDataset):
    item=dataset[0]
    return (item["text_embedding"].numel(),item["rgb"].shape[-1],item["rgb"].shape[2])


def _selection(dataset:VideoTextDataset, limit:int,seed:int):
    """Deterministic, diverse subset across videos for comparable val retrieval."""
    rng=np.random.default_rng(seed)
    ids={}
    for i,r in enumerate(dataset.refs):ids.setdefault(r.video_id,[]).append(i)
    pools={k:list(rng.permutation(v)) for k,v in ids.items()}
    keys=sorted(pools)
    result=[]
    while len(result)<limit:
        any_added=False
        for k in keys:
            if pools[k] and len(result)<limit:
                result.append(pools[k].pop());any_added=True
        if not any_added:break
    return result


@torch.no_grad()
def evaluate(model:VideoTextModel,dataset:VideoTextDataset,cfg:VideoTextConfig,device):
    """Cross-video retrieval on one fixed set of held-out queries.

    Scores use only video features + BGE-M3 text vectors. Reports retrieval,
    not subtitle IoU or video-to-text generation quality.
    """
    model.eval()
    ids=_selection(dataset,cfg.val_max_samples,cfg.seed)
    loader=DataLoader(Subset(dataset,ids),batch_size=max(1,min(cfg.batch_size,4)),
                      shuffle=False,collate_fn=collate_video_text,num_workers=0)
    texts=[];spans=[];masks=[];videos=[];text_keys=[]
    for batch in loader:
        data=move(batch,device)
        out=model(data)
        texts.append(out["text_vectors"].detach().cpu())
        spans.extend([out["span_vectors"][i,out["span_mask"][i]].detach().cpu()
                      for i in range(out["span_vectors"].shape[0])])
        videos.extend(batch["video_id"])
        text_keys.extend(batch["text_key"])
    if not texts:raise ValueError("Empty validation set")
    vectors=torch.cat(texts).to(device)
    # Across-val matrix computed in small blocks to bound GPU memory.
    result=[]
    for q in range(0,len(spans),16):
        query=vectors[q:q+16]
        batches=[]
        for s in range(0,len(spans),16):
            group=spans[s:s+16]
            maxlen=max(x.shape[0] for x in group)
            bag=vectors.new_zeros(len(group),maxlen,vectors.shape[-1]);mask=torch.zeros(
                len(group),maxlen,dtype=torch.bool,device=device)
            for j,item in enumerate(group):
                n=len(item);bag[j,:n]=item.to(device);mask[j,:n]=True
            # Each block: [num_text_queries, num_video_bags]
            tau=cfg.mil_temperature
            sim=torch.einsum('qd,bsd->qbs',query.float(),bag.float())
            sim=sim.masked_fill(~mask[None,:,:],-1e4)
            block=tau*torch.logsumexp(sim/tau,dim=-1)-tau*mask.sum(1).float().log()[None,:]
            batches.append(block)
        result.append(torch.cat(batches,dim=1))
    full=torch.cat(result,dim=0)
    loss,extra=contrastive_loss(full,videos,text_keys,cfg.contrastive_temperature)
    diag=retrieval_diagnostic(full,videos,text_keys)
    return {"retrieval_loss":float(loss.detach()),"r1":diag["r1"],
            "r5":diag["r5"],"queries":diag["queries"],
            "queries_at5":diag["queries_at5"],
            "num_pairs":len(videos),"negatives_masked":extra["masked_pairs"]}


def run_training(cfg:VideoTextConfig,resume:str|None=None):
    cfg.validate()
    seed_all(cfg.seed)
    splits=check_disjoint_splits(cfg.train_ids,cfg.val_ids,cfg.test_ids)
    ds=VideoTextDataset(cfg,cfg.train_ids)
    val=VideoTextDataset(cfg,cfg.val_ids)
    num_videos=len(set(r.video_id for r in ds.refs))
    if num_videos<cfg.batch_size:
        raise ValueError(f"Training has {num_videos} source videos, but batch_size={cfg.batch_size}; "
                         "reduce batch_size or add videos. Do NOT duplicate same-video queries as negatives.")
    if len(set(r.video_id for r in val.refs))<2:
        raise ValueError("Validation retrieval needs >= 2 distinct held-out source videos")
    sampler=VideoDiverseBatchSampler(ds.refs,cfg.batch_size,cfg.seed)
    loader=DataLoader(ds,batch_sampler=sampler,collate_fn=collate_video_text,
                      num_workers=cfg.num_workers,pin_memory=torch.cuda.is_available())
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dims=dims(ds)
    if model_dims!=dims(val):
        raise ValueError("Feature dimensionality differs between train/val caches")
    model=VideoTextModel(cfg,*model_dims).to(device)
    optim=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    mixed=bool(cfg.amp and device.type=='cuda')
    scaler=torch.amp.GradScaler('cuda',enabled=mixed)
    out_dir=Path(cfg.output_dir);out_dir.mkdir(parents=True,exist_ok=True)
    checkpoint=out_dir/cfg.checkpoint_name
    start_epoch=0;best_r1=-1.0;global_step=0
    if resume:
        state=torch.load(resume,map_location=device,weights_only=True)
        if state.get('dims')!=list(model_dims):
            raise ValueError("Resume checkpoint feature dimensions differ from caches")
        if checkpoint_architecture(state.get('model_cfg', {})) != architecture_config(cfg):
            raise ValueError("Resume checkpoint model architecture mismatch")
        verify_checkpoint_text_source(state.get('config', {}), cfg)
        model.load_state_dict(state['model']);optim.load_state_dict(state['optimizer'])
        start_epoch=int(state['epoch']);global_step=int(state['step'])
        best_r1=float(state['best_r1'])
    logfile=out_dir/'history.jsonl'
    print(f"[video_text] device={device} train={len(ds)} val={len(val)} "
          f"videos={num_videos} feature_dims={model_dims} split_videos={splits}")
    if num_videos < 8:
        warnings.warn("Few original videos: check that signer and session are separated across splits")
    if not len(loader):
        raise ValueError("No diverse train batches. Check video counts and batch_size")
    stop=False
    for epoch in range(start_epoch,cfg.epochs):
        sampler.set_epoch(epoch)
        model.train();running=[];timing=time.perf_counter();active=0
        for batch in loader:
            data=move(batch,device)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=mixed):
                obj=model(data)
                matrix=mil_similarity_matrix(obj['text_vectors'],obj['span_vectors'],
                                             obj['span_mask'],cfg.mil_temperature)
                loss,info=contrastive_loss(matrix,batch['video_id'],batch['text_key'],
                                           cfg.contrastive_temperature)
            if info['active_queries']==0 and info['active_videos']==0:
                continue
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite contrastive loss')
            scaler.scale(loss).backward();scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip_norm)
            scaler.step(optim);scaler.update()
            global_step+=1;active+=1;running.append(float(loss.detach()))
            if global_step%20==0:
                elapsed=max(time.perf_counter()-timing,1e-6)
                print(f"[video_text] epoch={epoch+1} step={global_step} "
                      f"loss={np.mean(running):.4f} steps/sec={active/elapsed:.2f}")
            if cfg.max_train_steps and global_step>=cfg.max_train_steps:
                stop=True;break
        if not running:
            raise ValueError('No informative batches: check text duplicates or number of videos')
        valstats=evaluate(model,val,cfg,device)
        metric=valstats['r1']
        row={'epoch':epoch+1,'step':global_step,'train_loss':float(np.mean(running)),
             'seconds':round(time.perf_counter()-timing,3),'val':valstats}
        with logfile.open('a',encoding='utf-8') as fh:
            fh.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(f"[video_text] {json.dumps(row,ensure_ascii=False)}")
        # Checkpoint tracks held-out cross-video retrieval; null metric -> fail clearly.
        if metric is None:
            raise ValueError('Validation has no reliable cross-video distractors')
        if metric>best_r1:
            best_r1=metric
            torch.save({
                'model':model.state_dict(),'optimizer':optim.state_dict(),
                'epoch':epoch+1,'step':global_step,'best_r1':best_r1,
                'dims':list(model_dims),
                'model_cfg':architecture_config(cfg),
                'config':asdict(cfg), 'val':valstats,
            },checkpoint)
            print(f"[checkpoint] {checkpoint}")
        if stop:break
    return checkpoint


def evaluate_checkpoint(cfg:VideoTextConfig,checkpoint_path:str|Path):
    check_disjoint_splits(cfg.train_ids,cfg.val_ids,cfg.test_ids)
    dataset=VideoTextDataset(cfg,cfg.test_ids)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt=torch.load(checkpoint_path,map_location=device,weights_only=True)
    stored=ckpt['config'];now=asdict(cfg)
    verify_checkpoint_text_source(stored, cfg)
    for k in MODEL_KEYS:
        if stored.get(k, 'mean' if k == 'span_pooling' else None)!=now[k]:
            raise ValueError(f"Model architecture differs on key {k}")
    model=VideoTextModel(cfg,*ckpt['dims']).to(device)
    model.load_state_dict(ckpt['model'])
    return evaluate(model,dataset,cfg,device)
