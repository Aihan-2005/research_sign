
from __future__ import annotations
import torch
import torch.nn.functional as F


def contrastive_loss(scores:torch.Tensor,video_ids:list[str],text_keys:list[str],
                     temperature:float=0.07):
    B=scores.shape[0]
    if scores.shape!=(B,B) or len(video_ids)!=B or len(text_keys)!=B:
        raise ValueError("Mismatch between score matrix and sample identifiers")
    if temperature<=0: raise ValueError("temperature must be positive")
    allowed=torch.ones((B,B),dtype=torch.bool,device=scores.device)
    for i in range(B):
        for j in range(B):
            if i!=j and (video_ids[i]==video_ids[j] or text_keys[i]==text_keys[j]):
                allowed[i,j]=False
    logits=scores.float()/temperature
    logits=logits.masked_fill(~allowed,-1e4)
    target=torch.arange(B,device=scores.device)
    row_valid=allowed.sum(1)>1
    col_valid=allowed.sum(0)>1
    terms=[]
    if row_valid.any():
        terms.append(F.cross_entropy(logits[row_valid],target[row_valid]))
    if col_valid.any():
        terms.append(F.cross_entropy(logits.T[col_valid],target[col_valid]))
    if not terms:
        return scores.sum()*0.0,{"active_queries":0,"active_videos":0,"masked_pairs":int((~allowed).sum())}
    return torch.stack(terms).mean(),{
        "active_queries":int(row_valid.sum()),
        "active_videos":int(col_valid.sum()),
        "masked_pairs":int((~allowed).sum())
    }


@torch.no_grad()
def retrieval_diagnostic(scores:torch.Tensor,video_ids:list[str],text_keys:list[str]):


    B=scores.shape[0]
    if B<2: return {"r1":None,"r5":None,"queries":0,"queries_at5":0}
    mask=torch.ones(B,B,dtype=torch.bool,device=scores.device)
    for i in range(B):
        for j in range(B):
            if i!=j and (video_ids[i]==video_ids[j] or text_keys[i]==text_keys[j]):
                mask[i,j]=False
    valid=mask.sum(1)>1
    if not valid.any(): return {"r1":None,"r5":None,"queries":0,"queries_at5":0}
    ranked=scores.masked_fill(~mask,-1e4).argsort(dim=1,descending=True)
    hits=(ranked==torch.arange(B,device=scores.device)[:,None])
    ranks=hits.float().argmax(1)[valid]
    valid5=mask.sum(1)>=5
    ranks5=hits.float().argmax(1)[valid5]
    return {"r1":float((ranks<1).float().mean()),
            "r5":float((ranks5<5).float().mean()) if valid5.any() else None,
            "queries":int(valid.sum()),
            "queries_at5":int(valid5.sum())}
