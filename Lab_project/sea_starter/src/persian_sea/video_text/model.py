

from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from .config import VideoTextConfig
from .span_pooling import SpanPooler


class VideoTower(nn.Module):
    def __init__(self, visual: nn.Module, dim:int, layers:int, heads:int, microbatch:int):
        super().__init__()
        self.visual = visual
        self.microbatch = int(microbatch)
        if layers:
            layer=nn.TransformerEncoderLayer(
                d_model=dim,nhead=heads,dim_feedforward=dim*2,
                batch_first=True,norm_first=True,dropout=.1,activation="gelu"
            )
            self.temporal=nn.TransformerEncoder(layer,num_layers=layers)
        else:
            self.temporal=None
        self.time_projection=nn.Sequential(nn.Linear(3,dim),nn.GELU(),nn.Linear(dim,dim))
        self.layer_norm=nn.LayerNorm(dim)
        self.dim=dim

    def forward(self,pose,rgb,starts,ends,mask):
        B,M=mask.shape
        flat_idx=torch.where(mask.reshape(-1))[0]
        if flat_idx.numel()==0:
            raise ValueError("Every batch needs at least one sign segment")
        P=pose.reshape(B*M,*pose.shape[2:]);R=rgb.reshape(B*M,*rgb.shape[2:])
        groups=[]
        for ids in flat_idx.split(self.microbatch):
            groups.append(self.visual(P[ids],R[ids])["fused"])
        encoded=torch.cat(groups,dim=0)
        if encoded.shape[-1]!=self.dim:
            raise ValueError(f"Visual output dimension must equal {self.dim}")
        flat=encoded.new_zeros(B*M,self.dim).index_copy(0,flat_idx,encoded)
        tokens=flat.reshape(B,M,self.dim)
        base=starts.masked_fill(~mask,float("inf")).amin(dim=1,keepdim=True)
        limit=ends.masked_fill(~mask,-float("inf")).amax(dim=1,keepdim=True)
        extent=(limit-base).clamp_min(1e-3)
        middle=(starts+ends)/2
        rel_middle=2*(middle-base)/extent-1
        length=(ends-starts).clamp_min(1e-3)
        prev=F.pad(ends[:,:-1],(1,0),value=0)
        prev[:,0]=starts[:,0]
        gap=(starts-prev).clamp_min(0)
        timing=torch.stack([rel_middle,torch.log1p(length),torch.log1p(gap)],dim=-1)
        timing=timing.masked_fill(~mask[...,None],0.0)
        tokens=tokens+self.time_projection(timing)
        if self.temporal is not None:
            tokens=self.temporal(tokens,src_key_padding_mask=~mask)
        tokens=self.layer_norm(tokens)
        return F.normalize(tokens,dim=-1)*mask[...,None]


class TextTower(nn.Module):
    def __init__(self,text_dim:int,output_dim:int):
        super().__init__()
        self.network=nn.Sequential(nn.LayerNorm(text_dim),nn.Linear(text_dim,output_dim),
                                   nn.GELU(),nn.Linear(output_dim,output_dim))
    def forward(self,x):
        return F.normalize(self.network(x.float()),dim=-1)


def multi_scale_spans(tokens,mask,sizes:tuple[int,...]):
    """Pool consecutive 1/2/4/8 sign segments, invalidating windows with padding.

    B,M,D -> B,S,D + B,S where S=sum(M-k+1 for k<=M).
    """
    B,M,D=tokens.shape
    sums=torch.cat((tokens.new_zeros((B,1,D)),tokens.cumsum(dim=1)),dim=1)
    counts=torch.cat((tokens.new_zeros((B,1,1)),mask.to(tokens.dtype).cumsum(dim=1)[...,None]),dim=1)
    spans=[];valid=[]
    for k in sizes:
        if k>M: continue
        pooled=(sums[:,k:]-sums[:,:-k])/k
        usable=(counts[:,k:]-counts[:,:-k]).squeeze(-1)==k
        spans.append(F.normalize(pooled,dim=-1))
        valid.append(usable)
    if not spans: raise ValueError("No valid span sizes")
    combined=torch.cat(spans,dim=1)
    mask_out=torch.cat(valid,dim=1)
    if not mask_out.any(dim=1).all():
        raise ValueError("Some video bags contain no valid spans")
    return combined,mask_out


class VideoTextModel(nn.Module):
    def __init__(self,cfg:VideoTextConfig,text_dim:int,rgb_dim:int,num_views:int,
                 visual:nn.Module|None=None):
        super().__init__()
        if visual is None:
            # Existing modular model registry stays the single source of truth.
            from persian_sea.model.builders import build_visual_encoder
            visual=build_visual_encoder(cfg.visual_backend,pose_input_dim=4,
                 rgb_input_dim=rgb_dim,num_views=num_views,
                 model_dim=cfg.model_dim,output_dim=cfg.embedding_dim)
        self.video=VideoTower(visual,cfg.embedding_dim,cfg.temporal_layers,
                              cfg.temporal_heads,cfg.visual_microbatch)
        self.text=TextTower(text_dim,cfg.embedding_dim)
        self.span_sizes=cfg.span_sizes
        self.span_pooler = SpanPooler(cfg.embedding_dim, cfg.span_sizes,
                                      backend=cfg.span_pooling)
        self.mil_temperature=cfg.mil_temperature

    def encode_video(self,batch:dict):
        """Public transfer interface for future translation models: [B,M,D]."""
        return self.video(batch["pose"],batch["rgb"],batch["starts"],
                          batch["ends"],batch["mask"])

    def forward(self,batch:dict):
        tokens=self.encode_video(batch)
        spans,span_mask=self.span_pooler(tokens,batch["mask"])
        text=self.text(batch["text_embedding"])
        return {"video_tokens":tokens,"span_vectors":spans,
                "span_mask":span_mask,"text_vectors":text}


def mil_similarity_matrix(text_vectors,span_vectors,span_mask,temperature:float):
    """Text i to video-bag j, BxB; normalized for different bag lengths."""
    temp=max(float(temperature),1e-4)
    sims=torch.einsum("id,jsd->ijs",text_vectors.float(),span_vectors.float())
    sims=sims.masked_fill(~span_mask[None,:,:],-1e4)
    # Normalise by number of valid windows, avoiding the bag-size shortcut.
    count=span_mask.sum(dim=1).clamp_min(1).float().log()
    return temp*torch.logsumexp(sims/temp,dim=2)-temp*count[None,:]
