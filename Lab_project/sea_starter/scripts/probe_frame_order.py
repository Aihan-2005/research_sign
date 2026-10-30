
from __future__ import annotations
import argparse
import json
import torch


def run(seed=42, frames=8, rgb_dim=768, views=4):
    from persian_sea.model.builders import build_visual_encoder
    torch.manual_seed(seed)
    p=torch.randn(1, frames,75,4)
    r=torch.randn(1, frames,views,rgb_dim)
    result={}
    for backend in ('pose_rgb','pose_rgb_positional'):
        torch.manual_seed(seed)
        model=build_visual_encoder(backend,pose_input_dim=4,rgb_input_dim=rgb_dim,
                                   model_dim=384, output_dim=256,num_views=views)
        model.eval()
        with torch.no_grad():
            z=model(p,r)['fused']
            zrev=model(p.flip(1),r.flip(1))['fused']
        result[backend]={'cosine_reversal':float(torch.nn.functional.cosine_similarity(z,zrev).item()),
                         'L2_difference':float(torch.linalg.vector_norm(z-zrev).item())}
    result['disclaimer']='Untrained random model probe, not trained sign recognition accuracy.'
    return result

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--seed',type=int,default=42)
    a.add_argument('--output',default='outputs/reports/frame_order_probe.json')
    args=a.parse_args()
    from pathlib import Path
    x=run(seed=args.seed)
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(x,indent=2))
