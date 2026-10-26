

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from persian_sea.io.subtitles import find_subtitle, read_subtitles
from persian_sea.translation.data import inference_windows


def interval_iou(a, b, c, d):
    inter = max(0., min(b,d) - max(a,c))
    union = max(b,d) - min(a,c)
    return inter / union if union > 0 else 0.


def audit(video_id, visual_dir, subtitle_dir, window_size, stride):
    path = Path(visual_dir) / f"{video_id}.npz"
    subtitle = find_subtitle(Path(subtitle_dir), video_id)
    if subtitle is None:
        raise FileNotFoundError(f"Raw VTT missing for {video_id}")
    with np.load(path, allow_pickle=False) as npz:
        data = {k:np.asarray(npz[k]) for k in ("starts","ends")}
    if (len(data['starts']) == 0 or len(data['starts']) != len(data['ends'])
        or not np.isfinite(data['starts']).all() or not np.isfinite(data['ends']).all()
        or np.any(data['ends'] <= data['starts']) or np.any(np.diff(data['starts']) < 0)):
        raise ValueError('Invalid segment times in feature cache')
    windows = inference_windows(data, window_size, stride)
    times = [(float(data['starts'][w[0]]), float(data['ends'][w[-1]])) for w in windows]
    cues = read_subtitles(subtitle)
    best = [max(interval_iou(c.start,c.end,a,b) for a,b in times) for c in cues]
    return {"video_id": video_id,"window_size":window_size,"stride":stride,
            "num_windows":len(windows),"num_cues":len(cues),
            "mean_best_iou_to_raw_vtt":sum(best)/len(best) if best else 0.,
            "recall_at_0_25":sum(v>=.25 for v in best)/len(best) if best else 0.,
            "recall_at_0_50":sum(v>=.5 for v in best)/len(best) if best else 0.,
            "recall_at_0_75":sum(v>=.75 for v in best)/len(best) if best else 0.,
            "warning":"Raw VTT is weak timing reference, not annotated ground truth."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video-id',required=True)
    p.add_argument('--visual-dir',default='outputs/features')
    p.add_argument('--subtitles-dir',default='data/subtitles_raw')
    p.add_argument('--window-size',type=int,default=32)
    p.add_argument('--stride',type=int,default=32)
    p.add_argument('--report',default=None)
    a=p.parse_args()
    r=audit(a.video_id,a.visual_dir,a.subtitles_dir,a.window_size,a.stride)
    dest=Path(a.report or f'outputs/reports/{a.video_id}_window_mismatch.json')
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(r,ensure_ascii=False,indent=2))
    print('Report:',dest)

if __name__=='__main__': main()
