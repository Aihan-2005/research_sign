
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np



def read_ids(path: Path) -> list[str]:
    ids = [x.strip() for x in path.read_text(encoding='utf-8').splitlines()
           if x.strip() and not x.lstrip().startswith('#')]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f'Empty or repeated video IDs: {path}')
    return ids


def audit(train: Path, val: Path, test: Path, visual_dir: Path,
          raw_dir: Path, sign_gt_dir: Path, subtitle_gt_dir: Path) -> dict:
    from persian_sea.io.subtitles import find_subtitle, read_subtitles
    splits = {'train': read_ids(train), 'val': read_ids(val), 'test': read_ids(test)}
    errors, warnings, per_video = [], [], []
    for a in splits:
        for b in splits:
            if a >= b: continue
            overlap = set(splits[a]) & set(splits[b])
            if overlap:
                errors.append(f'{a}/{b}: {len(overlap)} overlapping video IDs: {sorted(overlap)[:4]}')
    for split, vids in splits.items():
        for vid in vids:
            entry = {'split': split, 'video_id': vid, 'visual_ok': False,
                     'verified_subtitle_gt': False, 'segmentation_gt': False}
            raw = find_subtitle(raw_dir, vid)
            if raw is None:
                errors.append(f'{split}/{vid}: missing raw VTT')
            else:
                try:
                    cues = read_subtitles(raw)
                    if not cues: raise ValueError('empty VTT')
                    entry['cues'] = len(cues)
                except (ValueError, OSError) as exc:
                    errors.append(f'{split}/{vid}: VTT {exc}')
            visual = visual_dir / f'{vid}.npz'
            if not visual.is_file():
                errors.append(f'{split}/{vid}: missing visual cache {visual}')
            else:
                try:
                    with np.load(visual, allow_pickle=False) as f:
                        pose, rgb = f['pose'], f['rgb']
                        starts, ends = f['starts'], f['ends']
                    n = len(starts)
                    if not (pose.ndim == 4 and rgb.ndim == 4 and n > 0 and
                            pose.shape[0] == rgb.shape[0] == len(ends) == n and
                            pose.shape[-2:] == (75,4) and
                            pose.shape[1] == rgb.shape[1] and
                            np.isfinite(pose).all() and np.isfinite(rgb).all() and
                            np.isfinite(starts).all() and np.isfinite(ends).all() and
                            np.all(starts >= 0) and np.all(ends > starts) and
                            np.all(np.diff(starts) >= 0)):
                        raise ValueError('invalid shape, temporal order or nonfinite numbers')
                    entry['visual_ok'] = True
                    entry['segments'] = n
                    entry['cache_pose_shape'] = list(pose.shape)
                    entry['cache_rgb_shape'] = list(rgb.shape)
                except (OSError, KeyError, ValueError) as exc:
                    errors.append(f'{split}/{vid}: cache {exc}')
            entry['verified_subtitle_gt'] = find_subtitle(subtitle_gt_dir, vid) is not None
            entry['segmentation_gt'] = (sign_gt_dir / f'{vid}.eaf').is_file()
            per_video.append(entry)
    if not any(row['verified_subtitle_gt'] for row in per_video if row['split'] != 'train'):
        warnings.append('No independent val/test subtitle GT for IoU/Oracle Coverage')
    if not any(row['segmentation_gt'] for row in per_video):
        warnings.append('No E4s sign segmentation GT; cannot measure sign boundary F1')
    warnings.append('Signer/session independence NOT checked here: run signer audit with real metadata')
    warnings.append('Text-cache provenance NOT checked here: run scripts/pretrain_gate.py')
    return {'ok': not errors, 'errors': errors, 'warnings': warnings,
            'splits': {k:len(v) for k,v in splits.items()}, 'videos': per_video}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('train-ids','val-ids','test-ids'):
        p.add_argument('--'+k, default='data/'+k.replace('-ids','')+'_ids.txt')
    p.add_argument('--visual-dir', default='outputs/features')
    p.add_argument('--raw-dir', default='data/subtitles_raw')
    p.add_argument('--segmentation-gt-dir', default='data/segmentation_gt')
    p.add_argument('--subtitle-gt-dir', default='data/subtitles_gt')
    p.add_argument('--report', default='outputs/reports/phase12_audit.json')
    a=p.parse_args()
    result = audit(Path(a.train_ids), Path(a.val_ids), Path(a.test_ids),
        Path(a.visual_dir), Path(a.raw_dir), Path(a.segmentation_gt_dir), Path(a.subtitle_gt_dir))
    dest=Path(a.report)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'ok':result['ok'],'errors':result['errors'],'warnings':result['warnings']},ensure_ascii=False,indent=2))
    print('Report:',dest)
    if not result['ok']: raise SystemExit(2)

if __name__ == '__main__': main()
