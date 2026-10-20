

from __future__ import annotations
import csv
from pathlib import Path
from collections import defaultdict
from .dataset import check_disjoint_splits


def audit_signer_splits(csv_path: str|Path,train_ids:str|Path,val_ids:str|Path,test_ids:str|Path,
                        strict_signer:bool=False)->dict:
    video_counts=check_disjoint_splits(str(train_ids),str(val_ids),str(test_ids))
    def readids(path):
        return {s.strip() for s in Path(path).read_text(encoding='utf-8').splitlines()
                if s.strip() and not s.lstrip().startswith('#')}
    splits={name:readids(path) for name,path in
            (('train',train_ids),('val',val_ids),('test',test_ids))}
    meta={}
    with Path(csv_path).open(newline='',encoding='utf-8-sig') as file:
        reader=csv.DictReader(file)
        required={'video_id','signer_id','source_recording_id'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Missing required CSV headers: {required}')
        for row in reader:
            vid=(row['video_id'] or '').strip()
            if not vid or vid in meta:raise ValueError(f'Empty/duplicate video_id in metadata: {vid!r}')
            meta[vid]=row
    not_found=set.union(*splits.values())-set(meta)
    if not_found:raise ValueError(f'Missing metadata entries: {sorted(not_found)}')
    owners=defaultdict(set);sessions=defaultdict(set);unknown=[]
    for split,ids in splits.items():
        for vid in ids:
            row=meta[vid]
            signer=(row['signer_id'] or '').strip()
            source=(row['source_recording_id'] or '').strip()
            if signer:owners[signer].add(split)
            else:unknown.append(vid)
            if source:sessions[source].add(split)
            else:unknown.append(vid+':source_recording_id')
    conflicting_sessions={k:sorted(v) for k,v in sessions.items() if len(v)>1}
    if conflicting_sessions:
        raise ValueError(f'Same source recording leaked across splits: {conflicting_sessions}')
    conflicting_signers={k:sorted(v) for k,v in owners.items() if len(v)>1}
    if strict_signer and (conflicting_signers or unknown):
        raise ValueError(f'Signer-disjoint split invalid: {conflicting_signers}; unknown={unknown}')
    return {'video_counts':video_counts,'signers_across_splits':conflicting_signers,
            'unknown_metadata':unknown,'strict_signer':strict_signer,
            'unique_signers':len(owners),'unique_source_recordings':len(sessions)}
