import argparse,json
from persian_sea.video_text.config import read_config
from persian_sea.video_text.split_audit import audit_signer_splits


def main():
    p=argparse.ArgumentParser(description='Audit signer and source-video leakage')
    p.add_argument('--config',required=True)
    p.add_argument('--metadata',required=True,help='CSV: video_id,signer_id,source_recording_id')
    p.add_argument('--strict-signer',action='store_true')
    a=p.parse_args()
    c=read_config(a.config)
    print(json.dumps(audit_signer_splits(a.metadata,c.train_ids,c.val_ids,c.test_ids,
                        a.strict_signer),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
