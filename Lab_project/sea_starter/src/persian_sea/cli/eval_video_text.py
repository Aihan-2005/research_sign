
import argparse,json
from persian_sea.video_text.config import read_config
from persian_sea.video_text.train import evaluate_checkpoint


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--checkpoint',required=True)
    a=p.parse_args()
    print(json.dumps(evaluate_checkpoint(read_config(a.config),a.checkpoint),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
