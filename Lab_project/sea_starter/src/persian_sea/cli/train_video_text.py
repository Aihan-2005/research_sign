
import argparse
from persian_sea.video_text.config import read_config
from persian_sea.video_text.train import run_training


def main():
    p=argparse.ArgumentParser(description='Train weak video-text embeddings from cached sign features')
    p.add_argument('--config',required=True)
    p.add_argument('--resume',default=None,help='Checkpoint from same model architecture')
    a=p.parse_args()
    print('Best checkpoint:',run_training(read_config(a.config),resume=a.resume))

if __name__=='__main__':main()
