import argparse
import json
from pathlib import Path


from persian_sea.features.cache import FeatureCache

def main()-> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('cache')
    args = parser.parse_args()
    cache = FeatureCache(Path(args.cache))
    print(json.dumps(cache.summary(),indent=2,ensure_ascii=False))


if __name__ == '__main__':
    main()
    