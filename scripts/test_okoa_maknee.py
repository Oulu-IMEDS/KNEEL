import argparse
from kneel.inference.lc import GlobalSearcher

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='')
    parser.add_argument('--lc_snapshot_path', default='')
    parser.add_argument('--hc_snapshot_path', default='')
    parser.add_argument('--mean_std_path', default='')
    args = parser.parse_args()

    global_searcher = GlobalSearcher(snapshot_path=args.lc_snapshot_path,
                                     mean_std_path=args.mean_std_path)


