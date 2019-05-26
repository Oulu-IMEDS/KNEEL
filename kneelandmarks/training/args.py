import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='localized_rois/')
    parser.add_argument('--workdir', default='/media/lext/FAST/knee_landmarks/workdir')
    parser.add_argument('--meta', default='bf_landmarks_2_0.3.csv')
    parser.add_argument('--base_width', type=int, default=24)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--start_val', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=5e-5)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--val_bs', type=int, default=16)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--n_threads', type=int, default=12)
    parser.add_argument('--hc_spacing', type=float, default=0.3)
    parser.add_argument('--lc_spacing', type=int, default=2)
    parser.add_argument('--crop_size', type=int, default=2)
    parser.add_argument('--pad_size', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()
