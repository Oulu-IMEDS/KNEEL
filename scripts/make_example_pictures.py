import argparse
import pandas as pd
import glob
import os
import numpy as np
from kneel.evaluation import visualize_landmarks
import matplotlib.pyplot as plt
from kneel.data.utils import parse_landmarks
import cv2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', default='')
    parser.add_argument('--lc_images', default='')
    parser.add_argument('--hc_images', default='')
    parser.add_argument('--save_dir', default='')
    args = parser.parse_args()

    np.random.seed(1234567)
    annotations = pd.read_csv(args.annotations)
    pics_dir = os.path.join(args.save_dir, 'paper_pics')
    os.makedirs(pics_dir, exist_ok=True)
    for kl in range(5):
        subject_id, side, kl, t_lnd, f_lnd, _, center = annotations[annotations.kl == kl].sample(1, axis=0).iloc[0]
        t_lnd, f_lnd = parse_landmarks(t_lnd), parse_landmarks(f_lnd)
        fname = os.path.join(args.hc_images, f'{subject_id}_{kl}_{side}.png')
        print(fname, kl)
        img = cv2.imread(fname, 0)
        visualize_landmarks(img, t_lnd, f_lnd, save_path=os.path.join(pics_dir, f'hc-{kl}.pdf'))
