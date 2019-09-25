"""
This script generates low- and high-cost annotations datasets based on the refined landmarks
annotated in via.

(c) Aleksei Tiulpin, University of Oulu, 2019
"""
import numpy as np
import os
import cv2
import pandas as pd
import argparse
import json

from tqdm import tqdm
from joblib import Parallel, delayed
from kneel.data.utils import save_original_from_via_annotations
cv2.ocl.setUseOpenCL(False)


def get_image_cb(entry, spacing):
    return cv2.imread(os.path.join(args.data_dir, entry.filename), 0), spacing


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', default='/media/lext/FAST/knee_landmarks/workdir/'
                                                 'oai_landmarks_00_via_format_refined.csv')
    parser.add_argument('--kl_info', default='/media/lext/FAST/knee_landmarks/workdir/'
                                             'oai_00_kl_info.csv')
    parser.add_argument('--to_save_meta', default='/media/lext/FAST/knee_landmarks/workdir/')
    parser.add_argument('--data_dir', default='/media/lext/FAST/knee_landmarks/workdir/oai_images/00/')
    parser.add_argument('--pad', default=100)
    parser.add_argument('--to_save_low_cost_img', default='/media/lext/FAST/knee_landmarks/workdir/low_cost_data')
    parser.add_argument('--to_save_high_cost_img', default='/media/lext/FAST/knee_landmarks/workdir/high_cost_data')
    parser.add_argument('--high_cost_spacing', type=float, default=0.3)
    parser.add_argument('--low_cost_spacing', type=int, default=1)
    parser.add_argument('--n_per_grade', type=int, default=150)
    parser.add_argument('--num_threads', type=int, default=30)
    parser.add_argument('--seed', type=int, default=123456)
    parser.add_argument('--sizemm', type=int, default=140)
    args = parser.parse_args()

    os.makedirs(args.to_save_low_cost_img, exist_ok=True)
    os.makedirs(args.to_save_high_cost_img, exist_ok=True)

    metadata = pd.read_csv(args.annotations)
    metadata['x'] = metadata.region_shape_attributes.apply(lambda x: int(json.loads(x)['cx']), 1)
    metadata['y'] = metadata.region_shape_attributes.apply(lambda x: int(json.loads(x)['cy']), 1)
    metadata['Bone'] = metadata.region_attributes.apply(lambda x: json.loads(x)['Bone'], 1)
    metadata['Side'] = metadata.region_attributes.apply(lambda x: json.loads(x)['Side'], 1)

    kl_metadata = pd.read_csv(args.kl_info)

    to_process = []
    for filename, group in metadata.groupby('filename'):
        kl_meta_i = kl_metadata[kl_metadata.ID == int(filename.split('.')[0])]
        to_process.append([filename, group, int(kl_meta_i.KL_right), int(kl_meta_i.KL_left), float(kl_meta_i.spacing)])

    info = []
    res = Parallel(args.num_threads)(delayed(save_original_from_via_annotations)(data_entry, args, get_image_cb) for
                                     data_entry in tqdm(to_process, total=len(to_process)))
    for r in res:
        info.extend(r)

    df = pd.DataFrame(data=info, columns=['subject_id', 'side', 'kl', 'tibia', 'femur', 'bbox', 'center'])
    df.to_csv(os.path.join(args.to_save_meta,
                           f'bf_landmarks_{args.low_cost_spacing}_{args.high_cost_spacing}.csv'), index=False)
