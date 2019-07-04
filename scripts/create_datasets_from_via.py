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

cv2.ocl.setUseOpenCL(False)


def worker(data_entry, args):
    filename, annotations, klr, kll, spacing = data_entry
    subject_id = filename.split('.')[0]

    pad = args.pad
    sizemm = args.sizemm

    img_original = cv2.imread(os.path.join(args.data_dir, filename), 0)
    scale = spacing / args.high_cost_spacing
    scale_lc = spacing / args.low_cost_spacing
    spacing = args.high_cost_spacing

    bbox_width_pix = int(sizemm / spacing)
    img = cv2.resize(img_original, (int(img_original.shape[1] * scale), int(img_original.shape[0] * scale)))
    img_lc = cv2.resize(img_original, (int(img_original.shape[1] * scale_lc), int(img_original.shape[0] * scale_lc)))

    row, col = img.shape
    tmp = np.zeros((row + 2 * pad, col + 2 * pad))
    tmp[pad:pad + row, pad:pad + col] = img
    img = tmp
    row, col = img.shape

    landmarks = {}
    centers = {}
    bboxes = {}
    sides = []
    for side, grp_side in annotations.groupby('Side'):
        for bone, grp_side_bone in grp_side.groupby('Bone'):
            points = grp_side_bone[['x', 'y']].values * scale + pad
            landmarks[bone+side] = points
            if bone == 'T':
                # Defining the centers
                cx, cy = landmarks[f'T{side}'][landmarks[f'T{side}'].shape[0] // 2, :].astype(int)
                centers[side] = (cx, cy)
                # Defining the bounding boxes for the cropped images
                bboxes[side] = [cx - bbox_width_pix // 2, cy - bbox_width_pix // 2,
                                cx + bbox_width_pix // 2, cy + bbox_width_pix // 2]

        sides.append(side)

    res = []
    for side in sides:
        kl = klr if side == 'R' else kll

        if side == 'R':
            localized_img = img[bboxes['R'][1]:bboxes['R'][3], bboxes['R'][0]:bboxes['R'][2]]
        else:
            localized_img = cv2.flip(img[bboxes['L'][1]:bboxes['L'][3], bboxes['L'][0]:bboxes['L'][2]], 1)

        landmarks[f'T{side}'] -= bboxes[side][:2]
        landmarks[f'F{side}'] -= bboxes[side][:2]

        if side == 'L':
            # Inverting the left landmarks
            landmarks[f'T{side}'][:, 0] = bbox_width_pix - landmarks[f'T{side}'][:, 0]
            landmarks[f'F{side}'][:, 0] = bbox_width_pix - landmarks[f'F{side}'][:, 0]

        landmarks[f'T{side}'] = np.round(landmarks[f'T{side}']).astype(int)
        landmarks[f'F{side}'] = np.round(landmarks[f'F{side}']).astype(int)

        cv2.imwrite(os.path.join(args.to_save_high_cost_img, f'{subject_id}_{kl}_{side}.png'), localized_img)
        if not os.path.isfile(os.path.join(args.to_save_low_cost_img, f'{subject_id}.png')):
            cv2.imwrite(os.path.join(args.to_save_low_cost_img, f'{subject_id}.png'), img_lc)

        tibial_landmarks = ''.join(map(lambda x: '{},{},'.format(*x), landmarks[f'T{side}']))[:-1]
        femoral_landmarks = ''.join(map(lambda x: '{},{},'.format(*x), landmarks[f'F{side}']))[:-1]

        tmp = [subject_id, side, kl, tibial_landmarks, femoral_landmarks,
               f"{bboxes[side][0]},{bboxes[side][1]},{bboxes[side][2]},{bboxes[side][3]}",
               f"{centers[side][0]},{centers[side][1]}"]
        res.append(tmp)

    return res


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', default='/media/lext/FAST/knee_landmarks/workdir/'
                                                 'oai_landmarks_00_via_format_refined.csv')
    parser.add_argument('--kl_info', default='/media/lext/FAST/knee_landmarks/workdir/'
                                             'oai_00_kl_info.csv')
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
    res = Parallel(args.num_threads)(delayed(worker)(data_entry, args) for data_entry in tqdm(to_process,
                                                                                              total=len(to_process)))
    for r in res:
        info.append(r)

    df = pd.DataFrame(data=info, columns=['subject_id', 'side', 'folder', 'kl', 'tibia', 'femur', 'bbox', 'center'])
    df.to_csv(os.path.join(args.to_save_meta,
                           f'bf_landmarks_{args.low_cost_spacing}_{args.high_cost_spacing}.csv'), index=False)
