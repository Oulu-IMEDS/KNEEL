import numpy as np
import os
import cv2
import pandas as pd
import argparse
import glob
from tqdm import tqdm
from joblib import Parallel, delayed
from kneelandmarks.data.utils import read_dicom, process_xray, read_pts, read_sas7bdata_pd

cv2.ocl.setUseOpenCL(False)


def worker(data_entry, args):
    subject_id, side, folder, kl = data_entry

    info = []
    pad = args.pad
    sizemm = args.sizemm

    dicom_name = os.path.join(args.oai_data_dir, 'OAI_00m', folder, '001')

    # Not the most effective way, but works OK
    landmarks_dir = glob.glob(os.path.join(args.oai_data_dir,  'landmarks_oai', '00',
                                           folder.split('/')[0], '*', '/'.join(folder.split('/')[1:])))[0]

    res = read_dicom(dicom_name)
    if res is None:
        return info
    img, spacing, _ = res
    img_original = process_xray(img).astype(np.uint8)
    if img_original.shape[0] == 0 or img_original.shape[1] == 0:
        return info
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

    points = np.round(read_pts(os.path.join(landmarks_dir, '001.pts')) * 1 / spacing) + pad
    landmarks_fl = points[list(range(12, 25, 2)), :]
    landmarks_tl = points[list(range(47, 64, 2)), :]

    points = np.round(read_pts(os.path.join(landmarks_dir, '001_f.pts')) * 1 / spacing) + pad
    landmarks_fr = points[list(range(12, 25, 2)), :]
    landmarks_tr = points[list(range(47, 64, 2)), :]

    landmarks_fr[:, 0] = col - landmarks_fr[:, 0]
    landmarks_tr[:, 0] = col - landmarks_tr[:, 0]

    landmarks = {'TR': landmarks_tr, 'FR': landmarks_fr,
                 'TL': landmarks_tl, 'FL': landmarks_fl}

    # Low-cost annotations in
    # padded image coordinate system
    lcx, lcy = landmarks['TL'][landmarks['TL'].shape[0] // 2, :].astype(int)
    rcx, rcy = landmarks['TR'][landmarks['TR'].shape[0] // 2, :].astype(int)
    centers = {'L': (lcx, lcy), 'R': (rcx, rcy)}

    # Defining the bounding boxes for the cropped images
    bbox_r = [rcx - bbox_width_pix // 2, rcy - bbox_width_pix // 2,
              rcx + bbox_width_pix // 2, rcy + bbox_width_pix // 2]

    bbox_l = [lcx - bbox_width_pix // 2, lcy - bbox_width_pix // 2,
              lcx + bbox_width_pix // 2, lcy + bbox_width_pix // 2]

    bboxes = {'L': bbox_l, 'R': bbox_r}

    if side == 'R':
        localized_img = img[bbox_r[1]:bbox_r[3], bbox_r[0]:bbox_r[2]]
    else:
        localized_img = cv2.flip(img[bbox_l[1]:bbox_l[3], bbox_l[0]:bbox_l[2]], 1)

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

    return [subject_id, side, folder, kl,
            tibial_landmarks, femoral_landmarks,
            f"{bboxes[side][0]},{bboxes[side][1]},{bboxes[side][2]},{bboxes[side][3]}",
            f"{centers[side][0]},{centers[side][1]}"]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--oai_data_dir', default='/media/lext/FAST/knee_landmarks/Data/')
    parser.add_argument('--pad', default=100)
    parser.add_argument('--to_save_low_cost_img', default='/media/lext/FAST/knee_landmarks/workdir/low_cost_data')
    parser.add_argument('--to_save_high_cost_img', default='/media/lext/FAST/knee_landmarks/workdir/high_cost_data')
    parser.add_argument('--to_save_meta', default='/media/lext/FAST/knee_landmarks/workdir/')
    parser.add_argument('--high_cost_spacing', type=float, default=0.3)
    parser.add_argument('--low_cost_spacing', type=int, default=1)
    parser.add_argument('--n_per_grade', type=int, default=150)
    parser.add_argument('--num_threads', type=int, default=30)
    parser.add_argument('--seed', type=int, default=123456)
    parser.add_argument('--sizemm', type=int, default=140)
    args = parser.parse_args()

    os.makedirs(args.to_save_low_cost_img, exist_ok=True)
    os.makedirs(args.to_save_high_cost_img, exist_ok=True)

    # Getting the info about the file locations
    contents = pd.read_csv(os.path.join(args.oai_data_dir, 'OAI_00m', 'contents.csv'))
    contents.rename(index=str, columns={"ParticipantID": "ID"}, inplace=True)
    contents = contents[contents.SeriesDescription == 'Bilateral PA Fixed Flexion Knee']
    # Getting the info about the KL grade for each side and preparing these data form merge
    kl_data = read_sas7bdata_pd(os.path.join(args.oai_data_dir, 'kxr_sq_bu00.sas7bdat'))
    kl_data.drop_duplicates(subset=['ID', 'SIDE'], inplace=True)

    right = kl_data[kl_data.SIDE == 1][['ID', 'V00XRKL']].rename(index=str, columns={'V00XRKL': 'KL_right'})
    left = kl_data[kl_data.SIDE == 2][['ID', 'V00XRKL']].rename(index=str, columns={'V00XRKL': 'KL_left'})
    kl_data = pd.merge(right, left, on='ID')

    kl_data.ID = kl_data.ID.values.astype(int)
    contents.ID = contents.ID.values.astype(int)
    contents = pd.merge(contents, kl_data, on='ID').drop(['StudyDate', 'Barcode',
                                                          'StudyDescription',
                                                          'SeriesDescription'],
                                                         axis=1)

    contents = contents[(~contents.KL_right.isna()) & (~contents.KL_left.isna())]

    # Preparing the data for parallel processing
    to_process = []
    np.random.seed(args.seed)
    indices = np.arange(contents.shape[0], dtype=int)
    for kl in range(5):
        idx_left = np.random.choice(indices[contents.KL_left.values == kl], size=args.n_per_grade // 2, replace=False)
        idx_right = np.random.choice(indices[contents.KL_right.values == kl], size=args.n_per_grade // 2, replace=False)

        ids_left = contents.iloc[idx_left].ID.values
        ids_right = contents.iloc[idx_right].ID.values

        folders_left = contents.iloc[idx_left].Folder.values
        folders_right = contents.iloc[idx_right].Folder.values

        side_id_left = np.array(['L', ] * (args.n_per_grade // 2))
        side_id_right = np.array(['R', ] * (args.n_per_grade // 2))

        kls = np.zeros(args.n_per_grade // 2, dtype=int) + kl

        left = np.vstack((ids_left, side_id_left, folders_left, kls)).T.tolist()
        left = list(map(lambda x: [int(x[0]), x[1], x[2], int(x[3])], left))

        right = np.vstack((ids_right, side_id_right, folders_right, kls)).T.tolist()
        right = list(map(lambda x: [int(x[0]), x[1], x[2], int(x[3])], right))

        # We will run the loop over this array
        to_process.extend(left)
        to_process.extend(right)

    info = []
    res = Parallel(args.num_threads)(delayed(worker)(data_entry, args) for data_entry in tqdm(to_process,
                                                                                              total=len(to_process)))
    for r in res:
        info.append(r)

    df = pd.DataFrame(data=info, columns=['subject_id', 'side', 'folder', 'kl', 'tibia', 'femur', 'bbox', 'center'])
    df.to_csv(os.path.join(args.to_save_meta,
                           f'bf_landmarks_{args.low_cost_spacing}_{args.high_cost_spacing}.csv'), index=False)
