import argparse
import glob
import pandas as pd
import numpy as np
from kneelandmarks.data.utils import read_dicom, process_xray, read_pts, read_sas7bdata_pd
import os
from joblib import Parallel, delayed
import cv2
from tqdm import tqdm


def worker(data_entry, args):
    subject_id, folder = data_entry

    info = []

    dicom_name = os.path.join(args.oai_data_dir, f'OAI_{args.fu}m', folder, '001')

    # Not the most effective way, but works OK
    landmarks_dir = glob.glob(os.path.join(args.oai_data_dir,
                                           'landmarks_oai', args.fu,
                                           folder.split('/')[0], '*', '/'.join(folder.split('/')[1:])))[0]

    res = read_dicom(dicom_name)
    if res is None:
        return info
    img, spacing, _ = res
    img = process_xray(img).astype(np.uint8)
    if img.shape[0] == 0 or img.shape[1] == 0:
        return info

    img_saved_path = os.path.join(args.workdir, 'oai_images', args.fu, str(subject_id) + '.png')
    cv2.imwrite(img_saved_path, img)

    row, col = img.shape
    points = np.round(read_pts(os.path.join(landmarks_dir, '001.pts')) * 1 / spacing)
    landmarks_fl = points[list(range(12, 25, 2)), :]
    landmarks_tl = points[list(range(47, 64, 2)), :]

    points = np.round(read_pts(os.path.join(landmarks_dir, '001_f.pts')) * 1 / spacing)
    landmarks_fr = points[list(range(12, 25, 2)), :]
    landmarks_tr = points[list(range(47, 64, 2)), :]

    landmarks_fr[:, 0] = col - landmarks_fr[:, 0]
    landmarks_tr[:, 0] = col - landmarks_tr[:, 0]

    landmarks = {'TR': landmarks_tr, 'FR': landmarks_fr,
                 'TL': landmarks_tl, 'FL': landmarks_fl}

    result = []
    total_landmarks = sum([landmarks[key].shape[0] for key in landmarks])
    passed_through = 0
    for side in ['L', 'R']:
        for bone in ['T', 'F']:
            lndm = landmarks[bone + side]
            for pt_id in range(lndm.shape[0]):
                cx, cy = lndm[pt_id].astype(int)
                result.append([str(subject_id) + '.png',
                               os.path.getsize(img_saved_path),
                               '{}', total_landmarks, passed_through + pt_id,
                               '{"name":"point","cx":' + str(cx) + ',"cy":' + str(cy) + '}',
                               '{"Bone": "' + bone + '","Side":"' + side + '"}'])
            passed_through += lndm.shape[0]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--oai_data_dir', default='/media/lext/FAST/knee_landmarks/Data/')
    parser.add_argument('--fu', default='00')
    parser.add_argument('--workdir', default='/media/lext/FAST/knee_landmarks/workdir/')
    parser.add_argument('--n_per_grade', type=int, default=150)
    parser.add_argument('--num_threads', type=int, default=30)
    parser.add_argument('--seed', type=int, default=123456)
    args = parser.parse_args()

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
        left = list(map(lambda x: (int(x[0]), x[2]), left))

        right = np.vstack((ids_right, side_id_right, folders_right, kls)).T.tolist()
        right = list(map(lambda x: (int(x[0]), x[2]), right))

        taken = set()

        for case in left + right:
            if case not in taken:
                taken.update({case, })

        assert np.unique(np.hstack((ids_left, ids_right))).shape[0] == len(taken)
        to_process.extend(list(taken))

    os.makedirs(os.path.join(args.workdir, 'oai_images', args.fu), exist_ok=True)
    res = Parallel(args.num_threads)(delayed(worker)(data_entry, args) for data_entry in tqdm(to_process,
                                                                                              total=len(to_process)))
    info = []
    for r in res:
        info.extend(r)

    via_metadata_df = pd.DataFrame(data=info, columns=[
        'filename', 'file_size', 'file_attributes',
        'region_count', 'region_id', 'region_shape_attributes', 'region_attributes'
    ])

    via_metadata_df.to_csv(os.path.join(args.workdir, f'oai_landmarks_{args.fu}_via_format.csv'), index=None)



