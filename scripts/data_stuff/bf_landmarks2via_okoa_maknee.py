"""
This script converts the landmarks for OKOA/MAKNEE datasets (private) generated by
bonefinder to via format for further refinement.

(c) Aleksei Tiulpin, University of Oulu, 2019.

"""
import argparse
import glob
import pandas as pd
from kneel.data.utils import save_original_with_via_landmarks
import os
from joblib import Parallel, delayed
from tqdm import tqdm


def worker(data_entry, args_thread):
    subject_id, side = data_entry

    dicom_name = os.path.join(args_thread.data_dir, args_thread.dataset_name, subject_id)

    # Not the most effective way, but works OK
    landmarks_dir = glob.glob(os.path.join(args_thread.data_dir, f'landmarks_{args_thread.dataset_name.lower()}',
                                           subject_id.split('_')[-1]))[0]

    img_save_path = os.path.join(args_thread.workdir, f'{args_thread.dataset_name}_images', str(subject_id) + '.png')

    return save_original_with_via_landmarks(subject_id, side, dicom_name, img_save_path, landmarks_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/media/lext/FAST/knee_landmarks/Data/OKOA_MAKNEE')
    parser.add_argument('--dataset_name', default='OKOA')
    parser.add_argument('--workdir', default='/media/lext/FAST/knee_landmarks/workdir/')
    parser.add_argument('--n_per_grade', type=int, default=150)
    parser.add_argument('--num_threads', type=int, default=30)
    parser.add_argument('--seed', type=int, default=123456)
    args = parser.parse_args()
    to_process = []

    for pat_id in os.listdir(os.path.join(args.data_dir, args.dataset_name)):
        to_process.append([pat_id, 'R'])
        to_process.append([pat_id, 'L'])

    os.makedirs(os.path.join(args.workdir, f'{args.dataset_name}_images'), exist_ok=True)
    res = Parallel(args.num_threads)(delayed(worker)(data_entry, args) for data_entry in tqdm(to_process,
                                                                                              total=len(to_process)))
    landmarks = []
    spacings = []
    for r in res:
        if len(r) > 0:
            landmarks.extend(r[0])
            spacings.append([r[1], r[2]])

    via_metadata_df = pd.DataFrame(data=landmarks, columns=[
        'filename', 'file_size', 'file_attributes',
        'region_count', 'region_id', 'region_shape_attributes', 'region_attributes'
    ])

    via_metadata_df.to_csv(os.path.join(args.workdir, f'{args.dataset_name}_via_format.csv'), index=None)

    spacings = pd.DataFrame(spacings, columns=['ID', 'spacing'])
    spacings.drop_duplicates(inplace=True)
    spacings.to_csv(os.path.join(args.workdir, f'{args.dataset_name}_spacing_info.csv'), index = None)