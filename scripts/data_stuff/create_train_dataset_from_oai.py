from tqdm import tqdm
import os
import pandas as pd
import argparse
import numpy as np
from joblib import Parallel, delayed
import cv2
from kneel.data.utils import save_based_on_exising_annotations, read_dicom, process_xray

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def read_dicom_callback(dataset_entry):
    dicom_name = os.path.join(dataset_entry.oai_dir, dataset_entry.folder, '001')

    # Getting the raw DICOM image
    res = read_dicom(dicom_name)

    if res is None:
        raise RuntimeError("Can't read the DICOM file! Something is wrong with the paths or permissions.")
    # Pre-processing (histogram clipping and stretching)
    img_original, spacing, _ = res
    img_original = process_xray(img_original).astype(np.uint8)
    if img_original.shape[0] == 0 or img_original.shape[1] == 0:
        raise RuntimeError("Couldn't process the DICOM image!!!")

    return img_original, spacing


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--oai_xrays_path', default='')
    parser.add_argument('--annotations_path', default='')
    parser.add_argument('--pad', default=100)
    parser.add_argument('--sizemm', type=int, default=140)
    parser.add_argument('--num_threads', default=30)
    parser.add_argument('--high_cost_spacing', type=float, default=0.3)
    parser.add_argument('--low_cost_spacing', type=int, default=1)
    parser.add_argument('--to_save_low_cost_img', default='/media/lext/FAST/knee_landmarks/workdir/low_cost_data')
    parser.add_argument('--to_save_high_cost_img', default='/media/lext/FAST/knee_landmarks/workdir/high_cost_data')
    args = parser.parse_args()

    oai_contents = pd.read_csv(os.path.join(args.oai_xrays_path, 'contents.csv'))
    oai_contents = oai_contents[oai_contents.SeriesDescription == 'Bilateral PA Fixed Flexion Knee']
    oai_contents['subject_id'] = oai_contents.ParticipantID
    oai_contents['folder'] = oai_contents.Folder
    oai_contents = oai_contents[['subject_id', 'folder']]

    oai_contents['pad'] = args.pad
    oai_contents['sizemm'] = args.sizemm
    oai_contents['to_save_low_cost_img'] = args.to_save_low_cost_img
    oai_contents['to_save_high_cost_img'] = args.to_save_high_cost_img
    oai_contents['high_cost_spacing'] = args.high_cost_spacing
    oai_contents['low_cost_spacing'] = args.low_cost_spacing
    oai_contents['oai_dir'] = args.oai_xrays_path

    annotations = pd.read_csv(os.path.join(args.annotations_path,
                                           f'bf_landmarks_{args.low_cost_spacing}_{args.high_cost_spacing}.csv'))
    annotations = pd.merge(oai_contents, annotations, on='subject_id')

    res = Parallel(args.num_threads)(delayed(save_based_on_exising_annotations)(entry,
                                                                                read_dicom_callback)
                                     for _, entry in tqdm(annotations.iterrows(), total=annotations.shape[0]))




