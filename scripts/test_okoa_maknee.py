import matplotlib.pyplot as plt

import argparse
import glob
import os
import numpy as np
from kneel.inference import LandmarkAnnotator
from kneel.evaluation import visualize_landmarks



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='')
    parser.add_argument('--lc_snapshot_path', default='')
    parser.add_argument('--hc_snapshot_path', default='')
    parser.add_argument('--roi_size_mm', type=int, default=140)
    parser.add_argument('--pad_src_img', type=int, default=0)
    parser.add_argument('--mean_std_path', default='')
    args = parser.parse_args()

    global_searcher = LandmarkAnnotator(snapshot_path=args.lc_snapshot_path,
                                        mean_std_path=args.mean_std_path,
                                        device='cuda')

    local_searcher = LandmarkAnnotator(snapshot_path=args.hc_snapshot_path,
                                       mean_std_path=args.mean_std_path,
                                       device='cuda')

    imgs = glob.glob(os.path.join(args.dataset_path, '*'))
    for img_name in imgs:
        print(img_name)
        img, orig_spacing, h_orig, w_orig, img_orig = global_searcher.read_dicom(img_name,
                                                                                 new_spacing=global_searcher.img_spacing,
                                                                                 return_orig=True,
                                                                                 pad_img=args.pad_src_img)
        # First pass of knee joint center estimation
        roi_size_px = int(args.roi_size_mm * 1. / orig_spacing)
        global_coords = global_searcher.predict_img(img, h_orig, w_orig)
        landmarks, _, _ = local_searcher.predict_local(img_orig, global_coords,
                                                                                roi_size_px, orig_spacing)

        centers_d = np.array([roi_size_px // 2, roi_size_px // 2]) - landmarks[:, 5]
        global_coords -= centers_d
        landmarks, right_roi_orig, left_roi_orig = local_searcher.predict_local(img_orig, global_coords,
                                                                                roi_size_px, orig_spacing)

        visualize_landmarks(right_roi_orig, landmarks[0, :9], landmarks[0, 9:])
        visualize_landmarks(left_roi_orig, landmarks[1, :9], landmarks[1, 9:])