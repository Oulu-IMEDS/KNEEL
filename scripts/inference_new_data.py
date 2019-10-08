import argparse
import glob
import os
import numpy as np
from kneel.inference.pipeline import KneeAnnotatorPipeline
from tqdm import tqdm
import cv2
import logging

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='')
    parser.add_argument('--dataset', default='')
    parser.add_argument('--workdir', default='')
    parser.add_argument('--lc_snapshot_path', default='')
    parser.add_argument('--hc_snapshot_path', default='')
    parser.add_argument('--roi_size_mm', type=int, default=140)
    parser.add_argument('--pad', type=int, default=300)
    parser.add_argument('--device',  default='cuda')
    parser.add_argument('--refine', type=bool, default=False)
    parser.add_argument('--mean_std_path', default='')
    args = parser.parse_args()
    # Needed to silence the logger
    logger = logging.getLogger('DummyLogger')
    logger.setLevel(logging.ERROR)

    annotator = KneeAnnotatorPipeline(args.lc_snapshot_path, args.hc_snapshot_path,
                                      args.mean_std_path, args.device, logger)

    imgs = glob.glob(os.path.join(args.dataset_path, args.dataset, '*'))
    imgs.sort()
    predicted_landmarks = []
    case_names = []
    for img_name in tqdm(imgs, total=len(imgs), desc=f'Annotating..'):
        landmarks = annotator.predict(img_name, args.roi_size_mm, args.pad, args.refine)
        if landmarks is None:
            continue
        predicted_landmarks.append(landmarks)
        case_names.append(img_name.split('/')[-1].split('.')[0])

    predicted_landmarks = np.vstack(predicted_landmarks)
    save_dir = os.path.join(args.workdir, args.dataset+'_inference')
    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, f'preds_unpadded{"_refined" if args.refine else ""}.npz'),
             preds=predicted_landmarks,
             imgs=case_names)

