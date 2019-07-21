import os
import torch
import argparse
import glob
from tqdm import tqdm
import cv2
import numpy as np
import torchvision.transforms as standard_transforms

from kneel.baselines.chen.yolov2.utils.cython_yolo import yolo_to_bbox

from kneel.baselines.chen.yolov2.darknet import Darknet19
from kneel.baselines.chen.yolov2.cfgs.config_knee import cfg
import kneel.baselines.chen.yolov2.utils.yolo as yolo_utils
from kneel.inference import LandmarkAnnotator
from kneel.data.utils import read_dicom

import solt.transforms as slt
import solt.data as sld
import solt.core as slc
import matplotlib.pyplot as plt



cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def load_dicom(img_path):
    # res = read_dicom(img_path)
    # if res is None:
    #     return []
    # img, orig_spacing, _ = res
    # img *= 1.
    # img -= img.min()
    # img /= img.max()
    # img *= 255
    #
    # h_orig, w_orig = img.shape
    #
    # img = np.expand_dims(img, 2)
    # img = np.dstack((img, img, img))
    #
    # img = LandmarkAnnotator.resize_to_spacing(img, spacing=orig_spacing, new_spacing=0.14)
    img, orig_spacing, h_orig, w_orig = LandmarkAnnotator.read_dicom(img_path, 0.14)
    img = np.expand_dims(img, 2)
    img = np.dstack((img, img, img))
    img = img.astype(np.uint8)
    dc = sld.DataContainer((img, ), 'I')
    ppl = slc.Stream([
        slt.PadTransform(pad_to=(2560, 2048)),
        slt.CropTransform(crop_size=(2560, 2048))
    ])
    img_cropped = ppl(dc).data[0]

    return cv2.resize(img_cropped, (320, 256)), h_orig, w_orig


if __name__ == '__main__':
    # Config arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', default='')
    parser.add_argument('--dataset_path', default='')
    parser.add_argument('--dataset', default='')
    args = parser.parse_args()

    net = Darknet19(cfg)
    net.eval()
    weights_dict = torch.load(args.weights_path, map_location=lambda storage, loc: storage)
    net.load_state_dict(weights_dict)
    net.to('cuda')

    imgs = glob.glob(os.path.join(args.dataset_path, args.dataset, '*'))
    imgs.sort()
    predicted_landmarks = []
    case_names = []
    input_transform = standard_transforms.Normalize(cfg.rgb_mean, cfg.rgb_var)

    for img_name in tqdm(imgs, total=len(imgs), desc=f'Annotating {args.dataset}...'):
        img_cur, h_orig, w_orig = load_dicom(img_name)

        img_t = torch.from_numpy(img_cur.astype(np.float32)) / 255.
        img_t = img_t.transpose(0, 2).transpose(1, 2)
        img_trf = input_transform(img_t).to('cuda').unsqueeze(0)
        with torch.no_grad():
            bbox_pred, iou_pred, prob_pred = net(img_trf)

        H, W = net.out_size
        x_ratio, y_ratio = net.x_ratio, net.y_ratio

        bbox_pred = bbox_pred.to('cpu').numpy()
        iou_pred = iou_pred.to('cpu').numpy()
        prob_pred = prob_pred.to('cpu').numpy()

        bbox_pred = yolo_to_bbox(
            np.ascontiguousarray(bbox_pred, dtype=np.float),
            np.ascontiguousarray(cfg.anchors, dtype=np.float),
            H, W, x_ratio, y_ratio)

        bboxes, scores, cls_inds = yolo_utils.postprocess_bbox(bbox_pred, iou_pred,
                                                               prob_pred, img_trf.shape[2:],
                                                               cfg, thresh=0.12)
        for i in range(bboxes.shape[0]):
            cv2.rectangle(img_cur, tuple(bboxes[i, [0, 1]]),
                          tuple(bboxes[i, [1, 2]]),
                          (255, 0, 0), 1)
        #plt.imshow(img_cur)
        #plt.show()
