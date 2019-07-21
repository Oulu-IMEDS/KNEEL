# -*- coding: utf-8 -*-

import os, sys, pdb
import numpy as np
from scipy.misc import toimage
import matplotlib.pyplot as plt
import torch
import deepdish as dd
import json

from .utils.timer import Timer
from .utils import yolo as yolo_utils
from .proj_utils.plot_utils  import plot_scalar
from .proj_utils.local_utils import writeImg, mkdirs
from .proj_utils.torch_utils import tensor_to_img
from .knee_utils import knee_det_cls
from .knee_utils import evaluate_det_cls, save_pred_box_coors
from .knee_utils import overlay_bbox_iou


def test_eng(dataloader, model_root, save_root, net, args, cfg):
    net.eval()
    weightspath = os.path.join(model_root, args.model_name)
    weights_dict = torch.load(weightspath, map_location=lambda storage, loc: storage)
    print("===="*20)
    print('Model name is {}'.format(weightspath))
    net.load_state_dict(weights_dict)
    print("Num images: {}".format(len(dataloader)))
    _t = {'im_detect': Timer(), 'misc': Timer()}

    evalate_matrix = np.zeros((5, 5), dtype=np.int)
    true_box_num, total_box_num = 0, 0
    all_overlap = []

    # import pdb; pdb.set_trace()
    for ind, data in enumerate(dataloader):
        cur_img, cur_boxes, cur_classes, cur_name = data
        # detection
        _t['im_detect'].tic()
        result_dict = knee_det_cls(net, cur_img, cfg=cfg)
        bbox_pred, iou_pred, prob_pred  = result_dict['bbox'], result_dict['iou'], result_dict['prob']
        detect_time = _t['im_detect'].toc()
        # postprocessing
        _t['misc'].tic()
        bboxes, scores, cls_inds = yolo_utils.postprocess_bbox(bbox_pred, iou_pred, prob_pred, cur_img.shape[2:], cfg, thresh=0.12)
        utils_time = _t['misc'].toc()

        gt_boxes = cur_boxes.squeeze().numpy()
        gt_classes = cur_classes.squeeze().numpy()
        _, true_num, total_num, overlap_list, _ = evaluate_det_cls(gt_boxes, gt_classes, bboxes, cls_inds,
                                                                   cfg.num_classes, cfg.JIthresh)
        # save_pred_box_coors(save_root, gt_boxes, gt_classes, bboxes, cur_name[0])
        if total_num != 2: # Check wrong detection file
            print("Name: {}, num: {}".format(cur_name[0], total_num))
        true_box_num += true_num
        total_box_num += total_num
        all_overlap.extend(overlap_list)

        # if cur_name[0] in selection:
        #     box_dict = {}
        #     box_dict["det"] = bboxes.tolist()
        #     box_dict["gt"] = gt_boxes.tolist()
        #     select_dict[cur_name[0]] = box_dict


        if (ind+1) % 100 == 0:
            print('{}/{} detection time {:.4f}, post_processing time {:.4f}'.format(
                ind+1, len(dataloader), detect_time, utils_time))
            # Overlay gt box and prediction box and overlap value
            img_np = tensor_to_img(cur_img, cfg.rgb_mean, cfg.rgb_var)
            overlaid_img = overlay_bbox_iou(img_np, bboxes, gt_boxes)
            writeImg(overlaid_img, os.path.join(save_root, cur_name[0]+'.png'))

    # # save to json
    # with open(os.path.join(save_root, 'selection.json'), 'w') as outfile:
    #     json.dump(select_dict, outfile)

    print("---Detection accuracy---")
    print("Number of object: {}, Mean IoU is: {}".format(len(all_overlap), np.mean(all_overlap)))
    print("True predicted knee: {}\n All predicted knee: {}\n Total accuracy: {:.4f}\n".format(
        true_box_num, total_box_num, true_box_num*1.0/total_box_num))
