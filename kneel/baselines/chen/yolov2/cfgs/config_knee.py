# -*- coding: utf-8 -*-

import os, sys
import numpy as np

class config:
    def __init__(self):
        self.label_names = ["0", "1", "2", "3", "4"]
        self.num_classes = len(self.label_names)

        # w * h # 1
        self.anchors = np.asarray([[ 67.05,  65.61],])
        # # w * h # 2
        # self.anchors = np.asarray([[ 63.10,  60.43],
        #                            [ 71.71,  71.71]])
        # # w * h # 3
        # self.anchors = np.asarray([[ 65.3,  67.7],
        #                            [ 62.4,  57.5],
        #                            [ 75.0,  72.3]])
        # # w * h # 4
        # self.anchors = np.asarray([[61.29,  57.40],
        #                            [74.87,  75.23],
        #                            [71.61,  63.60],
        #                            [63.74, 69.12]])
        # # w * h # 5
        # self.anchors = np.asarray([[72.31, 63.91],
        #                            [76.06, 75.22],
        #                            [60.86, 55.39],
        #                            [63.01, 63.96],
        #                            [65.51, 73.18]])
        # # w * h # 6
        # self.anchors = np.asarray([[74.13, 67.21],
        #                            [67.81, 60.29],
        #                            [65.62, 73.95],
        #                            [76.32, 77.20],
        #                            [62.49, 65.32],
        #                            [59.94, 55.40]])
        self.num_anchors = len(self.anchors)

        # Image mean and std
        # self.rgb_mean = [0.5, 0.5, 0.5]
        # self.rgb_var = [0.5, 0.5, 0.5]
        self.rgb_mean = [0.431, 0.431, 0.431]
        self.rgb_var = [0.0872, 0.0872, 0.0872]

        # IOU scale
        self.iou_thresh = 0.6
        self.object_scale = 5.
        self.noobject_scale = 1.

        # BOX scale
        self.coord_scale = 1.

        # CLS scale
        # self.class_scale = 1.0  # for kl classification
        self.class_scale = 0.0    # only for detection

        # Regulable Ordinal Loss
        self.w_loss = False        # for weighted loss

        # Test JI index
        self.JIthresh = 0.75
        # self.JIthresh = 0.50

cfg = config()
