import os, sys, pdb
import torch
import torchvision.transforms as standard_transforms
import argparse
from torch.utils import data

from kneel.baselines.chen.yolov2.proj_utils.local_utils import mkdirs
from kneel.baselines.chen.yolov2.darknet import Darknet19
from kneel.baselines.chen.yolov2.cfgs.config_knee import cfg
from kneel.baselines.chen.yolov2.datasets.knee import Knee
from kneel.baselines.chen.yolov2.test_yolo import test_eng


def set_args():
    parser = argparse.ArgumentParser(description = 'Testing code for Knee bone detection')
    parser.add_argument('--device-id',       type=int, default=0)
    parser.add_argument('--batch-size',      type=int, default=1)
    parser.add_argument('--model-dir',       type=str, default="best_models")
    parser.add_argument('--model-name',      type=str, default="")
    args = parser.parse_args()

    return args


if  __name__ == '__main__':
    # Config arguments
    args = set_args()

    net = Darknet19(cfg)