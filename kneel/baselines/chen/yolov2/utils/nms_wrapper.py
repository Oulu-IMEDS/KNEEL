# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------



from .nms.cpu_nms import cpu_nms
from .nms.gpu_nms import gpu_nms


def nms(dets, thresh, force_cpu=True):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    if force_cpu:
        return cpu_nms(dets, thresh)
    return gpu_nms(dets, thresh)
