import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .torch_utils import *
from .local_utils import Indexflow, split_img, imshow
from collections import deque, OrderedDict
import functools


def match_tensor(out, refer_shape):

    skiprow,skipcol = refer_shape
    row, col = out.size()[2], out.size()[3]
    if skipcol >= col:
        pad_col = skipcol - col
        left_pad_col  = pad_col // 2
        right_pad_col = pad_col - left_pad_col
        out = F.pad(out, (left_pad_col, right_pad_col, 0,0), mode='reflect')
    else:
        crop_col = col - skipcol
        left_crop_col  = crop_col // 2
        right_col = left_crop_col + skipcol
        out = out[:,:,:, left_crop_col:right_col]

    if skiprow >= row:
        pad_row = skiprow - row
        left_pad_row  = pad_row // 2
        right_pad_row = pad_row - left_pad_row
        out = F.pad(out, (0,0, left_pad_row,right_pad_row), mode='reflect')
    else:
        crop_row = row - skiprow
        left_crop_row  = crop_row // 2

        right_row = left_crop_row + skiprow

        out = out[:,:,left_crop_row:right_row, :]
    return out
