import numpy as np
from copy import copy
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn


def to_variable(x, requires_grad=True,  var=True,volatile=False):

    if type(x) is Variable:
        return x
    if type(x) is np.ndarray:
        x = torch.from_numpy(x.astype(np.float32))
    if var:
        x = Variable(x, requires_grad=requires_grad, volatile=volatile)
    x.volatile = volatile

    return x


def to_device(src, ref, var = True, volatile = False, requires_grad=True):
    requires_grad = requires_grad and (not volatile)
    src = to_variable(src, var=var, volatile=volatile,requires_grad=requires_grad)
    return src.cuda(ref.get_device()) if ref.is_cuda else src


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def tensor_to_img(t_img, mean, std):
    t_img.squeeze_()
    for t, m, s in zip(t_img, mean, std):
        t.mul_(s).add_(m)

    np_img = t_img.numpy().transpose(1, 2, 0)
    np_img = (np_img * 255.0).astype(np.uint8)

    return np_img
