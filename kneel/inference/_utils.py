import torch
import numpy as np

import solt.data as sld

from kneel.data.utils import convert_img


def wrap_slt(img, annotator_type='lc'):
    if annotator_type == 'lc':
        img = np.dstack((img, img, img))
        row, col, _ = img.shape
        # right, left encoding
        img = (img[:, :col // 2 + col % 2], img[:, col // 2:])
    else:
        img_right = np.dstack((img[0], img[0], img[0]))
        img_left = np.dstack((img[1], img[1], img[1]))
        img = (img_right, img_left)

    return sld.DataContainer((img[0], img[1]), 'II')


def unwrap_slt(dc, norm_trf):
    return torch.stack(norm_trf(list(map(convert_img, dc.data))))


class NFoldInferenceModel(torch.nn.Module):
    def __init__(self, models):
        super(NFoldInferenceModel, self).__init__()
        modules = dict()
        for idx, m in enumerate(models):
            modules[f'model_{idx+1}'] = m
        self.n_models = len(models)
        self.__dict__['_modules'] = modules

    def forward(self, x):
        res = 0
        for model_id in range(1, self.n_models+1):
            res += getattr(self, f'model_{model_id}')(x)
        return res / self.n_models


