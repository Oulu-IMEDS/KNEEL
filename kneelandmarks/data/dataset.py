import torch
from torch.utils import data
import os
import cv2
import numpy as np
from kneelandmarks.data.utils import parse_landmarks


class LandmarkDataset(data.Dataset):
    def __init__(self, data_root, split, hc_spacing, lc_spacing, transform, ann_type='hc'):
        if self.ann_type not in ['hc', 'lc']:
            raise ValueError('Wrong annotation type')

        self.split = split
        self.transform = transform
        self.data_root = data_root
        self.ann_type = ann_type
        self.hc_spacing = hc_spacing
        self.lc_spacing = lc_spacing
        self.hc_lc_scale = self.lc_spacing / self.hc_spacing

    def __getitem__(self, index):
        subject_id, side, folder, kl, t_lnd, f_lnd, _, center = self.split.iloc[index]
        fname = os.path.join(self.data_root, f'{subject_id}_{kl}_{side}.png')

        img = cv2.imread(fname, 0)

        if self.ann_type == 'hc':
            t_lnd = parse_landmarks(t_lnd)
            f_lnd = parse_landmarks(f_lnd)
            img, target = self.transform((img, t_lnd, f_lnd))
        else:
            center = np.array(list(map(int, center.split(',')))) * self.hc_lc_scale
            img, target = self.transform((img, center))

        return {'img': img, 'label': target,
                'subject_id': subject_id, 'kl': kl,
                't_gt': torch.from_numpy(t_lnd).float(),
                'f_gt': torch.from_numpy(f_lnd).float()}

    def __len__(self):
        return self.split.shape[0]
