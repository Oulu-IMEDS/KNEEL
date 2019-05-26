from torch.utils import data
import os
import cv2
import numpy as np
from kneelandmarks.data.utils import parse_landmarks
import solt.data as sld


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
        kl = int(kl)

        if self.ann_type == 'hc':
            fname = os.path.join(self.data_root, f'{subject_id}_{kl}_{side}.png')
        else:
            fname = os.path.join(self.data_root, f'{subject_id}.png')

        img = cv2.imread(fname, 0)

        if self.ann_type == 'hc':
            kp_tibia = sld.KeyPoints(parse_landmarks(t_lnd), img.shape[0], img.shape[1])
            kp_femur = sld.KeyPoints(parse_landmarks(f_lnd), img.shape[0], img.shape[1])
            dc = sld.DataContainer((img, kp_tibia,kp_femur, kl), 'IPPL')
        else:
            center = np.array(list(map(int, center.split(',')))) * self.hc_lc_scale
            cr = np.expand_dims(center[:2], 0)
            cl = np.expand_dims(center[2:], 0)
            cr = sld.KeyPoints(cr, img.shape[0], img.shape[1])
            cl = sld.KeyPoints(cl, img.shape[0], img.shape[1])

            dc = sld.DataContainer((img, cr, cl, -1), 'IPPL')

        img, target_hm, target_kp, kl = self.transform(dc)

        return {'img': img, 'target_hm': target_hm,
                'subject_id': subject_id, 'kl': kl,
                'kp_gt': target_kp}

    def __len__(self):
        return self.split.shape[0]


def init_meta():
    pass
