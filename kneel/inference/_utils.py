import cv2
import torch
import glob
import os
import pickle
import numpy as np


from functools import partial
from torchvision import transforms as tvt
import solt.core as slc
import solt.transforms as slt
import solt.data as sld

from deeppipeline.common.transforms import apply_by_index
from deeppipeline.common.normalization import normalize_channel_wise

from kneel.model import init_model_from_args
from kneel.data.utils import read_dicom, process_xray, convert_img


def wrap_slt(img):
    return sld.DataContainer((img, ), 'I')


def unwrap_slt(dc):
    return dc.data[0]


class LandmarkAnnotator(object):
    def __init__(self, snapshot_path, mean_std_path, data_parallel=False):
        self.fold_snapshots = glob.glob(os.path.join(snapshot_path, 'fold_*.pth'))
        self.models = []
        self.data_parallel = data_parallel
        with open(os.path.join(snapshot_path, 'session.pkl'), 'rb') as f:
            snapshot_session = pickle.load(f)

        snp_args = snapshot_session['args'][0]

        for snp_name in self.fold_snapshots:
            net = init_model_from_args(snp_args)
            snp = torch.load(snp_name)['model']
            net.load_state_dict(snp)
            net.eval()
            self.models.append(net)

        mean_vector, std_vector = np.load(mean_std_path)

        norm_trf = partial(normalize_channel_wise, mean=mean_vector, std=std_vector)

        self.trf = tvt.Compose([
                   wrap_slt,
                   slc.Stream([
                       slt.PadTransform((snp_args.pad_x, snp_args.pad_y), padding='z'),
                       slt.CropTransform((snp_args.crop_x, snp_args.crop_y), crop_mode='c'),
                   ]),
                   unwrap_slt,
                   convert_img,
                    partial(apply_by_index, transform=norm_trf, idx=0)
        ])

    @staticmethod
    def read_dicom(img_path, new_spacing, pad):
        res = read_dicom(img_path)
        if res is None:
            return []
        img, spacing, _ = res
        img = process_xray(img).astype(np.uint8)

        scale = spacing / new_spacing
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
        if pad == 0:
            return img

        row, col = img.shape
        tmp = np.zeros((row + 2 * pad, col + 2 * pad))
        tmp[pad:pad + row, pad:pad + col] = img
        return tmp

    def predict(self, img):
        if isinstance(img, str):
            img = self.read_dicom(img)
        img = self.trf(img)
        print(img)



