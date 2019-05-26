from deeppipeline.kvs import GlobalKVS
from deeppipeline.common.transforms import apply_by_index, numpy2tens
import numpy as np
import torch
from torchvision import transforms

import solt.data as sld
import solt.core as slc
import solt.transforms as slt


def init_augs():
    kvs = GlobalKVS()
    args = kvs['args']
    ppl = transforms.Compose([
        slc.Stream([
            slt.RandomFlip(p=0.5, axis=1),
            slt.RandomScale(range_x=(0.7, 1.5), p=1),
            slt.RandomRotate(rotation_range=(-180, 180), p=0.2),
            slt.RandomProjection(affine_transforms=slc.Stream([
                slt.RandomScale(range_x=(0.8, 1.3), p=1),
                slt.RandomRotate(rotation_range=(-180, 180), p=1),
                slt.RandomShear(range_x=(-0.1, 0.1), range_y=(0, 0), p=0.5),
                slt.RandomShear(range_y=(-0.1, 0.1), range_x=(0, 0), p=0.5),
            ]), v_range=(1e-5, 2e-3), p=0.8),
            slt.PadTransform(args.pad_img, padding='z'),
            slt.CropTransform(args.crop_size, crop_mode='r'),
            slt.ImageGammaCorrection(p=1, gamma_range=(0.5, 1.5))
        ]),
    ])
    kvs.update('train_trf', ppl)


def init_data_processing():
    raise NotImplementedError


def l2m(lm, shape, sigma=1.5):
    # lm = (x,y)
    m = np.zeros(shape, dtype=np.uint8)

    if np.all(lm > 0) and lm[0] < shape[1] and lm[1] < shape[0]:
        x, y = np.meshgrid(np.linspace(-0.5, 0.5, m.shape[1]), np.linspace(-0.5, 0.5, m.shape[0]))
        mux = (lm[0]-m.shape[1]//2)/1./m.shape[1]
        muy = (lm[1]-m.shape[0]//2)/1./m.shape[0]
        s = sigma / 1. / m.shape[0]
        m = (x-mux)**2 / 2. / s**2 + (y-muy)**2 / 2. / s**2
        m = np.exp(-m)
        m -= m.min()
        m /= m.max()

    return m


def solt2torch(dc: sld.DataContainer, downsample=4, sigma=1.5):
    """
    Converts image and the landmarks in numpy into torch.
    The landmarks are converted into heatmaps as well.
    Covers both, low- and high-cost annotations cases.

    Parameters
    ----------
    dc : sld.DataContainer
        Data container
    downsample : int
        Downsampling factor to match the hourglass outputs.
    sigma : float
        Variance of the gaussian to fit at each landmark
    Returns
    -------
    out : tuple of torch.FloatTensor

    """
    if dc.data_format != 'IPPL':
        raise TypeError('Invalid type of data container')

    img, landmarks, label = dc.data
    target = []
    for lnd_cat in range(len(landmarks)):
        for i in range(landmarks[0].shape[0]):
            res = l2m((img.shape[0] // downsample, img.shape[1] // downsample),
                      landmarks[lnd_cat][i] // downsample, sigma)

            target.append(numpy2tens(res))
    target = torch.cat(target, 0).unsqueeze(0)
    assert target.size(0) == 1
    assert target.size(1) == len(landmarks) * landmarks.shape[0]
    assert target.size(2) == img.shape[0] // downsample
    assert target.size(3) == img.shape[0] // downsample

    return img, target, torch.from_numpy(np.vstack(landmarks) / downsample).float(), label

