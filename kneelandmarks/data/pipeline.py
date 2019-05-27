import numpy as np
import torch
from torchvision import transforms as tvt
import matplotlib.pyplot as plt
import cv2

from functools import partial
import os
import pandas as pd

import solt.data as sld
import solt.core as slc
import solt.transforms as slt

from deeppipeline.kvs import GlobalKVS
from deeppipeline.common.transforms import apply_by_index, numpy2tens
from deeppipeline.common.normalization import init_mean_std, normalize_channel_wise

from kneelandmarks.data.dataset import LandmarkDataset


def init_augs():
    kvs = GlobalKVS()
    args = kvs['args']
    ppl = tvt.Compose([
        slc.SelectiveStream([
            slc.Stream([
                slt.RandomFlip(p=0.5, axis=1),
                slt.RandomScale(range_x=(0.6, 3), p=0.5),
                slt.RandomRotate(rotation_range=(-180, 180), p=0.2),
                slt.RandomProjection(affine_transforms=slc.Stream([
                    slt.RandomScale(range_x=(0.8, 1.3), p=1),
                    slt.RandomRotate(rotation_range=(-180, 180), p=1),
                    slt.RandomShear(range_x=(-0.1, 0.1), range_y=(-0.1, 0.1), p=0.5),
                    slt.RandomShear(range_y=(-0.1, 0.1), range_x=(-0.1, 0.1), p=0.5),
                ]), v_range=(1e-5, 2e-3), p=0.5),

            ]),
            slc.Stream()
        ], probs=[0.7, 0.3]),
        slc.Stream([
            slt.PadTransform((args.pad_x, args.pad_y), padding='z'),
            slt.CropTransform((args.crop_x, args.crop_y), crop_mode='r'),
        ]),
        slc.SelectiveStream([
            slc.Stream([
                slt.ImageSaltAndPepper(p=0.5, gain_range=0.01),
                slt.ImageAdditiveGaussianNoise(p=0.5, gain_range=0.5),
                slc.Stream([
                    slt.ImageBlur(p=0.5, blur_type='m', k_size=(3, 3)),
                    slt.ImageSaltAndPepper(p=1, gain_range=0.01),
                ]),
            ]),
            slt.ImageGammaCorrection(p=0.5, gamma_range=(0.5, 1.5)),
            slc.Stream()
        ], probs=[0.1, 0.3, 0.6]),
        partial(solt2torchhm, downsample=4, sigma=kvs['args'].hm_sigma),
    ])
    kvs.update('train_trf', ppl)


def init_data_processing():
    kvs = GlobalKVS()

    dataset = LandmarkDataset(data_root=kvs['args'].dataset_root,
                              split=kvs['metadata'],
                              hc_spacing=kvs['args'].hc_spacing,
                              lc_spacing=kvs['args'].lc_spacing,
                              transform=kvs['train_trf'], ann_type=kvs['args'].annotations)

    tmp = init_mean_std(snapshots_dir=os.path.join(kvs['args'].workdir, 'snapshots'),
                        dataset=dataset,
                        batch_size=kvs['args'].bs,
                        n_threads=kvs['args'].n_threads,
                        n_classes=-1)

    if len(tmp) == 3:
        mean_vector, std_vector, class_weights = tmp
    elif len(tmp) == 2:
        mean_vector, std_vector = tmp
    else:
        raise ValueError('Incorrect format of mean/std/class-weights')

    norm_trf = partial(normalize_channel_wise, mean=mean_vector, std=std_vector)

    train_trf = tvt.Compose([
        kvs['train_trf'],
        partial(apply_by_index, transform=norm_trf, idx=0)
    ])

    val_trf = tvt.Compose([
        partial(solt2torchhm, downsample=4, sigma=kvs['args'].hm_sigma),
        partial(apply_by_index, transform=norm_trf, idx=0)
    ])

    kvs.update('train_trf', train_trf)
    kvs.update('val_trf', val_trf)


def init_meta():
    kvs = GlobalKVS()
    metadata = pd.read_csv(os.path.join(kvs['args'].workdir, kvs['args'].metadata))
    kvs.update('metadata', metadata)


def init_loaders():
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


def solt2torchhm(dc: sld.DataContainer, downsample=4, sigma=1.5):
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
    if dc.data_format != 'IPL':
        raise TypeError('Invalid type of data container')

    img, landmarks, label = dc.data

    target = []
    for i in range(landmarks.data.shape[0]):
        res = l2m(landmarks.data[i] // downsample,
                  (img.shape[0] // downsample, img.shape[1] // downsample), sigma)

        target.append(numpy2tens(res))

    #plt.imshow(img.squeeze(), cmap=plt.cm.Greys_r)
    #plt.imshow(cv2.resize(target[0].squeeze().numpy(),
    #                      (img.shape[1], img.shape[0])),
    #           alpha=0.5, cmap=plt.cm.jet)
    #plt.show()
    target = torch.cat(target, 0).unsqueeze(0)
    assert target.size(0) == 1
    assert target.size(1) == landmarks.data.shape[0]
    assert target.size(2) == img.shape[0] // downsample
    assert target.size(3) == img.shape[0] // downsample
    assert len(img.shape) == 3

    return torch.from_numpy(img).float(), target, torch.from_numpy(landmarks.data / downsample).float(), label

