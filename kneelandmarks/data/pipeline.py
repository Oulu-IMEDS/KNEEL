import numpy as np
import torch
from torchvision import transforms as tvt

from functools import partial
import os
from torch.utils.data import DataLoader

import solt.core as slc
import solt.transforms as slt

from deeppipeline.kvs import GlobalKVS
from deeppipeline.common.transforms import apply_by_index
from deeppipeline.common.normalization import init_mean_std, normalize_channel_wise

from kneelandmarks.data.dataset import LandmarkDataset
from kneelandmarks.data.utils import solt2torchhm


def init_augs():
    kvs = GlobalKVS()
    args = kvs['args']
    ppl = tvt.Compose([
        slc.SelectiveStream([
            slc.Stream([
                slt.RandomFlip(p=0.5, axis=1),
                slt.RandomProjection(affine_transforms=slc.Stream([
                    slt.RandomScale(range_x=(0.8, 1.3), p=1),
                    slt.RandomRotate(rotation_range=(-90, 90), p=1),
                    slt.RandomShear(range_x=(-0.1, 0.1), range_y=(-0.1, 0.1), p=0.5),
                ]), v_range=(1e-5, 2e-3), p=0.5),
                slt.RandomScale(range_x=(0.5, 2.5), p=0.5),
            ]),
            slc.Stream()
        ], probs=[0.7, 0.3]),
        slc.Stream([
            slt.PadTransform((args.pad_x, args.pad_y), padding='z'),
            slt.CropTransform((args.crop_x, args.crop_y), crop_mode='r'),
        ]),
        slc.SelectiveStream([
            slt.ImageSaltAndPepper(p=1, gain_range=0.01),
            slt.ImageBlur(p=1, blur_type='g', k_size=(3, 5)),
            slt.ImageBlur(p=1, blur_type='m', k_size=(3, 5)),
            slt.ImageAdditiveGaussianNoise(p=1, gain_range=0.5),
            slc.Stream([
                slt.ImageSaltAndPepper(p=1, gain_range=0.05),
                slt.ImageBlur(p=0.5, blur_type='m', k_size=(3, 5)),
            ]),
            slc.Stream([
                slt.ImageBlur(p=0.5, blur_type='m', k_size=(3, 5)),
                slt.ImageSaltAndPepper(p=1, gain_range=0.01),
            ]),
            slc.Stream()
        ], n=1),
        slt.ImageGammaCorrection(p=0.5, gamma_range=(0.5, 1.5)),
        partial(solt2torchhm, downsample=None, sigma=None),
    ])
    kvs.update('train_trf', ppl)


def init_data_processing():
    kvs = GlobalKVS()

    dataset = LandmarkDataset(data_root=kvs['args'].dataset_root,
                              split=kvs['metadata'],
                              hc_spacing=kvs['args'].hc_spacing,
                              lc_spacing=kvs['args'].lc_spacing,
                              transform=kvs['train_trf'],
                              ann_type=kvs['args'].annotations,
                              image_pad=kvs['args'].img_pad)

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
        slc.Stream([
            slt.PadTransform((kvs['args'].pad_x, kvs['args'].pad_y), padding='z'),
            slt.CropTransform((kvs['args'].crop_x, kvs['args'].crop_y), crop_mode='c'),
        ]),
        partial(solt2torchhm, downsample=None, sigma=None),
        partial(apply_by_index, transform=norm_trf, idx=0)
    ])

    kvs.update('train_trf', train_trf)
    kvs.update('val_trf', val_trf)


def init_loaders(x_train, x_val, sequential_val_sampler=False):
    kvs = GlobalKVS()
    train_ds = LandmarkDataset(data_root=kvs['args'].dataset_root,
                               split=x_train,
                               hc_spacing=kvs['args'].hc_spacing,
                               lc_spacing=kvs['args'].lc_spacing,
                               transform=kvs['train_trf'],
                               ann_type=kvs['args'].annotations,
                               image_pad=kvs['args'].img_pad)

    val_ds = LandmarkDataset(data_root=kvs['args'].dataset_root,
                             split=x_val,
                             hc_spacing=kvs['args'].hc_spacing,
                             lc_spacing=kvs['args'].lc_spacing,
                             transform=kvs['val_trf'],
                             ann_type=kvs['args'].annotations,
                             image_pad=kvs['args'].img_pad)

    train_loader = DataLoader(train_ds, batch_size=kvs['args'].bs,
                              num_workers=kvs['args'].n_threads, shuffle=True,
                              drop_last=True,
                              worker_init_fn=lambda wid: np.random.seed(np.uint32(torch.initial_seed() + wid)))

    if sequential_val_sampler:
        sampler = torch.utils.data.sampler.SequentialSampler(data_source=val_ds)
    else:
        sampler = None
    val_loader = DataLoader(val_ds, batch_size=kvs['args'].val_bs,
                            num_workers=kvs['args'].n_threads, sampler=sampler)

    return train_loader, val_loader

