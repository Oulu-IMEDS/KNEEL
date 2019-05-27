import torch
from tqdm import tqdm
import gc
import numpy as np
import os
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
from termcolor import colored

from deeppipeline.common.core import save_checkpoint, init_optimizer, init_session
from deeppipeline.common.dataset import init_folds
from deeppipeline.kvs import GlobalKVS


def pass_epoch(net, loader, optimizer, criterion):
    kvs = GlobalKVS()
    net.train(optimizer is not None)

    fold_id = kvs['cur_fold']
    epoch = kvs['cur_epoch']
    max_ep = kvs['args'].n_epochs

    running_loss = 0.0
    n_batches = len(loader)

    device = next(net.parameters()).device
    pbar = tqdm(total=n_batches, ncols=200)
    with torch.set_grad_enabled(optimizer is not None):
        for i, entry in enumerate(loader):
            if optimizer is not None:
                optimizer.zero_grad()

            inputs = entry['img'].to(device)
            target_hm = entry['target_hm'].to(device).squeeze().unsqueeze(1)
#            target_kp = entry['target_kp'].to(device)
            outputs = net(inputs)
            loss = criterion(outputs, target_hm)

            if optimizer is not None:
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.set_description(f"Fold [{fold_id}] [{epoch} | {max_ep}] | "
                                     f"Running loss {running_loss / (i + 1):.5f} / {loss.item():.5f}")
            else:
                running_loss += loss.item()
                pbar.set_description(desc=f"Fold [{fold_id}] [{epoch} | {max_ep}] | Validation progress")

            pbar.update()
            gc.collect()
        gc.collect()
        pbar.close()

    return running_loss / n_batches, None


def log_metrics(writer, train_loss, val_loss, val_results):
    kvs = GlobalKVS()

    print(colored('==> ', 'green') + 'Metrics:')
    print(colored('====> ', 'green') + 'Train loss:', train_loss)
    print(colored('====> ', 'green') + 'Val loss:', val_loss)

    to_log = {'train_loss': train_loss, 'val_loss': val_loss}
    val_metrics = {'epoch': kvs['cur_epoch']}
    val_metrics.update(to_log)

    writer.add_scalars(f"Losses_{kvs['args'].annotations}", to_log, kvs['cur_epoch'])
    kvs.update(f'losses_fold_[{kvs["cur_fold"]}]', to_log)
    kvs.update(f'val_metrics_fold_[{kvs["cur_fold"]}]', val_metrics)


def train_fold(net, train_loader, optimizer, criterion, val_loader, scheduler):
    kvs = GlobalKVS()
    fold_id = kvs['cur_fold']
    writer = SummaryWriter(os.path.join(kvs['args'].workdir, 'snapshots', kvs['snapshot_name'],
                                        'logs', 'fold_{}'.format(fold_id), kvs['snapshot_name']))

    for epoch in range(kvs['args'].n_epochs):
        print(colored('==> ', 'green') + f'Training epoch [{epoch}] with LR {scheduler.get_lr()}')
        kvs.update('cur_epoch', epoch)
        train_loss, _ = pass_epoch(net, train_loader, optimizer, criterion)
        val_loss, val_results = pass_epoch(net, val_loader, None, criterion)
        log_metrics(writer, train_loss, val_loss, val_results)
        save_checkpoint(net, optimizer, 'val_loss', 'lt')
        scheduler.step()


def init_scheduler(optimizer):
    kvs = GlobalKVS()
    return lr_scheduler.MultiStepLR(optimizer, kvs['args'].lr_drop)


def train_n_folds(init_args, init_metadata, init_augs,
                  init_data_processing,
                  init_loaders,
                  init_model, init_loss,
                  img_group_id_colname=None, img_class_colname=None):

    args = init_args()
    kvs = init_session(args)[-1]
    init_metadata()
    assert 'metadata' in kvs

    if init_augs is None:
        raise NotImplementedError('Train augmentations are not defined !!!')
    else:
        init_augs()
    assert 'train_trf' in kvs
    init_data_processing()
    init_folds(img_group_id_colname=img_group_id_colname, img_class_colname=img_class_colname)

    for fold_id, x_train, x_val in kvs['cv_split']:
        kvs.update('cur_fold', fold_id)
        kvs.update('prev_model', None)

        net = init_model()
        optimizer = init_optimizer(net)
        criterion = init_loss()
        scheduler = init_scheduler(optimizer)
        train_loader, val_loader = init_loaders(x_train, x_val)

        train_fold(net=net, train_loader=train_loader,
                   optimizer=optimizer, criterion=criterion,
                   val_loader=val_loader, scheduler=scheduler)
