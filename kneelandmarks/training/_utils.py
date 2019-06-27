import numpy as np
import torch
from tqdm import tqdm
import gc
import pandas as pd
from deeppipeline.kvs import GlobalKVS
from deeppipeline.common.core import mixup
from torch.distributions import beta

import cv2
import matplotlib.pyplot as plt


def pass_epoch(net, loader, optimizer, criterion):
    kvs = GlobalKVS()
    net.train(optimizer is not None)

    fold_id = kvs['cur_fold']
    epoch = kvs['cur_epoch']
    max_ep = kvs['args'].n_epochs

    running_loss = 0.0
    n_batches = len(loader)
    landmark_errors = {}
    device = next(net.parameters()).device
    pbar = tqdm(total=n_batches, ncols=200)
    mixup_sampler = beta.Beta(kvs['args'].mixup_alpha, kvs['args'].mixup_alpha)
    with torch.set_grad_enabled(optimizer is not None):
        for i, entry in enumerate(loader):
            if optimizer is not None:
                optimizer.zero_grad()

            inputs = entry['img'].to(device)
            target = entry['kp_gt'].to(device).float()

            if kvs['args'].use_mixup and optimizer is not None:
                lam = mixup_sampler.sample().item()
                mixed_inputs, shuffled_targets = mixup(inputs, target, lam)

                outputs = net(inputs)
                outputs_mixed = net(mixed_inputs)

                loss_orig = criterion(outputs, target)
                loss_mixed = criterion(outputs_mixed, shuffled_targets)

                loss = lam * loss_orig + (1 - lam) * loss_mixed
            else:
                outputs = net(inputs)
                loss = criterion(outputs, target)

            if optimizer is not None:
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.set_description(f"Fold [{fold_id}] [{epoch} | {max_ep}] | "
                                     f"Running loss {running_loss / (i + 1):.5f} / {loss.item():.5f}")
            else:
                running_loss += loss.item()
                pbar.set_description(desc=f"Fold [{fold_id}] [{epoch} | {max_ep}] | Validation progress")
            if optimizer is None:
                target_kp = entry['kp_gt'].numpy()
                h, w = inputs.size(2), inputs.size(3)
                if isinstance(outputs, tuple):
                    predicts = outputs[-1].to('cpu').numpy()
                else:
                    predicts = outputs.to('cpu').numpy()

                xy_batch = predicts
                xy_batch[:, :, 0] *= (w - 1)
                xy_batch[:, :, 1] *= (h - 1)

                target_kp = target_kp
                xy_batch = xy_batch

                target_kp[:, :, 0] *= (w - 1)
                target_kp[:, :, 1] *= (h - 1)

                for kp_id in range(target_kp.shape[1]):
                    spacing = getattr(kvs['args'], f"{kvs['args'].annotations}_spacing")
                    d = target_kp[:, kp_id] - xy_batch[:, kp_id]
                    err = np.sqrt(np.sum(d ** 2, 1)) * spacing
                    if kp_id not in landmark_errors:
                        landmark_errors[kp_id] = list()

                    landmark_errors[kp_id].append(err)

            pbar.update()
            gc.collect()
        gc.collect()
        pbar.close()

    if len(landmark_errors) > 0:
        for kp_id in landmark_errors:
            landmark_errors[kp_id] = np.hstack(landmark_errors[kp_id])
    else:
        landmark_errors = None

    return running_loss / n_batches, landmark_errors


def val_results_callback(writer, val_metrics, to_log, val_results):
    results = []
    precision = [1, 1.5, 2, 2.5, 3, 3.5, 4, 5]
    for kp_id in val_results:
        kp_res = val_results[kp_id]

        n_outliers = np.sum(kp_res < 0) / kp_res.shape[0]
        kp_res = kp_res[kp_res > 0]

        tmp = []
        for t in precision:
            tmp.append(np.sum((kp_res <= t)) / kp_res.shape[0])
        tmp.append(n_outliers)
        results.append(tmp)
    cols = list(map(lambda x: '@ {} mm'.format(x), precision)) + ["% out.", ]

    results = pd.DataFrame(data=results, columns=cols)
    print(results)
