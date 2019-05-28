import numpy as np
import torch
from tqdm import tqdm
import gc
import pandas as pd
from deeppipeline.kvs import GlobalKVS
from kneelandmarks.data.utils import get_landmarks_from_hm


def pass_epoch(net, loader, optimizer, criterion):
    kvs = GlobalKVS()
    net.train(optimizer is not None)

    fold_id = kvs['cur_fold']
    epoch = kvs['cur_epoch']
    max_ep = kvs['args'].n_epochs

    running_loss = 0.0
    n_batches = len(loader)
    landmark_errors = []
    device = next(net.parameters()).device
    pbar = tqdm(total=n_batches, ncols=200)
    with torch.set_grad_enabled(optimizer is not None):
        for i, entry in enumerate(loader):
            if optimizer is not None:
                optimizer.zero_grad()

            inputs = entry['img'].to(device)
            target_hm = entry['target_hm'].to(device).squeeze().unsqueeze(1)
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
            if optimizer is None:
                if not kvs['args'].sagm:
                    if isinstance(outputs, tuple):
                        target_kp = entry['kp_gt'].numpy()
                        predict = outputs[-1].to('cpu').numpy()
                        xy_batch = np.zeros((predict.shape[0], 2))
                        for j in range(predict.shape[0]):
                            try:
                                xy_batch[j] = get_landmarks_from_hm(predict[j],
                                                                    inputs.size()[-2:],
                                                                    2, 0.9)
                            except IndexError:
                                xy_batch[j] = -1
                        spacing = getattr(kvs['args'], f"{kvs['args'].annotations}_spacing")
                        err = np.sqrt(np.sum(((target_kp.squeeze() - xy_batch.squeeze())*spacing) ** 2, 1))
                        landmark_errors.append(err)
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
            pbar.update()
            gc.collect()
        gc.collect()
        pbar.close()

    if len(landmark_errors) > 0:
        landmark_errors = np.hstack(landmark_errors)
    else:
        landmark_errors = None

    return running_loss / n_batches, landmark_errors


def val_results_callback(writer, val_metrics, to_log, val_results):
    results = []
    precision = [1, 1.5, 2, 2.5, 3, 5, 10]
    n_outliers = np.sum(val_results < 0)
    val_results = val_results[val_results > 0]

    tmp = []
    for t in precision:
        tmp.append(np.sum((val_results <= t)) / val_results.shape[0])

    results.append(tmp)
    results = pd.DataFrame(data=results, columns=list(map(lambda x: '@ {} mm'.format(x), precision)))
    print('# outliers:', n_outliers)
    print(results)
