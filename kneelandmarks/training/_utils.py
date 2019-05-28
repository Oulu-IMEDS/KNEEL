import torch
from tqdm import tqdm
import gc

from deeppipeline.kvs import GlobalKVS


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
            #target_kp = entry['target_kp']
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
                    pass
                else:
                    raise NotImplementedError
            pbar.update()
            gc.collect()
        gc.collect()
        pbar.close()

    return running_loss / n_batches, None


def val_results_callback(writer, val_metrics, to_log, val_results):
    pass

