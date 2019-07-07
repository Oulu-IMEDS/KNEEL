import pickle
import argparse
import numpy as np
import os
import torch
from tqdm import tqdm
import pandas as pd
import glob

from kneelandmarks.model import init_model
from kneelandmarks.data.pipeline import init_loaders
from kneelandmarks.evaluation import visualize_landmarks, landmarks_report

from deeppipeline.common.evaluation import cumulative_error_plot
from deeppipeline.kvs import GlobalKVS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', default='')
    parser.add_argument('--snapshot', default='')
    parser.add_argument('--save_pics', type=bool, default=False)
    args = parser.parse_args()

    snp_full_path = os.path.join(args.workdir, 'snapshots', args.snapshot)
    mean_vector, std_vector = np.load(os.path.join(args.workdir, 'snapshots', 'mean_std.npy'))
    mean_vector = torch.from_numpy(mean_vector).unsqueeze(1).unsqueeze(1)
    std_vector = torch.from_numpy(std_vector).unsqueeze(1).unsqueeze(1)

    snp_session_full_path = os.path.join(snp_full_path, 'session.pkl')
    oof_results_dir = os.path.join(args.workdir, 'snapshots', args.snapshot, 'oof_inference')
    os.makedirs(os.path.join(oof_results_dir, 'pics'), exist_ok=True)

    with open(snp_session_full_path, 'rb') as f:
        snapshot_session = pickle.load(f)

    snp_args = snapshot_session['args'][0]
    for arg in vars(snp_args):
        if not hasattr(args, arg):
            setattr(args, arg, getattr(snp_args, arg))

    kvs = GlobalKVS()
    kvs.update('args', args)
    kvs.update('val_trf', snapshot_session['val_trf'][0])
    kvs.update('train_trf', snapshot_session['train_trf'][0])

    oof_inference = []
    oof_gt = []
    subject_ids = []
    kls = []
    with torch.no_grad():
        for fold_id, train_split, val_split in snapshot_session['cv_split'][0]:
            _, val_loader = init_loaders(train_split, val_split, sequential_val_sampler=True)
            net = init_model()
            snp_weigths_path = glob.glob(os.path.join(snp_full_path, f'fold_{fold_id}*.pth'))[0]
            net.load_state_dict(torch.load(snp_weigths_path)['model'])
            if torch.cuda.device_count() > 1:
                net = torch.nn.DataParallel(net)

            for batch in tqdm(val_loader, total=len(val_loader), desc=f'Processing fold [{fold_id}]:'):
                img = batch['img'].to('cuda')
                out = net(img).to('cpu').numpy()
                gt = batch['kp_gt'].numpy()
                h, w = img.size()[-2:]

                out[:, :, 0] *= (w - 1)
                out[:, :, 1] *= (h - 1)

                gt[:, :, 0] *= (w - 1)
                gt[:, :, 1] *= (h - 1)

                if args.save_pics:
                    for img_id in range(batch['img'].size(0)):
                        subj_id = batch['subject_id'][img_id]
                        side = batch['side'][img_id]
                        kl = batch['kl'][img_id]
                        img = batch['img'][img_id] * std_vector + mean_vector
                        img = img.transpose(0, 2).transpose(0, 1).numpy().astype(np.uint8)
                        save_path = os.path.join(oof_results_dir, 'pics', f'{subj_id}_{side}_{kl}.png')
                        visualize_landmarks(img, out[img_id, :9, :], out[img_id, 9:, :], save_path=save_path)

                oof_inference.append(out)
                oof_gt.append(gt)
                subject_ids.append(batch['subject_id'])
                kls.append(batch['kl'])

    oof_inference = np.round(np.vstack(oof_inference))
    oof_gt = np.vstack(oof_gt)
    subject_ids = np.hstack(subject_ids)
    kls = np.hstack(kls)

    outliers = np.zeros(oof_inference.shape[:-1])
    landmark_errors = np.sqrt(((oof_gt - oof_inference)**2).sum(2))
    spacing = getattr(kvs['args'], f"{kvs['args'].annotations}_spacing")
    landmark_errors *= spacing
    ref_distance = 10  # 10 mm distance for outliers
    outliers[landmark_errors >= ref_distance] = 1
    print('==> Outliers')
    print(subject_ids[outliers.any(1)])

    errs_t = np.expand_dims(landmark_errors[:, :9].mean(1), 1)
    errs_f = np.expand_dims(landmark_errors[:, 9:].mean(1), 1)
    errs = np.hstack((errs_t, errs_f))
    precision = [1, 1.5, 2, 2.5, 3]
    rep_all = landmarks_report(errs, outliers, precision, 'All grades')
    print(rep_all)
    """
    for kl in range(5):
        print(f'==> KL {kl}')
        idx = kls == kl
        errs = landmark_errors[idx]
        outliers_cur = outliers[idx, :]
        f'KL {kl}'

        print(res_grouped)
    """