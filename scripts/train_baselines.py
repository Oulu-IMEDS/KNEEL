import pickle
import argparse
import numpy as np
import os
from tqdm import tqdm
from menpo.shape import PointCloud
import menpo.io as mio
from kneel.data.utils import parse_landmarks
import menpofit.clm as clm
import menpofit.aam as aam
import menpo.feature as mpf
from kneel.evaluation import landmarks_report_full


def load_df_menpo(df, data_dir):
    imgs = []
    k = []
    kls = []

    for row_id, entry in tqdm(df.iterrows(), total=df.shape[0]):
        path_img = os.path.join(data_dir, f'{entry.subject_id}_{entry.kl}_{entry.side}.png')
        img = mio.import_image(path_img, normalize=True)
        kls.append(entry.kl)
        tl = parse_landmarks(entry.tibia)[:, [1, 0]]
        fl = parse_landmarks(entry.femur)[:, [1, 0]]
        stacked_kpts = np.vstack((tl, fl))
        img.landmarks['pts'] = PointCloud(stacked_kpts)
        imgs.append(img)
        k.append(np.expand_dims(stacked_kpts, 0))

    gt_landmarks = np.stack(k, 0).squeeze()
    return imgs, gt_landmarks[:, :, [1, 0]], PointCloud(gt_landmarks.mean(0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', default='')
    parser.add_argument('--reference_snapshot', default='')
    parser.add_argument('--save_pics', type=bool, default=False)
    args = parser.parse_args()

    snp_full_path = os.path.join(args.workdir, 'snapshots', args.reference_snapshot)
    snp_session_full_path = os.path.join(snp_full_path, 'session.pkl')

    with open(snp_session_full_path, 'rb') as f:
        snapshot_session = pickle.load(f)
    snp_args = snapshot_session['args'][0]
    for arg in vars(snp_args):
        if not hasattr(args, arg):
            setattr(args, arg, getattr(snp_args, arg))

    for model, fitter, model_name in [(aam.HolisticAAM, aam.LucasKanadeAAMFitter, 'AAM'),
                                      (clm.CLM, clm.GradientDescentCLMFitter, 'CLM')]:
        for feature, feature_name in [(mpf.double_igo, 'igo'),
                                      (mpf.lbp, 'lbp')]:

            save_res_path = os.path.join(args.workdir, 'baselines', model_name, feature_name)
            if not os.path.isfile(os.path.join(save_res_path, 'oof_results.npz')):
                os.makedirs(save_res_path, exist_ok=True)
                oof_gt = []
                oof_preds = []
                oof_kls = []
                oof_subject_ids = []
                oof_sides = []

                for fold_id, train_split, val_split in snapshot_session['cv_split'][0]:
                    print(f'==> Loading fold {fold_id} data')
                    train, _, mean_shape = load_df_menpo(train_split, snapshot_session['args'][0].dataset_root)
                    val, val_gt_landmarks, _ = load_df_menpo(val_split, snapshot_session['args'][0].dataset_root)
                    print(f'==> Training [{model_name} | {feature_name}]:')

                    model_trained = model(train, holistic_features=feature)
                    model_fitter = fitter(model_trained, n_shape=0.9)
                    val_pts_preds = []
                    for val_img in tqdm(val, total=len(val), desc='Validating'):
                        fr = model_fitter.fit_from_shape(val_img, mean_shape)
                        val_pts_preds.append(np.expand_dims(fr.final_shape.points[:, [1, 0]], 0))

                    oof_preds.append(np.vstack(val_pts_preds).squeeze())
                    oof_gt.append(val_gt_landmarks)
                    oof_kls.extend(val_split.kl.values.tolist())
                    oof_subject_ids.extend(val_split.subject_id.values.tolist())
                    oof_sides.extend(val_split.side.values.tolist())

                oof_gt = np.vstack(oof_gt)
                oof_inference = np.vstack(oof_preds)
                subject_ids = np.array(oof_subject_ids)
                kls = np.array(oof_kls)
                oof_sides = np.array(oof_sides)

                np.savez(os.path.join(save_res_path, 'oof_results.npz'),
                         oof_inference=oof_inference,
                         oof_gt=oof_gt,
                         oof_sides=oof_sides,
                         subject_ids=subject_ids,
                         kls=kls)
            else:
                tmp = np.load(os.path.join(save_res_path, 'oof_results.npz'))
                oof_inference = tmp['oof_inference']
                oof_gt = tmp['oof_gt']
                oof_kls = tmp['kls']

            landmarks_report_full(inference=oof_inference, gt=oof_gt,
                                  spacing=getattr(args, f'{args.annotations}_spacing'),
                                  kls=oof_kls,
                                  save_results_root=save_res_path,
                                  precision_array=[1, 1.5, 2, 2.5], report_kl=False,
                                  experiment_desc=f'{model_name} | {feature_name}')


