import argparse
import pandas as pd
import numpy as np
import json
import cv2
import os
from kneel.inference import LandmarkAnnotator
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle


from kneel.evaluation import make_test_report_comparison, visualize_landmarks
# Implants are excluded from the evaluation
data_ignore = {'OKOA': [('60', 'R')],
               'MAKNEE': [('PA_074', 'R'),
                          ('PA_097', 'L'),
                          ('PA_073', 'L'),
                          ('PA_087', 'L'),
                          ('PA_082', 'L')]}


def parse_via_annotations(annotations):
    landmarks_parsed = {}
    sides_parsed = []
    for side, grp_side in annotations.groupby('Side'):
        for bone, grp_side_bone in grp_side.groupby('Bone'):
            points = grp_side_bone[['x', 'y']].values
            landmarks_parsed[bone+side] = points
        sides_parsed.append(side)

    return landmarks_parsed, sides_parsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_results', default='')
    parser.add_argument('--gt_data', default='')
    parser.add_argument('--bf_data', default='')
    parser.add_argument('--spacings', default='')
    parser.add_argument('--dataset', default='')
    parser.add_argument('--kls', default='')
    parser.add_argument('--datasets_dir', default='')
    args = parser.parse_args()

    gt = pd.read_csv(args.gt_data)
    kls = pd.read_csv(args.kls)
    gt['x'] = gt.region_shape_attributes.apply(lambda x: int(json.loads(x)['cx']), 1)
    gt['y'] = gt.region_shape_attributes.apply(lambda x: int(json.loads(x)['cy']), 1)
    gt['Bone'] = gt.region_attributes.apply(lambda x: json.loads(x)['Bone'], 1)
    gt['Side'] = gt.region_attributes.apply(lambda x: json.loads(x)['Side'], 1)

    bf_preds = pd.read_csv(args.bf_data)
    bf_preds['x'] = bf_preds.region_shape_attributes.apply(lambda x: int(json.loads(x)['cx']), 1)
    bf_preds['y'] = bf_preds.region_shape_attributes.apply(lambda x: int(json.loads(x)['cy']), 1)
    bf_preds['Bone'] = bf_preds.region_attributes.apply(lambda x: json.loads(x)['Bone'], 1)
    bf_preds['Side'] = bf_preds.region_attributes.apply(lambda x: json.loads(x)['Side'], 1)

    spacings = pd.read_csv(args.spacings)
    if args.dataset == 'OKOA':
        spacings.ID = spacings.ID.apply(lambda x: f'{x:02d}', 1)
        kls.ID = kls.ID.apply(lambda x: f'{x:02d}', 1)

    if args.dataset == 'MAKNEE':
        kls.ID = kls.ID.apply(lambda x: f'PA_{x:03d}', 1)

    spacings = {entry.ID: float(entry.spacing) for _, entry in spacings.iterrows()}
    kls = {entry.ID: (entry.KLR, entry.KLL) for _, entry in kls.iterrows()}
    res = np.load(args.saved_results)
    preds = res['preds']
    imgs_ids = res['imgs']
    preds = {imgs_ids[i]: preds[i] for i in range(preds.shape[0])}
    exclude = data_ignore[args.dataset]

    sides = []
    inference_per_knee = []
    gt_per_knee = []
    spacings_arr = []
    bf_preds_per_knee = []
    bf_ann = {grp_name: grp for grp_name, grp in bf_preds.groupby('filename')}
    kls_per_knee = []
    cases = []

    for fname, grp in gt.groupby('filename'):
        img_id = fname.split('.')[0]
        print(f'==> {img_id}')
        preds_cur = preds[img_id]
        landmarks_cur, sides_cur = parse_via_annotations(grp)
        bf_preds_cur, bf_sides_cur = parse_via_annotations(bf_ann[fname])

        assert sides_cur == bf_sides_cur

        if (img_id, 'R') not in exclude:
            sides.append('R')
            inference_per_knee.append(np.expand_dims(preds_cur[0], 0))
            to_add = np.vstack((landmarks_cur['TR'], landmarks_cur['FR']))
            gt_per_knee.append(np.expand_dims(to_add, 0))
            spacings_arr.append(spacings[img_id])

            to_add = np.vstack((bf_preds_cur['TR'], bf_preds_cur['FR']))
            bf_preds_per_knee.append(np.expand_dims(to_add, 0))
            kls_per_knee.append(int(kls[img_id][0]))
            cases.append(img_id)

        if (img_id, 'L') not in exclude:
            sides.append('L')
            inference_per_knee.append(np.expand_dims(preds_cur[1], 0))
            to_add = np.vstack((landmarks_cur['TL'], landmarks_cur['FL']))
            gt_per_knee.append(np.expand_dims(to_add, 0))
            spacings_arr.append(spacings[img_id])

            to_add = np.vstack((bf_preds_cur['TL'], bf_preds_cur['FL']))
            bf_preds_per_knee.append(np.expand_dims(to_add, 0))
            kls_per_knee.append(int(kls[img_id][1]))
            cases.append(img_id)

    gt_per_knee = np.vstack(gt_per_knee)
    inference_per_knee = np.vstack(inference_per_knee)
    bf_inference_per_knee = np.vstack(bf_preds_per_knee)
    kls_per_knee = np.array(kls_per_knee)
    cases = np.array(cases)
    sides = np.array(sides)

    spacings_arr = np.expand_dims(np.array(spacings_arr), 1)

    landmark_errors_ours = np.sqrt(((gt_per_knee - inference_per_knee) ** 2).sum(2))
    landmark_errors_ours *= spacings_arr

    landmark_errors_bf = np.sqrt(((gt_per_knee - bf_inference_per_knee) ** 2).sum(2))
    landmark_errors_bf *= spacings_arr

    print(args.saved_results)

    make_test_report_comparison(args, landmark_errors_bf, landmark_errors_ours)
    make_test_report_comparison(args, landmark_errors_bf[kls_per_knee < 2, :],
                                landmark_errors_ours[kls_per_knee < 2, :], '-no-oa')

    make_test_report_comparison(args, landmark_errors_bf[kls_per_knee >= 2, :],
                                landmark_errors_ours[kls_per_knee >= 2, :], '-oa')

    for kl in range(1, 4):
        if kl == 1:
            idx = (kls_per_knee == 0) | (kls_per_knee == 1)
        else:
            idx = kls_per_knee == kl
        errs_cur = landmark_errors_ours[idx][:, [0, 4, 8, 12, 15]].mean(1)
        cases_cur = cases[idx]
        sides_cur = sides[idx]
        inf_cur = inference_per_knee[idx]
        inf_bf_cur = bf_inference_per_knee[idx]

        idx_srt = np.argsort(errs_cur)

        for case_idx, label in zip([idx_srt[-1], idx_srt[idx_srt.shape[0]//2], idx_srt[0]], ['worst', 'median', 'best']):
            case_worst_cur = cases_cur[case_idx]
            side_worst_cur = sides_cur[case_idx]
            gt_worst_cur, _ = parse_via_annotations(gt[gt.filename == cases_cur[case_idx] + '.png'])
            landmarks_t = gt_worst_cur[f'T{side_worst_cur}']
            landmarks_f = gt_worst_cur[f'F{side_worst_cur}']

            path = os.path.join(args.datasets_dir, args.dataset, cases_cur[case_idx])
            img, spacing, _, _ = LandmarkAnnotator.read_dicom(path, new_spacing=None)
            roi_size_px = int(115 * 1. / spacing)
            tmp = np.expand_dims(landmarks_t[4], 0).copy()

            _, roi = LandmarkAnnotator.localize_left_right_rois(img, roi_size_px, np.vstack((tmp, tmp)))


            # GT points
            landmarks_t -= tmp[0] - roi_size_px // 2
            landmarks_f -= tmp[0] - roi_size_px // 2

            roi = LandmarkAnnotator.resize_to_spacing(roi, spacing, new_spacing=0.3)
            scale = spacing / 0.3

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(roi, cmap=plt.cm.Greys_r)
            ax.plot(landmarks_t[:, 0] * scale, landmarks_t[:, 1] * scale, 'ro', alpha=0.3)
            ax.plot(landmarks_f[:, 0] * scale, landmarks_f[:, 1] * scale, 'go', alpha=0.3)

            # Inference (own)
            inf_t = inf_cur[case_idx, :9]
            inf_f = inf_cur[case_idx, 9:]

            inf_t -= tmp[0] - roi_size_px // 2
            inf_f -= tmp[0] - roi_size_px // 2

            ax.plot(inf_t[:, 0] * scale, inf_t[:, 1] * scale, 'rx')
            ax.plot(inf_f[:, 0] * scale, inf_f[:, 1] * scale, 'gx')

            # Inference (bone finder)
            inf_t = inf_bf_cur[case_idx, :9]
            inf_f = inf_bf_cur[case_idx, 9:]

            inf_t -= tmp[0] - roi_size_px // 2
            inf_f -= tmp[0] - roi_size_px // 2

            ax.plot(inf_t[:, 0] * scale, inf_t[:, 1] * scale, 'rv',)
            ax.plot(inf_f[:, 0] * scale, inf_f[:, 1] * scale, 'gv')

            ax.set_xticks([])
            ax.set_yticks([])
            plt.ylim(250, 150)
            plt.tight_layout()

            save_dir = '/'.join(args.saved_results.split('/')[:-1])

            plt.savefig(os.path.join(save_dir, f'{args.dataset}-example-{kl}-{label}.pdf'), bbox_inches='tight')
            plt.close()
