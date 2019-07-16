import argparse
import pandas as pd
import numpy as np
import json
from kneel.evaluation import landmarks_report_partial
import os
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
    args = parser.parse_args()

    gt = pd.read_csv(args.gt_data)
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
    spacings = {entry.ID: float(entry.spacing) for _, entry in spacings.iterrows()}
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

        if (img_id, 'L') not in exclude:
            sides.append('L')
            inference_per_knee.append(np.expand_dims(preds_cur[1], 0))
            to_add = np.vstack((landmarks_cur['TL'], landmarks_cur['FL']))
            gt_per_knee.append(np.expand_dims(to_add, 0))
            spacings_arr.append(spacings[img_id])

            to_add = np.vstack((bf_preds_cur['TL'], bf_preds_cur['FL']))
            bf_preds_per_knee.append(np.expand_dims(to_add, 0))

    gt = np.vstack(gt_per_knee)
    inference_per_knee = np.vstack(inference_per_knee)
    bf_inference_per_knee = np.vstack(bf_preds_per_knee)

    sides = np.array(sides)
    spacings_arr = np.expand_dims(np.array(spacings_arr), 1)

    landmark_errors_ours = np.sqrt(((gt - inference_per_knee) ** 2).sum(2))
    landmark_errors_ours *= spacings_arr

    landmark_errors_bf = np.sqrt(((gt - bf_inference_per_knee) ** 2).sum(2))
    landmark_errors_bf *= spacings_arr

    print(args.saved_results)
    for landmark_errors, label in zip([landmark_errors_bf, landmark_errors_ours], ['bf', 'ours']):
        outliers = np.zeros(landmark_errors.shape)
        outliers[landmark_errors >= 10] = 1
        precision = [1, 1.5, 2, 2.5]
        save_dir = '/'.join(args.saved_results.split('/')[:-1])
        errs_t = np.expand_dims(landmark_errors[:, [0, 5, 8]].mean(1), 1)
        errs_f = np.expand_dims(landmark_errors[:, [9, 12, 15]].mean(1), 1)
        errs = np.hstack((errs_t, errs_f))

        save_plot_path = os.path.join(save_dir, f'{label}_{args.dataset}_inference.pdf')
        res_aggregated, outliers_percentage = landmarks_report_partial(errs, precision, outliers, None,
                                                                       save_plot=save_plot_path)

        tmp = []
        for m, s in zip(res_aggregated['mean'].values, res_aggregated['std'].values):
            tmp.append(f'${m:.2f} \\pm {s:.2f}$')
        tmp.append(f'${outliers_percentage:.2f}$')
        print(label)
        print(' & '.join(tmp))
