import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from deeppipeline.common.evaluation import cumulative_error_plot
import os
import sys

def visualize_landmarks(img, landmarks_t, landmarks_f, figsize=8, radius=3, save_path=None):
    """
    Visualizes tibial and femoral landmarks

    Parameters
    ----------
    img : np.ndarray
        Image
    landmarks_t : np.ndarray
        Tibial landmarks
    landmarks_f : np.ndarray
        Femoral landmarks
    figsize : int
        The size of the figure
    radius : int
        The radius of the circle
    Returns
    -------
    out: None
        Makes and image plot with overlayed landmarks.

    """
    landmarks_t = PatchCollection(map(lambda x: Circle(x, radius=radius), landmarks_t), color='red')
    landmarks_f = PatchCollection(map(lambda x: Circle(x, radius=radius), landmarks_f), color='green')

    plt.figure(figsize=(figsize, figsize))
    plt.imshow(img, cmap=plt.cm.Greys_r)
    plt.axes().add_collection(landmarks_t)
    plt.axes().add_collection(landmarks_f)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


def assess_errors(val_results):
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
    return results


def landmarks_report_partial(errs, precision, outliers, plot_title=None, save_plot=None):
    results = []

    cumulative_error_plot(errs, labels=['Tibia', 'Femur'],
                          title=plot_title,
                          colors=['blue', 'red'],
                          save_plot=save_plot)

    for kp_id in range(errs.shape[1]):
        kp_res = errs[:, kp_id]

        tmp = []
        for t in precision:
            tmp.append(np.sum((kp_res <= t)) / kp_res.shape[0])
        results.append(tmp)
    cols = list(map(lambda x: '@ {} mm'.format(x), precision))

    results = pd.DataFrame(data=results, columns=cols)
    res_grouped = pd.concat(((results.mean(0) * 100).round(2),
                             (results.std(0) * 100).round(2)), keys=['mean', 'std'])

    outliers_percentage = 100. * (outliers.any(1)).sum() * 1. / outliers.shape[0]
    return res_grouped, outliers_percentage


def landmarks_report_full(inference, gt, spacing, kls, save_results_root, precision_array=None, report_kl=False,
                          experiment_desc=None):
    landmark_errors = np.sqrt(((gt - inference) ** 2).sum(2))
    landmark_errors *= spacing

    errs_t = np.expand_dims(landmark_errors[:, [0, 8]].mean(1), 1)
    errs_f = np.expand_dims(landmark_errors[:, [9, 15]].mean(1), 1)
    errs = np.hstack((errs_t, errs_f))
    if precision_array is None:
        precision = [1, 2, 3]
    else:
        precision = precision_array
    outliers = np.zeros(landmark_errors.shape)
    outliers[landmark_errors >= 10] = 1
    rep_all, outliers_percentage = landmarks_report_partial(errs, precision, outliers, None,
                                                            save_plot=os.path.join(save_results_root,
                                                                                   'all_grades.pdf'))
    outliers_percentage = np.round(outliers_percentage, 2)
    lines = list()
    lines.append('\\toprule')
    header = ['Setting', ] + rep_all['mean'].index.tolist() + ['\\% out', ]
    header = list(map(lambda x: '\\textbf{'f'{x}''}', header))
    lines.append(header)
    lines.append('\\midrule')
    tmp = list()
    if experiment_desc is None:
        tmp.append('All grades')
    else:
        tmp.append(experiment_desc)
    for m, s in zip(rep_all['mean'].values, rep_all['std'].values):
        tmp.append(f'${m:.2f} \\pm {s:.2f}$')
    tmp.append(f'${outliers_percentage:.2f}$')
    lines.append(tmp)
    if report_kl:
        for kl in range(5):
            idx = kls == kl
            errs_kl = errs[idx]
            outliers_kl = outliers[idx]
            rep_kl, outliers_percentage_kl = landmarks_report_partial(errs_kl,
                                                                      precision,
                                                                      outliers_kl,
                                                                      None,
                                                                      save_plot=os.path.join(save_results_root,
                                                                                             f'{kl}.pdf'))
            tmp = list()
            tmp.append(f'KL{kl}')
            for m, s in zip(rep_kl['mean'].values, rep_kl['std'].values):
                tmp.append(f'${m:.2f} \\pm {s:.2f}$')
            tmp.append(f'${outliers_percentage_kl:.2f}$')
            lines.append(tmp)
    lines.append('\\bottomrule')

    with open(os.path.join(save_results_root, 'cv_res.tex'), 'w') as results_file:
        for f_print in [sys.stdout, results_file]:
            print('\\begin{table}[ht!]', file=f_print)
            print('\\begin{tabular}{'f'{"c"*len(lines[1])}''}', file=f_print)
            for l in lines:
                if 'rule' not in l:
                    print(' & '.join(l).replace('@ ', '') + '\\\\', file=f_print)
                else:
                    print(l, file=f_print)
            print('\\end{tabular}', file=f_print)
            print('\\end{table}', file=f_print)