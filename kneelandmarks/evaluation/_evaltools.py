import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from deeppipeline.common.evaluation import cumulative_error_plot


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


def landmarks_report(errs, outliers, precision, plot_title=None):
    results = []

    cumulative_error_plot(errs, labels=['Tibia', 'Femur'], title=plot_title, colors=['red', 'green'])

    for kp_id in range(errs.shape[1]):
        kp_res = errs[:, kp_id]

        tmp = []
        for t in precision:
            tmp.append(np.sum((kp_res <= t)) / kp_res.shape[0])
        results.append(tmp)
    cols = list(map(lambda x: '@ {} mm'.format(x), precision))

    results = pd.DataFrame(data=results, columns=cols)
    res_grouped = pd.concat(((results.mean(0) * 100).round(2), (results.std(0) * 100).round(2)), keys=['mean', 'std'])
    return res_grouped
