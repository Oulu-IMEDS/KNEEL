import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
import numpy as np


def visualize_landmarks(img, landmarks_t, landmarks_f, figsize=8, radius=3):
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
    plt.show()


