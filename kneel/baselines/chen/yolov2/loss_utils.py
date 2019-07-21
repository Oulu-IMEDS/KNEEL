# -*- coding: utf-8 -*-

import os, sys, pdb
import numpy as np

def set_weights():
    init_weights = np.array([[1, 3, 5, 7, 9],
                             [3, 1, 3, 5, 7],
                             [5, 3, 1, 3, 5],
                             [7, 5, 3, 1, 3],
                             [9, 7, 5, 3, 1]], dtype=np.float)

    adjusted_weights = init_weights + 1.0
    np.fill_diagonal(adjusted_weights, 0)

    return adjusted_weights
