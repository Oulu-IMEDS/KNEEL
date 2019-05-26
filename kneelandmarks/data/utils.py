import numpy as np
import pydicom as dicom
from sas7bdat import SAS7BDAT
import pandas as pd


def read_pts(fname):
    with open(fname) as f:
        content = f.read()
    arr = np.array(list(map(lambda x: [float(x.split()[0]), float(x.split()[1])], content.split('\n')[3:-2])))
    return arr


def read_dicom(filename):
    """
    Reads a dicom file
    Parameters
    ----------
    filename : str
        Full path to the image
    Returns
    -------
    out : tuple
        Image itself as uint16, spacing, and the DICOM metadata
    """

    try:
        data = dicom.read_file(filename)
    except:
        return None
    img = np.frombuffer(data.PixelData, dtype=np.uint16).copy().astype(np.float64)

    if data.PhotometricInterpretation == 'MONOCHROME1':
        img = img.max() - img
    try:
        img = img.reshape((data.Rows, data.Columns))
    except:
        return None

    try:
        if isinstance(data.ImagerPixelSpacing, str):
            data.ImagerPixelSpacing = data.ImagerPixelSpacing.split()
    except:
        pass

    try:
        if isinstance(data.PixelSpacing, str):
            data.PixelSpacing = data.PixelSpacing.split()
    except:
        pass

    try:
        return img, float(data.ImagerPixelSpacing[0]), data
    except:
        pass
    try:
        return img, float(data.PixelSpacing[0]), data
    except:
        return None


def process_xray(img, cut_min=5, cut_max=99, multiplier=255):
    # This function changes the histogram of the image by doing global contrast normalization
    # cut_min - lowest percentile which is used to cut the image histogram
    # cut_max - highest percentile

    img = img.copy()
    lim1, lim2 = np.percentile(img, [cut_min, cut_max])
    img[img < lim1] = lim1
    img[img > lim2] = lim2

    img -= lim1
    img /= img.max()
    img *= multiplier

    return img


def parse_landmarks(txt):
    pts = np.array(list(map(int, txt.values[0].split(','))))
    return pts.reshape(pts.shape[0] // 2, 2)


def read_sas7bdata_pd(fname):
    data = []
    with SAS7BDAT(fname) as f:
        for row in f:
            data.append(row)

    return pd.DataFrame(data[1:], columns=data[0])
