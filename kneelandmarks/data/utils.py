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


def flip_left_landmarks(dataset_entry, imwidth):
    """
    Flips the BoneFinder landmarks inplace for pandas dataset entry

    Parameters
    ----------
    dataset_entry : pd.Series
        Row of a pandas dataframe
    imwidth : int
        The width of the image (whole dicom)

    Returns
    -------
    out : pd.Series
        Modified dataset row.

    """
    if dataset_entry.Side == 'L':
        tibial_landamrks = dataset_entry['T']
        femoral_landmarks = dataset_entry['F']

        tibial_landmarks_arr = parse_landmarks(tibial_landamrks)
        femoral_landmarks_array = parse_landmarks(femoral_landmarks)

        tibial_landmarks_arr[:, 0] = imwidth - tibial_landmarks_arr[:, 0]
        femoral_landmarks_array[:, 0] = imwidth - femoral_landmarks_array[:, 0]

        dataset_entry['T'] = ''.join(map(lambda x: '{},{},'.format(*x), tibial_landmarks_arr))[:-1]
        dataset_entry['F'] = ''.join(map(lambda x: '{},{},'.format(*x), femoral_landmarks_array))[:-1]

    return dataset_entry


def landmark2map(lm, shape, sigma=1.5):
    # lm = (x,y)
    m = np.zeros(shape, dtype=np.uint8)

    if np.all(lm > 0) and lm[0] < shape[1] and lm[1] < shape[0]:
        x, y = np.meshgrid(np.linspace(-0.5, 0.5, m.shape[1]), np.linspace(-0.5, 0.5, m.shape[0]))
        mux = (lm[0]-m.shape[1]//2)/1./m.shape[1]
        muy = (lm[1]-m.shape[0]//2)/1./m.shape[0]
        s = sigma / 1. / m.shape[0]
        m = (x-mux)**2 / 2. / s**2 + (y-muy)**2 / 2. / s**2
        m = np.exp(-m)
        m -= m.min()
        m /= m.max()

    return m*255


def generate_maps(lms, shape, sigma=1.5):
    res = []
    for i in range(lms.shape[0]):
        res.append(landmark2map(lms[i], shape, sigma))

    return res


def pre_filter_landmarks(t_landmarks, f_landmarks):
    # from all the bonefinder landmarks we select only a few relevant ones
    t_pt1, t_pt2, t_pt3 = t_landmarks[0, :], t_landmarks[t_landmarks.shape[0] // 2, :], t_landmarks[-1, :]
    f_pt1, f_pt2, f_pt3 = f_landmarks[0, :], f_landmarks[f_landmarks.shape[0] // 2, :], f_landmarks[-1, :]

    return np.array([t_pt1, t_pt2, t_pt3]), np.array([f_pt1, f_pt2, f_pt3])


def read_sas7bdata_pd(fname):
    data = []
    with SAS7BDAT(fname) as f:
        for row in f:
            data.append(row)

    return pd.DataFrame(data[1:], columns=data[0])
