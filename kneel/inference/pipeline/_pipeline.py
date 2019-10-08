import numpy as np
import logging
from kneel.inference.pipeline import LandmarkAnnotator


class KneeAnnotatorPipeline(object):
    def __init__(self, lc_snapshot_path, hc_snapshot_path, mean_std_path, device, jit_trace=True, logger=None):
        if logger is None:
            logger = logging.Logger('KneeAnnotatorPipeline')
            logger.setLevel(logging.DEBUG)

        if isinstance(logger, logging.Logger):
            self.logger = logger
            logger_lc = logger
            logger_hc = logger
        elif isinstance(logger, dict):
            self.logger = logger['kneel-backend:pipeline']
            logger_lc = logger['kneel-backend:roi-loc']
            logger_hc = logger['kneel-backend:landmarks-loc']
        else:
            raise TypeError('Unknown logger type!')
        self.logger.log(logging.INFO, 'Initializing the global searcher (ROI localizer)')
        self.global_searcher = LandmarkAnnotator(snapshot_path=lc_snapshot_path,
                                                 mean_std_path=mean_std_path,
                                                 device=device, jit_trace=jit_trace,
                                                 logger=logger_lc)
        self.logger.log(logging.INFO, 'Initializing the local searcher (landmark localizer)')
        self.local_searcher = LandmarkAnnotator(snapshot_path=hc_snapshot_path,
                                                mean_std_path=mean_std_path,
                                                device=device, jit_trace=jit_trace,
                                                logger=logger_hc)

    def predict(self, img_name, roi_size_mm=140, pad=300, refine=True):
        self.logger.log(logging.INFO, f'Loading the image with a new spacing of {self.global_searcher.img_spacing} mm.')
        res = self.global_searcher.read_dicom(img_name,
                                              new_spacing=self.global_searcher.img_spacing,
                                              return_orig=True)
        if len(res) > 0:
            img, orig_spacing, h_orig, w_orig, img_orig = res
        else:
            return None

        # First pass of knee joint center estimation
        self.logger.log(logging.INFO, 'Predicting knee joint centers')
        roi_size_px = int(roi_size_mm * 1. / orig_spacing)
        global_coords = self.global_searcher.predict_img(img, h_orig, w_orig)
        img_orig = LandmarkAnnotator.pad_img(img_orig, pad if pad != 0 else None)
        global_coords += pad
        self.logger.log(logging.INFO, 'Predicting knee landmarks')
        landmarks, right_roi_orig, left_roi_orig = self.local_searcher.predict_local(img_orig, global_coords,
                                                                                     roi_size_px, orig_spacing)

        if refine:
            # refinement
            self.logger.log(logging.INFO, 'Refining the predictions via second pass through the landmark localizer.')
            centers_d = np.array([roi_size_px // 2, roi_size_px // 2]) - landmarks[:, 4]
            global_coords -= centers_d
            # prediction for refined centers
            landmarks, right_roi_orig, left_roi_orig = self.local_searcher.predict_local(img_orig, global_coords,
                                                                                         roi_size_px, orig_spacing)
        landmarks -= pad
        landmarks[0, :, :] += global_coords[0, :] - roi_size_px // 2
        landmarks[1, :, :] += global_coords[1, :] - roi_size_px // 2
        landmarks = np.expand_dims(landmarks, 0)
        return landmarks
