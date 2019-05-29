import cv2

from deeppipeline.common.core import train_n_folds, init_ms_scheduler
from deeppipeline.common.dataset import init_folds, init_pd_meta

from kneelandmarks.training.args import parse_args
from kneelandmarks.model import init_model
from kneelandmarks.data.pipeline import init_augs, init_data_processing, init_loaders
from kneelandmarks.loss import init_loss
from kneelandmarks.training import pass_epoch, val_results_callback

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

if __name__ == "__main__":
    train_n_folds(init_args=parse_args,
                  init_metadata=init_pd_meta,
                  init_augs=init_augs,
                  init_data_processing=init_data_processing,
                  init_folds=init_folds,
                  init_loaders=init_loaders,
                  init_model=init_model,
                  init_loss=init_loss,
                  init_scheduler=init_ms_scheduler,
                  pass_epoch=pass_epoch, log_metrics_cb=val_results_callback,
                  img_group_id_colname='subject_id', img_class_colname='kl')
