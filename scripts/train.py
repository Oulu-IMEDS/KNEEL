import cv2

from deeppipeline.common.core import train_n_folds
from deeppipeline.common.dataset import init_folds, init_pd_meta

from kneel.training.args import parse_args
from kneel.model import init_model
from kneel.data.pipeline import init_augs, init_data_processing, init_loaders
from kneel.loss import init_loss
from kneel.training import pass_epoch, val_results_callback

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
                  init_optimizer=None,
                  init_scheduler=None,
                  pass_epoch=pass_epoch,
                  log_metrics_cb=val_results_callback,
                  img_key='img',
                  img_group_id_colname='subject_id',
                  img_class_colname='kl')
