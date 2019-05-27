from kneelandmarks.training import train_n_folds
from kneelandmarks.training.args import parse_args
from kneelandmarks.model import init_model
from kneelandmarks.data.pipeline import init_augs, init_data_processing, init_meta, init_loaders
from kneelandmarks.loss import init_loss

if __name__ == "__main__":
    train_n_folds(init_args=parse_args,
                  init_metadata=init_meta,
                  init_augs=init_augs,
                  init_data_processing=init_data_processing,
                  init_loaders=init_loaders,
                  init_model=init_model,
                  init_loss=init_loss,
                  img_group_id_colname='subject_id', img_class_colname='kl')
