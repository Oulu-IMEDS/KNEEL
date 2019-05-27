from deeppipeline.kvs import GlobalKVS
from ._losses import HGHMLoss, ElasticLoss


def init_loss():
    kvs = GlobalKVS()
    if kvs['args'].sagm:
        loss = ElasticLoss(w=kvs['args'].loss_weight)
    else:
        loss = HGHMLoss(hm_loss=kvs['args'].hm_loss, w=kvs['args'].loss_weight)
    return loss.to('cuda')
