from deeppipeline.kvs import GlobalKVS
from deeppipeline.common.losses import LNLoss, ElasticLoss, WingLoss


def init_loss():
    kvs = GlobalKVS()
    if kvs['args'].loss_type:
        loss = ElasticLoss(w=kvs['args'].loss_weight)
    elif kvs['args'].loss_type == 'l2':
        loss = LNLoss(space='l2')
    elif kvs['args'].loss_type == 'l1':
        loss = LNLoss(space='l1')
    elif kvs['args'].loss_type == 'wing':
        loss = WingLoss(width=kvs['args'].wing_w, curvature=kvs['args'].wing_c)
    else:
        raise NotImplementedError

    return loss.to('cuda')
