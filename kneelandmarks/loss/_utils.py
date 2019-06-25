from deeppipeline.kvs import GlobalKVS
from deeppipeline.common.losses import LNLoss, ElasticLoss, WingLoss, GeneralizedRobustLoss


def init_loss():
    kvs = GlobalKVS()
    if kvs['args'].loss_type == 'elastic':
        loss = ElasticLoss(w=kvs['args'].loss_weight)
    elif kvs['args'].loss_type == 'l2':
        loss = LNLoss(space='l2')
    elif kvs['args'].loss_type == 'l1':
        loss = LNLoss(space='l1')
    elif kvs['args'].loss_type == 'wing':
        loss = WingLoss(width=kvs['args'].wing_w, curvature=kvs['args'].wing_c)
    elif kvs['args'].loss_type == 'robust':
        loss = GeneralizedRobustLoss(num_dims=20 * 2 if kvs['args'].annotations == 'hc' else 2,
                                     alpha_init=kvs['args'].alpha_robust,
                                     scale_init=kvs['args'].c_robust,
                                     alpha_lo=kvs['args'].alpha_robust_min,
                                     alpha_hi=kvs['args'].alpha_robust_max)
    else:
        raise NotImplementedError

    return loss.to('cuda')
