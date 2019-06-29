from deeppipeline.kvs import GlobalKVS
from ._hourglass import HourglassNet


def init_model():
    kvs = GlobalKVS()
    if kvs['args'].annotations == 'lc':
        net = HourglassNet(3, 1, bw=kvs['args'].base_width,
                           upmode='bilinear',
                           multiscale_hg_block=kvs['args'].multiscale_hg,
                           se=kvs['args'].use_se,
                           se_ratio=kvs['args'].se_ratio)
    else:
        net = HourglassNet(3, 16, bw=kvs['args'].base_width,
                           upmode='bilinear',
                           multiscale_hg_block=kvs['args'].multiscale_hg,
                           se=kvs['args'].use_se,
                           se_ratio=kvs['args'].se_ratio)

    return net.to('cuda')
