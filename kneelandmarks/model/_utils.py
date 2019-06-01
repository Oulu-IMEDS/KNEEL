from torch import nn
from deeppipeline.kvs import GlobalKVS
from ._hourglass import HourglassNet


def init_model(ignore_data_parallel=False):
    kvs = GlobalKVS()
    if kvs['args'].annotations == 'lc':
        net = HourglassNet(3, 1, bw=kvs['args'].base_width,
                           upmode='bilinear', refinement=kvs['args'].hg_refine,
                           use_sagm=kvs['args'].sagm, fast_mode=kvs['args'].fast_hg,
                           multiscale_hg_block=kvs['args'].multiscale_hg)
    else:
        net = HourglassNet(3, 20, bw=kvs['args'].base_width,
                           upmode='bilinear', refinement=kvs['args'].hg_refine,
                           use_sagm=kvs['args'].sagm, fast_mode=kvs['args'].fast_hg,
                           multiscale_hg_block=kvs['args'].multiscale_hg)

    if not ignore_data_parallel:
        if kvs['gpus'] > 1:
            net = nn.DataParallel(net).to('cuda')

    return net.to('cuda')
