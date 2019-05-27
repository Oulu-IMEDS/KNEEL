from torch import nn
from deeppipeline.kvs import GlobalKVS
from ._hourglass import HourglassNet


def init_model(ignore_data_parallel=False):
    kvs = GlobalKVS()
    if kvs['args'].annotations == 'lc':
        net = HourglassNet(1, 1, bw=kvs['args'].base_width, upmode='bilinear')
    else:
        net = HourglassNet(1, 20, bw=kvs['args'].base_width, upmode='bilinear')

    if not ignore_data_parallel:
        if kvs['gpus'] > 1:
            net = nn.DataParallel(net).to('cuda')

    return net.to('cuda')
