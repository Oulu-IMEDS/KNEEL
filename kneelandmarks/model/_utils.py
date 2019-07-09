from deeppipeline.kvs import GlobalKVS
from ._hourglass import HourglassNet
import torch
import os
import glob


def init_model():
    kvs = GlobalKVS()
    if kvs['args'].annotations == 'lc':
        net = HourglassNet(3, 1, bw=kvs['args'].base_width,
                           upmode='bilinear',
                           multiscale_hg_block=kvs['args'].multiscale_hg,
                           se=kvs['args'].use_se,
                           se_ratio=kvs['args'].se_ratio,
                           use_drop=kvs['args'].use_drop)
    else:
        net = HourglassNet(3, 16, bw=kvs['args'].base_width,
                           upmode='bilinear',
                           multiscale_hg_block=kvs['args'].multiscale_hg,
                           se=kvs['args'].use_se,
                           se_ratio=kvs['args'].se_ratio,
                           use_drop=kvs['args'].use_drop)

        if kvs['args'].init_model_from != '':
            cur_fold = kvs['cur_fold']
            pattern_snp = os.path.join(kvs['args'].init_model_from,
                                       f'fold_{cur_fold}_*.pth')
            state_dict = torch.load(glob.glob(pattern_snp)[0])['model']
            pretrained_dict = {k: v for k, v in state_dict.items() if 'out_block' not in k}
            net_state_dict = net.state_dict()
            net_state_dict.update(pretrained_dict)
            net.load_state_dict(net_state_dict)

    return net.to('cuda')
