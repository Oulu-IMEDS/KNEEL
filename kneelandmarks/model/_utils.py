from deeppipeline.kvs import GlobalKVS
from ._hourglass import HourglassNet
import torch
import os
import glob


def init_model_from_args(args):
    if args.annotations == 'lc':
        net = HourglassNet(3, 1, bw=args.base_width,
                           upmode='bilinear',
                           multiscale_hg_block=args.multiscale_hg,
                           se=args.use_se,
                           se_ratio=args.se_ratio,
                           use_drop=args.use_drop)
    else:
        net = HourglassNet(3, 16, bw=args.base_width,
                           upmode='bilinear',
                           multiscale_hg_block=args.multiscale_hg,
                           se=args.use_se,
                           se_ratio=args.se_ratio,
                           use_drop=args.use_drop)

    return net


def init_model():
    kvs = GlobalKVS()
    net = init_model_from_args(kvs['args'])

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
