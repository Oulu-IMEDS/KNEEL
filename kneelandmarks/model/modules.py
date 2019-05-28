from torch import nn
from deeppipeline.common.modules import conv_block_1x1, conv_block_3x3
import torch
import torch.nn.functional as F


class Identity(nn.Module):
    def forward(self, x):
        return x


class HGResidual(nn.Module):
    def __init__(self, n_inp, n_out):
        super().__init__()
        self.bottleneck = conv_block_1x1(n_inp, n_out // 2, 'relu')
        self.conv = conv_block_3x3(n_out // 2, n_out // 2, 'relu')
        self.out = conv_block_1x1(n_out // 2, n_out, None)

        if n_inp != n_out:
            self.skip = conv_block_1x1(n_inp, n_out, None)
        else:
            self.skip = Identity()

    def forward(self, x):
        o1 = self.bottleneck(x)
        o2 = self.conv(o1)
        o3 = self.out(o2)

        return o3 + self.skip(x)


class SoftArgmax(nn.Module):
    def __init__(self, beta=1):
        super(SoftArgmax, self).__init__()
        self.beta = beta

    def forward(self, hm):
        hm = hm.mul(self.beta)
        bs, nc, h, w = hm.size()
        hm = hm.squeeze()

        softmax = F.softmax(hm, dim=-1)
        weights = torch.ones(h, w).float().to('cuda')
        w_x = torch.arange(w).float().div(w).to(hm.device).mul(weights)
        w_y = torch.arange(h).float().div(h).to(hm.device).mul(weights)

        approx_x = softmax.mul(w_x).sum(1).sum(1).unsqueeze(1)
        approx_y = softmax.mul(w_y).sum(1).sum(1).unsqueeze(1)

        return torch.cat([approx_x, approx_y], 1)



