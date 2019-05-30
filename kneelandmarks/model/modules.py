from torch import nn
from deeppipeline.common.modules import conv_block_1x1, conv_block_3x3
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

        softmax = F.softmax(hm.view(bs, nc, h*w), dim=2).view(bs, nc, h, w)

        weights = torch.ones(bs, nc, h, w).float().to(hm.device)
        w_x = torch.arange(1, w+1).float().div(w).add(-1/w)
        w_x = w_x.to(hm.device).mul(weights)

        w_y = torch.arange(1, h+1).float().div(h).add(-1/h)
        w_y = w_y.to(hm.device).mul(weights.transpose(2, 3)).transpose(2, 3)

        approx_x = softmax.mul(w_x).view(bs, nc, h*w).sum(2)
        approx_y = softmax.mul(w_y).view(bs, nc, h*w).sum(2)

        res_xy = torch.cat([approx_x, approx_y], 1)
        return res_xy



