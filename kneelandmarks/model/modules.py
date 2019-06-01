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


class MultiScaleHGResidual(nn.Module):
    """
    https://arxiv.org/pdf/1808.04803.pdf

    """
    def __init__(self, n_inp, n_out):
        super().__init__()
        self.scale1 = conv_block_3x3(n_inp, n_out // 2, 'relu')
        self.scale2 = conv_block_3x3(n_out // 2, n_out // 4, 'relu')
        self.scale3 = conv_block_3x3(n_out // 4, n_out - n_out // 4 - n_out // 2, None)

        if n_inp != n_out:
            self.skip = conv_block_1x1(n_inp, n_out, None)
        else:
            self.skip = Identity()

    def forward(self, x):
        o1 = self.scale1(x)
        o2 = self.scale2(o1)
        o3 = self.scale3(o2)

        return torch.cat([o1, o2, o3], 1) + self.skip(x)



class SoftArgmax2D(nn.Module):
    def __init__(self, beta=1):
        super(SoftArgmax2D, self).__init__()
        self.beta = beta

    def forward(self, hm):
        hm = hm.mul(self.beta)
        bs, nc, h, w = hm.size()
        hm = hm.squeeze()

        softmax = F.softmax(hm.view(bs, nc, h*w), dim=2).view(bs, nc, h, w)

        weights = torch.ones(bs, nc, h, w).float().to(hm.device)
        w_x = torch.arange(w).float().div(w)
        w_x = w_x.to(hm.device).mul(weights)

        w_y = torch.arange(h).float().div(h)
        w_y = w_y.to(hm.device).mul(weights.transpose(2, 3)).transpose(2, 3)

        approx_x = softmax.mul(w_x).view(bs, nc, h*w).sum(2).unsqueeze(2)
        approx_y = softmax.mul(w_y).view(bs, nc, h*w).sum(2).unsqueeze(2)

        res_xy = torch.cat([approx_x, approx_y], 2)
        return res_xy



