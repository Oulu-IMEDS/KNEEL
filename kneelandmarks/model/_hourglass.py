import torch
from torch import nn
from torch.nn import functional as F

from deeppipeline.common.modules import conv_block_1x1
from deeppipeline.keypoints.models.modules import Hourglass, HGResidual, MultiScaleHGResidual, SoftArgmax2D


class HourglassNet(nn.Module):
    def __init__(self, n_inputs=1, n_outputs=6, bw=64, hg_depth=4,
                 upmode='bilinear', multiscale_hg_block=False):

        super(HourglassNet, self).__init__()
        self.multiscale_hg_block = multiscale_hg_block

        self.layer1 = nn.Sequential(
            nn.Conv2d(n_inputs, bw, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(bw),
            nn.ReLU(inplace=True),
            self.__make_hg_block(bw, bw * 2),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            self.__make_hg_block(bw * 2, bw * 2),
            self.__make_hg_block(bw * 2, bw * 2),
        )

        self.res4 = self.__make_hg_block(bw * 2, bw * 4)

        self.hg1 = Hourglass(hg_depth, bw * 4, bw * 4, bw * 8, upmode, multiscale_hg_block)

        self.linear1 = nn.Sequential(conv_block_1x1(bw * 8, bw * 8, 'relu'),
                                     conv_block_1x1(bw * 8, bw * 4, 'relu'))

        self.out1 = nn.Conv2d(bw * 4, n_outputs, kernel_size=1, padding=0)

        self.sagm = SoftArgmax2D()

    def __make_hg_block(self, inp, out):
        if self.multiscale_hg_block:
            return MultiScaleHGResidual(inp, out)
        else:
            return HGResidual(inp, out)

    def forward(self, x):
        # Compressing the input
        o_1 = self.layer1(x)
        o_2 = self.layer2(o_1)

        o = self.res4(o_2)
        o = self.hg1(o)
        o = self.linear1(o)
        out = self.out1(o)

        return self.sagm(out)
