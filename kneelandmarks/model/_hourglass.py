import torch
from torch import nn

import torch.nn.functional as F

from kneelandmarks.model.modules import HGResidual, conv_block_1x1, SoftArgmax2D


class Hourglass(nn.Module):
    def __init__(self, n, hg_width, n_inp, n_out, upmode='nearest'):
        super().__init__()

        self.upmode = upmode

        self.lower1 = HGResidual(n_inp, hg_width)
        self.lower2 = HGResidual(hg_width, hg_width)
        self.lower3 = HGResidual(hg_width, hg_width)

        if n > 1:
            self.lower4 = Hourglass(n - 1, hg_width, hg_width, n_out, upmode)
        else:
            self.lower4 = HGResidual(hg_width, n_out)

        self.lower5 = HGResidual(n_out, n_out)

        self.upper1 = HGResidual(n_inp, hg_width)
        self.upper2 = HGResidual(hg_width, hg_width)
        self.upper3 = HGResidual(hg_width, n_out)

    def forward(self, x):
        o_pooled = F.max_pool2d(x, 2)

        o1 = self.lower1(o_pooled)
        o2 = self.lower2(o1)
        o3 = self.lower3(o2)

        o4 = self.lower4(o3)

        o1_u = self.upper1(x)
        o2_u = self.upper2(o1_u)
        o3_u = self.upper3(o2_u)
        return o3_u + F.interpolate(self.lower5(o4), x.size()[-2:], mode=self.upmode, align_corners=True)


class HourglassNet(nn.Module):
    def __init__(self, n_inputs=1, n_outputs=6, bw=64, hg_depth=4,
                 upmode='nearest', refinement=True, use_sagm=False):
        super(HourglassNet, self).__init__()
        self.refinement = refinement

        self.conv1 = nn.Sequential(
            nn.Conv2d(n_inputs, bw, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(bw),
            nn.ReLU(inplace=True)
        )

        self.res1 = HGResidual(bw, bw * 2)
        self.res2 = HGResidual(bw * 2, bw * 2)
        self.res3 = HGResidual(bw * 2, bw * 2)
        self.res4 = HGResidual(bw * 2, bw * 4)

        self.hg1 = Hourglass(hg_depth, bw * 4, bw * 4, bw * 8, upmode)

        self.linear1 = nn.Sequential(conv_block_1x1(bw * 8, bw * 8, 'relu'),
                                     conv_block_1x1(bw * 8, bw * 4, 'relu'))

        self.out1 = nn.Conv2d(bw * 4, n_outputs, kernel_size=1, padding=0)

        if self.refinement:
            # to match the concatenation after the first pooling and the first
            # set of predictions

            self.remap1 = nn.Conv2d(n_outputs, bw * 2 + bw * 4, kernel_size=1, padding=0)

            self.hg2 = Hourglass(hg_depth, bw * 4, bw * 2 + bw * 4, bw * 8, upmode)

            self.compression = nn.Sequential(conv_block_1x1(bw * 8, bw * 8, 'relu'),
                                             conv_block_1x1(bw * 8, bw * 4, 'relu'))

            self.out2 = nn.Conv2d(bw * 4, n_outputs, kernel_size=1, padding=0)

        self.use_sagm = use_sagm
        self.sagm = SoftArgmax2D()

    def forward(self, x):
        # Compressing the input
        o_1 = self.conv1(x)
        o_before_mp = self.res1(o_1)
        o_p = F.max_pool2d(o_before_mp, 2)

        # Performing first hourglass stage
        o = self.res2(o_p)
        o = self.res3(o)
        o = self.res4(o)
        o = self.hg1(o)
        # Producing the 1st set of predictions
        o = self.linear1(o)
        out1 = self.out1(o)
        if self.refinement:
            # Refining the outputs
            o = torch.cat([o, o_p], 1) + self.remap1(out1)
            o = self.hg2(o)
            o = self.compression(o)
            out2 = self.out2(o)

            if self.use_sagm:
                return self.sagm(out1), self.sagm(out2)
            else:
                return out1, out2
        else:
            if self.use_sagm:
                return self.sagm(out1)
            else:
                return out1
