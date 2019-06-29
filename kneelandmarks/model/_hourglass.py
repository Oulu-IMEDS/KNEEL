from torch import nn
from deeppipeline.common.modules import conv_block_1x1
from deeppipeline.keypoints.models.modules import Hourglass, HGResidual, MultiScaleHGResidual, SoftArgmax2D


class HourglassNet(nn.Module):
    def __init__(self, n_inputs=1, n_outputs=6, bw=64, hg_depth=4,
                 upmode='bilinear', multiscale_hg_block=False, se=False, se_ratio=16):

        super(HourglassNet, self).__init__()
        self.multiscale_hg_block = multiscale_hg_block
        self.se = se
        self.se_ratio = se_ratio

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
            self.__make_hg_block(bw * 2, bw * 4)
        )

        self.hourglass = Hourglass(hg_depth, bw * 4, bw * 4, bw * 8, upmode, multiscale_hg_block,
                                   se=se, se_ratio=se_ratio)

        self.mixer = nn.Sequential(conv_block_1x1(bw * 8, bw * 8),
                                   conv_block_1x1(bw * 8, bw * 4))

        self.out_block = nn.Sequential(nn.Conv2d(bw * 4, n_outputs, kernel_size=1, padding=0))
        self.sagm = SoftArgmax2D()

    def __make_hg_block(self, inp, out):
        if self.multiscale_hg_block:
            return MultiScaleHGResidual(inp, out, se=self.se, se_ratio=self.se_ratio)
        else:
            return HGResidual(inp, out, se=self.se, se_ratio=self.se_ratio)

    def forward(self, x):
        o_layer_1 = self.layer1(x)
        o_layer_2 = self.layer2(o_layer_1)

        o_hg = self.hourglass(o_layer_2)
        o_mixer = self.mixer(o_hg)
        out = self.out_block(o_mixer)

        return self.sagm(out)
