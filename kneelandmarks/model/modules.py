from torch import nn

def conv_block_3x3(inp, out, activation):
    """
    3x3 ConvNet building block

    Aleksei Tiulpin, Unversity of Oulu, 2017 (c).

    """
    if activation == 'relu':
        return nn.Sequential(
            nn.Conv2d(inp, out, kernel_size=3, padding=1),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(inp, out, kernel_size=1, padding=0),
            nn.BatchNorm2d(out),
        )


def conv_block_1x1(inp, out, activation):
    """
    3x3 ConvNet building block

    Aleksei Tiulpin, Unversity of Oulu, 2017 (c).

    """
    if activation == 'relu':
        return nn.Sequential(
            nn.Conv2d(inp, out, kernel_size=1, padding=0),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(inp, out, kernel_size=1, padding=0),
            nn.BatchNorm2d(out),
        )


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
