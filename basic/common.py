from torch import nn
from .utils import *


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = DWConv(c_, c2, 3, 1)
        # self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class MLP(nn.Module):
    ''' MLP多层感知机:
        features: 特征量 list/tuple
        dropout: Dropout 层的概率'''

    def __init__(self, features, activation=nn.LeakyReLU):
        super(MLP, self).__init__()
        self.id = f'MLP(features={features})'
        num = len(features) - 1
        self.in_features = features[0]
        layers = []
        for idx in range(num):
            in_features = features[idx]
            out_features = features[idx + 1]
            layers.append(nn.Linear(in_features=in_features, out_features=out_features))
            layers.append(activation(inplace=False))
        self.unit = nn.Sequential(*layers)

    def forward(self, data, reshape=False):
        if reshape:
            batch_size = data.shape[0]
            data = data.contiguous().view(batch_size, -1)
        out = self.unit(data)
        return out

    def __str__(self):
        return self.id

    __repr__ = __str__