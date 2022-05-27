from .common import *


class Net(nn.Module):

    def __init__(self, shrink=6, classes=8, expansion=0.5):
        super(Net, self).__init__()
        c2 = make_divisible(16)
        self.head = nn.ModuleList([
            nn.Identity(),
            Conv(3, c2, s=2)
        ])
        c1 = c2

        c2 = make_divisible(c1 * 2)
        self.neck = nn.ModuleList([
            Conv(1, c2, s=2),
            Conv(c1, c2, s=2)
        ])
        c1 = c2

        self.bone = []
        for i in range(shrink - 2):
            bottle = Bottleneck(c1, c2, e=expansion)
            maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            self.bone += [bottle, maxpool]
        self.bone = nn.Sequential(*self.bone)

        size = (512 // 2 ** shrink) ** 2 * c2
        self.mlp = MLP([size] + [2 ** p for p in range(
            10, int(math.log2(classes)) + 1, -2)] + [classes])

    def forward(self, x):
        mode = x.shape[1] // 2
        print(mode)
        print(x.shape)
        x = self.head[mode](x)
        x = self.neck[mode](x)
        x = self.bone(x)
        x = self.mlp(x, reshape=True)

        return x