# borrowed from https://github.com/julianstastny/VAE-ResNet18-PyTorch/blob/master/model.py

import torch
from torch import nn
from torch.nn import functional as F


class ResizeConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        scale_factor,
        mode="nearest",
        groups=1,
        bias=False,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=1,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class BasicBlockDec(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()

        planes = out_planes

        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                ResizeConv2d(
                    in_planes,
                    out_planes,
                    kernel_size=3,
                    scale_factor=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

        self.conv2 = nn.Conv2d(
            in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        hidden_dim = in_planes * 2

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_planes, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(hidden_dim),
        )

        if stride == 1:
            self.layer2 = nn.Sequential(
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    groups=hidden_dim,
                ),
                nn.BatchNorm2d(hidden_dim),
            )
        else:
            self.layer2 = nn.Sequential(
                ResizeConv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    scale_factor=stride,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
            )

        self.layer3 = nn.Sequential(
            nn.Conv2d(hidden_dim, out_planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_planes),
        )

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        out = self.layer3(out)
        return out + shortcut


class ResNet18Dec(nn.Module):
    def __init__(self, num_Blocks=[3, 3, 3], z_dim=10, nc=3):
        super().__init__()
        #         self.in_planes = 64 * 7 * 7

        #         self.linear = nn.Linear(z_dim, 512)
        self.nc = nc

        #         self.layer4 = self._make_layer(BasicBlockDec, 64, 64, num_Blocks[3], stride=1)
        self.layer3 = self._make_layer(BasicBlockDec, 512, 64, num_Blocks[2], stride=7)
        self.layer2 = self._make_layer(BasicBlockDec, 64, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, 32, num_Blocks[0], stride=2)
        self.conv1 = ResizeConv2d(32, nc, kernel_size=3, scale_factor=1, bias=True)

    def _make_layer(self, BasicBlockDec, in_planes, out_planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        planes = list(reversed([out_planes] + [in_planes] * (num_Blocks)))
        layers = []
        for i, stride in enumerate(reversed(strides)):
            layers += [BasicBlockDec(planes[i], planes[i + 1], stride)]
        return nn.Sequential(*layers)

    def forward(self, z):
        x = z.view(z.size(0), 512, 1, 1)
        #         x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), self.nc, 28, 28)
        return x
