from typing import List, Optional

import torch
from torch import nn
import numpy as np

from torch.nn import functional as F
from resnet import resnet18, resnet18ID
import resnet

ResNet18 = resnet18
ResNet18ID = resnet18ID


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128, 128)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        return out


class MeNet5(nn.Module):
    def __init__(
        self, output_dim: int = 64, input_channels: int = 1, width_factor: int = 1
    ):
        super().__init__()
        self.width_factor = width_factor
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                input_channels, 16 * width_factor, kernel_size=5, stride=2, padding=2
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16 * width_factor),
            nn.Conv2d(
                16 * width_factor, 32 * width_factor, kernel_size=5, stride=2, padding=2
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32 * width_factor),
            nn.Conv2d(
                32 * width_factor, 32 * width_factor, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32 * width_factor),
            nn.AvgPool2d(2, stride=2),
        )
        self.fc = nn.Linear(9 * 32 * width_factor, output_dim)

    def forward(self, x):
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


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
        padding=1,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class MeNet5Decoder(nn.Module):
    def __init__(self, embedding_size, output_channels: int = 1, width_factor: int = 1):
        super().__init__()
        self.width_factor = width_factor
        self.layers = nn.Sequential(
            ResizeConv2d(
                32 * width_factor,
                32 * width_factor,
                kernel_size=3,
                scale_factor=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32 * width_factor),
            ResizeConv2d(
                32 * width_factor,
                16 * width_factor,
                kernel_size=5,
                scale_factor=3,
                padding=2,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16 * width_factor),
            ResizeConv2d(
                16 * width_factor,
                output_channels,
                kernel_size=5,
                scale_factor=3,
                padding=2,
            ),
        )
        self.fc = nn.Linear(2 * embedding_size, 32 * 3 * 3 * self.width_factor)

    def forward(self, x: torch.Tensor, belief: torch.Tensor):
        x = torch.cat([x, belief], dim=1)
        x = self.fc(x)
        x = F.relu(x)
        x = x.view(x.shape[0], -1, 3, 3)
        x = F.interpolate(x, scale_factor=2, mode="nearest")  # 6 by 6, undo avg pool
        x = self.layers(x)
        x = F.interpolate(x, size=(28, 28), mode="bilinear")  # 27 by 27 to 28 by 28
        return x


class Canonical(nn.Module):
    def __init__(self, output_dim: int = 64):
        super().__init__()
        res = int(np.sqrt(output_dim / 64))
        assert (
            res * res * 64 == output_dim
        ), "canonical backbone resolution error: cant fit desired output_dim"

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((res, res)),
        )

    def forward(self, x):
        return self.backbone(x).flatten(1)


class MLPNet(nn.Module):
    def __init__(self, output_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, output_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        out = x.flatten(1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class PassThrough(nn.Module):
    def forward(self, x):
        return x.view(*x.shape[:-3], -1)


class PixelPredictorConv(torch.nn.Module):
    def __init__(self, action_dim=2):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
        )
        self.proj = nn.Linear(action_dim, 64)

    def forward(self, x, a):
        original_shape = x.shape
        if x.shape[-1] == 784:
            x = x.view(*x.shape[:-1], 1, 28, 28)
        a_proj = self.proj(a)
        e = self.enc(x)
        e = e + a_proj.view(*a_proj.shape, 1, 1)
        d = self.dec(e)
        return d.view(*original_shape)

    def predict_sequence(self, h: torch.Tensor, actions: torch.Tensor):
        outputs = []
        for i in range(len(actions)):
            h = self(h, actions[i])
            outputs.append(h)
        return outputs


class VAEDecoder(torch.nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.k1, self.k2, self.k3, self.k4 = (
            (3, 3),
            (3, 3),
            (3, 3),
            (3, 3),
        )  # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (
            (2, 2),
            (2, 2),
            (2, 2),
            (2, 2),
        )  # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        )  # 2d padding

        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=self.k4,
                stride=self.s4,
                padding=self.pd4,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16, momentum=0.01),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=8,
                kernel_size=self.k3,
                stride=self.s3,
                padding=self.pd3,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8, momentum=0.01),
        )

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=8,
                out_channels=8,
                kernel_size=self.k2,
                stride=self.s2,
                padding=self.pd2,
            ),
            nn.ReLU(),  # y = (y1, y2, y3) \in [0 ,1]^3
            nn.BatchNorm2d(8, momentum=0.01),
            nn.Conv2d(8, out_channels=1, kernel_size=3, padding=1),
        )

    #         self.fc1 = nn.Linear(embedding_size, embedding_size)

    def forward(self, z):
        x = z.view(-1, 32, 4, 4)
        x = self.convTrans6(x)
        x = self.convTrans7(x)
        x = self.convTrans8(x)
        x = F.interpolate(x, size=(28, 28), mode="bilinear")
        return x


class PixelEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        # self.dec = VAEDecoder(512)
        # self.proj = nn.Linear(action_dim, 512)

    def forward(self, x):
        # a_proj = self.proj(a)
        return self.enc(x).flatten(1)


def build_projector(arch: str, embedding: int):
    if arch == "id":
        return nn.Identity(), embedding
    else:
        f = [embedding] + list(map(int, arch.split("-")))
        return build_mlp(f), f[-1]


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1], bias=False))
    return nn.Sequential(*layers)


class Projector(torch.nn.Module):
    def __init__(self, arch: str, embedding: int, random: bool = False):
        super().__init__()

        self.arch = arch
        self.embedding = embedding
        self.random = random

        self.model, self.output_dim = build_projector(arch, embedding)

        if self.random:
            for param in self.parameters():
                param.requires_grad = False

    def maybe_reinit(self):
        if self.random and self.arch != "id":
            for param in self.parameters():
                torch.nn.init.xavier_uniform_(param)
                print("initialized")

    def forward(self, x: torch.Tensor):
        return self.model(x)


def build_backbone(
    arch: str,
    embedding_size: int,
    backbone_mlp: str,
    backbone_width_factor: int,
    channels: int = 1,
):
    backbone, embedding = None, None

    if arch == "resnet18":
        backbone, embedding = resnet.__dict__[arch](
            zero_init_residual=True,
            num_channels=channels,
        )
    elif arch == "resnet18ID":
        backbone, embedding = resnet.__dict__[arch](
            zero_init_residual=False, num_channels=channels
        )
    elif arch == "lenet5":
        backbone = LeNet5()
        embedding = 128
    elif arch == "id":
        backbone = PassThrough()
        embedding = 28 * 28
    elif arch == "menet5":
        backbone = MeNet5(
            embedding_size, width_factor=backbone_width_factor, input_channels=channels
        )
        embedding = embedding_size
    elif arch == "mlp":
        backbone = MLPNet(embedding_size)
        embedding = embedding_size
    elif arch == "canonical":
        backbone = Canonical(embedding_size)
        embedding = embedding_size
    elif arch == "pixel":
        backbone = PixelEncoder()
        embedding = 512
    else:
        raise NotImplementedError(f"backbone arch {arch} is unknown")

    if backbone_mlp is not None:
        backbone_mlp = Projector(backbone_mlp, embedding)
        embedding = backbone_mlp.output_dim
        backbone = nn.Sequential(backbone, backbone_mlp)

    return backbone, embedding


class Predictor(torch.nn.Module):
    def __init__(self, arch: str, num_features: int, action_dim: int = 2):
        super().__init__()
        layers = []
        f = (
            [num_features + action_dim]
            + (list(map(int, arch.split("-"))) if arch != "" else [])
            + [num_features]
        )
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1]))
            layers.append(nn.BatchNorm1d(f[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(f[-2], f[-1], bias=False))
        self.model = nn.Sequential(*layers)

    def forward(self, x, action):
        t = torch.concat([x, action], dim=1)
        return self.model(t)

    def predict_sequence(self, h: torch.Tensor, actions: torch.Tensor):
        outputs = []
        for i in range(len(actions)):
            h = h + self(h, actions[i])
            outputs.append(h)
        return torch.stack(outputs, dim=0)


class IDPredictor(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def burn_in(self, *args, **kwargs):
        return None

    def predict_sequence(
        self, enc: torch.Tensor, h: torch.Tensor, actions: torch.Tensor
    ):
        return enc.unsqueeze(0).repeat(actions.shape[0], 1, 1)


class RNNPredictor(torch.nn.Module):
    def __init__(
        self, hidden_size: int = 512, num_layers: int = 1, action_dim: int = 2
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = torch.nn.GRU(
            input_size=action_dim,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
        )

    def burn_in(self, *args, **kwargs):
        return None

    def predict_sequence(
        self, enc: torch.Tensor, h: torch.Tensor, actions: torch.Tensor
    ):
        # in this version, encoding is directly used as h, and the passed h is ignored.
        # since h is obtained from burn_in, it's actually None.
        h = enc
        return self.rnn(actions, h.unsqueeze(0).repeat(self.num_layers, 1, 1))[0]


class RNNPredictorBurnin(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int = 512,
        output_size: int = 512,
        num_layers: int = 1,
        action_dim: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.action_dim = action_dim

        self.rnn = torch.nn.GRU(
            input_size=action_dim + output_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
        )
        self.output_projector = nn.Linear(hidden_size, output_size)

    def burn_in(
        self,
        encs: torch.Tensor,
        actions: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ):
        """Runs a few iterations of RNN with the provided GT encodings to obtain h0"""
        if h is None:
            h = torch.zeros(self.num_layers, actions.shape[1], self.hidden_size).to(
                actions.device
            )

        for i in range(encs.shape[0]):
            rnn_input = torch.cat([encs[i], actions[i]], dim=1).unsqueeze(0)
            _, h = self.rnn(rnn_input, h)
        return h

    def predict_sequence(
        self, enc: torch.Tensor, actions: torch.Tensor, h: Optional[torch.Tensor] = None
    ):
        """Predicts the sequence given gt encoding for the current time step"""
        outputs = []
        if h is None:
            h = torch.zeros(self.num_layers, actions.shape[1], self.hidden_size).to(
                actions.device
            )
        for i in range(actions.shape[0]):
            rnn_input = torch.cat([enc, actions[i]], dim=1).unsqueeze(0)
            _, h = self.rnn(rnn_input, h)
            outputs.append(self.output_projector(h[-1]))
            enc = outputs[-1]  # autoregressive GRU
        outputs = torch.stack(outputs)
        return outputs


class ActionPredictor(torch.nn.Module):
    def __init__(self, embedding: int, action_dim: int = 3):
        super().__init__()
        self.model = nn.Linear(embedding * 2, action_dim)

    def forward(self, s, sn):
        t = torch.concat([s, sn], dim=1)
        return self.model(t)


def build_predictor(arch: str, embedding: int, action_dim: int, rnn_layers: int):
    if arch == "conv":
        predictor = PixelPredictorConv(action_dim=action_dim)
    elif arch == "rnn":
        predictor = RNNPredictor(
            hidden_size=embedding,
            num_layers=rnn_layers,
            action_dim=action_dim,
        )
    elif arch == "rnn_burnin":
        predictor = RNNPredictorBurnin(
            hidden_size=embedding,
            output_size=embedding,
            num_layers=rnn_layers,
            action_dim=action_dim,
        )
    elif arch == "id":
        predictor = IDPredictor()
    else:
        predictor = Predictor(arch, embedding, action_dim=action_dim)

    return predictor


class Prober(torch.nn.Module):
    def __init__(self, embedding: int, arch: str, output_shape: List[int]):
        super().__init__()

        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        # f = [embedding, embedding, embedding]
        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.BatchNorm1d(f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1], bias=False))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        # return self.prober(e)
        return self.prober(e).view(e.shape[0], *self.output_shape)


if __name__ == "__main__":
    model = LeNet5()
    test_in = torch.rand(1, 1, 28, 28)
    print(model(test_in).shape)
