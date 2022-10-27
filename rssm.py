from typing import Optional
from dataclasses import dataclass
import wandb
import torch
import numpy as np
from matplotlib import pyplot as plt

from typing import Any

from torch.distributions.kl import kl_divergence
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from configs import ConfigBase

import models


@dataclass
class RSSMConfig(ConfigBase):
    backbone_width_factor: int = 1
    action_coeff: float = 0.0
    arch: str = "resnet18"
    epochs: int = 100
    learning_rate: float = 0.001
    embedding_size: Optional[int] = 64  # only for MeNet5
    action_dim: int = 2
    min_var: float = 0.1  # RSSM only for postive variance in tf Normal dist
    rssm_adam_epsilon: float = 1e-8
    log_recontructed_image: bool = True
    channels: int = 1
    rnn_burnin: int = 1


class RSSMPredictor(torch.nn.Module):
    def __init__(self, hidden_size: int, embedding_size: int, action_dim: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.rnn_input_net = nn.Sequential(
            nn.Linear(self.embedding_size + action_dim, self.embedding_size),
            nn.ReLU(),
        )
        self.prior_mu_net = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU(),
        )
        self.prior_var_net = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU(),
        )
        self.rnn = torch.nn.GRUCell(self.embedding_size, self.hidden_size)

    def forward(self, sampled_prior_state, action, belief):
        sampled_state_action = torch.cat((sampled_prior_state, action), dim=1)
        rnn_input = self.rnn_input_net(sampled_state_action)
        belief_new = self.rnn(rnn_input, belief)
        prior_mu = self.prior_mu_net(belief_new)
        prior_var = self.prior_var_net(belief_new)
        return belief_new, prior_mu, prior_var

    def burn_in(self, *args, **kwargs):
        return None

    def predict_sequence(
        self, enc: torch.Tensor, actions: torch.Tensor, h: torch.Tensor
    ):
        initial_belief = enc
        prior_mu0 = self.prior_mu_net(initial_belief)
        prior_var0 = self.prior_var_net(initial_belief)
        prior_var0 = F.softplus(prior_var0)
        z0 = Normal(prior_mu0, (prior_var0))
        sampled_prior_state = z0.sample()
        T = actions.shape[0] + 1
        prior_mus = []
        prior_vars = []
        sampled_prior_states = []
        beliefs = []
        rnn_belief = initial_belief
        for i in range(T - 1):
            rnn_belief, prior_mu, prior_var = self(
                sampled_prior_state, actions[i], rnn_belief
            )
            prior_var = F.softplus(prior_var)  # +self.min_var
            z = Normal(prior_mu, (prior_var))
            sampled_prior_state = z.sample()
            prior_vars.append(prior_var)
            prior_mus.append(prior_mu)
            sampled_prior_states.append(sampled_prior_state)
            beliefs.append(rnn_belief)
        prior_vars = torch.stack(prior_vars, dim=0)
        prior_mus = torch.stack(prior_mus, dim=0)
        beliefs = torch.stack(beliefs, dim=0)
        sampled_prior_states = torch.stack(sampled_prior_states, dim=0)
        return beliefs


class Posterior(torch.nn.Module):
    def __init__(self, hidden_size: int, embedding_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.posterior_mu_net = nn.Sequential(
            nn.Linear(self.hidden_size + self.embedding_size, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU(),
        )
        self.posterior_var_net = nn.Sequential(
            nn.Linear(self.hidden_size + self.embedding_size, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU(),
        )

    def forward(self, encoded_state, belief):
        state_posterior = torch.cat((encoded_state, belief), dim=1)
        posterior_mu = self.posterior_mu_net(state_posterior)
        posterior_var = self.posterior_var_net(state_posterior)
        return posterior_mu, posterior_var


@dataclass
class LossInfo:
    total_loss: torch.Tensor
    kl_loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    diagnostics_info: Optional[Any] = None


class VAEDecoder(torch.nn.Module):
    def __init__(self, embedding_size=512, output_channels=1):
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
            nn.BatchNorm2d(16, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=8,
                kernel_size=self.k3,
                stride=self.s3,
                padding=self.pd3,
            ),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=8,
                out_channels=8,
                kernel_size=self.k2,
                stride=self.s2,
                padding=self.pd2,
            ),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.Sigmoid(),  # y = (y1, y2, y3) \in [0 ,1]^3
            nn.Conv2d(8, out_channels=output_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.fc1 = nn.Linear(2 * embedding_size, embedding_size)

    def forward(self, z, belief):
        z = self.fc1(torch.cat([z, belief], dim=1))
        x = z.view(-1, 32, 4, 4)
        x = self.convTrans6(x)
        x = self.convTrans7(x)
        x = self.convTrans8(x) * 20
        x = F.interpolate(x, size=(28, 28), mode="bilinear")
        return x


class RSSMPredMultistep(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.min_var = args.min_var

        self.backbone, self.embedding = models.build_backbone(
            args.arch,
            args.embedding_size,
            backbone_mlp=None,
            backbone_width_factor=args.backbone_width_factor,
            channels=self.args.channels,
        )

        print(f"backbone is {args.arch}")
        if args.arch == "menet5":
            self.decoder = models.MeNet5Decoder(
                embedding_size=self.embedding,
                output_channels=self.args.channels,
                width_factor=self.args.backbone_width_factor,
            )
        else:
            self.decoder = VAEDecoder(output_channels=self.args.channels)
        # self.action_predictor = ActionPredictor(self.embedding, self.args.action_dim)
        self.predictor = RSSMPredictor(
            hidden_size=self.embedding, embedding_size=self.embedding
        )
        self.posterior = Posterior(
            hidden_size=self.embedding, embedding_size=self.embedding
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, states, actions, step=None):
        """states [T, batch_size, 1, 28, 28]
        actions [T-1, batch_size]
        """
        T = states.shape[0]
        states_flatten = states.reshape(-1, *states.shape[2:])
        states_enc = self.backbone(states_flatten)
        states_encs = states_enc.view(*states.shape[:2], -1)
        multi_layer_rnn_beliefs = states_encs[0]
        prior_mu0 = self.predictor.prior_mu_net(states_encs[0])
        prior_var0 = self.predictor.prior_var_net(states_encs[0])
        prior_var0 = F.softplus(prior_var0)
        z0 = Normal(prior_mu0, (prior_var0))
        sampled_prior_state = z0.sample()
        loss = 0.0
        kl_loss = torch.tensor(0.0).cuda()
        reconstruction0 = self.decoder(sampled_prior_state, multi_layer_rnn_beliefs)
        loss = F.mse_loss(reconstruction0, states[0], reduction="mean")
        for i in range(T - 1):
            multi_layer_rnn_beliefs, prior_mu, prior_var = self.predictor(
                sampled_prior_state, actions[i], multi_layer_rnn_beliefs
            )
            prior_var = F.softplus(prior_var) + self.min_var
            z = Normal(prior_mu, (prior_var))
            sampled_prior_state = z.sample()
            posterior_mu, posterior_var = self.posterior(
                states_encs[i + 1], multi_layer_rnn_beliefs
            )
            posterior_var = F.softplus(posterior_var) + self.min_var
            post_z = Normal(posterior_mu, posterior_var)
            kl_loss = torch.mean(kl_divergence(z, post_z), dim=(0, 1))
            kl_loss = torch.max(kl_loss, kl_loss.new_full(kl_loss.size(), 3))
            reconstruction = self.decoder(sampled_prior_state, multi_layer_rnn_beliefs)
            sampled_prior_state = post_z.sample()
            loss += (kl_loss) + F.mse_loss(
                reconstruction, states[i + 1], reduction="mean"
            )

            """
            if self.args.action_coeff:
                pred_action = self.action_predictor(sampled_prior_state, post_z.sample)
                action_loss += F.mse_loss(actions[i], pred_action)
            """

        loss /= T - 1
        # action_loss /= T - 1

        t_loss = loss  # + # action_loss
        loss_info = LossInfo(
            total_loss=t_loss, kl_loss=kl_loss, reconstruction_loss=loss - kl_loss
        )
        if self.args.log_recontructed_image and step % 1000 == 0:
            # f = plt.imshow(s_f[0].data.cpu().numpy())
            fig, ax = plt.subplots(5, 2, dpi=200)
            r_f = reconstruction.clone().detach().permute(0, 2, 3, 1)
            s_f = states[i + 1].clone().detach().permute(0, 2, 3, 1)
            # normalize for plotting
            r_f -= r_f.min()
            r_f /= r_f.max()
            s_f -= s_f.min()
            s_f /= s_f.max()
            for i in range(5):
                k = np.random.randint(0, r_f.shape[0])
                ax[i][0].imshow(r_f[k].detach().cpu().numpy())
                ax[i][0].set_axis_off()
                ax[i][1].imshow(s_f[k].detach().cpu().numpy())
                ax[i][1].set_axis_off()
            log_ = {}
            log_["r_f " + str(step)] = fig
            if wandb.run is not None:
                wandb.log(log_, step=step)
            plt.close(fig)
        return loss_info
