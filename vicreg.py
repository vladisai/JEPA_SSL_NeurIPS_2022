from dataclasses import dataclass
from typing import Optional, NamedTuple

import torch
from torch.nn import functional as F
import wandb
import numpy as np
from matplotlib import pyplot as plt

from configs import ConfigBase
import models
import diagnostics


@dataclass
class VICRegConfig(ConfigBase):
    projector: str = "512-512-512"
    random_projector: bool = False
    predictor: str = "515-512-512"
    backbone_mlp: Optional[str] = None
    sim_coeff: float = 25.0
    std_coeff: float = 25.0
    cov_coeff: float = 1.0
    encoding_std_coeff: float = 0.0
    encoding_cov_coeff: float = 0.0
    decoding_coeff: float = 0.0
    action_coeff: float = 0.0
    prediction_diff_coeff: float = 0.0
    arch: str = "resnet18"
    epochs: int = 100
    base_lr: float = 0.2
    repr_loss_after_projector: bool = True
    # Don't apply variance-covariance loss on the first step (maybe necessary because
    # first step doesn't have any repr loss component.
    skip_first_step_vc: bool = False
    embedding_size: Optional[int] = 64  # only for MeNet5
    action_dim: int = 2
    rnn_layers: int = 1
    backbone_width_factor: int = 1
    rnn_burnin: int = 1
    channels: int = 1


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


@dataclass
class LossInfo:
    total_loss: torch.Tensor
    diagnostics_info: diagnostics.VICRegDiagnostics


class CovStdLoss(NamedTuple):
    cov_loss: torch.Tensor
    std_loss: torch.Tensor


def get_cov_std_loss(x: torch.Tensor) -> CovStdLoss:
    batch_size = x.shape[0]
    num_features = x.shape[-1]

    x = x - x.mean(dim=0)

    std = torch.sqrt(x.var(dim=0) + 0.0001)
    std_loss = torch.mean(F.relu(1 - std))

    cov = (x.T @ x) / (batch_size - 1)
    cov_loss = off_diagonal(cov).pow_(2).sum().div(num_features)
    return CovStdLoss(cov_loss, std_loss)


class VICRegPredMultistep(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.backbone, self.embedding = models.build_backbone(
            args.arch,
            args.embedding_size,
            args.backbone_mlp,
            args.backbone_width_factor,
            channels=args.channels,
        )

        print(f"backbone is {args.arch}")

        self.projector = models.Projector(
            args.projector, self.embedding, random=args.random_projector
        )
        self.num_features = self.projector.output_dim

        self.predictor = models.build_predictor(
            args.predictor, self.embedding, self.args.action_dim, self.args.rnn_layers
        )

        if args.decoding_coeff > 0:
            self.decoder = models.VAEDecoder()

        self.action_predictor = models.ActionPredictor(
            self.embedding, self.args.action_dim
        )

    def forward(self, states, actions, step=None):
        """states [T, batch_size, 1, 28, 28]
        actions [T-1, batch_size]
        """

        states_flatten = states.reshape(-1, *states.shape[2:])
        all_states_enc = self.backbone(states_flatten)
        all_states_enc = all_states_enc.view(*states.shape[:2], -1)

        burnin_states_enc = all_states_enc[: self.args.rnn_burnin - 1]
        states_enc = all_states_enc[self.args.rnn_burnin - 1 :]

        T = states_enc.shape[0]

        burnin_actions = actions[: self.args.rnn_burnin - 1]
        actions = actions[self.args.rnn_burnin - 1 :]

        states_enc_flatten = states_enc.view(-1, *states_enc.shape[2:])
        states_proj = self.projector(states_enc_flatten).view(*states_enc.shape[:2], -1)
        states_proj = states_proj.view(*states_enc.shape[:2], -1)

        current_enc = states_enc[0]
        h0 = self.predictor.burn_in(burnin_states_enc, burnin_actions)

        pred_encs = self.predictor.predict_sequence(
            enc=current_enc, actions=actions, h=h0
        )
        pred_projs = []

        # e_t -> predictor -> e_t+1 -> predictor -> e_t+2 ...
        # ^                     v(with proj)
        # encoder              loss
        # ^                     ^
        # x_t                proj(enc(x_t+1))

        repr_loss = 0.0
        action_loss = torch.tensor(0.0).cuda()

        for i in range(T - 1):
            pred_enc = pred_encs[i]
            pred_proj = self.projector(pred_enc)
            pred_projs.append(pred_proj)

            if self.args.repr_loss_after_projector:
                repr_loss += F.mse_loss(pred_proj, states_proj[i + 1])
            else:
                repr_loss += F.mse_loss(pred_enc, states_enc[i + 1])

            if self.args.action_coeff:
                pred_action = self.action_predictor(current_enc, states_enc[i + 1])
                action_loss += F.mse_loss(actions[i], pred_action)

            current_enc = pred_enc

        reconstruction_loss = torch.tensor(0.0).cuda()

        if self.args.decoding_coeff > 0:
            flattened_states_enc = states_enc.view(-1, self.embedding)
            flattened_states = states.reshape(-1, *states.shape[2:])
            decodings = self.decoder(flattened_states_enc)
            reconstruction_loss = F.mse_loss(decodings, flattened_states)
            # if step is not None and step % 100 == 0:
            if step is not None and step % 100 == 0:
                fig, ax = plt.subplots(5, 2, dpi=200)
                for i in range(5):
                    k = np.random.randint(0, flattened_states_enc.shape[0])
                    ax[i][0].imshow(flattened_states[k, 0].detach().cpu())
                    ax[i][0].set_axis_off()
                    ax[i][1].imshow(decodings[k, 0].detach().cpu())
                    ax[i][1].set_axis_off()
                wandb.log({"reconstructions": fig}, commit=False, step=step)
                plt.close(fig)

        repr_loss /= T - 1
        action_loss /= T - 1

        total_proj_std_loss = torch.tensor(0.0).cuda()
        total_proj_cov_loss = torch.tensor(0.0).cuda()
        total_enc_std_loss = torch.tensor(0.0).cuda()
        total_enc_cov_loss = torch.tensor(0.0).cuda()

        if not self.args.skip_first_step_vc:
            enc_projs = zip(states_enc, states_proj)
            denominator = T
        else:
            enc_projs = zip(states_enc[1:], states_proj[1:])
            denominator = T - 1

        for enc, proj in enc_projs:
            proj_losses = get_cov_std_loss(proj)
            total_proj_std_loss += proj_losses.std_loss
            total_proj_cov_loss += proj_losses.cov_loss

            if self.args.encoding_std_coeff > 0.0 or self.args.encoding_cov_coeff > 0.0:
                enc_losses = get_cov_std_loss(enc)
                total_enc_std_loss += enc_losses.std_loss
                total_enc_cov_loss += enc_losses.cov_loss

        total_proj_cov_loss /= denominator
        total_proj_std_loss /= denominator
        total_enc_cov_loss /= denominator
        total_enc_std_loss /= denominator

        loss = (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * total_proj_std_loss
            + self.args.cov_coeff * total_proj_cov_loss
            + self.args.encoding_std_coeff * total_enc_std_loss
            + self.args.encoding_cov_coeff * total_enc_cov_loss
            + self.args.action_coeff * action_loss
            + self.args.decoding_coeff * reconstruction_loss
        )

        loss_info = LossInfo(
            total_loss=loss,
            diagnostics_info=diagnostics.VICRegDiagnostics(
                std_loss=total_proj_std_loss,
                cov_loss=total_proj_cov_loss,
                repr_loss=repr_loss,
                action_loss=action_loss,
                reconstruction_loss=reconstruction_loss,
                enc_std_loss=total_enc_std_loss,
                enc_cov_loss=total_enc_cov_loss,
                enc=states_enc,
                proj=states_proj,
                pred_enc=pred_encs,
                pred_proj=pred_projs,
            ),
        )
        return loss_info
