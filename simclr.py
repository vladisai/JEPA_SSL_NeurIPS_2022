from dataclasses import dataclass
from typing import Optional, Any

import torch
from torch.nn import functional as F
import wandb
import numpy as np
from matplotlib import pyplot as plt

from configs import ConfigBase

import models


@dataclass
class SimCLRConfig(ConfigBase):
    projector: str = "512-512-512"
    random_projector: bool = False
    predictor: str = "rnn"
    backbone_mlp: Optional[str] = None
    epochs: int = 100
    base_lr: float = 0.2
    decoding_coeff: float = 0.0
    action_coeff: float = 0.0
    arch: str = "menet5"
    embedding_size: Optional[int] = 64  # only for MeNet5
    action_dim: int = 2
    rnn_layers: int = 1
    rnn_burnin: int = 1
    backbone_width_factor: int = 1

    normalize_z: bool = False
    loss_temp: float = 1.0
    channels: int = 1


@dataclass
class LossInfo:
    total_loss: torch.Tensor
    diagnostics_info: Any


class SimCLR(torch.nn.Module):
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

    def info_nce(self, pred, encs, reduction="mean"):
        if self.args.normalize_z:
            pred = F.normalize(pred, dim=-1)
            encs = F.normalize(encs, dim=-1)

        agreement = pred @ encs.T / self.args.loss_temp

        # Positive keys are the entries on the diagonal
        labels = torch.arange(pred.shape[0], device=pred.device)

        return (1 / 2) * (
            F.cross_entropy(agreement, labels, reduction=reduction)
            + F.cross_entropy(agreement.T, labels, reduction=reduction)
        )

    def info_nce_simclr(self, pred, encs, reduction="mean"):
        if self.args.normalize_z:
            pred = F.normalize(pred, dim=-1)  # batch, nfeats
            encs = F.normalize(encs, dim=-1)  # batch, nfeats

        features = torch.cat([pred, encs], dim=0)

        labels = torch.cat(
            [torch.arange(pred.shape[0]) for i in range(2)],
            dim=0,
        )
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(pred.device)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(pred.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1
        )
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1
        )

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(pred.device)

        logits = logits / self.args.loss_temp
        return F.cross_entropy(logits, labels, reduction=reduction)

    def forward(self, states, actions, step=None):
        """states [T, batch_size, 1, 28, 28]
        actions [T-1, batch_size]
        """
        T = states.shape[0]

        states_flatten = states.reshape(-1, *states.shape[2:])
        states_enc = self.backbone(states_flatten)
        states_proj = self.projector(states_enc)

        states_enc = states_enc.view(*states.shape[:2], -1)
        states_proj = states_proj.view(*states.shape[:2], -1)

        current_enc = states_enc[0]
        pred_encs = self.predictor.predict_sequence(
            enc=current_enc, actions=actions, h=None
        )
        pred_projs = []

        action_loss = torch.tensor(0.0).cuda()
        info_nce_loss = torch.tensor(0.0).cuda()

        for i in range(T - 1):
            pred_enc = pred_encs[i]
            pred_proj = self.projector(pred_enc)
            pred_projs.append(pred_proj)

            info_nce_loss += self.info_nce_simclr(pred_proj, states_proj[i + 1])

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
                if wandb.run is not None:
                    wandb.log({"reconstructions": fig}, commit=False, step=step)
                plt.close(fig)

        info_nce_loss /= T - 1
        action_loss /= T - 1

        loss = (
            info_nce_loss
            + self.args.action_coeff * action_loss
            + self.args.decoding_coeff * reconstruction_loss
        )

        return LossInfo(total_loss=loss, diagnostics_info=None)
