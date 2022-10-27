import data
from typing import Optional
import os
from dataclasses import dataclass
import dataclasses
from pathlib import Path
import random
from enum import Enum, auto

import torch
import numpy as np
import wandb
from omegaconf import OmegaConf, MISSING
from matplotlib import pyplot as plt
import matplotlib
import enum

from configs import ConfigBase
from rssm import RSSMConfig, RSSMPredMultistep
from simclr import SimCLRConfig, SimCLR
from vicreg import VICRegConfig, VICRegPredMultistep
import probing
import models

# os.environ['WANDB_DISABLED'] = "true"


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class ModelType(enum.Enum):
    VICReg = enum.auto()
    RSSM = enum.auto()
    SimCLR = enum.auto()


class DatasetType(Enum):
    Single = auto()
    Multiple = auto()


@dataclass
class TrainConfig(ConfigBase):
    model_type: ModelType = MISSING
    n_steps: int = 17
    val_n_steps: int = 17
    dataset_size: int = 10000
    dataset_noise: float = 0.0
    val_dataset_size: int = 10000
    wandb: bool = True
    run_name: Optional[str] = None
    run_group: Optional[str] = None
    output_path: Optional[str] = None
    safe_every_n_epochs: int = 10
    eval_mpcs: int = 20
    quick_debug: bool = False
    seed: int = 42
    load_checkpoint_path: Optional[str] = None
    eval_only: bool = False
    probe_mpc: bool = False
    dataset_batch_size: int = 128
    dataset_static_noise: float = 0.8
    dataset_static_noise_speed: float = 0.0
    dataset_dot_std: float = 1.3
    dataset_normalize: bool = False
    vicreg: VICRegConfig = VICRegConfig()
    rssm: RSSMConfig = RSSMConfig()
    simclr: SimCLRConfig = SimCLRConfig()
    eval_at_the_end_only: bool = False
    dataset_type: DatasetType = DatasetType.Single

    probing_cfg: probing.ProbingConfig = probing.ProbingConfig()

    id_predictor: bool = False


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        if self.config.model_type == ModelType.RSSM:
            self.model_config = self.config.rssm
            self.pred_ms = RSSMPredMultistep(config.rssm)
            self.pred_ms = self.pred_ms.cuda()
        elif self.config.model_type == ModelType.VICReg:
            self.model_config = self.config.vicreg
            self.pred_ms = VICRegPredMultistep(self.config.vicreg)
            self.pred_ms = self.pred_ms.cuda()
        elif self.config.model_type == ModelType.SimCLR:
            self.model_config = self.config.simclr
            self.pred_ms = SimCLR(config.simclr)
            self.pred_ms = self.pred_ms.cuda()
        else:
            raise ValueError(f"No valid model type : {self.config.model_type}")
        if config.wandb:
            wandb.init(
                project="HJEPA-untrained-baselines",
                name=config.run_name,
                group=config.run_group,
                config=dataclasses.asdict(config),
            )

        if self.config.id_predictor:
            self.pred_ms.predictor = models.IDPredictor()

        seed_everything(config.seed)

        self.sample_step = 0
        self.epoch = 0
        self.step = 0

        self.save_config()
        self.init_dataset()

        assert (not config.eval_only or config.probing_cfg.full_finetune) or (
            self.maybe_load_model()
        ), "Running eval only without full finetune but couldn't load checkpoint"

        self.n_parameters = sum(
            p.numel() for p in self.pred_ms.parameters() if p.requires_grad
        )
        print("number of params:", self.n_parameters)
        if config.wandb:
            wandb.run.summary["n_params"] = self.n_parameters
            wandb.run.summary["actual_repr_size"] = self.pred_ms.embedding

    def init_dataset(self):
        if self.config.dataset_type == DatasetType.Single:
            assert self.pred_ms.args.channels == 1, (
                "encoder and dataset type are incompatible,"
                f"single dot datset provides one channel, while {self.pred_ms.args.channels} were expected"
            )
            self.ds = data.ContinuousMotionDataset(
                self.config.val_dataset_size // self.config.dataset_batch_size,
                batch_size=self.config.dataset_batch_size,
                n_steps=self.config.n_steps + self.pred_ms.args.rnn_burnin - 1,
                noise=self.config.dataset_noise,
                static_noise=self.config.dataset_static_noise,
                static_noise_speed=self.config.dataset_static_noise_speed,
                std=self.config.dataset_dot_std,
                normalize=self.config.dataset_normalize,
                device=torch.device("cuda"),
            )
            self.val_ds = data.ContinuousMotionDataset(
                self.config.val_dataset_size // self.config.dataset_batch_size,
                batch_size=self.config.dataset_batch_size,
                n_steps=self.config.val_n_steps + self.pred_ms.args.rnn_burnin - 1,
                noise=self.config.dataset_noise,
                static_noise=self.config.dataset_static_noise,
                static_noise_speed=self.config.dataset_static_noise_speed,
                std=self.config.dataset_dot_std,
                normalize=self.config.dataset_normalize,
                device=torch.device("cuda"),
            )
        elif self.config.dataset_type == DatasetType.Multiple:
            sum_image = self.pred_ms.args.channels == 1
            self.ds = data.create_three_datasets(
                self.config.val_dataset_size // self.config.dataset_batch_size,
                batch_size=self.config.dataset_batch_size,
                n_steps=self.config.n_steps + self.pred_ms.args.rnn_burnin - 1,
                noise=self.config.dataset_noise,
                static_noise=self.config.dataset_static_noise,
                static_noise_speed=self.config.dataset_static_noise_speed,
                std=self.config.dataset_dot_std,
                normalize=self.config.dataset_normalize,
                sum_image=sum_image,
                device=torch.device("cuda"),
            )
            self.val_ds = data.create_three_datasets(
                self.config.val_dataset_size // self.config.dataset_batch_size,
                batch_size=self.config.dataset_batch_size,
                n_steps=self.config.val_n_steps + self.pred_ms.args.rnn_burnin - 1,
                noise=self.config.dataset_noise,
                static_noise=self.config.dataset_static_noise,
                static_noise_speed=self.config.dataset_static_noise_speed,
                std=self.config.dataset_dot_std,
                normalize=self.config.dataset_normalize,
                sum_image=sum_image,
                device=torch.device("cuda"),
            )
        else:
            raise NotImplementedError(
                f"dataset type {self.config.dataset_type} is not supported"
            )

    def maybe_load_model(self):
        if self.config.load_checkpoint_path is not None:
            checkpoint = torch.load(self.config.load_checkpoint_path)
            self.pred_ms.load_state_dict(checkpoint["model_state_dict"])
            return True
        return False

    def save_config(self):
        if self.config.output_path is not None:
            os.makedirs(self.config.output_path, exist_ok=True)
            p = Path(self.config.output_path) / "config.yaml"
            with p.open("w") as f:
                OmegaConf.save(config=self.config, f=f)
                print("saved config")

    def validate(self):
        if not self.config.probing_cfg.full_finetune:
            self.pred_ms.eval()

        probing_result = probing.probe_pred_position(
            self.pred_ms.backbone,
            dataset=self.val_ds,
            embedding=self.pred_ms.embedding,
            predictor=self.pred_ms.predictor,
            visualize=False,
            quick_debug=self.config.quick_debug,
            burn_in=self.pred_ms.args.rnn_burnin,
            config=self.config.probing_cfg,
            name_suffix=f"_{self.epoch}",
        )
        probing_enc_result = probing.probe_enc_position(
            backbone=self.pred_ms.backbone,
            embedding=self.pred_ms.embedding,
            dataset=self.val_ds,
            quick_debug=self.config.quick_debug,
            config=self.config.probing_cfg,
            name_suffix=f"_{self.epoch}",
        )

        log_dict = {
            "avg_eval_loss": probing_result.average_eval_loss,
            "avg_eval_loss_rmse": np.sqrt(probing_result.average_eval_loss),
            "avg_eval_enc_loss": probing_enc_result,
            "avg_eval_enc_loss_rmse": np.sqrt(probing_enc_result),
            "epoch": self.epoch,
            "sample_step": self.sample_step,
        }

        for i in range(probing_result.eval_losses_per_step.shape[0]):
            for j in range(probing_result.eval_losses_per_step.shape[1]):
                log_dict[f"eval/loss_{i}_{j}"] = probing_result.eval_losses_per_step[
                    i, j
                ].item()
                log_dict[f"eval/loss_{i}_{j}_rmse"] = np.sqrt(
                    probing_result.eval_losses_per_step[i, j].item()
                )

        for j in range(probing_result.eval_losses_per_step.shape[1]):
            log_dict[f"eval/loss_{j}"] = (
                probing_result.eval_losses_per_step[:, j].mean().item()
            )
            log_dict[f"eval/loss_{j}_rmse"] = np.sqrt(
                probing_result.eval_losses_per_step[:, j].mean().item()
            )

        log_dict["custom_step"] = self.step

        if self.config.wandb:
            wandb.log(log_dict)
            wandb.run.summary.update(log_dict)

        for v in log_dict.values():
            if isinstance(v, matplotlib.figure.Figure):
                plt.close(v)

        self.pred_ms.train()

        return probing_result

    def save_model(self):
        if self.config.output_path is not None:
            os.makedirs(self.config.output_path, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": self.pred_ms.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                os.path.join(
                    self.config.output_path,
                    f"epoch={self.epoch}_sample_step={self.sample_step}.ckpt",
                ),
            )


def main(config: TrainConfig):
    trainer = Trainer(config)
    trainer.validate()


if __name__ == "__main__":
    cfg = TrainConfig.parse_from_command_line()
    main(cfg)
