import random
from typing import NamedTuple, List, Any, Optional
from itertools import chain
from dataclasses import dataclass

import torch
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import numpy as np
import wandb

import data
from models import Prober
from configs import ConfigBase


@dataclass
class ProbingConfig(ConfigBase):
    full_finetune: bool = False
    lr: float = 1e-3
    epochs: int = 3
    epochs_enc: int = 3
    schedule: Optional[str] = None
    prober_arch: str = ""


@torch.no_grad()
def probe_enc_position_visualize(
    backbone: torch.nn.Module, prober: torch.nn.Module, dataset
):
    batch = next(iter(dataset))
    enc = backbone(batch.states[:, 0].cuda())
    pred_loc = prober(enc)

    plt.figure(dpi=200)
    pidx = 1
    for row in range(5):
        for col in range(5):
            plt.subplot(5, 5, pidx)
            pidx += 1
            idx = random.randint(0, batch.states.shape[0] - 1)

            plt.imshow(batch.states[idx, 0, 0].cpu(), origin="lower")
            plt.axis("off")

            pred_x_loc = pred_loc[idx, 0]
            pred_y_loc = pred_loc[idx, 1]

            plt.plot(
                pred_x_loc.item(),
                pred_y_loc.item(),
                marker="o",
                markersize=2,
                markeredgecolor="red",
                markerfacecolor="green",
                alpha=0.5,
            )


def probe_enc_position(
    backbone: torch.nn.Module,
    embedding: int,
    dataset,
    *,
    visualize: bool = False,
    prober_arch: str = "",
    quick_debug: bool = False,
    config: ProbingConfig = ProbingConfig(),
    name_suffix: str = "",
):
    test_batch = dataset[0]

    prober_output_shape = test_batch.locations[0, 0].shape
    prober = Prober(embedding, prober_arch, output_shape=prober_output_shape)
    prober = prober.cuda()

    if not config.full_finetune:
        optimizer_pred_prober = torch.optim.Adam(prober.parameters(), config.lr)
    else:
        optimizer_pred_prober = torch.optim.Adam(
            chain(prober.parameters(), backbone.parameters()),
            config.lr,
        )

    losses = []

    if quick_debug:
        config.epochs_enc = 1

    step = 0
    sample_step = 0

    for epoch in tqdm(range(config.epochs_enc)):
        for batch in dataset:
            target_loc = batch.locations[:, 0].cuda().float()

            e = backbone(batch.states[:, 0].cuda())
            pred_loc = prober(e)

            loss = location_losses(pred_loc, target_loc)

            losses.append(loss.mean().item())

            optimizer_pred_prober.zero_grad()
            loss.mean().backward()
            optimizer_pred_prober.step()

            if wandb.run is not None and step % 100 == 0:
                log_dict = {
                    f"finetune_enc{name_suffix}/loss": loss.mean().item(),
                    f"finetune_enc{name_suffix}/step": step,
                    f"finetune_enc{name_suffix}/sample_step": sample_step,
                    f"finetune_enc{name_suffix}/epoch": epoch,
                }
                per_dot_losses = loss
                for i, val in enumerate(per_dot_losses):
                    log_dict[f"finetune_enc{name_suffix}/loss_dot_{i}"] = val.item()
                wandb.log(log_dict)

            step += 1
            sample_step += batch.locations.shape[0]
            if quick_debug:
                break

    with torch.no_grad():
        eval_losses = []
        for batch in dataset:
            target_loc = batch.locations[:, 0].cuda().float()
            e = backbone(batch.states[:, 0].cuda())
            pred_loc = prober(e)
            loss = location_losses(pred_loc, target_loc).mean()
            eval_losses.append(loss.item())

    if visualize:
        plt.figure(dpi=200)
        plt.plot(losses)
        plt.grid()
        plt.show()
        probe_enc_position_visualize(backbone, prober, dataset)
        plt.show()

    avg_loss = np.mean(eval_losses)
    unnormalized_avg_loss = dataset.unnormalize_mse(avg_loss)

    return unnormalized_avg_loss


def probe_pred_position_visualize(
    model: torch.nn.Module,
    *,
    embedding: int,
    burn_in: int,
    predictor: torch.nn.Module,
    prober: torch.nn.Module,
    dataset,
):
    batch = next(iter(dataset))

    burnin_states = batch.states[:, : burn_in - 1].cuda().permute(1, 0, 2, 3, 4)
    states = batch.states[:, burn_in - 1 :].cuda().permute(1, 0, 2, 3, 4)

    # drop actions of other spheres, put time first
    burnin_actions = batch.actions[:, : burn_in - 1, 0].cuda().permute(1, 0, 2)
    actions = batch.actions[:, burn_in - 1 :, 0].cuda().permute(1, 0, 2)

    if burn_in > 1:
        burnin_encodings = model(burnin_states.flatten(0, 1)).view(
            *burnin_actions.shape[:2], -1
        )
        h0 = predictor.burn_in(burnin_encodings, burnin_actions)
    else:
        h0 = None

    e = model(states[0].cuda())
    pred_encs = predictor.predict_sequence(enc=e, actions=actions, h=h0)

    # for i in range(batch.actions.shape[1]):
    #     e = predictor(e, batch.actions[:, i].cuda())
    pred_loc = dataset.unnormalize_location(prober(pred_encs[-1])[:, 0])
    target_loc = dataset.unnormalize_location(batch.locations[:, -1, 0])

    fig, ax = plt.subplots(5, 5, dpi=200)
    pidx = 1
    for row in range(5):
        for col in range(5):
            # plt.subplot(5, 5, pidx)
            pidx += 1
            idx = random.randint(0, batch.states.shape[0] - 1)

            ax[row][col].imshow(batch.states[idx, 0, 0].cpu(), origin="lower")
            ax[row][col].set_axis_off()

            pred_x_loc = pred_loc[idx, 0]
            pred_y_loc = pred_loc[idx, 1]

            ax[row][col].plot(
                pred_x_loc.item(),
                pred_y_loc.item(),
                marker="o",
                markersize=2,
                markeredgecolor="red",
                markerfacecolor="green",
                alpha=0.5,
            )
            ax[row][col].plot(
                target_loc[idx, 0].item(),
                target_loc[idx, 1].item(),
                marker="x",
                markersize=2,
                markeredgecolor="red",
                markerfacecolor="yellow",
                alpha=0.5,
            )
    return fig


class ProbeResult(NamedTuple):
    model: torch.nn.Module
    average_eval_loss: float
    eval_losses_per_step: List[float]
    plots: List[Any]


def location_losses(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (pred - target).pow(2).sum(dim=-1).mean(dim=0)
    return mse


def probe_pred_position(
    backbone: torch.nn.Module,
    dataset,
    embedding: int,
    predictor: torch.nn.Module,
    *,
    visualize: bool = False,
    quick_debug: bool = False,
    burn_in: int = 0,
    config: ProbingConfig = ProbingConfig(),
    name_suffix: str = "",
):
    if quick_debug:
        config.epochs = 1
    test_batch = next(iter(dataset))

    prober_output_shape = test_batch.locations[0, 0].shape
    prober = Prober(embedding, config.prober_arch, output_shape=prober_output_shape)
    prober = prober.cuda()

    if not config.full_finetune:
        optimizer_pred_prober = torch.optim.Adam(prober.parameters(), config.lr)
    else:
        optimizer_pred_prober = torch.optim.Adam(
            chain(prober.parameters(), backbone.parameters(), predictor.parameters()),
            config.lr,
        )

    sample_step = 0
    step = 0

    for epoch in tqdm(range(config.epochs)):
        for batch in dataset:
            # put time first
            burnin_states = batch.states[:, : burn_in - 1].cuda().permute(1, 0, 2, 3, 4)
            states = batch.states[:, burn_in - 1 :].cuda().permute(1, 0, 2, 3, 4)

            # drop actions of other spheres, put time first
            burnin_actions = batch.actions[:, : burn_in - 1, 0].cuda().permute(1, 0, 2)
            actions = batch.actions[:, burn_in - 1 :, 0].cuda().permute(1, 0, 2)

            if burn_in > 1:
                burnin_encodings = backbone(burnin_states.flatten(0, 1)).view(
                    *burnin_actions.shape[:2], -1
                )
                h0 = predictor.burn_in(burnin_encodings, burnin_actions)
            else:
                h0 = None

            e = backbone(states[0])
            pred_encs = predictor.predict_sequence(enc=e, actions=actions, h=h0)
            if not config.full_finetune:
                pred_encs = pred_encs.detach()

            all_encs = torch.cat([e.unsqueeze(0), pred_encs], dim=0)
            pred_locs = torch.stack([prober(x) for x in all_encs], dim=1)

            losses = location_losses(
                pred_locs, batch.locations[:, burn_in - 1 :].cuda()
            )

            loss = losses.mean()
            optimizer_pred_prober.zero_grad()
            loss.backward()
            optimizer_pred_prober.step()

            if wandb.run is not None and step % 100 == 0:
                log_dict = {
                    f"finetune_pred{name_suffix}/loss": loss.item(),
                    f"finetune_pred{name_suffix}/step": step,
                    f"finetune_pred{name_suffix}/sample_step": sample_step,
                    f"finetune_pred{name_suffix}/epoch": epoch,
                }
                per_dot_losses = losses.mean(dim=0)
                for i, val in enumerate(per_dot_losses):
                    log_dict[f"finetune_pred{name_suffix}/loss_dot_{i}"] = val.item()

                wandb.log(log_dict)

            step += 1
            sample_step += states.shape[0]

            if quick_debug:
                break

    with torch.no_grad():
        eval_losses = []
        for batch in dataset:
            # put time first
            burnin_states = batch.states[:, : burn_in - 1].cuda().permute(1, 0, 2, 3, 4)
            states = batch.states[:, burn_in - 1 :].cuda().permute(1, 0, 2, 3, 4)

            # drop actions of other spheres, put time first
            burnin_actions = batch.actions[:, : burn_in - 1, 0].cuda().permute(1, 0, 2)
            actions = batch.actions[:, burn_in - 1 :, 0].cuda().permute(1, 0, 2)

            if burn_in > 1:
                burnin_encodings = backbone(burnin_states.flatten(0, 1)).view(
                    *burnin_actions.shape[:2], -1
                )
                h0 = predictor.burn_in(burnin_encodings, burnin_actions)
            else:
                h0 = None

            e = backbone(states[0])
            pred_encs = predictor.predict_sequence(
                enc=e, actions=actions, h=h0
            ).detach()
            all_encs = torch.cat([e.unsqueeze(0), pred_encs], dim=0)

            pred_locs = torch.stack([prober(x) for x in all_encs], dim=1)

            losses = location_losses(
                pred_locs, batch.locations[:, burn_in - 1 :].cuda()
            )

            eval_losses.append(losses)

            if quick_debug:
                break

        losses_t = torch.stack(eval_losses, dim=0).mean(dim=0)
        losses_t = dataset.unnormalize_mse(losses_t)

    if visualize:
        fig = probe_pred_position_visualize(
            backbone,
            dataset=dataset,
            embedding=embedding,
            burn_in=burn_in,
            predictor=predictor,
            prober=prober,
        )
    else:
        fig = None

    return ProbeResult(prober, losses_t.mean().item(), losses_t, [fig])


class ProbeMPCResult(NamedTuple):
    average_diff: float
    figures: List[Any]


def normalize_actions(actions):
    actions_n = actions.clone()
    actions_n[..., :2] = (
        actions[..., :2] / actions[..., :2].norm(dim=-1).unsqueeze(-1).detach()
    )
    actions_n[..., -1] = actions[..., -1].clamp(min=0, max=4)
    return actions_n


def probe_mpc(
    backbone: torch.nn.Module,
    *,
    embedding: int,
    predictor: torch.nn.Module,
    prober: torch.nn.Module,
    plan_size: int = 17,
    n_iters: int = 20,
):
    prober.eval()
    figs = []
    diffs = []
    for i in range(n_iters):
        state1, location1 = data.ContinuousMotionDataset.generate_state()
        state2, location2 = data.ContinuousMotionDataset.generate_state()

        # plt.subplot(1, 2, 1)
        # plt.imshow(state1[0], origin='lower')
        # plt.subplot(1, 2, 2)
        # plt.imshow(state2[0], origin=''lower)

        enc1 = backbone(state1.unsqueeze(0).cuda())
        # enc2 = backbone(state2.unsqueeze(0).cuda())

        directions = torch.rand((plan_size, 2), device="cuda") * 2 - 1
        speeds = torch.rand((plan_size, 1), device="cuda") * 4
        actions = normalize_actions(torch.cat([directions, speeds], dim=-1))
        actions.requires_grad = True

        opt = torch.optim.Adam((actions,), lr=0.1)

        losses = []
        for _ in range(100):
            current_enc = enc1.detach()
            actions_n = normalize_actions(actions)
            actions_2 = actions_n[:, :2] * actions[:, 2].unsqueeze(-1)

            pred_encs = predictor.predict_sequence(
                enc=current_enc, actions=actions_2.unsqueeze(1).cuda()
            )
            # for i in range(actions.shape[0]):
            #     # t = torch.concat([current_enc, actions_2[i].unsqueeze(0)], dim=1)
            #     # current_enc = predictor.model(t)
            #     current_enc = predictor(current_enc, actions_2[i].unsqueeze(0))

            pred_loc = prober(pred_encs[-1])
            #     target_loc = prober(enc2)
            target_loc = location2.cuda().float().unsqueeze(0)
            #     diff = torch.nn.functional.mse_loss(current_enc, enc2.detach())
            diff = torch.nn.functional.mse_loss(pred_loc, target_loc.detach())
            #     diff = -1 * current_enc[0].T @ enc2[0].detach()
            #     print(diff.shape, current_enc.shape)
            #     print(pred_loc, target_loc)
            opt.zero_grad()
            diff.backward()
            opt.step()
            losses.append(diff.item())

        actions_n = normalize_actions(actions)
        actions_2 = actions_n[:, :2] * actions[:, 2].unsqueeze(-1)

        seq = data.ContinuousMotionDataset.generate_transitions(
            state1, location1, actions_2.cpu()
        )

        # fig, ax = plt.subplots(1, 5, dpi=200)

        fig = plt.figure(dpi=200)
        gs = fig.add_gridspec(2, 4)
        ax0 = fig.add_subplot(gs[0, :])
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[1, 1])
        ax3 = fig.add_subplot(gs[1, 2])
        ax4 = fig.add_subplot(gs[1, 3])

        ax0.plot(losses)
        ax0.set_title("loss")
        ax0.grid(True)

        ax1.imshow(state1[0], origin="lower")
        ax1.set_xlabel(f"loc={list(map(lambda x: round(x, 1), location1.tolist()))}")
        ax1.set_title("start")

        ax2.imshow(state2[0], origin="lower")
        ax2.set_xlabel(f"loc={list(map(lambda x: round(x, 1), location2.tolist()))}")
        ax2.set_title("target")

        ax3.imshow(seq.states[-1, 0].detach().cpu(), origin="lower")
        ax3.set_xlabel(
            f"loc={list(map(lambda x: round(x, 1), seq.locations[-1].tolist()))}"
        )
        ax3.set_title("reached")

        ax4.set_title("path")
        ax4.scatter(
            seq.locations[:, 0].detach().cpu(),
            seq.locations[:, 1].detach().cpu(),
            c=range(seq.locations.shape[0]),
        )
        ax4.plot(location2[0], location2[1], marker="x", markersize=10, c="r")
        ax4.set_xlim(0, 28)
        ax4.set_ylim(0, 28)
        ax4.grid(visible=True)
        ax4.set_aspect("equal")

        fig.set_tight_layout(True)

        figs.append(fig)

        diff = (seq.locations[-1] - location2).pow(2).mean().item()
        diffs.append(diff)
    return ProbeMPCResult(np.mean(diffs), figs)
