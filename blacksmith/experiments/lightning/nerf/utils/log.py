# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch

from blacksmith.experiments.lightning.nerf.utils.metrics import psnr


def log_gradients(log, model_coarse, model_fine):
    for name, param in model_coarse.named_parameters():
        # log weight to make sure it changes
        log(f"weight_coarse/{name}", param.norm(), on_step=True)
        if param.grad is not None:
            log(f"grad_coarse/{name}/norm", param.grad.norm(), on_step=True)
            log(f"grad_coarse/{name}/mean", param.grad.mean(), on_step=True)
            log(f"grad_coarse/{name}/max", param.grad.max(), on_step=True)
        else:
            log(f"grad_coarse/{name}", 0, on_step=True)
            log(f"grad_coarse/{name}/mean", 0, on_step=True)
            log(f"grad_coarse/{name}/max", 0, on_step=True)

    for name, param in model_fine.named_parameters():
        if param.grad is not None:
            log(f"grad_fine/{name}", param.grad.norm(), on_step=True)
            log(f"grad_fine/{name}/mean", param.grad.mean(), on_step=True)
            log(f"grad_fine/{name}/max", param.grad.max(), on_step=True)
        else:
            log(f"grad_fine/{name}", 0, on_step=True)
            log(f"grad_fine/{name}/mean", 0, on_step=True)
            log(f"grad_fine/{name}/max", 0, on_step=True)


def log_training_metrics(log, rgbs, results, loss_rgb, loss_total):
    log("train/loss_rgb", loss_rgb, on_step=True)

    log("train/loss_total", loss_total, on_step=True)

    if "num_samples_coarse" in results:
        log(
            "train/num_samples_coarse",
            results["num_samples_coarse"].mean(),
            on_step=True,
        )

    if "num_samples_fine" in results:
        log("train/num_samples_fine", results["num_samples_fine"].mean(), on_step=True)

    typ = "fine" if "rgb_fine" in results else "coarse"

    with torch.no_grad():
        psnr_fine = psnr(results[f"rgb_{typ}"], rgbs)
        log("train/psnr_fine", psnr_fine, on_step=True, prog_bar=True)

        if "rgb_coarse" in results:
            psnr_coarse = psnr(results["rgb_coarse"], rgbs)
            log("train/psnr_coarse", psnr_coarse, on_step=True)
