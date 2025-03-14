from torch.optim import SGD, Adam, AdamW

from configs import NerfConfig
from thomas.experiments.lightning.nerf.utils.optimizers import *
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, LambdaLR
from thomas.experiments.lightning.nerf.utils.warmup_scheduler import GradualWarmupScheduler


def get_optimizer(hparams: NerfConfig, models: list[torch.nn.Module]):
    assert hparams.training.optimizer in ["sgd", "adam", "radam", "adamw"], "Invalid optimizer"
    
    parameters = []
    for model in models:
        parameters.extend(model.parameters())

    optimizers = {
        "sgd": SGD,
        "adam": Adam,
        "radam": RAdam,
        "adamw": AdamW,
    }
    optimizer = optimizers[hparams.training.optimizer](
        parameters,
        **hparams.training.optimizer_kwargs,
    )
    return optimizer


def get_scheduler(hparams, optimizer):
    assert hparams.lr_scheduler in ["steplr", "cosine", "poly"], "Invalid scheduler"

    schedulers = {
        "steplr": MultiStepLR,
        "cosine": CosineAnnealingLR,
        "poly": LambdaLR,
    }
    scheduler = schedulers[hparams.lr_scheduler](
        optimizer,
        **hparams.lr_scheduler_kwargs,
    )

    if hparams.training.warmup_epochs > 0 and hparams.training.optimizer != "radam":
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=hparams.training.warmup_multiplier,
            total_epoch=hparams.training.warmup_epochs,
            after_scheduler=scheduler,
        )

    return scheduler


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
