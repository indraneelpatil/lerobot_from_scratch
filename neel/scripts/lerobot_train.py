"""
Created by Indraneel on 11/10/2025



"""
import logging

import torch
from torch.amp import GradScaler
from termcolor import colored

from lerobot.utils.utils import init_logging
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, format_big_number
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.optim.factory import make_optimizer_and_scheduler


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    """
        Training!
    """
    cfg.validate()

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info("Wandb logger is disabled")
    
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check if device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    # Tried different convolutional algorithms for your hardware and then picks the best one
    torch.backends.cudnn.benchmark = True
    # Useful for newer nvidia gpu, using tf32 math is faster at the cost of some accuracy
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    # TODO eval env

    logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta
    )

    # Create processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    processor_kwargs["dataset_stats"] = dataset.meta.stats

    processor_kwargs["preprocessor_overrides"] = {
        "device_processor": {"device" : device.type},
        "normalizer_processor": {
            "stats": dataset.meta.stats,
            "features": {**policy.config.input_features, **policy.config.output_features},
            "norm_map": policy.config.normalization_mapping
        },
    }
    postprocessor_kwargs["postprocessor_overrides"] = {
        "unnormalizer_processor": {
            "stats": dataset.meta.stats,
            "features": policy.config.output_features,
            "norm_map": policy.config.normalization_mapping
        }
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Helps scale the loss before back propogation to ensure gradients dont go to zero
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0 # Number of policy updates

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f"{cfg.output_dir}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # Create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2,
    )
    dl_iter = cycle(dataloader)

    # Set the module in training mode
    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", "0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f")
    }
    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )
    logging.info("Start offline training on a fixed dataset")
    for _ in range(step, cfg.steps):
        pass


def main():
    init_logging()
    train()
    
if __name__== "__main__":
    main()