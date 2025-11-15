"""
Created by Indraneel on 11/10/2025


python scripts/lerobot_train.py  \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id=indraneelpatil/smolVLA-fine-tuning1 \
  --dataset.repo_id=indraneelpatil/lerobot-teleop-with-cameras \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/train/my_smolvla \
  --job_name=my_smolvla_training \
  --policy.device=cuda \
  --wandb.enable=true

"""
import logging

import torch
from torch.amp import GradScaler
from termcolor import colored
import time
from typing import Any
from torch.optim import Optimizer
from contextlib import nullcontext

from lerobot.utils.utils import init_logging
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, format_big_number, has_method
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.utils.train_utils import get_step_checkpoint_dir, save_checkpoint, update_last_checkpoint


def update_policy(
        train_metrics: MetricsTracker,
        policy: PreTrainedPolicy,
        batch: Any,
        optimizer: Optimizer,
        grad_clip_norm: float,
        grad_scaler: GradScaler,
        lr_scheduler=None,
        use_amp: bool=False,
        lock=None,
) -> tuple[MetricsTracker, dict]:
    """ Single Training step to update the policy's weights"""
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()

    # This means we are doing mixed precision training
    # Training your neural network in both 16 bit and 32 bit floating point numbers
    # Normally pytorch does everything inn 32 bit but GPUs have special tensor cores
    # which can do math faster in 16 bit
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
    
    grad_scaler.scale(loss).backward()

    # Unscale the gradient of the optimizers assigned params
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False
    )

    # Skips optimizer.step if gradient container infs or NaNs
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    
    # Update scale for next iteration
    grad_scaler.update()

    optimizer.zero_grad()
    
    #Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()
    
    if has_method(policy, "update"):
        # Maybe to update an internal buffer
        policy.update()
    
    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict

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
    # Only use for mixed precision training
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
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp
        )

        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()    

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step) 
            save_checkpoint(
                checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler, preprocessor, postprocessor
            )
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

    
    logging.info("End of training")
    
    if cfg.policy.push_to_hub:
        policy.push_model_to_hub(cfg)
        preprocessor.push_to_hub(cfg.policy.repo_id)
        postprocessor.push_to_hub(cfg.policy.repo_id)


def main():
    init_logging()
    train()
    
if __name__== "__main__":
    main()