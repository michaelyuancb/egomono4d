import hydra
import torch
import pdb
import time
import os
import sys
from pathlib import Path
from jaxtyping import install_import_hook
from lightning import Trainer
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig

from .dataset import get_dataset
from torch.utils.data import DataLoader

# Configure beartype and jaxtyping.
with install_import_hook(
    ("flowmap",),
    ("beartype", "beartype"),
):
    from .config.common import get_typed_root_config
    from .config.pretrain import PretrainCfg
    from .dataset.data_module_pretrain import DataModulePretrain
    from .loss import get_losses
    from .misc.common_training_setup import run_common_training_setup
    from .misc.data_util import compute_patch_cropped_shape
    from .model.model import Model
    from .model.model_wrapper_pretrain import ModelWrapperPretrain
    from .visualization import get_visualizers


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="pretrain",
)
def pretrain(cfg_dict: DictConfig) -> None:
    cfg = get_typed_root_config(cfg_dict, PretrainCfg)                                # get the PretrainCfg
    enable_checkpoints_after = 0
    loss_name_list = [cfg_item.name for cfg_item in cfg.loss]
    print(f"USE-GT-DEP:{cfg.use_gt_depth}")

    cfg.model.backbone.cache_dir = cfg.base_cache_dir
    cfg.flow.cache_dir = cfg.base_cache_dir
    cfg.tracking.cache_dir = cfg.base_cache_dir
    for dataset_cfg in cfg.dataset:
        dataset_cfg.resize_shape = cfg.preprocess.resize_shape
        dataset_cfg.patch_size = cfg.preprocess.patch_size
        dataset_cfg.num_frames = cfg.preprocess.num_frames
        dataset_cfg.cache_dir = cfg.base_cache_dir
        dataset_cfg.use_consistency_loss = ('cc' in loss_name_list)
        if hasattr(dataset_cfg, "mask_flow_model"):
            dataset_cfg.mask_flow_model = cfg.flow 
    
    patch_size = cfg.preprocess.patch_size
    patch_crop_shape = compute_patch_cropped_shape(cfg.preprocess.resize_shape, patch_size)
    num_frames = cfg.preprocess.num_frames

    for loss_cfg in cfg.loss:
        if hasattr(loss_cfg, "enable_after"):
            if loss_cfg.enable_after >= enable_checkpoints_after:
                enable_checkpoints_after = loss_cfg.enable_after
        if hasattr(loss_cfg, "decay_end_epochs"):
            if loss_cfg.decay_end_epochs >= enable_checkpoints_after:
                enable_checkpoints_after = loss_cfg.decay_end_epochs 

    callbacks, logger, checkpoint_path, _ = run_common_training_setup(cfg, cfg_dict)

    # Set up the model.
    print("setup model...")
    model = Model(cfg.model, num_frames=num_frames, image_shape=patch_crop_shape,
                  patch_size=patch_size)
    print("setup losses...")
    losses = get_losses(cfg.loss)
    print("setup visualizers...")
    visualizers = get_visualizers(cfg.visualizer)
    print("setup wrapper_pretrain...")

    # abandon_first = (cfg.checkpoint.load is not None)
    abandon_first = False

    model_wrapper = ModelWrapperPretrain(
        cfg=cfg.model_wrapper,
        cfg_flow=cfg.flow,
        model=model,
        cfg_track=cfg.tracking,
        losses=losses,
        visualizers=visualizers,
        enable_checkpoints_after=enable_checkpoints_after,
        abandon_first=abandon_first
    )
    trainer = Trainer(
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_epochs=cfg.trainer.max_epochs,
        accelerator="gpu",
        logger=logger,
        num_nodes=cfg.trainer.num_nodes,
        devices=cfg.trainer.gpus,
        strategy=(
            "ddp_find_unused_parameters_true"
        ),
        callbacks=callbacks,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        val_check_interval=cfg.trainer.val_check_interval,
        max_steps=-1,                                    
        log_every_n_steps=1,
    )

    trainer.fit(
        model_wrapper,
        datamodule=DataModulePretrain(
            cfg.dataset,
            cfg.data_module,
            trainer.global_rank,
            trainer.world_size,
            cfg.data_ratio
        ),
        ckpt_path=checkpoint_path,
    )



if __name__ == "__main__":

    pretrain()

# python -m egomono4d.pretrain wandb.name=egomono4d_train