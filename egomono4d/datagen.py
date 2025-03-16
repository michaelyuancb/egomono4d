import hydra
import torch
from jaxtyping import install_import_hook
from lightning import Trainer
import pdb
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig

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
    from .model.model import Model
    from .model.model_wrapper_pretrain import ModelWrapperPretrain
    from .visualization import get_visualizers

from .dataset import get_dataset

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="datagen_pov_surgery",
)
def pretrain(cfg_dict: DictConfig) -> None:
    cfg = get_typed_root_config(cfg_dict, PretrainCfg)                     
    cfg.flow.cache_dir = cfg.base_cache_dir
    loss_name_list = [cfg_item.name for cfg_item in cfg.loss]

    for dataset_cfg in cfg.dataset:
        dataset_cfg.resize_shape = cfg.preprocess.resize_shape
        dataset_cfg.patch_size = cfg.preprocess.patch_size
        dataset_cfg.num_frames = cfg.preprocess.num_frames
        dataset_cfg.cache_dir = cfg.base_cache_dir
        dataset_cfg.use_consistency_loss = ('cc' in loss_name_list)
        if hasattr(dataset_cfg, "mask_flow_model"):
            dataset_cfg.mask_flow_model = cfg.flow 

    cfg.trainer.gpus = 1

    dataset_train = get_dataset(cfg.dataset, 'train', debug=False, global_rank=0, world_size=1)
    dataset_val = get_dataset(cfg.dataset, 'val', debug=False, global_rank=0, world_size=1)
    dataset_test = get_dataset(cfg.dataset, 'test', debug=False, global_rank=0, world_size=1)
    train = iter(dataset_train).__next__()
    val = iter(dataset_val).__next__()
    test = iter(dataset_test).__next__()
    pdb.set_trace()


if __name__ == "__main__":
    pretrain()

# CUDA_VISIBLE_DEVICES=0,1 python -m egomono4d.data