import random
import os
from typing import Callable, Optional, List

import numpy as np
import torch
from torch import Generator
from torch.utils.data import DataLoader, Dataset, IterableDataset, DistributedSampler

from . import DatasetCfg, get_dataset
from .types import Stage
from lightning.pytorch import LightningDataModule as LightningDataModule
from .data_module_pretrain_cfg import DataLoaderStageCfg, DataModulePretrainCfg

DatasetShim = Callable[[Dataset, Stage], Dataset]


def worker_init_fn(worker_id: int) -> None:
    random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))


class DataModulePretrain(LightningDataModule):
    def __init__(
        self,
        dataset_cfgs: List[DatasetCfg],
        data_module_cfg: DataModulePretrainCfg,
        global_rank: int,
        world_size: int,
        data_ratio: Optional[float]=1.0
    ) -> None:
        super().__init__()
        self.dataset_cfgs = dataset_cfgs
        self.data_module_cfg = data_module_cfg
        self.global_rank = global_rank
        self.world_size = world_size
        self.data_ratio = data_ratio

    def get_persistent(self, loader_cfg: DataLoaderStageCfg):
        return None if loader_cfg.num_workers == 0 else loader_cfg.persistent_workers

    def get_generator(self, loader_cfg: DataLoaderStageCfg):
        if loader_cfg.seed is None:
            return None
        generator = Generator()
        generator.manual_seed(loader_cfg.seed + self.global_rank)
        return generator

    def train_dataloader(self):
        dataset = get_dataset(self.dataset_cfgs, "train", global_rank=self.global_rank, world_size=self.world_size, data_ratio=self.data_ratio)
        print(f"train_batch_size = {self.data_module_cfg.train.batch_size}")
        return DataLoader(
            dataset,
            self.data_module_cfg.train.batch_size,
            shuffle=not isinstance(dataset, IterableDataset),
            num_workers=self.data_module_cfg.train.num_workers,
            generator=self.get_generator(self.data_module_cfg.train),
            worker_init_fn=worker_init_fn,
            persistent_workers=self.get_persistent(self.data_module_cfg.train),
        )

    def val_dataloader(self):
        dataset = get_dataset(self.dataset_cfgs, "val", global_rank=self.global_rank, world_size=self.world_size, data_ratio=self.data_ratio)
        print(f"validation_batch_size = {self.data_module_cfg.val.batch_size}")
        return DataLoader(
            dataset,
            self.data_module_cfg.val.batch_size,
            num_workers=self.data_module_cfg.val.num_workers,
            generator=self.get_generator(self.data_module_cfg.val),
            worker_init_fn=worker_init_fn,
            persistent_workers=self.get_persistent(self.data_module_cfg.val),
        )
