from typing import Optional
from dataclasses import dataclass

@dataclass
class DataLoaderStageCfg:
    batch_size: int = 1
    num_workers: int = 1
    persistent_workers: bool = True
    seed: Optional[int] = None


@dataclass
class DataModulePretrainCfg:
    train: DataLoaderStageCfg
    val: DataLoaderStageCfg