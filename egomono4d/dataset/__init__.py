from ..frame_sampler import get_frame_sampler
from typing import Union, Optional, List
import os

from .dataset_merged import DatasetMerged
from .types import Stage
from .dataset_arctic import DatasetArctic, DatasetArcticCfg
from .dataset_pov_surgery import DatasetPOVSurgery, DatasetPOVSurgeryCfg
from .dataset_hoi4d import DatasetHOI4D, DatasetHOI4DCfg
from .dataset_fpha import DatasetFPHA, DatasetFPHACfg
from .dataset_h2o import DatasetH2O, DatasetH2OCfg
from .dataset_egopat3d import DatasetEgoPAT3D, DatasetEgoPAT3DCfg
from .dataset_epic_kitchen import DatasetEpicKitchen, DatasetEpicKitchenCfg

DATASETS = {
    "arctic": DatasetArctic,
    "pov_surgery": DatasetPOVSurgery,
    "hoi4d": DatasetHOI4D,
    "h2o": DatasetH2O,
    "fpha": DatasetFPHA,
    "egopat3d": DatasetEgoPAT3D,
    "epic_kitchen": DatasetEpicKitchen
}

DatasetCfg = Union[
    DatasetArcticCfg,
    DatasetPOVSurgeryCfg,
    DatasetHOI4DCfg,
    DatasetH2OCfg,
    DatasetEgoPAT3DCfg,
    DatasetFPHACfg,
    DatasetEpicKitchenCfg
]

def get_dataset(
    dataset_cfgs: List[DatasetCfg],
    stage: Stage,
    global_rank: int,
    world_size: int,
    data_ratio: Optional[float]=1.0,
    debug: Optional[bool]=False,
) -> DatasetMerged:

    datasets = []
    for cfg in dataset_cfgs:
        frame_sampler = get_frame_sampler(cfg.frame_sampler, cfg.num_frames, stage)
        dataset = DATASETS[cfg.name](cfg, stage, frame_sampler, global_rank, world_size, debug) 
        datasets.append(dataset)
        
    return DatasetMerged(datasets, stage=stage, global_rank=global_rank, world_size=world_size, data_ratio=data_ratio)
