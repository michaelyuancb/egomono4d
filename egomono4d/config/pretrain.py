from dataclasses import dataclass

from ..misc.data_util import PreProcessingCfg
from ..dataset.data_module_pretrain_cfg import DataModulePretrainCfg
from ..model.model_pretrain_cfg import ModelWrapperPretrainCfg
from .common import CommonCfg


@dataclass
class StageCfg:
    batch_size: int = 1
    num_workers: int = 1


@dataclass
class PretrainCfg(CommonCfg):
    model_wrapper: ModelWrapperPretrainCfg = None
    data_module: DataModulePretrainCfg = None
    preprocess: PreProcessingCfg = None
