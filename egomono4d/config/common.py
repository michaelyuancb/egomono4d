from dataclasses import dataclass
import os
import pdb
from typing import Type, TypeVar, Optional, List, Union # , Literal

from omegaconf import DictConfig

from ..dataset import DatasetCfg
from ..misc.cropping import CroppingCfg
from .tools import get_typed_config, separate_multiple_defaults


try:
    EVAL = os.environ['EVAL_MODE']
except:
    EVAL = 'False'
try:
    INFER = os.environ['INFER_MODE']
except:
    INFER = 'False'
if (EVAL not in ['True']):
    from ..flow import FlowPredictorCfg
    from ..loss import LossCfg
    from ..model.model import ModelCfg
    from ..visualization import VisualizerCfg
    from ..tracking import TrackPredictorCfg
    print("Install Training Cfg.")
else:
    FlowPredictorCfg, TrackPredictorCfg = None, None
    LossCfg, ModelCfg, VisualizerCfg = None, None, None

@dataclass
class WandbCfg:
    project: str = "egomono4d"
    mode: str = "disabled"
    name: Optional[str] = None
    group: Optional[str] = None
    tags: Optional[List[str]] = None


@dataclass
class CheckpointCfg:
    load: Optional[str] = None  # str instead of Path, since it could be wandb://...


@dataclass
class TrainerCfg:
    val_check_interval: Union[int, float] = 1.0
    # check_val_every_n_epoch: int
    gradient_clip_val: float = 10.0
    max_steps: Optional[int] = None 
    max_epochs: Optional[int] = None
    accumulate_grad_batches: Optional[int] = None
    num_nodes: int = 1
    gpus: int = 1


@dataclass
class CommonCfg:
    base_cache_dir: str = None
    save_dir: str = None
    data_ratio: float = None
    use_gt_depth: bool = False
    wandb: WandbCfg = None
    checkpoint: CheckpointCfg = None
    trainer: TrainerCfg = None
    flow: Optional[FlowPredictorCfg] = None
    tracking: Optional[TrackPredictorCfg] = None
    dataset: List[DatasetCfg] = None
    model: ModelCfg = None
    loss: List[LossCfg] = None
    visualizer: List[VisualizerCfg] = None
    cropping: Optional[CroppingCfg] = None


T = TypeVar("T")


def get_typed_root_config(cfg_dict: DictConfig, cfg_type: Type[T]) -> T:
    return get_typed_config(
        cfg_type,
        cfg_dict,
        {
            List[DatasetCfg]: separate_multiple_defaults(DatasetCfg),
            List[LossCfg]: separate_multiple_defaults(LossCfg),
            List[VisualizerCfg]: separate_multiple_defaults(VisualizerCfg),
        },
    )
