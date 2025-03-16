from .intrinsics import Intrinsics
from .intrinsics_ground_truth import IntrinsicsGroundTruth, IntrinsicsGroundTruthCfg
from .intrinsics_regressed import IntrinsicsRegressed, IntrinsicsRegressedCfg
from .intrinsics_softmin import IntrinsicsSoftmin, IntrinsicsSoftminCfg
from .intrinsics_model import IntrinsicsModel, IntrinsicsModelCfg

INTRINSICS = {
    "ground_truth": IntrinsicsGroundTruth,
    "regressed": IntrinsicsRegressed,
    "softmin": IntrinsicsSoftmin,
    "model": IntrinsicsModel
}

IntrinsicsCfg = IntrinsicsRegressedCfg | IntrinsicsGroundTruthCfg | IntrinsicsSoftminCfg | \
    IntrinsicsModelCfg


def get_intrinsics(cfg: IntrinsicsCfg) -> Intrinsics:
    return INTRINSICS[cfg.name](cfg)
