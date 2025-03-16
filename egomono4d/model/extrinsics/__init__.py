from .extrinsics import Extrinsics
from .extrinsics_procrustes_flow import ExtrinsicsProcrustesFlow, ExtrinsicsProcrustesFlowCfg
from .extrinsics_regressed import ExtrinsicsRegressed, ExtrinsicsRegressedCfg
from .extrinsics_procrustes_ransac import ExtrinsicsProcrustesRANSAC, ExtrinsicsProcrustesRANSACCfg

EXTRINSICS = {
    "regressed": ExtrinsicsRegressed,
    "procrustes_flow": ExtrinsicsProcrustesFlow,
    "procrustes_ransac": ExtrinsicsProcrustesRANSAC
}

ExtrinsicsCfg = ExtrinsicsRegressedCfg | ExtrinsicsProcrustesFlowCfg | ExtrinsicsProcrustesRANSACCfg


def get_extrinsics(
    cfg: ExtrinsicsCfg,
    num_frames: int | None,
) -> Extrinsics:
    return EXTRINSICS[cfg.name](cfg, num_frames)
