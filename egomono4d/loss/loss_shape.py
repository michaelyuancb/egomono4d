from dataclasses import dataclass
from typing import Literal
import torch
import pdb

from jaxtyping import Float
from torch import Tensor
import torch.nn.functional as F
from torchvision.utils import save_image

from ..dataset.types import Batch
from ..flow import Flows
from ..model.model import ModelOutput
from ..tracking import Tracks
from .loss import Loss, LossCfgCommon

from ..model.procrustes import align_scaled_rigid

@dataclass
class LossShapeCfg(LossCfgCommon):
    name: Literal["shape"]
    dynamic_coef: float
    decay_end_epochs: int
    decay_low_weight: float


def loss_shape_func(ref_pcds, surfaces, flys, loss_func, return_val=False, inf_mode=False, cfg=None):
    b, f, h, w, _ = ref_pcds.shape
    device = ref_pcds.device

    surfaces = surfaces.reshape(b*f, h*w, 3)
    pcd_r = ref_pcds.reshape(b*f, h*w, 3)

    # we keep all points the same weight to conduct constraint on shape rather than points.
    weights = flys.reshape(b*f,h*w)
    transform, scale = align_scaled_rigid(surfaces, pcd_r, weights=weights)

    surfaces_transformed = torch.matmul(transform[..., :3,:3], surfaces.mT).mT + transform[..., None, :3, 3]

    loss_map = loss_func(surfaces_transformed, pcd_r).sum(dim=-1) * weights 
    loss_map = loss_map.reshape(b, f, h, w)
    loss = loss_map.sum() / weights.sum()

    return loss, {"shape": loss}


class LossShape(Loss[LossShapeCfg]):
    def __init__(self, cfg: LossShapeCfg) -> None:
        super().__init__(cfg)
        self.loss = torch.nn.MSELoss(reduction="none")

    def compute_unweighted_loss(
        self,
        batch: Batch,
        flows: Flows,
        tracks: list[Tracks] | None,
        model_output: ModelOutput,
        current_epoch: int,
        return_val: bool,
    ) -> tuple[Float[Tensor, ""], dict]:
        return loss_shape_func(batch.pcds, model_output.surfaces, batch.flys, self.loss, return_val=return_val, cfg=self.cfg)
        
