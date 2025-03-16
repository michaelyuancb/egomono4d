from dataclasses import dataclass
from typing import Literal
import torch
import pdb
from einops import einsum, rearrange

from jaxtyping import Float
from torch import Tensor
import torch.nn.functional as F
from torchvision.utils import save_image

from ..dataset.types import Batch
from ..flow import Flows
from ..model.model import ModelOutput
from ..model.projection import sample_image_grid

from ..tracking import Tracks
from .loss import Loss, LossCfgCommon
from .mapping import MappingCfg, get_mapping

earlier = lambda x: x[:, :-1]  # noqa
later = lambda x: x[:, 1:]  # noqa


@dataclass
class LossDynamicAreaCfg(LossCfgCommon):
    name: Literal["dynamic_area"]


class LossDynamicArea(Loss[LossDynamicAreaCfg]):
    def __init__(self, cfg: LossDynamicAreaCfg) -> None:
        super().__init__(cfg)
        self.bce_loss = torch.nn.BCELoss(reduction="none")

    def compute_unweighted_loss(
        self,
        batch: Batch,
        flows: Flows,
        tracks: list[Tracks] | None,
        model_output: ModelOutput,
        current_epoch: int,
        return_val: bool
    ) -> tuple[Float[Tensor, ""], dict]:
        
        surfaces = model_output.surfaces
        device = surfaces.device
        b, f, h, w, _ = surfaces.shape
        xy, _ = sample_image_grid((h, w), device=device)

        later_mask = later(batch.masks)        # (b, f-1, h, w)
        b_xy_earlier = rearrange(xy + flows.backward, "b f h w xy -> (b f) h w xy")
        earlier_mask = F.grid_sample(
            rearrange(earlier(batch.masks), "b f h w -> (b f) () h w"),
            b_xy_earlier * 2 - 1,
            align_corners=True, 
            mode='bilinear', 
            padding_mode="zeros"
        )
        earlier_mask = rearrange(earlier_mask, "(b f) () h w -> b f h w", b=b, f=f-1)
        gt_mask = later_mask * earlier_mask

        loss = self.bce_loss(model_output.backward_correspondence_weights, gt_mask)
        valid = h * w * (f-1) * b


        loss = loss.sum() / (valid or 1)
        return loss, {"dynamic_area": loss}