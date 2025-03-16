from dataclasses import dataclass
from typing import Literal
import torch
import pdb

from jaxtyping import Float
from torch import Tensor
import torch.nn.functional as F
from einops import einsum, rearrange
from torchvision.utils import save_image

from ..dataset.types import Batch
from ..flow import Flows
from ..model.model import ModelOutput
from ..model.projection import (
    compute_backward_flow,
    compute_forward_flow,
    sample_image_grid,
)
from ..tracking import Tracks
from .loss import Loss, LossCfgCommon
from ..misc.fly import calculate_scale_torch


INF = 1e12
NAN_THRESHOLD = 1e-4
earlier = lambda x: x[:, :-1]  # noqa
later = lambda x: x[:, 1:]  # noqa


@dataclass
class LossFlow3DCfg(LossCfgCommon):
    name: Literal["flow_3d"]
    # upper_threshold: float


def loss_flow_3d_func(flows: Flows, surfaces, extrinsics, masks, loss_func, return_val=False, conf_threshold=0.9):        
    b, f, h, w, _ = surfaces.shape
    device = surfaces.device
    xyz_base_pre = surfaces.reshape(b, f, h, w, 3)
        
    scale_edge = calculate_scale_torch(xyz_base_pre)

    xyz_base = xyz_base_pre.clone()
    xyz_base[masks < conf_threshold] = INF

    ##################################### Forward Loss #####################################

    # pdb.set_trace()
    # get PCD[later] from Interpolation of Flow
    xy, _ = sample_image_grid((h, w), device)
    xy = xy.reshape(1, 1, h*w, 2)  
    xy_forward = xy + flows.forward.reshape(b, f-1, h*w, 2)
    xy_forward_inp = rearrange(xy_forward * 2 - 1, "b f p xyz -> (b f) () p xyz")
    xyz_forward = F.grid_sample(
        rearrange(later(xyz_base), "b f h w xyz -> (b f) xyz h w"),
        xy_forward_inp,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    xyz_pre_forward = F.grid_sample(
        rearrange(later(xyz_base_pre), "b f h w xyz -> (b f) xyz h w"),
        xy_forward_inp,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    xyz_forward = rearrange(xyz_forward, "(b f) xyz () p -> b f p xyz", b=b, f=f-1)            # (b, f, n, 3)
    xyz_pre_forward = rearrange(xyz_pre_forward, "(b f) xyz () p -> b f p xyz", b=b, f=f-1)    # (b, f, n, 3)

    # get PCD[later] from PCD[earlier] projection                                   # (b, f, 4, 4)
    ones = torch.ones((b, f-1, h*w, 1), device=xyz_forward.device)
    xyz_homogeneous_earlier = torch.cat([earlier(surfaces).reshape(b,f-1,-1,3), ones], dim=-1)   # (b, f-1, n, 4)
    forward_transformation = later(extrinsics).inverse() @ earlier(extrinsics)   # (b, f-1, 4, 4)
    forward_transformation_expanded = forward_transformation.unsqueeze(2)        # (b, f, 1, 4, 4)
    world_coords_homogeneous_forward = torch.matmul(forward_transformation_expanded, xyz_homogeneous_earlier.unsqueeze(-1))  # (b, f-1, n, 4, 1)
    world_coords_homogeneous_forward = world_coords_homogeneous_forward.squeeze(-1)  # (b, f-1, n, 4)
    xyz_w_forward = world_coords_homogeneous_forward[..., :3] / (world_coords_homogeneous_forward[..., 3:4] + 1e-8)  # (b, f-1, n, 3)

    # get forward masks
    nan_error_forward = torch.norm(xyz_forward-xyz_pre_forward, dim=-1)                     # (b, f, n)
    nan_masks_forward = (nan_error_forward > NAN_THRESHOLD).sum(axis=1) > 0    # (b, n)
    nan_masks_forward = nan_masks_forward[:,None].repeat(1,f-1,1)
        
    forward_dist = torch.norm(xyz_forward-xyz_w_forward, dim=-1)
    sampled_masks_forward = (1.0 - nan_masks_forward.float()) * flows.forward_mask.reshape(b,f-1,-1)
        
    forward_dist_normalized = forward_dist / scale_edge[:, :-1, None]

    forward_loss = loss_func(forward_dist_normalized, torch.zeros_like(forward_dist_normalized, device=forward_dist.device))
    w_forward_loss = forward_loss * sampled_masks_forward


    ##################################### Backward Loss #####################################

    # pdb.set_trace()
    # get PCD[earlier] from Interpolation of Flow
    xy, _ = sample_image_grid((h, w), device)
    xy = xy.reshape(1, 1, h*w, 2)  
    xy_backward = xy + flows.backward.reshape(b, f-1, h*w, 2)
    xy_backward_inp = rearrange(xy_backward * 2 - 1, "b f p xyz -> (b f) () p xyz")
    xyz_backward = F.grid_sample(
        rearrange(earlier(xyz_base), "b f h w xyz -> (b f) xyz h w"),
        xy_backward_inp,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    xyz_pre_backward = F.grid_sample(
        rearrange(earlier(xyz_base_pre), "b f h w xyz -> (b f) xyz h w"),
        xy_backward_inp,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    xyz_backward = rearrange(xyz_backward, "(b f) xyz () p -> b f p xyz", b=b, f=f-1)            # (b, f, n, 3)
    xyz_pre_backward = rearrange(xyz_pre_backward, "(b f) xyz () p -> b f p xyz", b=b, f=f-1)    # (b, f, n, 3)

    # get PCD[later] from PCD[earlier] projection
    xyz_homogeneous_later = torch.cat([later(surfaces).reshape(b,f-1,-1,3), ones], dim=-1)   # (b, f-1, n, 4)
    backward_transformation = earlier(extrinsics).inverse() @ later(extrinsics)   # (b, f-1, 4, 4)
    backward_transformation_expanded = backward_transformation.unsqueeze(2)        # (b, f, 1, 4, 4)
    world_coords_homogeneous_backward = torch.matmul(backward_transformation_expanded, xyz_homogeneous_later.unsqueeze(-1))  # (b, f-1, n, 4, 1)
    world_coords_homogeneous_backward = world_coords_homogeneous_backward.squeeze(-1)  # (b, f-1, n, 4)
    xyz_w_backward = world_coords_homogeneous_backward[..., :3] / (world_coords_homogeneous_backward[..., 3:4] + 1e-8)  # (b, f-1, n, 3)

    # get backward masks
    nan_error_backward = torch.norm(xyz_backward-xyz_pre_backward, dim=-1)                     # (b, f, n)
    nan_error_backward = (nan_error_backward > NAN_THRESHOLD).sum(axis=1) > 0    # (b, n)
    nan_error_backward = nan_error_backward[:,None].repeat(1,f-1,1)
        
    backward_dist = torch.norm(xyz_backward-xyz_w_backward, dim=-1)
    sampled_masks_backward = (1.0 - nan_error_backward.float()) * flows.backward_mask.reshape(b,f-1,-1)
        
    backward_dist_normalized = backward_dist / scale_edge[:, 1:, None]

    backward_loss = loss_func(backward_dist_normalized, torch.zeros_like(backward_dist_normalized, device=forward_dist.device))
    w_backward_loss = backward_loss * sampled_masks_backward

    # get final loss
    loss_sum = w_forward_loss.sum() + w_backward_loss.sum()
    valid_sum = sampled_masks_forward.sum() + sampled_masks_backward.sum()


    flow_loss = loss_sum / (valid_sum or 1)
    return flow_loss, {"flow_3d": flow_loss}



class LossFlow3D(Loss[LossFlow3DCfg]):
    def __init__(self, cfg: LossFlow3DCfg) -> None:
        super().__init__(cfg)
        self.mse_loss = torch.nn.MSELoss(reduction="none")

    def compute_unweighted_loss(
        self,
        batch: Batch,
        flows: Flows,
        tracks: list[Tracks] | None,
        model_output: ModelOutput,
        current_epoch: int,
        return_val: bool
    ) -> tuple[Float[Tensor, ""], dict]:
        return loss_flow_3d_func(flows, model_output.surfaces, model_output.extrinsics, batch.masks, self.mse_loss, return_val=return_val)
        

