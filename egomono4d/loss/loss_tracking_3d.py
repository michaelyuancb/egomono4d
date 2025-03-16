from dataclasses import dataclass
from typing import Literal
import torch
import numpy as np
import pdb
from einops import einsum, rearrange

from jaxtyping import Float
from torch import Tensor
import torch.nn.functional as F
from torchvision.utils import save_image

from ..dataset.types import Batch
from ..flow import Flows
from ..model.model import ModelOutput
from ..model.projection import sample_image_grid, homogenize_points
from ..misc.fly import calculate_scale_torch

from ..tracking import Tracks
from .loss import Loss, LossCfgCommon


earlier = lambda x: x[:, :-1]  # noqa
later = lambda x: x[:, 1:]  # noqa


INF = 1e12
NAN_THRESHOLD = 1e-4


@dataclass
class LossTracking3DCfg(LossCfgCommon):
    name: Literal["tracking_3d"]
    # upper_threshold: float


def loss_tracking_3d_func(track_xy, track_visibility, surfaces, extrinsics, masks, loss_func, return_val=False, conf_threshold=0.9, return_traj=False):
    
    _, _, h, w, _ = surfaces.shape

    # TODO: (michael) I only support left->right tracking currently.
    (b, f, n, _) = track_xy.shape

    xyz_base_pre = surfaces.reshape(b, f, h, w, 3)

    scale_edge = calculate_scale_torch(xyz_base_pre)
    scale_edge = rearrange(scale_edge, "b f -> b f () () ()")

    xyz_base = xyz_base_pre.clone()
    xyz_base[masks < conf_threshold] = INF
    segment_tracks_xy_normalized = rearrange(track_xy * 2 - 1, "b f p xyz -> (b f) () p xyz")

    xyz_base = F.grid_sample(
        rearrange(xyz_base, "b f h w xyz -> (b f) xyz h w"),
        segment_tracks_xy_normalized,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    xyz = F.grid_sample(
        rearrange(xyz_base_pre, "b f h w xyz -> (b f) xyz h w"),
        segment_tracks_xy_normalized,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    xyz_base = rearrange(xyz_base, "(b f) xyz () p -> b f p xyz", b=b, f=f)              # (b, f, n, 3)
    xyz = rearrange(xyz, "(b f) xyz () p -> b f p xyz", b=b, f=f)      # (b, f, n, 3)

    nan_error = torch.norm(xyz_base-xyz, dim=-1)                     # (b, f, n)
    nan_masks = (nan_error > NAN_THRESHOLD).sum(axis=1) > 0          # (b, n)

    xy_source = rearrange(track_xy, "b fs p xy -> b fs () p xy")
    xyz_source = rearrange(xyz, "b fs p xyz -> b fs () p xyz")
    xyz_target = rearrange(xyz, "b ft p xyz -> b () ft p xyz")
    extrinsics_source = rearrange(extrinsics, "b fs i j -> b fs () () i j")
    extrinsics_target = rearrange(extrinsics, "b ft i j -> b () ft () i j")
    visibility_source = rearrange(track_visibility, "b fs p -> b fs () p")
    visibility_target = rearrange(track_visibility, "b ft p -> b () ft p")

    relative_transformations = extrinsics_target.inverse() @ extrinsics_source

    xyz_proj = einsum(
        relative_transformations,
        homogenize_points(xyz_source),
        "... i j, ... j -> ... i",
    )[..., :3]                                            # (b fs ft p xyz)

    visibility = visibility_source & visibility_target   
    source_in_frame = (xy_source >= 0).all(dim=-1) & (xy_source < 1).all(dim=-1)
    visibility = visibility & source_in_frame             # (b fs ft p)

    # calculate mask
    segment_tracks_xy_normalized = track_xy.clone()
    segment_tracks_xy_normalized = track_xy * 2 - 1
    out_of_bounds = ((segment_tracks_xy_normalized[..., 0] < -1) | (segment_tracks_xy_normalized[..., 0] > 1) | (segment_tracks_xy_normalized[..., 1] < -1) | (segment_tracks_xy_normalized[..., 1] > 1))

    segment_tracks_xy_normalized = segment_tracks_xy_normalized.view(b*f, n, 1, 2)
    batch_masks_reshaped = masks.view(b*f, 1, h, w)
    sampled_masks = F.grid_sample(batch_masks_reshaped, segment_tracks_xy_normalized, align_corners=True, mode='bilinear', padding_mode="zeros")
    sampled_masks = sampled_masks.view(b, f, n)
    sampled_masks[out_of_bounds] = 0

    sampled_masks_s = rearrange(sampled_masks, "b f p -> b f () p")
    sampled_masks_t = rearrange(sampled_masks, "b f p -> b () f p")
    nan_masks_st = rearrange(1.0 - nan_masks.float(), "b p -> b () () p")

    self_mask = torch.eye(f, device=nan_masks_st.device)
    self_mask = rearrange(1.0 - self_mask, "f t -> () f t ()")

    final_mask = sampled_masks_s * sampled_masks_t * visibility * nan_masks_st * self_mask


    track_loss = loss_func(xyz_proj / scale_edge, xyz_target / scale_edge).sum(dim=-1)
    track_loss = track_loss * final_mask

    grid_n = int(np.sqrt(n))

    track_loss = track_loss.sum() / (final_mask.sum() or 1)
    return track_loss, {"tracking_3d": track_loss}


class LossTracking3D(Loss[LossTracking3DCfg]):
    def __init__(self, cfg: LossTracking3DCfg) -> None:
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
        return loss_tracking_3d_func(tracks[0].xy, tracks[0].visibility, model_output.surfaces, model_output.extrinsics, batch.masks, self.mse_loss, return_val=return_val)
        
       