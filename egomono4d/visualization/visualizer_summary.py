from dataclasses import dataclass
from typing import Literal

import torch
import pdb
from einops import rearrange
from flow_vis_torch import flow_to_color
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import Batch
from ..flow import Flows
from ..model.model import Model, ModelOutput
from ..model.projection import compute_backward_flow, sample_image_grid
from ..tracking import Tracks
from .color import apply_color_map_to_image
from .depth import color_map_depth
from .layout import add_border, add_label, hcat, vcat
from .visualizer import Visualizer


def flow_with_key(
    flow: Float[Tensor, "frame height width 2"],
) -> Float[Tensor, "3 height vis_width"]:
    _, h, w, _ = flow.shape
    length = min(h, w)
    x = torch.linspace(-1, 1, length, device=flow.device)
    y = torch.linspace(-1, 1, length, device=flow.device)
    key = torch.stack(torch.meshgrid((x, y), indexing="xy"), dim=0)
    flow = rearrange(flow, "f h w xy -> f xy h w")
    return hcat(
        *(flow_to_color(flow) / 255),
        flow_to_color(key) / 255,
    )


@dataclass
class VisualizerSummaryCfg:
    name: Literal["summary"]
    num_vis_frames: int


class VisualizerSummary(Visualizer[VisualizerSummaryCfg]):
    def visualize(
        self,
        batch: Batch,
        flows: Flows,
        tracks: list[Tracks] | None,
        model_output: ModelOutput,
        model: Model,
        loss_packages: dict, 
        global_step: int,
        current_epoch: int,
        inf_mode: bool=False,
    ) -> dict[str, Float[Tensor, "3 _ _"] | Float[Tensor, ""]]:
        # For now, only support batch size 1 for visualization.
        b, f, _, h, w = batch.videos.shape

        if inf_mode is False:
            if b > 1:
                batch.videos = batch.videos[:1]
                # batch.extrinsics = batch.extrinsics[:1]
                flows.forward = flows.forward[:1]
                flows.forward_mask = flows.forward_mask[:1]
                flows.backward = flows.backward[:1]
                flows.backward_mask = flows.backward_mask[:1]
                model_output.surfaces = model_output.surfaces[:1]
                model_output.extrinsics = model_output.extrinsics[:1]
                model_output.intrinsics = model_output.intrinsics[:1]
                b = 1
            assert b == 1

            if self.select_scenes is None:
                self.select_scenes = batch[0].scenes
                self.select_indices = batch[0].indices
            if batch[0].scenes != self.select_scenes:
                return 
            if not torch.equal(batch[0].indices, self.select_indices):
                return

        # Pick a random interval to visualize.
        frames = torch.ones(f, dtype=torch.bool, device=batch.videos.device)
        pairs = torch.ones(f - 1, dtype=torch.bool, device=batch.videos.device)
        if self.cfg.num_vis_frames < f:
            start = torch.randint(f - self.cfg.num_vis_frames, (1,)).item()
            frames[:] = False
            frames[start : start + self.cfg.num_vis_frames] = True
            pairs[:] = False
            pairs[start : start + self.cfg.num_vis_frames - 1] = True

        # Color-map the ground-truth optical flow.
        # fwd_gt = flow_with_key(flows.forward[0, pairs])
        bwd_gt = flow_with_key(flows.backward[0, pairs])

        # Color-map the pose-induced optical flow.
        xy_flowed_backward = compute_backward_flow(
            model_output.surfaces,
            model_output.extrinsics,
            model_output.intrinsics,
        )

        xy, _ = sample_image_grid((h, w), batch.videos.device)
        bwd_hat = flow_with_key((xy_flowed_backward - xy)[0, pairs])

        # Color-map the depth, for metric depth, we do not need to calculate log first.
        depth = color_map_depth(model_output.depths[0, frames], log_first=True)

        # Color-map the correspondence weights.
        bwd_weights = apply_color_map_to_image(
            model_output.backward_correspondence_weights[0, pairs], "gray"
        )
        epoch_prefix = "[Epoch " + str(current_epoch) + "] "
        vcat_list = [
            add_label(hcat(*batch.videos[0, frames]), epoch_prefix+"Video (Ground Truth)"),
            add_label(hcat(*depth), epoch_prefix+"Depth (Predicted)"),
            add_label(bwd_hat, epoch_prefix+"Backward Flow (Predicted)"),
            add_label(bwd_gt, epoch_prefix+"Backward Flow (Ground Truth)"),
            add_label(hcat(*(bwd_weights)), epoch_prefix+"Backward Correspondence Weights"),
        ]
        for loss_package in loss_packages:
            key, value = loss_package['name'], loss_package['package']
            if value is None:
                continue
            xy = value['xy']
            vcat_list.append(add_label(hcat(*batch.videos[0, frames]), epoch_prefix+"Loss Reference Video (Ground Truth)"))
            for lmap_name, lmap in value['loss_map'].items():
                lmap_name  = key + "_loss: " + lmap_name
                if key in ['flow']:
                    lmap_lg = (lmap > 3.5)
                    lmap = lmap_lg * 3.5 + (~lmap_lg) * lmap
                lmap_color = color_map_depth(lmap[0])

                max_idx = torch.argmax(lmap[0,0])
                min_idx = torch.argmin(lmap[0,0])
                _, _, h, w = lmap.shape
                max_coord = (max_idx // w).item(), (max_idx % w).item()
                min_coord = (min_idx // w).item(), (min_idx % w).item()

                lmap_name = lmap_name + "(clip[0,3.5])    |   max_xy-f0:"+str(max_coord)+"  |  min_xy-f0" + str(min_coord)

                # forward_loss: larger lmap = smaller lmap_color
                vcat_list.append(add_label(hcat(*lmap_color), epoch_prefix+lmap_name))

        visualization = vcat(*vcat_list)

        return {"summary": add_border(visualization)}
