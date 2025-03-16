import torch
import pdb
import os
from einops import reduce
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import optim

from ..dataset.types import Batch, BatchInference
from ..flow import FlowPredictorCfg, Flows, get_flow_predictor
from ..tracking import TrackPredictorCfg, Tracks, get_track_predictor, generate_pretrain_video_tracks, \
    generate_pretrain_video_flows_tracks
from ..loss import Loss
import time
from ..misc.image_io import prep_image
from ..model.projection import sample_image_grid
from ..visualization import Visualizer, VisualizerCoTracker
from .model import Model
from .model_pretrain_cfg import ModelWrapperPretrainCfg

def get_device():
    return f"cuda" if torch.cuda.is_available() else "cpu"


def visualize_batch_flows_tracks(batch, flows, tracks, stage='train'):
    
    save_dir = "./outputs/tracker"
    track_filename = 'track_'+str(batch.indices[0,1].item())+"i"+str((batch.indices[0,1]-batch.indices[0,0]).item())+'_'+stage+'_'+batch.scenes[0].replace("/", "_")
    if not os.path.exists(save_dir+"/"+track_filename+".mp4"):
        # tracks
        vis_tracks = VisualizerCoTracker(pad_value=120, linewidth=1, save_dir=save_dir)
        vis_tracks_video = vis_tracks.visualize(batch.videos * 255, tracks[0].xy, tracks[0].visibility, filename=track_filename)

    save_dir = "./outputs/flow_forward"
    flow_forward_filename = 'flow_forward_'+str(batch.indices[0,1].item())+"i"+str((batch.indices[0,1]-batch.indices[0,0]).item())+'_'+stage+'_'+batch.scenes[0].replace("/", "_")
    if not os.path.exists(save_dir+"/"+flow_forward_filename+".mp4"):
        # forward flows
        b, f, _, h, w = batch.videos.shape
        xy, _ = sample_image_grid((h, w), get_device())  # (b, 1, h, w, xy)
        xy = xy.unsqueeze(0).unsqueeze(0).repeat(b, 1, 1, 1, 1)
        mask = torch.ones(b, f, h, w).to(get_device())
        xy_forward = xy.repeat(1, f-1, 1, 1, 1) + flows.forward.cumsum(dim=1)
        xy = torch.cat((xy, xy_forward), dim=1)
        mask[:, 1:] = flows.forward_mask
        mask = mask.cumprod(dim=1)
        mask = mask > 0.25

        grid_size = (48, 48)
        sampled_h_indices = torch.linspace(0, h - 1, grid_size[0], dtype=torch.long)
        sampled_w_indices = torch.linspace(0, w - 1, grid_size[0], dtype=torch.long)
        sampled_xy = xy[:, :, sampled_h_indices[:, None, None], sampled_w_indices[None, None, :]]
        sampled_mask = mask[:, :, sampled_h_indices[:, None, None], sampled_w_indices[None, None, :]]
        sampled_xy = sampled_xy.reshape(b, f, -1, 2)
        sampled_mask = sampled_mask.reshape(b, f, -1)

        vis_tracks = VisualizerCoTracker(pad_value=120, linewidth=1, save_dir=save_dir)
        vis_flows_video = vis_tracks.visualize(batch.videos * 255, sampled_xy, sampled_mask, filename=flow_forward_filename)

    save_dir = "./outputs/flow_backward"
    flow_backward_filename = 'flow_backward_'+str(batch.indices[0,1].item())+"i"+str((batch.indices[0,1]-batch.indices[0,0]).item())+'_'+stage+'_'+batch.scenes[0].replace("/", "_")
    if not os.path.exists(save_dir+"/"+flow_backward_filename+".mp4"):
        # backward flows
        b, f, _, h, w = batch.videos.shape
        xy, _ = sample_image_grid((h, w), get_device())  # (b, 1, h, w, xy)
        xy = xy.unsqueeze(0).unsqueeze(0).repeat(b, 1, 1, 1, 1)
        mask = torch.ones(b, f, h, w).to(get_device())
        xy_backward = xy.repeat(1, f-1, 1, 1, 1) + flows.backward.flip(dims=[1]).cumsum(dim=1)
        xy = torch.cat((xy, xy_backward), dim=1)
        mask[:, 1:] = flows.backward_mask.flip(dims=[1])
        mask = mask.cumprod(dim=1)
        mask = mask > 0.25

        grid_size = (48, 48)
        sampled_h_indices = torch.linspace(0, h - 1, grid_size[0], dtype=torch.long)
        sampled_w_indices = torch.linspace(0, w - 1, grid_size[0], dtype=torch.long)
        sampled_xy = xy[:, :, sampled_h_indices[:, None, None], sampled_w_indices[None, None, :]]
        sampled_mask = mask[:, :, sampled_h_indices[:, None, None], sampled_w_indices[None, None, :]]
        sampled_xy = sampled_xy.reshape(b, f, -1, 2)
        sampled_mask = sampled_mask.reshape(b, f, -1)

        vis_tracks = VisualizerCoTracker(pad_value=120, linewidth=1, save_dir=save_dir)
        vis_flows_video = vis_tracks.visualize(batch.videos.flip(dims=[1]) * 255, sampled_xy, sampled_mask, filename=flow_backward_filename)

    # pdb.set_trace()


class ModelWrapperPretrain(LightningModule):
    def __init__(
        self,
        cfg: ModelWrapperPretrainCfg,
        cfg_flow: FlowPredictorCfg,
        model: Model,
        cfg_track: TrackPredictorCfg | None,
        losses: list[Loss] | None,
        visualizers: list[Visualizer] | None,
        enable_checkpoints_after: int | None,
        abandon_first: bool = False,
        device=None
    ) -> None:
        super().__init__()
        self.cfg = cfg
        device = get_device()
        if device == "cuda":
            device = device + ":" + str(self.global_rank)
        print("get flow_predictor...")
        print(f"self.global_rank={self.global_rank}, device={device}")
        self.flow_predictor = get_flow_predictor(cfg_flow)
        print("get track_predictor...")
        if cfg_track is not None:
            self.track_predictor = get_track_predictor(cfg_track)
        else:
            self.track_predictor = None
        print(f"device: {device}")
        self.model = model
        self.losses = losses
        self.visualizers = visualizers
        self.enable_checkpoints_after = 0 if enable_checkpoints_after is None else enable_checkpoints_after
        self.abandon_first = abandon_first

    @torch.no_grad()
    def preprocess_batch(self, batch_dict: dict) -> tuple[Batch, Flows]:
        batch_dict.pop("frame_paths", None)
        batch = Batch(**batch_dict)

        if batch.videos.ndim == 5:
            b, _, _, h, w = batch.videos.shape
        else:
            # For Camera Geometry Consistency Loss
            b, n, f, c, h, w = batch.videos.shape
            assert n == 2
            batch.videos = batch.videos.reshape(b*n, f, c, h, w)
            batch.depths = batch.depths.reshape(b*n, f, h, w)
            batch.flys = batch.flys.reshape(b*n, f, h, w)
            batch.masks = batch.masks.reshape(b*n, f, h, w)
            batch.indices = batch.indices.reshape(b*n, f)
            batch.pcds = batch.pcds.reshape(b*n, f, h, w, 3)
            batch.hoi_masks = batch.hoi_masks.reshape(b*n, f, h, w)
            batch.gt_extrinsics = batch.gt_extrinsics.reshape(b*n, f, 4, 4)
            batch.use_gt_depth = batch.use_gt_depth.reshape(b*n,)
            batch.scenes = [x for x in batch.scenes for _ in range(2)]
            batch.datasets = [x for x in batch.datasets for _ in range(2)]

        flows = self.flow_predictor.compute_bidirectional_flow(batch, (h, w))
        tracks = generate_pretrain_video_tracks(self.track_predictor, batch, cache=self.cfg.cache_track)
        # visualize_batch_flows_tracks(batch, flows, tracks)
        return batch, flows, tracks


    def training_step(self, batch):

        batch, flows, tracks = self.preprocess_batch(batch)    
        model_output = self.model(batch, flows, self.global_step)

        total_loss = 0
        for loss_fn in self.losses:
            loss, loss_package = loss_fn.forward(batch, flows, tracks, model_output, self.current_epoch, return_unweighted=False)
            for k, v in loss_package.items():
                self.log(f"train/loss/{k}", loss_fn.cfg.weight * v, sync_dist=True)
            total_loss = total_loss + loss
            
        self.log(f"train/loss/total_loss", total_loss, sync_dist=True)
        return total_loss

    def validation_step(self, batch):

        if (self.current_epoch >= self.enable_checkpoints_after) and (self.abandon_first is False):
            batch, flows, tracks = self.preprocess_batch(batch)

            # Compute depths, poses, and intrinsics using the model.
            with torch.no_grad():
                model_output = self.model(batch, flows, self.global_step)

            self.log(f"val/focal_x", model_output.intrinsics[0,0,0,0], on_epoch=True, sync_dist=True)
            self.log(f"val/focal_y", model_output.intrinsics[0,0,1,1], on_epoch=True, sync_dist=True)
            self.log(f"val/principle_x", model_output.intrinsics[0,0,0,2], on_epoch=True, sync_dist=True)
            self.log(f"val/principle_y", model_output.intrinsics[0,0,1,2], on_epoch=True, sync_dist=True)

            # Compute and log the loss.
            total_loss = 0
            loss_packages = []
            for loss_fn in self.losses:
                (loss, uloss), loss_package = loss_fn.forward(batch, flows, tracks, model_output, self.current_epoch, return_unweighted=True)
                for k, v in loss_package.items():
                    self.log(f"val/loss/{k}", v, sync_dist=True)
                total_loss = total_loss + loss
            
            self.log(f"val/loss/total_loss", total_loss, on_epoch=True, sync_dist=True)
            return total_loss
        else:
            self.log(f"val/loss/total_loss", torch.inf, on_epoch=True, sync_dist=True)
            self.abandon_first = False
            return None

        return model_output


    def inference(self, batch, masks_hoi_aux=None, return_flow=False):

        batch = BatchInference(**batch)
        b, _, _, h, w = batch.videos.shape
        flows = self.flow_predictor.compute_bidirectional_flow(batch, (h, w))
        with torch.no_grad():
            model_output = self.model(batch, flows, self.global_step, masks_hoi_aux=masks_hoi_aux)
        if return_flow is True:
            return model_output, flows.forward, flows.forward_mask
        else:
            return model_output


    def inference_flows(self, batch):

        batch = BatchInference(**batch)
        b, _, _, h, w = batch.videos.shape
        flows = self.flow_predictor.compute_bidirectional_flow(batch, (h, w))
        return flows, flows.forward, flows.forward_mask, batch


    def inference_output(self, batch, flows, masks_hoi_aux=None):

        with torch.no_grad():
            model_output = self.model(batch, flows, self.global_step, masks_hoi_aux=masks_hoi_aux)
        return model_output

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return optim.Adam(self.model.parameters(), lr=self.cfg.lr)
