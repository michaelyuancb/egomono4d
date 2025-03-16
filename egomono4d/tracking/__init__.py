from dataclasses import dataclass
from pathlib import Path

import torch
import time
import copy
import pdb
from jaxtyping import Float, Int64
from torch import Tensor
from tqdm import trange

from ..dataset.types import Batch
from ..misc.disk_cache import make_cache
from ..misc.nn_module_tools import convert_to_buffer
from .track_predictor import TrackPredictor, Tracks
from .track_predictor_cotracker import (
    TrackPredictorCoTracker,
    TrackPredictorCoTrackerCfg,
)
TRACKERS = {
    "cotracker": TrackPredictorCoTracker,
}

TrackPredictorCfg = TrackPredictorCoTrackerCfg


def get_track_predictor(cfg: TrackPredictorCfg) -> TrackPredictor:
    tracker = TRACKERS[cfg.name](cfg)
    convert_to_buffer(tracker, persistent=False)
    return tracker


def get_cache_key(
    dataset: str,
    scene: str,
    indices: Int64[Tensor, " frame"],
) -> tuple[str, str, int, int, int, int]:
    first, *_, last = indices
    return (
        dataset,
        scene,
        first.item(),
        last.item(),
    )


def generate_video_tracks(
    tracker: TrackPredictor,
    videos: Float[Tensor, "batch frame 3 height width"],
    interval: int,
    radius: int,
) -> list[Tracks]:
    segment_tracks = []

    _, f, _, _, _ = videos.shape
    for middle_frame in trange(0, f, interval, desc="Computing tracks"):
        # Retrieve the video segment we want to compute tracks for.
        start_frame = max(0, middle_frame - radius)
        end_frame = min(f, middle_frame + radius + 1)
        segment = videos[:, start_frame:end_frame]

        # Compute tracks on the segment, then mark the tracks with the segment's
        # starting frame so that they can be matched to the segment.
        tracks = tracker.forward(segment, middle_frame - start_frame)
        tracks.start_frame = start_frame
        segment_tracks.append(tracks)

    return segment_tracks


def generate_pretrain_video_tracks_cache(
    tracker: TrackPredictor,
    videos: Float[Tensor, "batch frame 3 height width"],
) -> list[Tracks]:

    segment_tracks = []
    _, f, _, _, _ = videos.shape

    track_base = [0]   
    # track_base = [i for i in range(f)]         
    # we only compute 0->num_frames tracking for pretraining, and leave flip in data_augmentation. 
    
    num_frames = videos.shape[1]
    if num_frames < 5:
        videos = torch.concat([videos]+[videos[:,-1:]]*(5-num_frames), dim=1)
    
    with torch.no_grad():
        for middle_frame in track_base:
            tracks = tracker.forward(videos, middle_frame)
            if num_frames < 5:
                tracks.xy = tracks.xy[:, :num_frames]
                tracks.visibility = tracks.visibility[:, :num_frames]
            tracks.start_frame = 0
            segment_tracks.append(tracks)

    return segment_tracks


def generate_pretrain_video_tracks(
    tracker: TrackPredictor,
    batch: Batch,
    cache: bool
) -> list[Tracks]:
    if cache is False:
        return generate_pretrain_video_tracks_cache(tracker, batch.videos)
    else:
        # raise ValueError("Please set cache=False.")
        try:
            device = batch.videos.device

            batch_size = len(batch.datasets)
            cache_key_list = []
            disk_cache_dict = dict()
            for i in range(batch_size):
                dataset = batch.datasets[i]
                cache_key = get_cache_key(
                    dataset,
                    batch.scenes[i],
                    batch.indices[i],
                )
                if dataset not in disk_cache_dict.keys():
                    disk_cache_dict[dataset] = make_cache(f"{tracker.cache_dir}/cotracker_result_cache/{dataset}")
                cache_key_list.append(cache_key)

            track_result_single = []
            for i in range(batch_size):
                dataset = batch.datasets[i]
                track_result = disk_cache_dict[dataset](
                    cache_key_list[i],
                    device,
                    lambda: generate_pretrain_video_tracks_cache(tracker, batch.videos[i:i+1])
                )
                track_result_single.append(track_result)
            track_merge_list = []
            # pdb.set_trace()
            n_track = len(track_result_single[0])
            for i in range(n_track):
                track_result_i = [track_result_single[j][i] for j in range(batch_size)]
                track_merge = Tracks(None, None, 0)
                track_merge.build_from_track_list(track_result_i, device=batch.videos.device)
                track_merge_list.append(track_merge)
            return track_merge_list
        except:
            return generate_pretrain_video_tracks_cache(tracker, batch.videos)


def generate_pretrain_video_flows_tracks(
    tracker: TrackPredictor,
    batch: Batch,
    cache: bool
) -> tuple[list[Tracks], list[Tracks]]:

    segment_tracks = []
    _, f, _, _, _ = batchs.videos.shape

    flows = []

    track_base = [0]            
    # we only compute 0->num_frames tracking for pretraining, and leave flip in data_augmentation. 
    with torch.no_grad():
        for middle_frame in track_base:
            tracks = tracker.forward(batch.videos, middle_frame)
            tracks.start_frame = 0
            segment_tracks.append(tracks)

        flow = copy.deepcopy(tracks)
        flow.xy = flow.xy[:, :1]
        flow.visibility = flow.visibility[:, :1]
        flows.append(flow)
        
        for st in range(1, f-1):
            flow = tracker.forward(batch.videos[:, st:st+1], 0)
            flows.append(flow)


    return flow, segment_tracks


@dataclass
class TrackPrecomputationCfg:
    cache_path: Path | None
    interval: int
    radius: int


def compute_tracks(
    batch: Batch,
    device: torch.device,
    tracking_cfg: TrackPredictorCfg,
    precomputation_cfg: TrackPrecomputationCfg,
) -> list[Tracks]:
    # Set up the tracker.
    tracker = get_track_predictor(tracking_cfg)
    tracker.to(device)

    # Since we only use tracks for overfitting, assert that the batch size is 1.
    b, _, _, _, _ = batch.videos.shape
    assert b == 1

    cache_key = get_cache_key(
        batch.datasets[0],
        batch.scenes[0],
        batch.indices[0],
        precomputation_cfg.interval,
        precomputation_cfg.radius,
    )
    disk_cache = make_cache(precomputation_cfg.cache_path)
    return disk_cache(
        cache_key,
        lambda: generate_video_tracks(
            tracker,
            batch.videos[:1].to(device),
            precomputation_cfg.interval,
            precomputation_cfg.radius,
        ),
    )
