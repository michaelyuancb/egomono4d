from typing import Any

from .frame_sampler import FrameSampler
from .frame_sampler_pretrain import FrameSamplerPretrainNeighbor, FrameSamplerPretrainInterval

FRAME_SAMPLER = {
    "pretrain_neighbor": FrameSamplerPretrainNeighbor,   # pick num_frames neighborhood
    "pretrain_interval": FrameSamplerPretrainInterval,   # pick random index (with random interval)
}


def get_frame_sampler(fs_name, num_frames, stage) -> FrameSampler[Any]:
    return FRAME_SAMPLER[fs_name](num_frames, stage)
