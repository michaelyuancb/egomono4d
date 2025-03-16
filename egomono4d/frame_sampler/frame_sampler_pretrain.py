from dataclasses import dataclass
# from typing import Literal
from typing_extensions import Literal

import torch
from jaxtyping import Int64
from torch import Tensor
import random

from .frame_sampler import FrameSampler


class FrameSamplerPretrainNeighbor(FrameSampler):
    def sample(
        self,
        num_frames_in_video: int,
        device: torch.device,
    ) -> Int64[Tensor, " frame"]:
        # If the video doesn't have enough frames, just repeat the last frame.
        if num_frames_in_video < self.num_frames:
            indices = torch.arange(self.num_frames, device=device)
            indices[indices >= num_frames_in_video] = num_frames_in_video - 1
            return indices

        # If the video has enough frames, pick a random starting point.
        if self.stage == 'train':
            start = torch.randint(0, num_frames_in_video - self.num_frames + 1, tuple())
        else:
            start = 0
        return torch.arange(start, start + self.num_frames, device=device)


class FrameSamplerPretrainInterval(FrameSampler):
    def sample(
        self,
        num_frames_in_video: int,
        device: torch.device,
        max_interval: int=1
    ) -> Int64[Tensor, " frame"]:
        # If the video doesn't have enough frames, just repeat the last frame.
        if num_frames_in_video < self.num_frames:
            indices = torch.arange(self.num_frames, device=device)
            indices[indices >= num_frames_in_video] = num_frames_in_video - 1
            return indices
        
        if num_frames_in_video - 1 < max_interval * (self.num_frames-1):
            max_interval = (num_frames_in_video - 1) // (self.num_frames-1)

        if self.stage == 'train':
            interval = random.randint(1, max_interval)
            start = torch.randint(0, (num_frames_in_video-1)-interval*(self.num_frames-1), tuple())
        else:
            interval = (max_interval + 1) // 2    # we test the middle state as representative performance (between easiest and hardest).
            start = ((num_frames_in_video-1)-interval*(self.num_frames-1)) // 2    # fixed it to eliminate uncertainty.
        # print(f"interval: {interval}")
        res_idx = torch.tensor([start+i*interval for i in range(self.num_frames)], device=device)
        # print(f"max_interval={max_interval}, interval={interval}, res={res_idx}")
        return res_idx