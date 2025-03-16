from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from typing import Generic, TypeVar, Optional

from jaxtyping import Bool, Float
from torch import Tensor, nn

from ..misc.manipulable import Manipulable

T = TypeVar("T")


def sample_image_grid_tracker(
    shape,
    device: torch.device = torch.device("cpu"),
):
    """Get normalized (range 0 to 1) coordinates and integer indices for an image."""
    indices = [torch.arange(length, device=device) for length in shape]
    stacked_indices = torch.stack(torch.meshgrid(*indices, indexing="ij"), dim=-1)
    coordinates = [(idx + 0.5) / length for idx, length in zip(indices, shape)]
    coordinates = reversed(coordinates)
    coordinates = torch.stack(torch.meshgrid(*coordinates, indexing="xy"), dim=-1)
    return coordinates


@dataclass
class Tracks(Manipulable):
    xy: Optional[Float[Tensor, "batch frame point 2"]] = None
    visibility: Optional[Bool[Tensor, "batch frame point"]] = None

    # This is the first frame in the track sequence, not the query frame used to
    # generate the sequence, which is often different.
    start_frame: int = 0

    def build_from_track_list(self, track_list, device='cpu'):
        self.xy = torch.concatenate([track.xy for track in track_list], dim=0)
        self.visibility = torch.concatenate([track.visibility for track in track_list], dim=0)


class TrackPredictor(nn.Module, ABC, Generic[T]):
    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        videos: Float[Tensor, "batch frame 3 height width"],
        query_frame: int,
    ) -> Tracks:
        pass
