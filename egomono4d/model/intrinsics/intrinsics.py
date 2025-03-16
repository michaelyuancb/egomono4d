from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from jaxtyping import Float
from torch import Tensor, nn

from ...dataset.types import Batch
from ...flow.flow_predictor import Flows
from ...tracking.track_predictor import Tracks
from ..backbone.backbone import BackboneOutput

T = TypeVar("T")


class Intrinsics(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        batch: Batch,
        flows: Flows | list[Tracks],
        backbone_output: BackboneOutput,
        global_step: int,
    ) -> Float[Tensor, "batch frame 3 3"]:
        pass
