from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
import pdb
from jaxtyping import Float
from torch import Tensor, nn

from ..dataset.types import Batch
from ..flow import Flows
from ..model.model import ModelOutput
from ..tracking import Tracks


@dataclass
class LossCfgCommon:
    enable_after: int
    weight: float


T = TypeVar("T", bound=LossCfgCommon)


class Loss(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        batch: Batch,
        flows: Flows | None,
        tracks: list[Tracks] | None,
        model_output: ModelOutput,
        current_epoch: int,
        return_unweighted=False
    ) -> Float[Tensor, ""]:

        if current_epoch < self.cfg.enable_after:
            zr_loss = torch.tensor(0, dtype=torch.float32, device=batch.videos.device)
            if return_unweighted is True:
                return (zr_loss, zr_loss), None
            else:
                return zr_loss, None

        loss, loss_package = self.compute_unweighted_loss(
            batch, flows, tracks, model_output, current_epoch, return_unweighted
        )
        if return_unweighted is True:
            return (self.cfg.weight * loss, 100 * loss), loss_package
        else:
            return self.cfg.weight * loss, loss_package

    @abstractmethod
    def compute_unweighted_loss(
        self,
        batch: Batch,
        flows: Flows,
        tracks: list[Tracks] | None,
        model_output: ModelOutput,
        global_step: int,
    ) -> tuple[Float[Tensor, ""], dict]:
        pass
