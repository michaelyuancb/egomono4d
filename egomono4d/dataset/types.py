from dataclasses import dataclass
# from typing import Literal
from typing_extensions import Literal
from typing import List, Union, Optional

from jaxtyping import Float, Int64, Int32
from torch import Tensor

from ..misc.manipulable import Manipulable

Stage = Literal["train", "test", "val"]


@dataclass
class Batch(Manipulable):
    videos: Float[Tensor, "batch frame 3 height width"]
    depths: Float[Tensor, "batch frame height width"]  
    pcds: Float[Tensor, "batch frame height width 3"] 
    flys: Float[Tensor, "batch frame height width"] 
    masks: Float[Tensor, "batch frame height width"] 
    indices: Int64[Tensor, "batch frame"]

    scenes: Union[List[str], str] 
    datasets: Union[List[str], str] 
    use_gt_depth: bool
    
    intrinsics: Optional[Float[Tensor, "batch frame 3 3"]] = None

    gt_depths: Optional[Float[Tensor, "batch frame height width"]] = None
    gt_intrinsics: Optional[Float[Tensor, "batch frame 3 3"]] = None
    gt_extrinsics: Optional[Float[Tensor, "batch frame 4 4"]] = None
    hoi_masks: Optional[Float[Tensor, "batch frame height width"]] = None


@dataclass
class BatchInference(Manipulable):
    videos: Float[Tensor, "batch frame 3 height width"]
    start_indice: Int32
    aux_masks: Optional[Float[Tensor, "batch frame height width"]]  = None
