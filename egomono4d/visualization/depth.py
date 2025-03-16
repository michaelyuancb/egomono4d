from jaxtyping import Float
from torch import Tensor

from .color import apply_color_map_to_image


def color_map_depth(
    depth: Float[Tensor, "batch height width"],
    cmap: str = "inferno",
    invert: bool = True,
    log_first: bool = False
) -> Float[Tensor, "batch 3 height width"]:
    mask = (depth == 0)
    if log_first is True:
        # for depth estimation, we first get log for convinient visualization.
        depth = depth.log()
    # Normalize the depth.
    far = depth.max()
    depth = depth + mask * 1e9
    near = depth.min()
    depth = (depth - near) / (far - near)
    depth = depth.clip(min=0, max=1)
    depth[mask] = 0
    if invert:
        depth = 1 - depth
    return apply_color_map_to_image(depth, cmap)
