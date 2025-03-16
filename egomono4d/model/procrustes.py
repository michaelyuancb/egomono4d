import torch
import pdb
from einops import einsum
from jaxtyping import Float
from torch import Tensor


def align_rigid(
    p: Float[Tensor, "*batch point 3"],
    q: Float[Tensor, "*batch point 3"],
    weights: Float[Tensor, "*batch point"],
) -> Float[Tensor, "*batch 4 4"]:
    """Compute a rigid transformation that, when applied to p, minimizes the weighted
    squared distance between transformed points in p and points in q. See "Least-Squares
    Rigid Motion Using SVD" by Olga Sorkine-Hornung and Michael Rabinovich for more
    details (https://igl.ethz.ch/projects/ARAP/svd_rot.pdf).
    """

    device = p.device
    dtype = p.dtype
    *batch, _, _ = p.shape

    # 1. Compute the centroids of both point sets.
    weights_normalized = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
    p_centroid = (weights_normalized[..., None] * p).sum(dim=-2)
    q_centroid = (weights_normalized[..., None] * q).sum(dim=-2)

    # 2. Compute the centered vectors.
    p_centered = p - p_centroid[..., None, :]
    q_centered = q - q_centroid[..., None, :]

    # 3. Compute the 3x3 covariance matrix.
    covariance = (q_centered * weights[..., None]).transpose(-1, -2) @ p_centered

    # 4. Compute the singular value decomposition and then the rotation.
    u, _, vt = torch.linalg.svd(covariance)
    s = torch.eye(3, dtype=dtype, device=device)
    s = s.expand((*batch, 3, 3)).contiguous()
    s[..., 2, 2] = (u.det() * vt.det()).sign()
    rotation = u @ s @ vt

    # 5. Compute the optimal translation.
    translation = q_centroid - einsum(rotation, p_centroid, "... i j, ... j -> ... i")

    # Compose the results into a single transformation matrix.
    shape = (*rotation.shape[:-2], 4, 4)
    r = torch.eye(4, dtype=torch.float32, device=device).expand(shape).contiguous()
    r[..., :3, :3] = rotation
    t = torch.eye(4, dtype=torch.float32, device=device).expand(shape).contiguous()
    t[..., :3, 3] = translation

    return t @ r


def align_rigid_unweighted(
    p: Float[Tensor, "*batch point 3"],
    q: Float[Tensor, "*batch point 3"]
) -> Float[Tensor, "*batch 4 4"]:
    device = p.device
    dtype = p.dtype
    *batch, num_points, _ = p.shape

    p_centroid = p.mean(dim=-2)
    q_centroid = q.mean(dim=-2)

    p_centered = p - p_centroid[..., None, :]
    q_centered = q - q_centroid[..., None, :]

    covariance = (q_centered.transpose(-1, -2) @ p_centered)

    u, _, vt = torch.linalg.svd(covariance)
    s = torch.eye(3, dtype=dtype, device=device)
    s = s.expand((*batch, 3, 3)).contiguous()
    s[..., 2, 2] = (u.det() * vt.det()).sign()
    rotation = u @ s @ vt

    translation = q_centroid - einsum(rotation, p_centroid, "... i j, ... j -> ... i")

    shape = (*rotation.shape[:-2], 4, 4)
    r = torch.eye(4, dtype=torch.float32, device=device).expand(shape).contiguous()
    r[..., :3, :3] = rotation
    t = torch.eye(4, dtype=torch.float32, device=device).expand(shape).contiguous()
    t[..., :3, 3] = translation
    return t @ r


def align_scaled_rigid_unweighted(
    p: Float[Tensor, "*batch point 3"],
    q: Float[Tensor, "*batch point 3"]
) -> Float[Tensor, "*batch 4 4"]:
    device = p.device
    dtype = p.dtype
    *batch, num_points, _ = p.shape

    p_centroid = p.mean(dim=-2)
    q_centroid = q.mean(dim=-2)

    p_centered = p - p_centroid[..., None, :]
    q_centered = q - q_centroid[..., None, :]

    covariance = (q_centered.transpose(-1, -2) @ p_centered)

    u, _, vt = torch.linalg.svd(covariance)
    s = torch.eye(3, dtype=dtype, device=device)
    s = s.expand((*batch, 3, 3)).contiguous()
    s[..., 2, 2] = (u.det() * vt.det()).sign()
    rotation = u @ s @ vt

    rotated_p_centered = torch.einsum('...ij,...nj->...ni', rotation, p_centered)
    numerator = torch.einsum('...ni,...ni->...', q_centered, rotated_p_centered)
    denominator = (p_centered ** 2).sum(dim=(-2, -1))
    scale = numerator / (denominator + 1e-8)  # Add a small epsilon for numerical stability

    translation = q_centroid - scale[..., None] * (rotation @ p_centroid[..., None]).squeeze(-1)

    transformation_matrix = torch.eye(4, dtype=dtype, device=device).expand((*batch, 4, 4)).contiguous()
    transformation_matrix[..., :3, :3] = rotation * scale[..., None, None]
    transformation_matrix[..., :3, 3] = translation

    return transformation_matrix, scale


def align_scaled_rigid(
    p: Float[Tensor, "*batch point 3"],
    q: Float[Tensor, "*batch point 3"],
    weights: Float[Tensor, "*batch point"]=None,
) -> Float[Tensor, "*batch 4 4"]:
    
    device = p.device
    dtype = p.dtype
    *batch, _, _ = p.shape

    # 1. Compute the centroids of both point sets.
    weights_normalized = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
    p_centroid = (weights_normalized[..., None] * p).sum(dim=-2)
    q_centroid = (weights_normalized[..., None] * q).sum(dim=-2)

    # 2. Compute the centered vectors.
    p_centered = p - p_centroid[..., None, :]
    q_centered = q - q_centroid[..., None, :]  # (b, n, 3)

    # 3. Compute the 3x3 covariance matrix.
    covariance = (q_centered * weights[..., None]).transpose(-1, -2) @ p_centered

    # 4. Compute the singular value decomposition and then the rotation.
    u, _, vt = torch.linalg.svd(covariance)
    s = torch.eye(3, dtype=dtype, device=device)
    s = s.expand((*batch, 3, 3)).contiguous()
    s[..., 2, 2] = (u.det() * vt.det()).sign()
    rotation = u @ s @ vt

    # 5. Compute the scale factor s.
    # pdb.set_trace()
    rotated_p_centered = torch.einsum('...ij,...nj->...ni', rotation, p_centered)
    numerator = torch.einsum('...ni,...ni->...', q_centered * weights[..., None], rotated_p_centered)
    denominator = (p_centered ** 2 * weights[..., None]).sum(dim=(-2, -1))
    scale = numerator / (denominator + 1e-8)  # Add a small epsilon for numerical stability

    # 6. Compute the translation.
    translation = q_centroid - scale[..., None] * (rotation @ p_centroid[..., None]).squeeze(-1)

    # 7. Construct the transformation matrix.
    transformation_matrix = torch.eye(4, dtype=dtype, device=device).expand((*batch, 4, 4)).contiguous()
    transformation_matrix[..., :3, :3] = rotation * scale[..., None, None]
    transformation_matrix[..., :3, 3] = translation

    return transformation_matrix, scale


def align_scaled_rigid_timeavg(
    p: Float[Tensor, "*batch point 3"],
    q: Float[Tensor, "*batch point 3"],
    weights: Float[Tensor, "*batch point"]=None,
) -> Float[Tensor, "*batch 4 4"]:
    
    device = p.device
    dtype = p.dtype
    *batch, _, _ = p.shape

    # 1. Compute the centroids of both point sets.
    weights_normalized = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
    p_centroid = (weights_normalized[..., None] * p).sum(dim=-2)
    q_centroid = (weights_normalized[..., None] * q).sum(dim=-2)

    # 2. Compute the centered vectors.
    p_centered = p - p_centroid[..., None, :]
    q_centered = q - q_centroid[..., None, :]  # (b, n, 3)

    # 3. Compute the 3x3 covariance matrix.
    covariance = (q_centered * weights[..., None]).transpose(-1, -2) @ p_centered

    # 4. Compute the singular value decomposition and then the rotation.
    u, _, vt = torch.linalg.svd(covariance)
    s = torch.eye(3, dtype=dtype, device=device)
    s = s.expand((*batch, 3, 3)).contiguous()
    s[..., 2, 2] = (u.det() * vt.det()).sign()
    rotation = u @ s @ vt

    # 5. Compute the scale factor s.
    # pdb.set_trace()
    rotated_p_centered = torch.einsum('...ij,...nj->...ni', rotation, p_centered)
    numerator = torch.einsum('...ni,...ni->...', q_centered * weights[..., None], rotated_p_centered)
    denominator = (p_centered ** 2 * weights[..., None]).sum(dim=(-2, -1))
    scale = numerator / (denominator + 1e-8)  # Add a small epsilon for numerical stability

    # 6. Compute the translation.
    translation = q_centroid - scale[..., None] * (rotation @ p_centroid[..., None]).squeeze(-1)

    # 7. Construct the transformation matrix.
    transformation_matrix = torch.eye(4, dtype=dtype, device=device).expand((*batch, 4, 4)).contiguous()
    f = scale.shape[-1]
    scale = scale.mean(dim=-1, keepdim=True)
    transformation_matrix[..., :3, :3] = rotation * scale[..., None, None]
    transformation_matrix[..., :3, 3] = translation

    return transformation_matrix, scale


