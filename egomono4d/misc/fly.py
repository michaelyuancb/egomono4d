import numpy as np
import cv2
import torch


def detect_flying_pixels(depth_map, threshold=10):
    # # depth_map: (h, w)

    depth_dx, depth_dy = np.gradient(depth_map)
    depth_grad = np.sqrt(depth_dx**2 + depth_dy**2)
    flying_pixels = depth_grad > threshold

    return flying_pixels


def detect_sequence_flying_pixels(depth_sequence, threshold=10):
    """
    Process a sequence of depth maps to detect flying pixels.

    Parameters:
    depth_sequence (numpy.ndarray): The input depth sequence with shape (f, h, w).
    threshold (int): The threshold for detecting depth discontinuities.

    Returns:
    numpy.ndarray: A binary sequence where flying pixels are marked as 1.
    """
    f, h, w = depth_sequence.shape
    flying_pixels_sequence = np.zeros((f, h, w), dtype=np.uint8)

    for i in range(f):
        flying_pixels_sequence[i] = detect_flying_pixels(depth_sequence[i], threshold)

    return flying_pixels_sequence


def calculate_edge_scale_torch(surfaces, fly_masks):
    """
    Get the scale of the point cloud defined with mean edge distance.
    scale = \sigma_{(i,j) in edges} ||p_i - p_j||

    Inputs:
        surfaces:  torch.Tensor[batch*, h, w, 3]
        fly_masks:   torch.Tensor[batch*, h, w]

    Return:
        scale: torch.Tensor[batch*]
    """

    dist_right = torch.norm(surfaces[..., :, 1:, :] - surfaces[..., :, :-1, :], dim=-1)
    dist_down = torch.norm(surfaces[..., 1:, :, :] - surfaces[..., :-1, :, :], dim=-1)
    mask_right = fly_masks[..., :, 1:] * fly_masks[..., :, :-1]
    mask_down = fly_masks[..., 1:, :] * fly_masks[..., :-1, :]

    scale_right = (dist_right * mask_right).sum(dim=[-1,-2]) / mask_right.sum(dim=[-1,-2])
    scale_left = (dist_down * mask_down).sum(dim=[-1,-2]) / mask_down.sum(dim=[-1,-2])

    scale_edge = (scale_right + scale_left) * 0.5
    return scale_edge


def calculate_scale_pts(pts):  # (n, 3)
    
    n, _ = pts.shape 
    surfaces_flat = pts[None]

    centroids = torch.mean(surfaces_flat, dim=1, keepdim=True)
    centered_points = surfaces_flat - centroids

    cov_matrices = torch.bmm(centered_points.transpose(1, 2), centered_points) / n
    eigenvalues, _ = torch.linalg.eigh(cov_matrices)  # (batch*, 3)
    scale = torch.sqrt(eigenvalues[:, -1])            # pick the largest PCA item

    return scale[0]


def calculate_scale_torch(surfaces):
    """
    Get the scale of the point cloud defined with PCA analysis.
    scale = \sigma_{(i,j) in edges} ||p_i - p_j||

    Inputs:
        surfaces:  torch.Tensor[batch*, h, w, 3]
        fly_masks:   torch.Tensor[batch*, h, w]

    Return:
        scale: torch.Tensor[batch*]
    """
    
    batch_shape = surfaces.shape[:-3]
    h, w = surfaces.shape[-3:-1]
    surfaces_flat = surfaces.view(-1, h * w, 3)  # (batch*, h*w, 3)

    centroids = torch.mean(surfaces_flat, dim=1, keepdim=True)
    centered_points = surfaces_flat - centroids

    cov_matrices = torch.bmm(centered_points.transpose(1, 2), centered_points) / (h * w)
    eigenvalues, _ = torch.linalg.eigh(cov_matrices)  # (batch*, 3)
    scale = torch.sqrt(eigenvalues[:, -1])            # pick the largest PCA item
    scale = scale.reshape(batch_shape)

    return scale



