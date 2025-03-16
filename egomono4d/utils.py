import pickle
from einops import einsum, rearrange
from .model.projection import sample_image_grid, unproject, homogenize_points


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def save_pickle(pickle_file, data):
    with open(pickle_file, 'wb') as pfile:
        pickle.dump(data, pfile)


def batch_recover_pointclouds_sequence(depths, intrinsics, extrinsics, target_frame=0):
    b, f, h, w = depths.shape
    xy, _ = sample_image_grid((h, w), device=depths.device)
    gt_pcds_unp = unproject(xy, depths, rearrange(intrinsics, "b f i j -> b f () () i j"))

    extrinsics_source = rearrange(extrinsics, "b fs i j -> b fs () () i j")
    extrinsics_target = rearrange(extrinsics[:, target_frame:target_frame+1], "b ft i j -> b () ft () i j")
    relative_transformations = extrinsics_target.inverse() @ extrinsics_source

    pcds = einsum(
        relative_transformations,
        homogenize_points(gt_pcds_unp),
        "... i j, ... j -> ... i",
    )[..., :3]

    return pcds