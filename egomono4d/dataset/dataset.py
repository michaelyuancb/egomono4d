from dataclasses import dataclass
import torch
import random
import pdb
import os
import numpy as np
from einops import einsum, rearrange
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as tf
from typing import Optional
try:
    import torchvision.transforms.v2 as tfv2  
except:
    print("Not import tfv2 for video augmentation.")

try:
    EVAL = os.environ['EVAL_MODE']
except:
    EVAL = 'False'
if EVAL in ['True']:
    FlowPredictor = None
else:
    from ..flow.flow_predictor import FlowPredictor

from .types import Stage
from ..frame_sampler.frame_sampler import FrameSampler
from ..model.projection import unproject, sample_image_grid
from ..misc.fly import detect_sequence_flying_pixels

FLY_THRESHOLD = 0.05



def egomono4d_data_augmentation(data, color_augmentor, pcd_calc=False):
    data['videos'][0] = color_augmentor(data['videos'][0])
    data['videos'][1] = color_augmentor(data['videos'][1])
    
    # flip_dims = [0, 1, 2]   
    flip_dims = [0, 2]         # ignore "height" dimension.  (F, H, W)
    pcd_transform = False
    for flip_dim in flip_dims:
        frame_flip = random.randint(0, 2)
        if frame_flip == 0:
            # pdb.set_trace()
            if flip_dim > 0:
                data['videos'] = torch.flip(data['videos'], [flip_dim+2])  # (n, f, c, h, w)
            if flip_dim == 0:
                data['indices'] = torch.flip(data['indices'], [1])         # (n, f, )
                data['videos'] = torch.flip(data['videos'], [1])
            data['depths'] = torch.flip(data['depths'], [flip_dim+1])        # (n, f, h, w)
            data['masks'] = torch.flip(data['masks'], [flip_dim+1])          # (n, f, h, w)
            data['flys'] = torch.flip(data['flys'], [flip_dim+1])            # (n, f, h, w)
            data['datasets'] = data['datasets'] + f"_f{flip_dim}"
            if flip_dim == 1:
                data['intrinsics'][..., 1, 2] = 1.0 - data['intrinsics'][..., 1, 2]     
                pcd_transform = True  
            elif flip_dim == 2:
                data['intrinsics'][..., 0, 2] = 1.0 - data['intrinsics'][..., 0, 2]       
                pcd_transform = True
    
    return data


@dataclass
class DatasetCfgCommon:
    scene: Optional[str] = None
    cache_dir: Optional[str] = None
    resize_shape: Optional[tuple] = None
    patch_size: Optional[int] = None
    num_frames: Optional[int] = None
    all_frames: bool = False
    use_gt_depth: bool = False

    mask_estimation: Optional[list] = None   # ['epipolar', 'egohos', 'egomodel', 'maskrcnn'] 
    mask_flow_model: Optional[FlowPredictor] = None
    mask_binary_open_value: Optional[float] = None 

    frame_sampler: str = None
    frame_max_interval: Optional[int] = None


class Datasetegomono4d(Dataset):

    def __init__(
        self,
        cfg: DatasetCfgCommon,
        stage: Stage,
        frame_sampler: FrameSampler,
        global_rank: int,
        world_size: int,
        debug=False
    ) -> None:

        self.use_consistency_loss = cfg.use_consistency_loss
        self.world_size = world_size
        self.global_rank = global_rank
        self.cfg = cfg
        self.frame_sampler = frame_sampler
        self.all_frames = cfg.all_frames
        self.frame_max_interval = cfg.frame_max_interval
        self.stage = stage
        if stage in ['train']:
            self.color_jitter = tfv2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25)


    
    def get_aux_filename(self, rgb_filename):
        rgb_filename = rgb_filename.split('/')
        base = '/'.join(rgb_filename[:-1])
        idx = int(rgb_filename[-1].split('.')[0][3:])
        return {"edep": base + '/edep' + str(idx) + ".npy", 
                "intrinsics": base + '/edep' + str(idx) + "_canonical_intrinsic.npy"}


    def get_item_wrapper_single(self, index: int):
        meta = self.sequence[index]
        num_frames = len(meta["rgbs"])
        # Run the frame sampler.

        if (self.all_frames is True) and (self.stage in ['val', 'test']):
            indices = torch.Tensor([i for i in range(num_frames)], device=torch.device('cpu')).int()
        else:
            indices = self.frame_sampler.sample(num_frames, torch.device("cpu"), self.frame_max_interval)
            # print(f"stage={self.stage}, indices={indices}")

        rgbs_fp = [meta['rgbs'][i] for i in indices]
        videos = torch.stack([tf.ToTensor()(Image.open(path)) for path in rgbs_fp])

        deps_fp = []
        intr_fp = []
        for rgb_fp in rgbs_fp:
            aux_fp = self.get_aux_filename(rgb_fp)
            deps_fp.append(aux_fp['edep'])
            intr_fp.append(aux_fp['intrinsics'])
        depths = torch.stack([torch.Tensor(np.load(path)) for path in deps_fp])
        intrinsics = torch.stack([torch.Tensor(np.load(path)) for path in intr_fp])

        fly_mask = detect_sequence_flying_pixels(depths.numpy(), threshold=FLY_THRESHOLD)
        fly_mask = 1.0 - torch.Tensor(fly_mask)

        masks_fp = [meta['emasks'][i] for i in indices]    # 1 for valid procrutes, 0 for invalid
        masks = []
        for fp in masks_fp:
            msk = np.load(fp)
            if msk.ndim == 3: msk = msk.sum(2)
            masks.append(msk)
        masks = torch.Tensor(np.stack(masks))

        f = masks.shape[0]

        data = {
            "videos": videos,      # (F, C, H, W)
            "depths": depths,    
            "flys": fly_mask,
            "masks": torch.minimum(masks, fly_mask), 
            "indices": indices,
            "scenes": meta['id'],
            "datasets": "data",
            "intrinsics": intrinsics,
            "use_gt_depth": False,
            "hoi_masks": 1.0 - masks,
            "gt_depths": torch.zeros_like(masks), 
            "gt_intrinsics": torch.eye(3)[None].repeat(f,1,1),
            "gt_extrinsics": torch.eye(4)[None].repeat(f,1,1),
        }

        return data
    

    def get_item_wrapper(self, save_index):
        if self.stage in ['train', 'val'] and (self.use_consistency_loss is True):
            data_1 = self.get_item_wrapper_single(save_index)
            data_2 = self.get_item_wrapper_single(save_index)
            data = {
                "videos": torch.stack([data_1['videos'], data_2['videos']]),
                "depths": torch.stack([data_1['depths'], data_2['depths']]),
                "flys": torch.stack([data_1['flys'], data_2['flys']]),
                "masks": torch.stack([data_1['masks'], data_2['masks']]),
                "indices": torch.stack([data_1['indices'], data_2['indices']]),
                "scenes": data_1['scenes'],
                "datasets": data_1['datasets'],
                "intrinsics": torch.stack([data_1['intrinsics'], data_2['intrinsics']]),
                "use_gt_depth": torch.Tensor([data_1['use_gt_depth'], data_2['use_gt_depth']]),
                "hoi_masks": torch.stack([data_1['hoi_masks'], data_2['hoi_masks']]),
                "gt_extrinsics": torch.stack([data_1['gt_extrinsics'], data_2['gt_extrinsics']])
            }
            if self.stage == 'train':
                data = egomono4d_data_augmentation(data, self.color_jitter)
            _, _, _, h, w = data['videos'].shape
            xy, _ = sample_image_grid((h, w), data['videos'].device)
            surfaces = unproject(xy[None, None], data['depths'], rearrange(data['intrinsics'], "n f i j -> n f () () i j"),)        
            data['pcds'] = surfaces
            return data
        else:
            data = self.get_item_wrapper_single(save_index)
            _, _, h, w = data['videos'].shape
            xy, _ = sample_image_grid((h, w), data['videos'].device)
            surfaces = unproject(xy[None], data['depths'], rearrange(data['intrinsics'], "f i j -> f () () i j"),)        
            data['pcds'] = surfaces
            return data
        

    def __getitem__(self, index: int):
        # save get item function, there are some broken png in HOI4D dataset. 
        max_try = 50
        save_index = index
        while max_try > 0:
            max_try = max_try - 1
            try:
                data = self.get_item_wrapper(save_index)
                return data
            except Exception as e:
                new_index = (save_index + 10) % len(self.sequence)
                print(f"fail to get data[{save_index}] for {self.stage} stage, error: [{e}], retry data[{new_index}].")
                save_index = new_index


    def __len__(self) -> int:
        return len(self.sequence)