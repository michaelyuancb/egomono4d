from dataclasses import dataclass
# from typing import Literal
from typing_extensions import Literal
import multiprocessing as mp
import copy
from multiprocessing import current_process

import torch
import os
import time
import json
import random
import numpy as np

import torchvision.transforms as tf
from PIL import Image
import pdb
from tqdm import tqdm

from .dataset import DatasetCfgCommon
from .dataset import Datasetegomono4d

from .types import Stage
from ..misc.fly import detect_sequence_flying_pixels
from ..frame_sampler.frame_sampler import FrameSampler

try:
    EVAL = os.environ['EVAL_MODE']
except:
    EVAL = 'False'
if EVAL not in ['True']:
    import open3d as o3d
    from ..misc.data_util import pil_resize_to_center_crop, resize_crop_intrinisic, compute_patch_cropped_shape, canonicalize_intrinisic
    from ..misc.depth import get_depth_estimator


FLY_THRESHOLD = 0.05

@dataclass
class DatasetHOI4DCfg(DatasetCfgCommon):
    name: Literal["hoi4d"] = "hoi4d"
    # name: str = "hoi4d"
    mask_blur_radius: float = None
    clip_t: float = None
    clip_interval: float = None
    clip_max_n: int = None
    meta_file: str = None
    rgb_root: str = None
    depth_root: str = None
    anno_root: str = None
    cam_root: str = None
    pre_save_root: str = None


def num2d(x):
    if x > 99999:
        raise ValueError("x should be less than 99999")
    else:
        return "0" * (5 - len(str(x))) + str(x)


def get_hoi4d_duration_fps(action_fp):

    with open(action_fp, "r") as fp_json:
        actions = json.load(fp_json)
    # There are 104 video's fps=30 & duration=10.0 , while others fps=15 & duration=20.0
    if 'duration' in actions['info'].keys():
        duration_t = actions['info']['duration']
    elif 'Duration' in actions['info'].keys():
        duration_t = actions['info']['Duration']
    else:
        raise ValueError(f"duration not in clip: {actions['info']}")
        
    if duration_t != 20:
        fps = 30
    else:
        fps = 15

    return duration_t, fps


cls_name = {
    'C1': 'Toy Car', 'C2': 'Mug', 'C3': 'Laptop', 'C4': 'Storage Furniture',
    'C5': 'Bottle', 'C6': 'Safe', 'C7': 'Bowl', 'C8': 'Bucket', 'C9': 'Scissors',
    'C11': 'Pliers', 'C12': 'Kettle', 'C13': 'Knife', 'C14': 'Trash Can',
    'C17': 'Lamp', 'C18': 'Stapler', 'C20': 'Chair'
}
def get_hoi4d_object(index):
    index = index.split("/")
    return cls_name[index[2]]
    

def generate_shape_preprocess(iseq, data, data_dir, resize_shape, patch_crop_shape, n_data, num_procs):
    st = time.time()
    try:
        # pdb.set_trace()
        rgb_list = [Image.open(path) for path in data['rgbs']]
        dep_list = [Image.open(path) for path in data['deps']]
        msk_list = [Image.open(path) for path in data['masks']]
        data['rgbs'] = []
        data['deps'] = []
        data['masks'] = []
        ext_list = []
        timestamps = data['timestamps']
        camera_fp = data['exts']
        outCam = o3d.io.read_pinhole_camera_trajectory(camera_fp).parameters
        extrinsics = [outCam[i] for i in timestamps]
        num_frames = len(rgb_list)

        data_point_dir = os.path.join(data_dir, str(iseq))
        os.makedirs(data_point_dir, exist_ok=True)

        for i, (rgb, dep, mask) in enumerate(zip(rgb_list, dep_list, msk_list)):
            assert rgb.size == dep.size     # it needs to first convert the size of depth & image the same.
            w_org, h_org = rgb.size
            rgb, (h_scaled, w_scaled) = pil_resize_to_center_crop(rgb, resize_shape, patch_crop_shape)
            h_new, w_new = patch_crop_shape
            dep, _ = pil_resize_to_center_crop(dep, resize_shape, patch_crop_shape, depth_process=True)
            mask, _ = pil_resize_to_center_crop(mask, resize_shape, patch_crop_shape, depth_process=True)
            ext_list.append(extrinsics[i].extrinsic)
            rgb_fp = os.path.join(data_point_dir, "rgb"+str(i)+".png")
            dep_fp = os.path.join(data_point_dir, "dep"+str(i)+".png")
            msk_fp = os.path.join(data_point_dir, "mask"+str(i)+".png")
            rgb.save(rgb_fp)
            dep.save(dep_fp)
            mask.save(msk_fp)
            data['rgbs'].append(rgb_fp)
            data['deps'].append(dep_fp)
            data['masks'].append(msk_fp)

        extrinsics = np.stack(ext_list)
        ext_fp = os.path.join(data_point_dir, "extrinsic.npy")
        np.save(ext_fp, extrinsics)
        data['exts'] = ext_fp

        time_cost = time.time() - st
        rmin = int(int((n_data - iseq) / num_procs) * time_cost / 60.0)
        print(f"[{iseq}/{n_data}] with {current_process().name}: rest_time {rmin//60}hr {rmin%60}min.")
        return data
    except Exception as e:
        print(f"Fail to preprocess {data['idx']}: {e}")
        return None


class DatasetHOI4D(Datasetegomono4d):
    def __init__(
        self,
        cfg: DatasetHOI4DCfg,
        stage: Stage,
        frame_sampler: FrameSampler,
        global_rank: int,
        world_size: int,
        debug=False
    ) -> None:
        super().__init__(cfg, stage, frame_sampler, global_rank, world_size)

        pre_save_root = cfg.pre_save_root

        if (pre_save_root is not None):
            pre_save_root = pre_save_root + '/' + "egomono4d_hoi4d_" + str(cfg.resize_shape[0]) + "_" + str(cfg.resize_shape[1]) + "_patch" + str(cfg.patch_size)
        else:
            raise ValueError("For HOI4D dataset, you must appoint a pre_save_root for preprocessing (not None).")
        self.pre_save_root = pre_save_root
        
        if (not os.path.exists(pre_save_root)) or (debug is True):        
            
            os.makedirs(self.pre_save_root, exist_ok=True)
            sequence = []
            with open(cfg.meta_file, "r") as fp:
                data_index_list = fp.readlines()

            if debug is True:
                data_index_list = data_index_list[:10]

            print("Generate Original Sequence")
            for data_index in tqdm(data_index_list):

                if data_index.endswith("\n"):
                    data_index = data_index[:-1]
                rgb_root = os.path.join(cfg.rgb_root, data_index, "align_rgb")
                dep_root = os.path.join(cfg.depth_root, data_index, "align_depth")
                msk_root = os.path.join(cfg.anno_root, data_index, "2Dseg")
                if not os.path.exists(msk_root):
                    continue

                folder_name = iter(os.scandir(msk_root)).__next__()
                msk_root = os.path.join(msk_root, folder_name)
                obj = get_hoi4d_object(data_index)
                num_frames = sum(1 for entry in os.scandir(dep_root) if entry.is_file()) - 1

                # get the sub-clip of each HOI4D video
                action_fp = os.path.join(cfg.anno_root, data_index, "action/color.json")
                duration, fps = get_hoi4d_duration_fps(action_fp)
                subclip_t = int(cfg.clip_t * fps)
                subclip_interval_t = int(cfg.clip_interval * fps)
                n_subclip = (num_frames - subclip_t) // subclip_interval_t
                subclip = [(i*subclip_interval_t, i*subclip_interval_t + subclip_t) for i in range(n_subclip)]
                subclip.append((num_frames-subclip_t, num_frames))
                camera_fp = os.path.join(cfg.anno_root, data_index, '3Dseg', 'output.log')

                for iidx, (st, ed) in enumerate(subclip):
                    index = "hoi4d_" + data_index + "_" + str(iidx) + "_(" + str(st) + "_" + str(ed) + ")"
                    rgb_list, dep_list, msk_list = [], [], []
                    if ed-st < cfg.clip_max_n:
                        indices = range(st, ed)
                    else:
                        indices = np.linspace(st, ed-1, cfg.clip_max_n).tolist()
                        indices = [int(idc) for idc in indices]
                    ts = []
                    for img_idx in indices:
                        rgb_list.append(os.path.join(rgb_root, f'{num2d(img_idx)}.jpg'))
                        dep_list.append(os.path.join(dep_root, f'{num2d(img_idx)}.png'))
                        msk_list.append(os.path.join(msk_root, f'{num2d(img_idx)}.png'))
                        ts.append(img_idx)
                    data = {
                        "idx": index, "object": obj,
                        "rgbs": rgb_list, "deps": dep_list, "masks": msk_list,
                        "timestamps": ts
                    }
                    data['exts'] = camera_fp
                    sequence.append(data)

            self.sequence = sequence
            def get_obj_instance(x):
                return x['idx'].split("/")[2] + "_" + x['idx'].split("/")[3]

            instance_list = [get_obj_instance(x) for x in self.sequence]
            instance_list = list(set(instance_list))
            n_instance = len(instance_list)
            print(f"Number of Instance: {n_instance}")

            with open(os.path.join(self.pre_save_root, "initial_sequence.json"), "w") as f:
                json.dump(sequence, f, indent=4)

            self._get_preprocess_save(debug=debug)

        with open(os.path.join(self.pre_save_root, f"sequence_{stage}.json"), "r") as f:
            self.sequence = json.load(f)


    def _get_preprocess_save(self, num_procs=20, debug=False):

        with open(os.path.join(self.pre_save_root, "initial_sequence.json"), "r") as f:
            self.sequence = json.load(f)

        if debug is True:
            self.sequence = self.sequence[:10]
        
        sequence = []
        resize_shape = self.cfg.resize_shape
        patch_size = self.cfg.patch_size
        patch_crop_shape = compute_patch_cropped_shape(resize_shape, patch_size)

        data_dir = os.path.join(self.pre_save_root, "data")
        os.makedirs(data_dir, exist_ok=True)

        print("Generating Preprocess Sequence... ")
        pool = mp.Pool(processes=num_procs)
        results = []
        n_data = len(self.sequence)

        for iseq, data in enumerate(self.sequence):
            result = pool.apply_async(generate_shape_preprocess, args=(iseq, data, data_dir, resize_shape, patch_crop_shape, n_data, num_procs))
            results.append(result)
        pool.close()
        pool.join()
        sequence = [result.get() for result in results if result.get() is not None]

        print("Finish")

        rgb = Image.open(self.sequence[0]['rgbs'][0])
        w_org, h_org = rgb.size
        rgb, (h_scaled, w_scaled) = pil_resize_to_center_crop(rgb, resize_shape, patch_crop_shape)
        h_new, w_new = patch_crop_shape
        with open(os.path.join(self.pre_save_root, "process_sequence.json"), "w") as f:
            json.dump(sequence, f, indent=4)
        self.sequence = sequence

        intr_fp = os.path.join(self.pre_save_root, "intrinsic")
        os.makedirs(intr_fp, exist_ok=True)

        cam_id_list = os.listdir(self.cfg.cam_root)
        for cam_id in cam_id_list:
            intr_org = np.load(os.path.join(self.cfg.cam_root, cam_id, "intrin.npy"))
            intr_org = torch.Tensor(intr_org)
            intrinsic = resize_crop_intrinisic(intr_org, (h_org, w_org), (h_scaled, w_scaled), (h_new, w_new))
            canonical_intrinsic = canonicalize_intrinisic(intrinsic, (h_new, w_new)).cpu()
            np.save(os.path.join(intr_fp, cam_id+".npy"), canonical_intrinsic)

        with open(os.path.join(self.pre_save_root, "process_sequence.json"), "r") as f:
            sequence = json.load(f)
        self.sequence = sequence

        print("Generating Depth Estimation Sequence... ")
        depth_estimator = get_depth_estimator(cache_dir=self.cfg.cache_dir)
        h_new, w_new = patch_crop_shape
        # pdb.set_trace()
        for ib, data in enumerate(tqdm(sequence)):
            # pdb.set_trace()
            n_image = len(data['rgbs'])
            n_base = n_image // num_procs
            seqs = [(i*num_procs, (i+1)*num_procs) for i in range(n_base)]
            if n_base * num_procs < n_image:
                seqs.append((n_base*num_procs, n_image))
            for iseq, seq in  enumerate(seqs):
                image_list = [np.array(Image.open(data['rgbs'][i])) for i in range(seq[0], seq[1])]               

                image_batch = torch.from_numpy(np.stack(image_list)).permute(0, 3, 1, 2)
                predictions = depth_estimator.infer(image_batch)
                predictions['depth'] = predictions['depth'].cpu().detach().numpy()[:,0]
                predictions['intrinsics'] = predictions['intrinsics'].cpu().detach().numpy()
                predictions['points'] = predictions['points'].cpu().detach().numpy().transpose(0,2,3,1)

                rgb_fp = data['rgbs'][0]
                for i in range(seq[0], seq[1]):
                    e_dep = predictions['depth'][i]
                    e_dep_fp_np = '/'.join(rgb_fp.split('/')[:-1]) + "/edep" + str(i) + '.npy'
                    np.save(e_dep_fp_np, e_dep)

                    e_dep_intr = predictions['intrinsics'][i]
                    e_dep_intr_can = canonicalize_intrinisic(torch.tensor(e_dep_intr), (h_new, w_new))
                    e_dep_intr_np = '/'.join(rgb_fp.split('/')[:-1]) + "/edep" + str(i) + '_canonical_intrinsic.npy'
                    np.save(e_dep_intr_np, e_dep_intr_can)
                    

        with open(os.path.join(self.pre_save_root, "sequence_full.json"), "w") as f:
            json.dump(copy.deepcopy(self.sequence), f, indent=4)

        self._split_sequence_stage()

    
    def _split_sequence_stage(self):

        with open(os.path.join(self.pre_save_root, "sequence_full.json"), "r") as f:
            self.sequence = json.load(f)

        def get_obj_scene(x):
            return x['idx'].split("/")[4]

        scene_list = [get_obj_scene(x) for x in self.sequence]
        scene_list = list(set(scene_list))
        n_scene = len(scene_list)
        indices = list(range(n_scene))
        random.shuffle(indices)
        random.shuffle(scene_list)

        train_size, val_size = int(0.88*n_scene), int(0.09*n_scene)
        train_scene = [scene_list[idx] for idx in indices[:train_size]]
        val_scene = [scene_list[idx] for idx in indices[train_size:train_size + val_size]]
        test_scene = [scene_list[idx] for idx in indices[train_size + val_size:]]

        train_seq = [x for x in self.sequence if get_obj_scene(x) in train_scene]
        val_seq = [x for x in self.sequence if get_obj_scene(x) in val_scene]
        test_seq = [x for x in self.sequence if get_obj_scene(x) in test_scene]
        with open(os.path.join(self.pre_save_root, "sequence_train.json"), "w") as f:
            json.dump(train_seq, f, indent=4)
        with open(os.path.join(self.pre_save_root, "sequence_val.json"), "w") as f:
            json.dump(val_seq, f, indent=4)
        with open(os.path.join(self.pre_save_root, "sequence_test.json"), "w") as f:
            json.dump(test_seq, f, indent=4)
        
        print(f"n_train: {len(train_seq)}")
        print(f"n_val: {len(val_seq)}")
        print(f"n_test: {len(test_seq)}")


    def get_item_wrapper_single(self, index: int):
        
        meta = self.sequence[index]
        num_frames = len(meta["rgbs"])
        if (self.all_frames is True) and (self.stage in ['val', 'test']):
            indices = torch.Tensor([i for i in range(num_frames)], device=torch.device('cpu')).int()
        else:
            indices = self.frame_sampler.sample(num_frames, torch.device("cpu"), self.frame_max_interval)
        rgbs_fp = [meta['rgbs'][i] for i in indices]
        videos = torch.stack([tf.ToTensor()(Image.open(path)) for path in rgbs_fp])
        deps_fp = []
        intr_fp = []
        for rgb_fp in rgbs_fp:
            aux_fp = self.get_aux_filename(rgb_fp)
            deps_fp.append(aux_fp['edep'])
            intr_fp.append(aux_fp['intrinsics'])

        if self.cfg.use_gt_depth is True:
            deps_fp = [meta['deps'][i] for i in indices]
            # pdb.set_trace()
            depths = torch.stack([tf.ToTensor()(Image.open(path)) for path in deps_fp]).squeeze(1)
            fly_mask = 1.0 - (depths == 0).float()
            depths = depths / 1000.0
            uid = meta['idx']
            cam_id = uid.split('/')[0].split('_')[1]
            intrinsic = np.load(os.path.join(self.pre_save_root, "intrinsic", cam_id+'.npy'))
            intrinsics = torch.stack([torch.Tensor(intrinsic)]*videos.shape[0])
        else:
            depths = torch.stack([torch.Tensor(np.load(path)) for path in deps_fp])
            fly_mask = detect_sequence_flying_pixels(depths.numpy(), threshold=FLY_THRESHOLD)
            fly_mask = 1.0 - torch.Tensor(fly_mask)
            intrinsics = torch.stack([torch.Tensor(np.load(path)) for path in intr_fp])

        masks_fp = [meta['masks'][i] for i in indices]
        masks = []
        for fp in masks_fp:
            msk = np.asarray(Image.open(fp))
            if msk.ndim == 3: msk = msk.sum(2)
            masks.append(msk)
        masks = np.stack([(msk == 0).astype(np.float32) for msk in masks])
        masks = torch.Tensor(masks)
        data = {
            "videos": videos,      # (F, C, H, W)
            "depths": depths,    
            "flys": fly_mask,
            "masks": torch.minimum(masks, fly_mask), 
            "indices": indices,
            "scenes": meta['idx'],
            "datasets": "hoi4d",
            "intrinsics": intrinsics,
            "use_gt_depth": self.cfg.use_gt_depth,
            "hoi_masks": 1.0 - masks,
        }

        # HOI4D test setting
        if (self.stage in ['test']) or (self.cfg.use_gt_depth is True):
            deps_fp = [meta['deps'][i] for i in indices]
            gt_depths = torch.stack([tf.ToTensor()(Image.open(path)) for path in deps_fp]).squeeze(1)
            data['gt_depths'] = gt_depths / 1000.0
            f = videos.shape[0]
            uid = meta['idx']
            cam_id = uid.split('/')[0].split('_')[1]
            intrinsic = np.load(os.path.join(self.pre_save_root, "intrinsic", cam_id+'.npy'))
            data['gt_intrinsics'] = torch.stack([torch.Tensor(intrinsic)]*f)
            data['gt_extrinsics'] = torch.Tensor(np.load(meta['exts']))[indices.long()].inverse()

        return data