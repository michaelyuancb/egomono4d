from dataclasses import dataclass
import os
import re
import pdb
import random
import json
from tqdm import tqdm
from typing import Optional
from typing_extensions import Literal

import torch
import torchvision.transforms as tf

from ..frame_sampler.frame_sampler import FrameSampler
from .dataset import DatasetCfgCommon, Datasetegomono4d
from .types import Stage
from ..misc.fly import detect_sequence_flying_pixels
from ..model.projection import sample_image_grid, unproject

try:
    EVAL = os.environ['EVAL_MODE']
except:
    EVAL = 'False'
if EVAL not in ['True']:
    from .mpreprocess_images import parallel_preprocess
    


FLY_THRESHOLD = 0.05

@dataclass
class DatasetEpicKitchenCfg(DatasetCfgCommon):
    name: Literal["epic_kitchen"] = "epic_kitchen"
    max_clip_per_video: int = None
    clip_frame: int = None

    original_base_root: str = None
    intrinsic_root: Optional[str] = None
    pre_save_root: str = None


class DatasetEpicKitchen(Datasetegomono4d):
    def __init__(
        self,
        cfg: DatasetEpicKitchenCfg,
        stage: Stage,
        frame_sampler: FrameSampler,
        global_rank: int,
        world_size: int,
        debug: bool=False
    ) -> None:
        super().__init__(cfg, stage, frame_sampler, global_rank, world_size)
        pre_save_root = str(cfg.pre_save_root)

        if (pre_save_root is not None):
            pre_save_root = pre_save_root + '/' + "egomono4d_epickitchen_" + str(cfg.resize_shape[0]) + "_" + str(cfg.resize_shape[1]) + "_patch" + str(cfg.patch_size)
        else:
            raise ValueError("For EpicKitchen dataset, you must appoint a pre_save_root for preprocessing (not None).")
        self.pre_save_root = pre_save_root

        os.makedirs(self.pre_save_root, exist_ok=True)
        print(f"preprocess_dir = {self.pre_save_root}")
        print(f"len = {len(os.listdir(self.pre_save_root))}")
        if len(os.listdir(self.pre_save_root)) == 0:
            self._get_preprocess_save(cfg.original_base_root)
        
        with open(os.path.join(self.pre_save_root, f"sequence_{stage}.json"), "r") as f:
            self.sequence = json.load(f)


    def _get_preprocess_save(self, root_dir):
        
        os.makedirs(self.pre_save_root, exist_ok=True)
        save_data_dir = os.path.join(self.pre_save_root, "epic_kitchen_preprocessed")
        os.makedirs(save_data_dir, exist_ok=True)

        ########################### Get Original Sequence ##########################
        
        sequence_original = []
        P_list = os.listdir(root_dir)
        for P in tqdm(P_list, desc="P"):
            P_dir = os.path.join(root_dir, P, "rgb_frames")
            os.makedirs(os.path.join(save_data_dir, P), exist_ok=True)
            PV_list = os.listdir(P_dir)
            for V in tqdm(PV_list, desc="V"): 
                V_dir = os.path.join(P_dir, V)          # EPIC-KITCHENS/P27/rgb_frames/P27_01
                os.makedirs(os.path.join(save_data_dir, P, V), exist_ok=True)
                rgb_fp_list = []
                rgb_fp_list = os.listdir(V_dir)   
                def extract_number(filename):
                    match = re.search(r'\d+', filename)  
                    return int(match.group()) if match else 0  
                rgb_fp_list.sort(key=extract_number)
                sequence_original.append({"id": f"EpicKitchen_{V}", "P": P, "V": V, "rgbs": rgb_fp_list})

        with open(os.path.join(self.pre_save_root, "sequence_original.json"), "w") as f:
            json.dump(sequence_original, f, indent=4)

        ################## Preprare Sequential List for Parallel Preprocess ####################
        with open(os.path.join(self.pre_save_root, "sequence_original.json"), "r") as f:
            sequence_original = json.load(f)
        
        sequence_preprocess = []
        for seq in sequence_original:
            uid_path = os.path.join(save_data_dir, seq['P'], seq['V'])
            rgbs_list = seq['rgbs']
            if len(rgbs_list) < 1000:
                continue
            rgbs_list = rgbs_list[1000:]     # Erase the Dark (Light Turn off) frames of Epic-Kitchen
            if len(rgbs_list) < 1000:
                continue
            rgbs_list = rgbs_list[:-1000]    # Erase the last 1000 frames of Epic-Kitchen

            n_images = len(rgbs_list)
            spl = seq['id'].split('_')
            V = '_'.join(spl[1:])
            P = spl[1]
            base_img_dir = os.path.join(root_dir, P, "rgb_frames", V)
            n_seq_data = 0
            if n_images <= 200:
                continue
            if self.cfg.max_clip_per_video * self.cfg.clip_frame <= n_images:
                seq_l_base = (n_images - self.cfg.clip_frame) // (self.cfg.max_clip_per_video - 1)
                for seqi in range(self.cfg.max_clip_per_video):
                    seq_l = seq_l_base * seqi
                    seq_r = seq_l + self.cfg.clip_frame
                    data_dict = {"id": seq['id']+"_"+str(seqi), "save_dir": os.path.join(uid_path, str(seqi)), "base_id": 0}
                    rgbs_clip = []
                    for i in range(seq_l, seq_r):
                        rgbs_clip.append(os.path.join(base_img_dir, rgbs_list[i]))
                    data_dict['rgbs'] = rgbs_clip
                    sequence_preprocess.append(data_dict)
                    n_seq_data = self.cfg.max_clip_per_video
            else:
                seq_l = 0
                while seq_l + self.cfg.clip_frame <= n_images:
                    seq_r = seq_l + self.cfg.clip_frame
                    data_dict = {"id": seq['id']+"_"+str(n_seq_data), "save_dir": os.path.join(uid_path, str(n_seq_data)), "base_id": 0}
                    rgbs_clip = []
                    for i in range(seq_l, seq_r):
                        rgbs_clip.append(os.path.join(base_img_dir, rgbs_list[i]))
                    data_dict['rgbs'] = rgbs_clip
                    seq_l = seq_r
                    sequence_preprocess.append(data_dict)
                    n_seq_data = n_seq_data + 1
            print(f"P: {P} ; V:{V} ; n_seq_data={n_seq_data}")
        print(f"len_sequence={len(sequence_preprocess)}")

        with open(os.path.join(self.pre_save_root, "sequence_parallel_prepare.json"), "w") as f:
            json.dump(sequence_preprocess, f, indent=4)

        ########################## Conduct Parallel Preprocess ###########################

        with open(os.path.join(self.pre_save_root, "sequence_parallel_prepare.json"), "r") as f:
            sequence_preprocess = json.load(f)

        print(f"num_gpu: {torch.cuda.device_count()}")
        parallel_preprocess(
            n_cpu_procs=256,
            n_gpu_procs=12*torch.cuda.device_count(),   # for 80G GPU
            num_gpu=torch.cuda.device_count(),
            cfg=self.cfg,
            sequence=sequence_preprocess,
            pre_save_root=self.pre_save_root
        )

        self._split_sequence_stage()


    def _split_sequence_stage(self):

        with open(os.path.join(self.pre_save_root, "sequence_processed.json"), "r") as f:
            self.sequence = json.load(f)


        def get_pat_scene(x):
            return x['id'].split("_")[1]

        scene_list = [get_pat_scene(x) for x in self.sequence]
        scene_list = list(set(scene_list))
        n_scene = len(scene_list)
        indices = list(range(n_scene))
        random.shuffle(indices)
        random.shuffle(scene_list)

        train_size, val_size = int(0.96*n_scene), n_scene - int(0.96*n_scene)
        train_scene = [scene_list[idx] for idx in indices[:train_size]]
        val_scene = [scene_list[idx] for idx in indices[train_size:train_size + val_size]]
        test_scene = [scene_list[idx] for idx in indices[train_size + val_size:]]

        train_seq = [x for x in self.sequence if get_pat_scene(x) in train_scene]
        val_seq = [x for x in self.sequence if get_pat_scene(x) in val_scene]

        test_seq = [x for x in self.sequence if get_pat_scene(x) in test_scene]
        with open(os.path.join(self.pre_save_root, "sequence_train.json"), "w") as f:
            json.dump(train_seq, f, indent=4)
        with open(os.path.join(self.pre_save_root, "sequence_val.json"), "w") as f:
            json.dump(val_seq, f, indent=4)
        with open(os.path.join(self.pre_save_root, "sequence_test.json"), "w") as f:
            json.dump(test_seq, f, indent=4)
        
        print(f"n_train: {len(train_seq)}")
        print(f"n_val: {len(val_seq)}")
        print(f"n_test: {len(test_seq)}")