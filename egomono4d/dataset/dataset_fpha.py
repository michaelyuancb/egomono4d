from dataclasses import dataclass
import os
import re
import pdb
import random
import json
from tqdm import tqdm
from typing_extensions import Literal

import torch

from ..frame_sampler.frame_sampler import FrameSampler
from .dataset import DatasetCfgCommon, Datasetegomono4d
from .types import Stage

try:
    EVAL = os.environ['EVAL_MODE']
except:
    EVAL = 'False'
try:
    INFER = os.environ['INFER_MODE']
except:
    INFER = 'False'
if EVAL not in ['True']:
    from .mpreprocess_images import parallel_preprocess


FLY_THRESHOLD = 0.05

@dataclass
class DatasetFPHACfg(DatasetCfgCommon):
    name: Literal["fpha"] = "fpha"
    clip_frame: int = None

    original_base_root: str = None
    pre_save_root: str = None


class DatasetFPHA(Datasetegomono4d):
    def __init__(
        self,
        cfg: DatasetFPHACfg,
        stage: Stage,
        frame_sampler: FrameSampler,
        global_rank: int,
        world_size: int,
        debug: bool=False
    ) -> None:
        super().__init__(cfg, stage, frame_sampler, global_rank, world_size)
        pre_save_root = str(cfg.pre_save_root)

        if (pre_save_root is not None):
            pre_save_root = pre_save_root + '/' + "egomono4d_fpha_" + str(cfg.resize_shape[0]) + "_" + str(cfg.resize_shape[1]) + "_patch" + str(cfg.patch_size)
        else:
            raise ValueError("For HOI4D dataset, you must appoint a pre_save_root for preprocessing (not None).")
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
        save_data_dir = os.path.join(self.pre_save_root, "fpha_preprocessed")
        os.makedirs(save_data_dir, exist_ok=True)
        root_dir = os.path.join(root_dir, 'Video_files')

        # ########################### Get Original Sequence ##########################

        def extract_number(filename):
            filename = filename.split('/')[-1]
            match = re.search(r'\d+', filename)  
            return int(match.group()) if match else 0  
        
        ban_action = ['give_card', 'give_coin', 'handshake', 'high_five', 'open_wallet', 'pour_wine', 'receive_coin', 'receive_card', 'toast_wine']

        sequence_original = []
        S_list = os.listdir(root_dir)
        for S in S_list:
            S_dir = os.path.join(root_dir, S)
            os.makedirs(os.path.join(save_data_dir, S), exist_ok=True)
            SA_list = os.listdir(S_dir)
            for A in SA_list:
                if A in ban_action:
                    continue
                A_dir = os.path.join(S_dir, A)
                os.makedirs(os.path.join(save_data_dir, S, A), exist_ok=True)
                SAP_list = os.listdir(A_dir)
                for P in SAP_list:
                    os.makedirs(os.path.join(os.path.join(save_data_dir, S, A, P)), exist_ok=True)
                    rgb_list = os.listdir(os.path.join(A_dir, P, "color"))
                    rgb_list = [os.path.join(A_dir, P, "color", x) for x in rgb_list]
                    rgb_list.sort(key=extract_number)
                    data = {
                        "id": f"fpha_{S}{P}_{A}",
                        "S": S, "A": A, "P": P,
                        "rgbs": rgb_list,
                    }
                    if os.path.exists(os.path.join(A_dir, P, "depth")):
                        dep_list = os.listdir(os.path.join(A_dir, P, "depth"))
                        dep_list = [os.path.join(A_dir, P, "depth", x) for x in dep_list]
                        dep_list.sort(key=extract_number)
                        data['deps'] = dep_list
                    else:
                        data['deps'] = []

                    sequence_original.append(data)

        with open(os.path.join(self.pre_save_root, "sequence_original.json"), "w") as f:
            json.dump(sequence_original, f, indent=4)

        #################### Preprare Sequential List for Parallel Preprocess ####################
        with open(os.path.join(self.pre_save_root, "sequence_original.json"), "r") as f:
            sequence_original = json.load(f)
        
        sequence_preprocess = []
        for seq in tqdm(sequence_original):
            uid_path = os.path.join(save_data_dir, seq['S'], seq['A'], seq['P'])
            rgbs_list = seq['rgbs']
            deps_list = seq['deps']

            n_images = len(rgbs_list)
            n_seq_data = 0

            seq_l = 0
            while seq_l + self.cfg.clip_frame <= n_images:
                seq_r = seq_l + self.cfg.clip_frame
                data_dict = {"id": seq['id']+"_"+str(n_seq_data), "save_dir": os.path.join(uid_path, str(n_seq_data)), "base_id": 0, "action": seq['A']}
                rgbs_clip = []
                deps_clip = []
                for i in range(seq_l, seq_r):
                    rgbs_clip.append(seq['rgbs'][i])
                    if len(deps_list) > 0:
                        deps_clip.append(seq['deps'][i])
                data_dict['rgbs'] = rgbs_clip
                if len(deps_clip) > 0:
                    data_dict['deps'] = deps_clip
                seq_l = seq_l + self.cfg.clip_frame // 5
                sequence_preprocess.append(data_dict)
                n_seq_data = n_seq_data + 1
        print(f"len_sequence={len(sequence_preprocess)}")

        with open(os.path.join(self.pre_save_root, "sequence_parallel_prepare.json"), "w") as f:
            json.dump(sequence_preprocess, f, indent=4)

        ########################### Conduct Parallel Preprocess ###########################

        with open(os.path.join(self.pre_save_root, "sequence_parallel_prepare.json"), "r") as f:
            sequence_preprocess = json.load(f)

        sequence = parallel_preprocess(
            n_cpu_procs=64,
            n_gpu_procs=4,
            num_gpu=torch.cuda.device_count(),
            cfg=self.cfg,
            sequence=sequence_preprocess,
            pre_save_root=self.pre_save_root
        )

        with open(os.path.join(self.pre_save_root, f"sequence_processed.json"), "r") as f:
            sequence = json.load(f)

        action_list = [x['action'] for x in sequence]
        action_list = list(set(action_list))
        n_action = len(action_list)
        indices = list(range(n_action))
        random.shuffle(indices)
        random.shuffle(action_list)

        train_size, val_size = int(0.92*n_action), n_action - int(0.92*n_action)
        train_action = [action_list[idx] for idx in indices[:train_size]]
        val_action = [action_list[idx] for idx in indices[train_size:train_size + val_size]]
        test_action = [action_list[idx] for idx in indices[train_size + val_size:]]

        print(f"n_train_action={len(train_action)}")
        print(f"n_val_action={len(val_action)}")
        print(f"n_test_action={len(test_action)}")

        sequence_train = [x for x in sequence if x['action'] in train_action]
        sequence_val = [x for x in sequence if x['action'] in val_action]
        sequence_test = [x for x in sequence if x['action'] in test_action]

        with open(os.path.join(self.pre_save_root, "sequence_train.json"), "w") as f:
            json.dump(sequence_train, f, indent=4)
        with open(os.path.join(self.pre_save_root, "sequence_val.json"), "w") as f:
            json.dump(sequence_val, f, indent=4)
        with open(os.path.join(self.pre_save_root, "sequence_test.json"), "w") as f:
            json.dump(sequence_test, f, indent=4)

        print(f"len(train)={len(sequence_train)} ; len(val)={len(sequence_val)} ; len(test)={len(sequence_test)}")
        print(f"len_all={len(sequence)}")