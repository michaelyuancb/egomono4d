from .dataset import DatasetCfgCommon
import os
import re
import pdb
import torch
import random
import pdb
import copy
import time
import json
import numpy as np
from PIL import Image
import multiprocessing as mp
import torch.multiprocessing as t_mp
from multiprocessing import current_process

try:
    EVAL = os.environ['EVAL_MODE']
except:
    EVAL = 'False'
try:
    INFER = os.environ['INFER_MODE']
except:
    INFER = 'False'
if (EVAL not in ['True']) and (INFER not in ['True']):
    from ..misc.data_util import pil_resize_to_center_crop, compute_patch_cropped_shape, canonicalize_intrinisic
    from ..hoi.ego_hos_wrapper import EgoHOSWrapper
    from ..flow import get_flow_predictor
    from ..misc.mask import get_epipolar_error_masks
    from ..misc.depth import get_depth_estimator, estimate_relative_depth
else:
    from ..misc.data_util import pil_resize_to_center_crop, compute_patch_cropped_shape, canonicalize_intrinisic


def cpu_preprocess(iseq, data, resize_shape, patch_crop_shape, n_data, num_procs):
    
    if not os.path.exists(data['save_dir']):
        os.makedirs(data['save_dir'], exist_ok=True)

    st = time.time()
    try:
        meta = copy.deepcopy(data)
        meta['rgbs'] = []
        dep_flag, msk_flag = False, False
        if "deps" in data.keys():
            dep_flag = True 
            meta['deps'] = []
            dep_list = [Image.open(dep_fp) for dep_fp in data['deps']]
        if "masks" in data.keys():
            msk_flag = True
            meta['masks'] = []
            msk_list = [Image.open(msk_fp) for msk_fp in data['masks']]

        for i, rgb_fp in enumerate(data['rgbs']):
            rgb = Image.open(rgb_fp)
            rgb, (h_scaled, w_scaled) = pil_resize_to_center_crop(rgb, resize_shape, patch_crop_shape)

            rgb_fp = os.path.join(data['save_dir'], "rgb"+str(i+data['base_id'])+".png")
            meta['rgbs'].append(str(rgb_fp))
            rgb.save(rgb_fp)
            if dep_flag is True:
                dep = dep_list[i]
                dep, _ = pil_resize_to_center_crop(dep, resize_shape, patch_crop_shape, depth_process=True)
                dep_fp = os.path.join(data['save_dir'], "dep"+str(i+data['base_id'])+".png")
                dep.save(dep_fp)
                meta['deps'].append(str(dep_fp))
            if msk_flag is True:
                mask = msk_list[i]
                mask, _ = pil_resize_to_center_crop(mask, resize_shape, patch_crop_shape)
                msk_fp = os.path.join(data['save_dir'], "mask"+str(i+data['base_id'])+".png")
                mask.save(msk_fp)
                meta['masks'].append(str(msk_fp))

        time_cost = time.time() - st
        rmin = int(int((n_data - iseq) / num_procs) * time_cost / 60.0)
        print(f"[{iseq}/{n_data}] with {current_process().name}: rest_time {rmin//60}hr {rmin%60}min.")    
        return meta
    
    except Exception as e:
        print(f"Fail to preprocess {data['id']}_base_id_{data['base_id']}: {e}")
        return None
    

def parallel_cpu_process(
    cfg: DatasetCfgCommon, 
    num_cpu_procs,
    sequence,
    pre_save_root,
    ):
    
    print("Generating Preprocess Sequence... ")
    pool = mp.Pool(processes=num_cpu_procs)
    results = []
    n_data = len(sequence)
    resize_shape = cfg.resize_shape
    patch_size = cfg.patch_size
    patch_crop_shape = compute_patch_cropped_shape(resize_shape, patch_size)

    for iseq, data in enumerate(sequence):
        result = pool.apply_async(cpu_preprocess, args=(iseq, data, resize_shape, patch_crop_shape, n_data, num_cpu_procs))
        results.append(result)
    pool.close()
    pool.join()
    sequence = [result.get() for result in results if result.get() is not None]
    
    with open(os.path.join(pre_save_root, "sequence_cpu_process_finished.json"), "w") as f:
        json.dump(sequence, f, indent=4)
    return sequence


def gpu_process_conductor(iseq, 
                          data, 
                          depth_estimator, 
                          patch_crop_shape, 
                          mask_estimation,
                          mask_flow_model,
                          mask_binary_open_value,
                          mask_egohos_model,
                          device,
                          n_data
                          ):
    st = time.time()
    try:
        h_new, w_new = patch_crop_shape

        meta = copy.deepcopy(data)
        if depth_estimator is not None:
            meta['edeps'] = []
            for i, rgb_fp in enumerate(meta['rgbs']):
                e_dep_pack = estimate_relative_depth(Image.open(rgb_fp), depth_estimator)
                e_dep = e_dep_pack['depth']
                if 'intrinsics' in e_dep_pack.keys():
                    e_dep_intr = e_dep_pack['intrinsics']
                    e_dep_intr_can = canonicalize_intrinisic(torch.tensor(e_dep_intr), (h_new, w_new))
                    e_dep_intr_np = data['save_dir'] + "/edep" + str(i+data['base_id']) + '_canonical_intrinsic.npy'
                    np.save(e_dep_intr_np, e_dep_intr_can)
                e_dep_fp_np = data['save_dir'] + "/edep" + str(i+data['base_id']) + '.npy'
                np.save(e_dep_fp_np, e_dep)
                meta['edeps'].append(e_dep_fp_np)


        if (mask_estimation is None) or (len(mask_estimation) == 0):
            return meta

        dy_mask_list = []
        ego_hand = False

        if mask_estimation == ['epipolar', 'egohos']:
            mask_estimation = ['egohos', 'epipolar']

        for es_method in mask_estimation:
            if es_method == 'egohos':
                dy_mask = []
                ego_hand = True
                for rgb_fp in meta['rgbs']:
                    hand, obj, cb = mask_egohos_model.segment(rgb_fp)
                    dy_mask_res = (hand + obj + cb) > 0
                    if dy_mask_res.sum() == 0:
                        ego_hand = False
                    dy_mask.append(dy_mask_res)
            elif es_method == 'epipolar':
                if ego_hand is True:       
                    continue
                rgb_list = meta['rgbs']
                dy_mask = get_epipolar_error_masks(rgb_list, (h_new, w_new), mask_flow_model, mask_binary_open_value, 
                                                device=device)
            else:
                raise ValueError(f"Dynamic Mask Estimation Method [{es_method}] has not be coded;")
            dy_mask_list.append(dy_mask)

        dy_mask_list = np.array(dy_mask_list)
        mask_list = (dy_mask_list.sum(axis=0) == 0).astype(np.float32)  # [n_mask, (h, w)]
        meta['emasks'] = []
        n_image = len(meta['rgbs'])
        assert mask_list.shape[0] == n_image
        for i in range(n_image):
            e_msk_fp_np = data['save_dir'] + "/emask" + str(i+data['base_id']) + '.npy'
            np.save(e_msk_fp_np, mask_list[i])
            meta['emasks'].append(e_msk_fp_np)
            

        time_cost = time.time() - st
        rmin = int(int((n_data - iseq)) * time_cost / 60.0)
        rday = rmin // (60 * 24)
        rmin = rmin % (60 * 24)
        print(f"[{iseq}/{n_data}] with {current_process().name}: time_cost {np.round(time_cost,1)} secs ; rest_time {rday}day {rmin//60}hr {rmin%60}min.")    
        return meta
    
    except Exception as e:
        print(f"Fail to preprocess {data['id']}_base_id_{data['base_id']}: {e}")
        return None


def gpu_process(cfg, data_list, patch_crop_shape, device, proc_id, pre_save_root):

    torch.cuda.set_device(device)
    print(f"Start GPU Process, device: {device}")
    print(cfg.mask_estimation)
    print(cfg.mask_flow_model)

    if 'epipolar' in cfg.mask_estimation:
        mask_flow_model = get_flow_predictor(cfg.mask_flow_model).to(device)
    else:
        mask_flow_model = None 
    print(f"get flow model {device}")
    if 'egohos' in cfg.mask_estimation:
        os.makedirs(f"{cfg.cache_dir}/ego_hos_cache", exist_ok=True)
        mask_egohos_model = EgoHOSWrapper(cache_path=os.path.abspath(f"{cfg.cache_dir}/ego_hos_cache"), 
                                            repo_path=f"{cfg.cache_dir}/ego_hos_checkpoints", device=device)
    else:
        mask_egohos_model = None
    print(f"get ego model {device}")
    depth_estimator = get_depth_estimator(cache_dir=cfg.cache_dir, device=device)

    
    print(f"get depth model {device}")
    
    print(f"Device: {device}")

    meta_list = []
    n_proc_data = len(data_list)
    for iseq, data in enumerate(data_list):
        meta = gpu_process_conductor(iseq, data, 
                                     depth_estimator, 
                                     patch_crop_shape, 
                                     mask_estimation=cfg.mask_estimation,
                                     mask_flow_model=mask_flow_model,
                                     mask_binary_open_value=cfg.mask_binary_open_value,
                                     mask_egohos_model=mask_egohos_model,
                                     device=device,
                                     n_data=n_proc_data
                                    )
        if meta is not None:
            meta_list.append(meta)
    
    with open(os.path.join(pre_save_root, f"sequence_processed_{proc_id}.json"), "w") as f:
        json.dump(meta_list, f, indent=4)

    return meta_list


def parallel_gpu_process(cfg, num_gpu_procs, num_gpu, sequence, pre_save_root, commit_id=0):

    resize_shape = cfg.resize_shape
    patch_size = cfg.patch_size
    patch_crop_shape = compute_patch_cropped_shape(resize_shape, patch_size)

    for es_method in cfg.mask_estimation:
        if es_method not in ['epipolar', 'egohos', 'egomodel', 'maskrcnn']:
            raise ValueError(f"Unsupport Dynamic Mask Estimation Method: {es_method} ;")
    
    n_seq = len(sequence)
    chunk_size = n_seq // num_gpu_procs
    chunks = [(i*chunk_size, (i+1)*chunk_size) for i in range(num_gpu_procs)]
    if chunks[-1][1] != n_seq:
        chunks[-1] = (chunks[-1][0], n_seq)

    t_mp.set_start_method('spawn', force=True)

    processes = []
    if commit_id >= 0:
        pre_save_root_split = os.path.join(pre_save_root, "gpu_result_split", str(commit_id))
    else:
        pre_save_root_split = os.path.join(pre_save_root, "gpu_result_split")
    os.makedirs(pre_save_root_split, exist_ok=True)

    for i, chunk in enumerate(chunks):
        device = f'cuda:{i % num_gpu}' 
        p = t_mp.Process(target=gpu_process, args=(cfg, sequence[chunk[0]:chunk[1]], patch_crop_shape, device, i, pre_save_root_split))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()

    sequence = []
    for proc_id in range(len(chunks)):
        with open(os.path.join(pre_save_root_split, f"sequence_processed_{proc_id}.json"), "r") as f:
            seq_split = json.load(f)
        sequence = sequence + seq_split

    with open(os.path.join(pre_save_root_split, f"sequence_processed.json"), "w") as f:
        json.dump(sequence, f, indent=4)
    with open(os.path.join(pre_save_root, f"sequence_processed.json"), "w") as f:
        json.dump(sequence, f, indent=4)

    return sequence


#############################
# sequence_data= {
#     "id": uid
#     "rgbs": [rgb file path],
#     "deps": [dep file path] (optional),
#     "masks": [mask file path] (optional),
#     "save_dir": directory for result saving,
#     "base_id": save_idx = base_id + [order in rgb_file_path]
# }

# Return
# sequence_data = {
#     ...
#     "edeps":  [numpy file path],
#     "emasks": [numpy file path],
# }


def parallel_preprocess(n_cpu_procs: int,
                        n_gpu_procs: int,
                        num_gpu: int,
                        cfg: DatasetCfgCommon, 
                        sequence: list,
                        pre_save_root: str
                        ):

    sequence = parallel_cpu_process(cfg=cfg, num_cpu_procs=n_cpu_procs, sequence=sequence, pre_save_root=pre_save_root)
    print("Finish CPU Process")

    with open(os.path.join(pre_save_root, "sequence_cpu_process_finished.json"), "r") as f:
        sequence = json.load(f)

    idx_machine = -1
    sequence = parallel_gpu_process(cfg=cfg, num_gpu_procs=n_gpu_procs, num_gpu=num_gpu, sequence=sequence, pre_save_root=pre_save_root, commit_id=idx_machine)
    print("Finish GPU Process")

    return sequence