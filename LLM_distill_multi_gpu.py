# ====================================================================================
#  MoE SVD DISTILLATION SCRIPT v13.1
#
# This version is the final, high-performance script. It will NEVER delete the
# temporary worker files, giving you full control for verification and safety.
# ====================================================================================
import torch
import torch.fft
from safetensors.torch import load_file, save_file, safe_open
from tqdm.auto import tqdm
import os
import json
import re
import numpy as np
from sklearn.cluster import KMeans
import contextlib
import torch.multiprocessing as mp
from collections import defaultdict

# ==============================================================================
#                                 CONFIGURATION
# ==============================================================================
# --- 1. PATHS ---
TEACHER_MODEL_FOLDER = "/media/workstation/crucial/Qwen"
STUDENT_BASE_FOLDER = "/home/workstation/Desktop/qwen30b"
OUTPUT_LORA_PATH = "/media/workstation/crucial/distilled_qwen3_advanced_loraHIGH.safetensors"
OUTPUT_LORA_CONFIG_PATH = "/media/workstation/crucial/adapter_config.json" # Path for the LoRA config

# --- 2. MODEL ARCHITECTURE ---
MODEL_ARCHITECTURE_CONFIG = {
    "teacher_layers": 62,
    "student_layers": 48,
    "teacher_experts_per_layer": 160,
    "student_experts_per_layer": 128,
}

# --- 3. LORA RANK CONFIGURATION very high ranks are required for SVD distillation to distill the most amount of information. Lower ranks were tested but were ultimately worse than 2048---
RANK_MAP = { "self_attn": 2048, "mlp": 2048, "block_sparse_moe": 2048, "default": 2048 }
LORA_ALPHA = 1.0

# --- 4. MULTI-GPU CONFIGURATION ---
NUM_GPUS = 2 # Set the number of GPUs you want to use

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def get_tensors_from_shards(keys_to_find, model_folder, weight_map, device="cpu"):
    shards_to_load = defaultdict(list)
    for key in keys_to_find:
        if key in weight_map:
            shards_to_load[weight_map[key]].append(key)
    tensors = {}
    for shard_file, keys_in_shard in shards_to_load.items():
        shard_path = os.path.join(model_folder, shard_file)
        with safe_open(shard_path, framework="pt", device=device) as f:
            for key in keys_in_shard:
                tensors[key] = f.get_tensor(key).to(torch.float32)
    return tensors

def project_tensor_fourier(teacher_tensor, student_shape):
    teacher_len = teacher_tensor.shape[0]; student_len = student_shape[0]
    teacher_freq = torch.fft.rfft(teacher_tensor.float())
    if student_len < teacher_len:
        student_freq = teacher_freq[:student_len // 2 + 1]
    else:
        pad_size = student_len // 2 + 1 - len(teacher_freq)
        student_freq = torch.cat((teacher_freq, torch.zeros(pad_size, dtype=teacher_freq.dtype, device=teacher_freq.device)))
    projected_tensor = torch.fft.irfft(student_freq, n=student_len)
    teacher_std = torch.std(teacher_tensor); projected_std = torch.std(projected_tensor)
    if projected_std > 1e-9: projected_tensor *= (teacher_std / projected_std)
    return projected_tensor

def project_tensor_fourier_2d(teacher_tensor, student_shape):
    teacher_h, teacher_w = teacher_tensor.shape
    student_h, student_w = student_shape
    teacher_freq = torch.fft.rfft2(teacher_tensor.float())
    student_freq = torch.zeros(student_h, student_w // 2 + 1, dtype=teacher_freq.dtype, device=teacher_tensor.device)
    h_min = min(teacher_freq.shape[0], student_freq.shape[0])
    w_min = min(teacher_freq.shape[1], student_freq.shape[1])
    student_freq[:h_min, :w_min] = teacher_freq[:h_min, :w_min]
    projected_tensor = torch.fft.irfft2(student_freq, s=(student_h, student_w))
    teacher_std = torch.std(teacher_tensor)
    projected_std = torch.std(projected_tensor)
    if projected_std > 1e-9:
        projected_tensor *= (teacher_std / projected_std)
    return projected_tensor

def project_tensor(teacher_tensor, student_shape):
    original_device = teacher_tensor.device
    if teacher_tensor.dim() == 2:
        try:
            teacher_mean = teacher_tensor.float().mean()
            teacher_std = teacher_tensor.float().std() + 1e-9
            normalized_teacher = (teacher_tensor.float() - teacher_mean) / teacher_std
            U, S, Vh = torch.linalg.svd(normalized_teacher, full_matrices=False)
            del normalized_teacher
            torch.cuda.empty_cache()
            target_out, target_in = student_shape
            k = min(len(S), target_out, target_in)
            U_p, S_p, Vh_p = U[:, :k], torch.diag(S[:k]), Vh[:k, :]
            proj_norm = U_p @ S_p @ Vh_p
            final_norm = torch.zeros(student_shape, dtype=proj_norm.dtype, device=original_device)
            copy_out, copy_in = min(proj_norm.shape[0], target_out), min(proj_norm.shape[1], target_in)
            final_norm[:copy_out, :copy_in] = proj_norm[:copy_out, :copy_in]
            return (final_norm * teacher_std) + teacher_mean
        except torch.OutOfMemoryError:
            print(f"\n!!! SVD OOM on {original_device}. Using fast Fourier fallback. !!!")
            torch.cuda.empty_cache()
            return project_tensor_fourier_2d(teacher_tensor, student_shape)
        except torch.linalg.LinAlgError:
            print(f"\n!!! SVD LinAlgError on {original_device}. Using fast Fourier fallback. !!!")
            torch.cuda.empty_cache()
            return project_tensor_fourier_2d(teacher_tensor, student_shape)
    elif teacher_tensor.dim() == 1:
        return project_tensor_fourier(teacher_tensor, student_shape)
    else:
        return torch.zeros(student_shape, device=original_device)

def distill_moe_layer(student_layer_idx, cfg, teacher_folder, teacher_weight_map, student_folder, student_weight_map, device):
    synthetic_moe_weights = {}
    layer_ratio = cfg['teacher_layers'] / cfg['student_layers']
    teacher_float_idx = student_layer_idx * layer_ratio
    teacher_idx_floor = int(np.floor(teacher_float_idx))
    teacher_idx_ceil = min(int(np.ceil(teacher_float_idx)), cfg['teacher_layers'] - 1)
    interp_weight = teacher_float_idx - teacher_idx_floor
    expert_parts = ['gate_proj', 'up_proj', 'down_proj']
    all_teacher_keys = []
    for expert_idx in range(cfg['teacher_experts_per_layer']):
        for part in expert_parts:
            all_teacher_keys.append(f"model.layers.{teacher_idx_floor}.block_sparse_moe.experts.{expert_idx}.{part}.weight")
            all_teacher_keys.append(f"model.layers.{teacher_idx_ceil}.block_sparse_moe.experts.{expert_idx}.{part}.weight")
    all_teacher_tensors = get_tensors_from_shards(all_teacher_keys, teacher_folder, teacher_weight_map, device=device)
    interpolated_fingerprints = []
    for expert_idx in range(cfg['teacher_experts_per_layer']):
        fp_floor_parts, fp_ceil_parts = [], []
        for part in expert_parts:
            key_floor = f"model.layers.{teacher_idx_floor}.block_sparse_moe.experts.{expert_idx}.{part}.weight"
            key_ceil = f"model.layers.{teacher_idx_ceil}.block_sparse_moe.experts.{expert_idx}.{part}.weight"
            if key_floor in all_teacher_tensors and key_ceil in all_teacher_tensors:
                fp_floor_parts.append(all_teacher_tensors[key_floor].flatten())
                fp_ceil_parts.append(all_teacher_tensors[key_ceil].flatten())
        if fp_floor_parts and fp_ceil_parts:
            fp_floor, fp_ceil = torch.cat(fp_floor_parts), torch.cat(fp_ceil_parts)
            interpolated_fingerprint = (1 - interp_weight) * fp_floor + interp_weight * fp_ceil
            interpolated_fingerprints.append(interpolated_fingerprint.cpu().numpy())
    if not interpolated_fingerprints: return {}
    kmeans = KMeans(n_clusters=cfg['student_experts_per_layer'], random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(np.array(interpolated_fingerprints))
    expert_map = defaultdict(list)
    for teacher_idx, cluster_id in enumerate(cluster_labels):
        expert_map[cluster_id].append(teacher_idx)
    for student_expert_idx in range(cfg['student_experts_per_layer']):
        assigned_teacher_indices = expert_map[student_expert_idx]
        if not assigned_teacher_indices: continue
        for part in expert_parts:
            student_key = f"model.layers.{student_layer_idx}.block_sparse_moe.experts.{student_expert_idx}.{part}.weight"
            student_tensor_shape = get_tensors_from_shards([student_key], student_folder, student_weight_map, device=device)[student_key].shape
            blended_tensor = torch.zeros(student_tensor_shape, dtype=torch.float32, device=device)
            total_norm = 0.0
            for teacher_idx in assigned_teacher_indices:
                key_floor = f"model.layers.{teacher_idx_floor}.block_sparse_moe.experts.{teacher_idx}.{part}.weight"
                key_ceil = f"model.layers.{teacher_idx_ceil}.block_sparse_moe.experts.{teacher_idx}.{part}.weight"
                if key_floor in all_teacher_tensors and key_ceil in all_teacher_tensors:
                    interp_teacher_tensor = (1 - interp_weight) * all_teacher_tensors[key_floor] + interp_weight * all_teacher_tensors[key_ceil]
                    projected_teacher_tensor = project_tensor(interp_teacher_tensor, student_tensor_shape)
                    norm = torch.linalg.vector_norm(projected_teacher_tensor)
                    blended_tensor += projected_teacher_tensor * norm
                    total_norm += norm
            if total_norm > 0:
                synthetic_moe_weights[student_key] = blended_tensor / total_norm
    return synthetic_moe_weights

def get_rank_for_key(key, rank_map):
    for map_key, rank in rank_map.items():
        if map_key != "default" and map_key in key: return rank
    return rank_map["default"]

def extract_lora_from_diff(diff_tensor, rank):
    if not torch.all(torch.isfinite(diff_tensor)): return None, None
    if torch.linalg.vector_norm(diff_tensor) < 1e-9: return None, None
    try:
        if diff_tensor.dim() != 2: return None, None
        U, S, Vh = torch.linalg.svd(diff_tensor.float(), full_matrices=False)
        if torch.any(S < 1e-9): return None, None
        lora_A = Vh[:rank, :].contiguous()
        lora_B = (U[:, :rank] @ torch.diag(S[:rank])).contiguous()
        lora_B *= (LORA_ALPHA / rank)
        return lora_A.to(torch.bfloat16), lora_B.to(torch.bfloat16)
    except torch.linalg.LinAlgError:
        return None, None

def distillation_worker(rank, world_size, all_student_keys, temp_file_path):
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)
    print(f"--- Worker {rank} started, assigned to {device} ---")
    keys_per_worker = len(all_student_keys) // world_size
    start_idx = rank * keys_per_worker
    end_idx = None if rank == world_size - 1 else (rank + 1) * keys_per_worker
    student_keys_subset = all_student_keys[start_idx:end_idx]
    if not student_keys_subset:
        print(f"--- Worker {rank} has no keys to process. Exiting. ---")
        return
    print(f"--- Worker {rank} will process {len(student_keys_subset)} tensors. ---")
    cfg = MODEL_ARCHITECTURE_CONFIG
    with open(os.path.join(STUDENT_BASE_FOLDER, "model.safetensors.index.json"), 'r') as f:
        student_weight_map = json.load(f)['weight_map']
    with open(os.path.join(TEACHER_MODEL_FOLDER, "model.safetensors.index.json"), 'r') as f:
        teacher_weight_map = json.load(f)['weight_map']
    worker_lora_weights = {}
    processed_moe_layers = set()
    student_keys_by_shard = defaultdict(list)
    for key in student_keys_subset:
        if key in student_weight_map:
            student_keys_by_shard[student_weight_map[key]].append(key)
    print(f"--- Worker {rank} has grouped keys into {len(student_keys_by_shard)} shards. ---")
    for shard_file, keys_in_shard in tqdm(student_keys_by_shard.items(), desc=f"GPU {rank} Processing Shards", position=rank):
        teacher_keys_by_shard = defaultdict(list)
        for student_key in keys_in_shard:
            key_match = re.search(r'model\.layers\.(\d+)\.', student_key)
            if 'block_sparse_moe' in student_key: continue
            if not key_match:
                if student_key in teacher_weight_map:
                    teacher_shard = teacher_weight_map[student_key]
                    teacher_keys_by_shard[teacher_shard].append(student_key)
            else:
                student_layer_idx = int(key_match.group(1))
                layer_ratio = cfg['teacher_layers'] / cfg['student_layers']
                teacher_float_idx = student_layer_idx * layer_ratio
                teacher_idx_floor = int(np.floor(teacher_float_idx))
                teacher_idx_ceil = min(int(np.ceil(teacher_float_idx)), cfg['teacher_layers'] - 1)
                key_part = student_key.split(f'layers.{student_layer_idx}.')[1]
                key_floor = f'model.layers.{teacher_idx_floor}.{key_part}'
                key_ceil = f'model.layers.{teacher_idx_ceil}.{key_part}'
                if key_floor in teacher_weight_map:
                    teacher_shard = teacher_weight_map[key_floor]
                    teacher_keys_by_shard[teacher_shard].append(key_floor)
                if key_ceil in teacher_weight_map:
                    teacher_shard = teacher_weight_map[key_ceil]
                    teacher_keys_by_shard[teacher_shard].append(key_ceil)
        with contextlib.ExitStack() as exit_stack:
            student_shard_path = os.path.join(STUDENT_BASE_FOLDER, shard_file)
            f_student = exit_stack.enter_context(safe_open(student_shard_path, framework="pt", device="cpu"))
            teacher_handles = {
                t_shard: exit_stack.enter_context(safe_open(os.path.join(TEACHER_MODEL_FOLDER, t_shard), framework="pt", device="cpu"))
                for t_shard in teacher_keys_by_shard.keys()
            }
            for student_key in keys_in_shard:
                if not ('.weight' in student_key and 'norm' not in student_key): continue
                key_match = re.search(r'model\.layers\.(\d+)\.', student_key)
                if key_match and 'block_sparse_moe' in student_key:
                    student_layer_idx = int(key_match.group(1))
                    if student_layer_idx in processed_moe_layers: continue
                    moe_weights = distill_moe_layer(student_layer_idx, cfg, TEACHER_MODEL_FOLDER, teacher_weight_map, STUDENT_BASE_FOLDER, student_weight_map, device)
                    for key, synthetic_tensor in moe_weights.items():
                        student_tensor_moe = get_tensors_from_shards([key], STUDENT_BASE_FOLDER, student_weight_map, device=device)[key]
                        diff_tensor = synthetic_tensor - student_tensor_moe
                        rank_val = get_rank_for_key(key, RANK_MAP)
                        lora_A, lora_B = extract_lora_from_diff(diff_tensor, rank_val)
                        if lora_A is not None:
                            worker_lora_weights[f"base_model.model.{key}.lora_A.weight"] = lora_A.cpu()
                            worker_lora_weights[f"base_model.model.{key}.lora_B.weight"] = lora_B.cpu()
                    processed_moe_layers.add(student_layer_idx)
                    continue
                student_tensor = f_student.get_tensor(student_key).to(device, non_blocking=True)
                synthetic_tensor = None
                if not key_match:
                    if student_key in teacher_weight_map:
                        teacher_shard = teacher_weight_map[student_key]
                        teacher_tensor = teacher_handles[teacher_shard].get_tensor(student_key).to(device, non_blocking=True)
                        synthetic_tensor = project_tensor(teacher_tensor, student_tensor.shape)
                else:
                    student_layer_idx = int(key_match.group(1))
                    layer_ratio = cfg['teacher_layers'] / cfg['student_layers']
                    teacher_float_idx = student_layer_idx * layer_ratio
                    teacher_idx_floor = int(np.floor(teacher_float_idx))
                    teacher_idx_ceil = min(int(np.ceil(teacher_float_idx)), cfg['teacher_layers'] - 1)
                    interp_weight = teacher_float_idx - teacher_idx_floor
                    key_part = student_key.split(f'layers.{student_layer_idx}.')[1]
                    key_floor = f'model.layers.{teacher_idx_floor}.{key_part}'
                    key_ceil = f'model.layers.{teacher_idx_ceil}.{key_part}'
                    if key_floor in teacher_weight_map and key_ceil in teacher_weight_map:
                        t_shard_floor = teacher_weight_map[key_floor]
                        t_shard_ceil = teacher_weight_map[key_ceil]
                        t_floor = teacher_handles[t_shard_floor].get_tensor(key_floor).to(device, non_blocking=True)
                        t_ceil = teacher_handles[t_shard_ceil].get_tensor(key_ceil).to(device, non_blocking=True)
                        blended = (1 - interp_weight) * t_floor + interp_weight * t_ceil
                        synthetic_tensor = project_tensor(blended, student_tensor.shape)
                if synthetic_tensor is not None:
                    diff_tensor = synthetic_tensor - student_tensor
                    rank_val = get_rank_for_key(student_key, RANK_MAP)
                    lora_A, lora_B = extract_lora_from_diff(diff_tensor, rank_val)
                    if lora_A is not None:
                        worker_lora_weights[f"base_model.model.{student_key}.lora_A.weight"] = lora_A.cpu()
                        worker_lora_weights[f"base_model.model.{student_key}.lora_B.weight"] = lora_B.cpu()
    try:
        worker_file_path = f"{temp_file_path}_{rank}.safetensors"
        save_file(worker_lora_weights, worker_file_path)
        file_size = os.path.getsize(worker_file_path) / (1024*1024)
        print(f"--- Worker {rank} successfully saved {worker_file_path} ({file_size:.2f} MB) ---")
    except Exception as e:
        print(f"!!! CRITICAL ERROR IN WORKER {rank}: FAILED TO SAVE TEMPORARY FILE !!!")
        print(f"Error: {e}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    print("--- Starting Multi-GPU MoE Distillation (v13.1 - Definitive with Manual Cleanup) ---")
    print("Loading student model index to divide work...")
    with open(os.path.join(STUDENT_BASE_FOLDER, "model.safetensors.index.json"), 'r') as f:
        student_weight_map = json.load(f)['weight_map']
    student_keys = sorted(student_weight_map.keys())
    temp_file_path = "temp_lora_weights"
    spawn_args = (NUM_GPUS, student_keys, temp_file_path)
    mp.spawn(
        distillation_worker,
        args=spawn_args,
        nprocs=NUM_GPUS,
        join=True
    )
    print("\n--- All workers finished. Consolidating LoRA weights... ---")
    final_lora_weights = {}
    for i in range(NUM_GPUS):
        worker_file = f"{temp_file_path}_{i}.safetensors"
        if os.path.exists(worker_file):
            print(f"Loading weights from {worker_file}...")
            worker_weights = load_file(worker_file)
            final_lora_weights.update(worker_weights)
            # --- SAFETY MODIFICATION: Automatic deletion is now disabled ---
            # os.remove(worker_file)
            print(f"--- Finished with {worker_file}. It will NOT be deleted automatically. ---")
        else:
            print(f"!!! ERROR: Worker {i} did not produce a temporary file. Check worker logs for errors. !!!")
            
    if not final_lora_weights:
        print("\n\n❌ CRITICAL FAILURE: No LoRA weights were generated or consolidated.")
    else:
        print(f"--- Saving final consolidated LoRA weights to {OUTPUT_LORA_PATH} ---")
        save_file(final_lora_weights, OUTPUT_LORA_PATH)
    
    module_names = set()
    if final_lora_weights:
        module_names = set(re.search(r'\.([^.]+)\.lora_A', key).group(1) for key in final_lora_weights.keys())
    
    target_modules = sorted(list(module_names))
    adapter_config = {
        "base_model_name_or_path": STUDENT_BASE_FOLDER,
        "peft_type": "LORA",
        "r": RANK_MAP['default'],
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": 0.0,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM", 
        "bias": "none"
    }
    with open(OUTPUT_LORA_CONFIG_PATH, 'w') as f:
        json.dump(adapter_config, f, indent=4)
    print(f"--- Saving LoRA adapter config to {OUTPUT_LORA_CONFIG_PATH} ---")
    print("\n\n✅ MULTI-GPU DISTILLATION COMPLETE!")
    print("\n--- MANUAL CLEANUP REQUIRED ---")
    print("The temporary worker files have been preserved for safety.")
    print("Once you have verified the final LoRA file, you can delete them by running:")
    print(f"rm {temp_file_path}_*.safetensors")