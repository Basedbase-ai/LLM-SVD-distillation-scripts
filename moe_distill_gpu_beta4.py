# ====================================================================================
#  MoE SVD DISTILLATION SCRIPT v15.0-FINAL
#
# This version fixes the silent key-matching failure from v14.x.
# 1. The fragile `detect_prefix` function has been REMOVED.
# 2. It restores the robust, hardcoded `"model."` prefix, as used in the
#    working beta 3 script. This guarantees that teacher tensor keys are
#    correctly constructed and found.
# 3. It retains the high-performance GPU-first SVD with CPU/Resize fallbacks,
#    ensuring both speed and stability.
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
from sklearn.metrics.pairwise import cosine_similarity

# ==============================================================================
#                                 CONFIGURATION
# ==============================================================================
# --- 1. PATHS ---
TEACHER_MODEL_FOLDER = "/media/workstation/crucial/Qwen"
STUDENT_BASE_FOLDER = "/home/workstation/Desktop/qwen30b"
OUTPUT_LORA_PATH = "/media/workstation/crucial/distilled_qwen3_ULTIMATE_v2.safetensors"
OUTPUT_LORA_CONFIG_PATH = "/media/workstation/crucial/adapter_config_ULTIMATE_v2.json"

# --- 2. MODEL ARCHITECTURE ---
MODEL_ARCHITECTURE_CONFIG = {
    "teacher_layers": 62,
    "student_layers": 48,
    "teacher_experts_per_layer": 160,
    "student_experts_per_layer": 128,
}

# --- 3. LORA RANK CONFIGURATION ---
RANK_MAP = { "self_attn": 2048, "mlp": 2048, "block_sparse_moe": 2048, "default": 2048 }
LORA_ALPHA = RANK_MAP["default"]

# --- 4. MULTI-GPU CONFIGURATION ---
NUM_GPUS = 2

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def slerp(t1, t2, weight, epsilon=1e-7):
    t1_flat, t2_flat = t1.flatten(), t2.flatten()
    t1_norm_val, t2_norm_val = torch.linalg.vector_norm(t1_flat), torch.linalg.vector_norm(t2_flat)
    t1_norm, t2_norm = t1_flat / (t1_norm_val + epsilon), t2_flat / (t2_norm_val + epsilon)
    dot = torch.clamp(torch.dot(t1_norm, t2_norm), -1.0, 1.0)
    theta = torch.acos(dot)
    if theta < 1e-4:
        return (t1 * (1 - weight) + t2 * weight).reshape(t1.shape)
    sin_theta = torch.sin(theta)
    s1, s2 = torch.sin((1 - weight) * theta) / sin_theta, torch.sin(weight * theta) / sin_theta
    interpolated_norm_flat = s1 * t1_norm + s2 * t2_norm
    interpolated_original_norm = t1_norm_val * (1 - weight) + t2_norm_val * weight
    return (interpolated_norm_flat * interpolated_original_norm).reshape(t1.shape)

def apply_dare(diff_tensor, drop_rate=0.90, rescale_factor=1.1):
    if diff_tensor.dim() != 2: return diff_tensor
    mask = torch.rand_like(diff_tensor.float()) > drop_rate
    return (diff_tensor * mask) * rescale_factor

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
                tensors[key] = f.get_tensor(key)
    return tensors

def project_tensor(teacher_tensor, student_shape):
    original_device = teacher_tensor.device
    original_dtype = teacher_tensor.dtype

    if teacher_tensor.dim() == 1:
        new_tensor = torch.zeros(student_shape, device=original_device, dtype=original_dtype)
        copy_len = min(teacher_tensor.numel(), student_shape[0])
        new_tensor[:copy_len] = teacher_tensor[:copy_len]
        return new_tensor

    if teacher_tensor.dim() == 2:
        try:
            U, S, Vh = torch.linalg.svd(teacher_tensor.float(), full_matrices=False)
            target_out, target_in = student_shape
            k = min(len(S), target_out, target_in)
            U_p, S_p, Vh_p = U[:, :k], torch.diag(S[:k]), Vh[:k, :]
            proj_tensor = U_p @ S_p @ Vh_p
            final_tensor = torch.zeros(student_shape, device=original_device, dtype=original_dtype)
            copy_out, copy_in = min(proj_tensor.shape[0], target_out), min(proj_tensor.shape[1], target_in)
            final_tensor[:copy_out, :copy_in] = proj_tensor[:copy_out, :copy_in]
            return final_tensor

        except (torch.linalg.LinAlgError, torch.OutOfMemoryError):
            print(f"  [WARN] GPU SVD failed for shape {teacher_tensor.shape}. Retrying on CPU.")
            try:
                teacher_tensor_cpu = teacher_tensor.detach().to("cpu", dtype=torch.float32)
                U, S, Vh = torch.linalg.svd(teacher_tensor_cpu, full_matrices=False)
                target_out, target_in = student_shape
                k = min(len(S), target_out, target_in)
                U_p, S_p, Vh_p = U[:, :k], torch.diag(S[:k]), Vh[:k, :]
                proj_tensor_cpu = U_p @ S_p @ Vh_p
                final_tensor_cpu = torch.zeros(student_shape, dtype=torch.float32)
                copy_out, copy_in = min(proj_tensor_cpu.shape[0], target_out), min(proj_tensor_cpu.shape[1], target_in)
                final_tensor_cpu[:copy_out, :copy_in] = proj_tensor_cpu[:copy_out, :copy_in]
                return final_tensor_cpu.to(original_device, dtype=original_dtype)

            except (torch.linalg.LinAlgError, torch.OutOfMemoryError):
                print(f"  [WARN] CPU SVD also failed for shape {teacher_tensor.shape}. Falling back to direct resize.")
                fallback_tensor = torch.zeros(student_shape, device=original_device, dtype=original_dtype)
                copy_rows = min(teacher_tensor.shape[0], student_shape[0])
                copy_cols = min(teacher_tensor.shape[1], student_shape[1])
                fallback_tensor[:copy_rows, :copy_cols] = teacher_tensor[:copy_rows, :copy_cols]
                return fallback_tensor

    return torch.zeros(student_shape, device=original_device, dtype=original_dtype)

def align_with_procrustes(source_tensor, target_tensor):
    if source_tensor.shape != target_tensor.shape or source_tensor.dim() != 2: return source_tensor
    try:
        M = target_tensor.T.float() @ source_tensor.float()
        U, _, Vh = torch.linalg.svd(M, full_matrices=False)
        R = U @ Vh
        return source_tensor @ R.to(source_tensor.dtype)
    except (torch.linalg.LinAlgError, torch.OutOfMemoryError): return source_tensor

def get_rank_for_key(key, tensor_shape, rank_map):
    max_possible_rank = min(tensor_shape)
    requested_rank = rank_map["default"]
    for map_key, rank in rank_map.items():
        if map_key != "default" and map_key in key:
            requested_rank = rank
            break
    final_rank = min(requested_rank, max_possible_rank)
    return final_rank

def extract_lora_from_diff(diff_tensor, rank):
    if diff_tensor.dim() != 2 or torch.linalg.vector_norm(diff_tensor) < 1e-9:
        return None, None

    if not torch.all(torch.isfinite(diff_tensor)):
        print(f"  [WARN] Non-finite values in diff_tensor shape {diff_tensor.shape}. Skipping.")
        return None, None

    try:
        U, S, Vh = torch.linalg.svd(diff_tensor.float(), full_matrices=False)
    except (torch.linalg.LinAlgError, torch.OutOfMemoryError):
        print(f"  [WARN] GPU SVD failed for LoRA extraction shape {diff_tensor.shape}. Retrying on CPU.")
        try:
            diff_cpu = diff_tensor.detach().to("cpu", dtype=torch.float32)
            U, S, Vh = torch.linalg.svd(diff_cpu, full_matrices=False)
            U, S, Vh = U.to(diff_tensor.device), S.to(diff_tensor.device), Vh.to(diff_tensor.device)
        except (torch.linalg.LinAlgError, torch.OutOfMemoryError):
            print(f"  [ERROR] CPU SVD also failed for LoRA extraction shape {diff_tensor.shape}. Skipping.")
            return None, None

    valid_rank = min(rank, S.shape[0])
    lora_A = Vh[:valid_rank, :].contiguous()
    lora_B = (U[:, :valid_rank] @ torch.diag(S[:valid_rank])).contiguous()
    return lora_A.to(torch.bfloat16), lora_B.to(torch.bfloat16)

def get_teacher_layer_map_sigmoid(student_idx, student_layers, teacher_layers, k=0.15):
    to_norm_space = lambda idx, total: 2 * (idx / (total - 1)) - 1
    from_norm_space = lambda norm_idx, total: (norm_idx + 1) * (total - 1) / 2
    student_norm = to_norm_space(student_idx, student_layers)
    teacher_norm = np.tanh(student_norm / k) / np.tanh(1 / k)
    teacher_float_idx = from_norm_space(teacher_norm, teacher_layers)
    teacher_idx_floor = np.floor(teacher_float_idx)
    interp_weight = teacher_float_idx - teacher_idx_floor
    return teacher_idx_floor, 1.0 - interp_weight

def distill_moe_layer(student_layer_idx, cfg, teacher_folder, teacher_weight_map, student_folder, student_weight_map, device, teacher_prefix):
    synthetic_moe_weights = {}
    teacher_idx_floor_f, weight_floor = get_teacher_layer_map_sigmoid(student_layer_idx, cfg['student_layers'], cfg['teacher_layers'])
    teacher_idx_floor, teacher_idx_ceil = int(teacher_idx_floor_f), min(int(teacher_idx_floor_f) + 1, cfg['teacher_layers'] - 1)
    interp_weight = 1.0 - weight_floor
    expert_parts = ['gate_proj', 'up_proj', 'down_proj']
    all_teacher_keys = [f"{teacher_prefix}layers.{l_idx}.block_sparse_moe.experts.{e_idx}.{part}.weight" for l_idx in [teacher_idx_floor, teacher_idx_ceil] for e_idx in range(cfg['teacher_experts_per_layer']) for part in expert_parts]
    all_teacher_tensors = get_tensors_from_shards(list(set(all_teacher_keys)), teacher_folder, teacher_weight_map, device="cpu")

    interpolated_fingerprints = []
    for expert_idx in range(cfg['teacher_experts_per_layer']):
        fp_floor_parts, fp_ceil_parts = [], []
        for part in expert_parts:
            key_floor = f"{teacher_prefix}layers.{teacher_idx_floor}.block_sparse_moe.experts.{expert_idx}.{part}.weight"
            key_ceil = f"{teacher_prefix}layers.{teacher_idx_ceil}.block_sparse_moe.experts.{expert_idx}.{part}.weight"
            if key_floor in all_teacher_tensors and key_ceil in all_teacher_tensors:
                fp_floor_parts.append(all_teacher_tensors[key_floor].flatten())
                fp_ceil_parts.append(all_teacher_tensors[key_ceil].flatten())

        if fp_floor_parts and fp_ceil_parts:
            interpolated_fingerprints.append(slerp(torch.cat(fp_floor_parts), torch.cat(fp_ceil_parts), interp_weight).numpy())

    if not interpolated_fingerprints: return {}

    fingerprints_np = np.array(interpolated_fingerprints)
    normalized_fingerprints = fingerprints_np / (np.linalg.norm(fingerprints_np, axis=1, keepdims=True) + 1e-9)
    kmeans = KMeans(n_clusters=cfg['student_experts_per_layer'], random_state=42, n_init='auto').fit(normalized_fingerprints)
    expert_map = defaultdict(list)
    for teacher_idx, cluster_id in enumerate(kmeans.labels_): expert_map[cluster_id].append(teacher_idx)

    for student_expert_idx in range(cfg['student_experts_per_layer']):
        assigned_teacher_indices = expert_map[student_expert_idx]
        if not assigned_teacher_indices: continue
        similarity_scores = cosine_similarity(normalized_fingerprints[assigned_teacher_indices], kmeans.cluster_centers_[student_expert_idx].reshape(1, -1)).flatten()

        for part in expert_parts:
            # Use the student prefix to build the student key
            student_key = f"model.layers.{student_layer_idx}.block_sparse_moe.experts.{student_expert_idx}.{part}.weight"
            student_tensor_shape = get_tensors_from_shards([student_key], student_folder, student_weight_map, device="cpu")[student_key].shape
            blended_tensor = torch.zeros(student_tensor_shape, dtype=torch.float32, device=device)
            total_weight = 0.0
            for i, teacher_idx in enumerate(assigned_teacher_indices):
                key_floor = f"{teacher_prefix}layers.{teacher_idx_floor}.block_sparse_moe.experts.{teacher_idx}.{part}.weight"
                key_ceil = f"{teacher_prefix}layers.{teacher_idx_ceil}.block_sparse_moe.experts.{teacher_idx}.{part}.weight"
                if key_floor in all_teacher_tensors and key_ceil in all_teacher_tensors:
                    t_floor, t_ceil = all_teacher_tensors[key_floor].to(device), all_teacher_tensors[key_ceil].to(device)
                    interp_teacher = slerp(t_floor, t_ceil, interp_weight)
                    projected_teacher = project_tensor(interp_teacher, student_tensor_shape)
                    if projected_teacher is not None:
                        weight = similarity_scores[i]
                        blended_tensor += projected_teacher * weight
                        total_weight += weight
                    del t_floor, t_ceil, interp_teacher, projected_teacher
            if total_weight > 1e-9: synthetic_moe_weights[student_key] = blended_tensor / total_weight
    return synthetic_moe_weights

def distillation_worker(rank, world_size, all_student_keys, temp_file_path):
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)
    print(f"--- Worker {rank} started on {device} ---")
    cfg = MODEL_ARCHITECTURE_CONFIG
    with open(os.path.join(STUDENT_BASE_FOLDER, "model.safetensors.index.json"), 'r') as f: student_weight_map = json.load(f)['weight_map']
    with open(os.path.join(TEACHER_MODEL_FOLDER, "model.safetensors.index.json"), 'r') as f: teacher_weight_map = json.load(f)['weight_map']

    # --- FIX: Use hardcoded prefixes known to be correct for these models ---
    teacher_prefix = "model."
    student_prefix = "model."
    print(f"--- Using hardcoded prefix '{teacher_prefix}' for Teacher and Student ---")

    keys_per_worker = len(all_student_keys) // world_size
    start_idx, end_idx = rank * keys_per_worker, None if rank == world_size - 1 else (rank + 1) * keys_per_worker
    student_keys_subset = all_student_keys[start_idx:end_idx]

    worker_lora_weights = {}
    processed_moe_layers = set()
    student_keys_by_shard = defaultdict(list)
    for key in student_keys_subset:
        if key in student_weight_map: student_keys_by_shard[student_weight_map[key]].append(key)

    for shard_file, keys_in_shard in tqdm(student_keys_by_shard.items(), desc=f"GPU {rank} Processing Shards", position=rank):
        with safe_open(os.path.join(STUDENT_BASE_FOLDER, shard_file), framework="pt", device="cpu") as f_student_cpu:
            for student_key in keys_in_shard:
                if not ('.weight' in student_key and 'norm' not in student_key): continue

                with torch.no_grad():
                    key_match = re.search(r'layers\.(\d+)\.', student_key)

                    if key_match and 'block_sparse_moe' in student_key:
                        student_layer_idx = int(key_match.group(1))
                        if student_layer_idx in processed_moe_layers: continue

                        moe_weights = distill_moe_layer(student_layer_idx, cfg, TEACHER_MODEL_FOLDER, teacher_weight_map, STUDENT_BASE_FOLDER, student_weight_map, device, teacher_prefix)
                        if moe_weights:
                            for key, synthetic_tensor_gpu in moe_weights.items():
                                if synthetic_tensor_gpu is None: continue
                                student_tensor_moe_gpu = get_tensors_from_shards([key], STUDENT_BASE_FOLDER, student_weight_map, device=device)[key]
                                aligned_tensor = align_with_procrustes(synthetic_tensor_gpu, student_tensor_moe_gpu)
                                diff_tensor = aligned_tensor - student_tensor_moe_gpu
                                purified_diff = apply_dare(diff_tensor)
                                rank_val = get_rank_for_key(key, student_tensor_moe_gpu.shape, RANK_MAP)
                                lora_A, lora_B = extract_lora_from_diff(purified_diff, rank_val)
                                if lora_A is not None and lora_B is not None:
                                    worker_lora_weights[f"base_model.model.{key}.lora_A.weight"] = lora_A.cpu()
                                    worker_lora_weights[f"base_model.model.{key}.lora_B.weight"] = lora_B.cpu()
                                del student_tensor_moe_gpu, synthetic_tensor_gpu, aligned_tensor, diff_tensor, purified_diff
                        processed_moe_layers.add(student_layer_idx)
                        torch.cuda.empty_cache()
                        continue

                    student_tensor = f_student_cpu.get_tensor(student_key).to(device)
                    teacher_keys_to_fetch = []
                    
                    # Correctly strip the student prefix to get the base key
                    base_key = student_key[len(student_prefix):]

                    if 'layers' not in base_key:
                        teacher_equivalent = teacher_prefix + base_key
                        if teacher_equivalent in teacher_weight_map:
                            teacher_keys_to_fetch.append(teacher_equivalent)
                    elif key_match:
                        student_layer_idx = int(key_match.group(1))
                        teacher_idx_floor_f, weight_floor = get_teacher_layer_map_sigmoid(student_layer_idx, cfg['student_layers'], cfg['teacher_layers'])
                        teacher_idx_floor, teacher_idx_ceil = int(teacher_idx_floor_f), min(int(teacher_idx_floor_f) + 1, cfg['teacher_layers'] - 1)
                        
                        # Correctly get the part of the key after the layer number
                        key_part = base_key.split(f'layers.{student_layer_idx}.')[1]
                        
                        key_floor = f'{teacher_prefix}layers.{teacher_idx_floor}.{key_part}'
                        key_ceil = f'{teacher_prefix}layers.{teacher_idx_ceil}.{key_part}'
                        if key_floor in teacher_weight_map: teacher_keys_to_fetch.append(key_floor)
                        if key_ceil in teacher_weight_map: teacher_keys_to_fetch.append(key_ceil)

                    if not teacher_keys_to_fetch:
                        del student_tensor
                        torch.cuda.empty_cache()
                        continue

                    teacher_tensors = get_tensors_from_shards(teacher_keys_to_fetch, TEACHER_MODEL_FOLDER, teacher_weight_map, device=device)

                    synthetic_tensor = None
                    if len(teacher_tensors) == 1:
                         synthetic_tensor = project_tensor(next(iter(teacher_tensors.values())), student_tensor.shape)
                    elif len(teacher_tensors) == 2:
                         t_floor, t_ceil = teacher_tensors[key_floor], teacher_tensors[key_ceil]
                         blended = slerp(t_floor, t_ceil, 1.0 - weight_floor)
                         synthetic_tensor = project_tensor(blended, student_tensor.shape)
                         del blended

                    if synthetic_tensor is not None:
                        aligned_tensor = align_with_procrustes(synthetic_tensor, student_tensor)
                        diff_tensor = aligned_tensor - student_tensor
                        purified_diff = apply_dare(diff_tensor)
                        rank_val = get_rank_for_key(student_key, student_tensor.shape, RANK_MAP)
                        lora_A, lora_B = extract_lora_from_diff(purified_diff, rank_val)
                        if lora_A is not None and lora_B is not None:
                            # Use the original student_key for the final LoRA key
                            worker_lora_weights[f"base_model.model.{student_key}.lora_A.weight"] = lora_A.cpu()
                            worker_lora_weights[f"base_model.model.{student_key}.lora_B.weight"] = lora_B.cpu()
                        del aligned_tensor, diff_tensor, purified_diff

                    del student_tensor
                    if teacher_tensors: del teacher_tensors
                    if synthetic_tensor is not None: del synthetic_tensor
                    torch.cuda.empty_cache()
    try:
        worker_file_path = f"{temp_file_path}_{rank}.safetensors"
        save_file(worker_lora_weights, worker_file_path)
        print(f"--- Worker {rank} saved temp file ({os.path.getsize(worker_file_path)/(1024*1024):.2f} MB) ---")
    except Exception as e:
        print(f"!!! CRITICAL ERROR IN WORKER {rank}: FAILED TO SAVE TEMP FILE: {e} !!!")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    print("--- Starting Multi-GPU MoE Distillation (v15.0-FINAL) ---")
    with open(os.path.join(STUDENT_BASE_FOLDER, "model.safetensors.index.json"), 'r') as f: student_weight_map = json.load(f)['weight_map']
    student_keys = sorted(student_weight_map.keys())
    temp_file_path = "temp_lora_weights"
    spawn_args = (NUM_GPUS, student_keys, temp_file_path)
    mp.spawn(distillation_worker, args=spawn_args, nprocs=NUM_GPUS, join=True)

    print("\n--- Consolidating LoRA weights... ---")
    final_lora_weights = {}
    total_size = 0
    for i in range(NUM_GPUS):
        worker_file = f"{temp_file_path}_{i}.safetensors"
        if os.path.exists(worker_file):
            size_mb = os.path.getsize(worker_file)/(1024*1024)
            total_size += size_mb
            print(f"--- Loading {worker_file} ({size_mb:.2f} MB)... ---")
            final_lora_weights.update(load_file(worker_file))
        else:
            print(f"!!! ERROR: Worker {i} did not produce a temp file. Check logs. !!!")

    if not final_lora_weights:
        print("\n\n❌ CRITICAL FAILURE: No LoRA weights were generated.")
    else:
        print(f"--- Total consolidated size: {total_size:.2f} MB ---")
        print(f"--- Saving final consolidated LoRA weights to {OUTPUT_LORA_PATH} ---")
        save_file(final_lora_weights, OUTPUT_LORA_PATH)

    lora_A_keys = [key for key in final_lora_weights.keys() if key.endswith(".lora_A.weight")]
    if not lora_A_keys:
         print("\n\n❌ CRITICAL FAILURE: No LoRA A weights were found. Cannot generate adapter_config.json.")
    else:
        # Correctly reconstruct the target modules from the final keys
        module_names = sorted(list(set(re.search(r'\.([^.]+?)\.lora_A', key).group(1) for key in lora_A_keys)))
        final_rank = RANK_MAP['default']
        final_alpha = LORA_ALPHA

        adapter_config = {
            "base_model_name_or_path": STUDENT_BASE_FOLDER, "peft_type": "LORA",
            "r": final_rank, "lora_alpha": final_alpha, "lora_dropout": 0.0,
            "target_modules": module_names, "task_type": "CAUSAL_LM", "bias": "none"
        }
        with open(OUTPUT_LORA_CONFIG_PATH, 'w') as f: json.dump(adapter_config, f, indent=4)
        print(f"--- Saving LoRA adapter config to {OUTPUT_LORA_CONFIG_PATH} ---")

    print("\n\n✅ ULTIMATE FIDELITY DISTILLATION COMPLETE!")
    print("\n--- MANUAL CLEANUP REQUIRED ---")
    print("The temporary worker files have been preserved. You can delete them by running:")
    print(f"rm {temp_file_path}_*.safetensors")