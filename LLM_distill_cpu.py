# ====================================================================================
# ADVANCED MoE DISTILLATION SCRIPT (v7.4 - Final Complete Version)
#
# This is the full, complete, and final script. It incorporates all fixes:
# 1. Truly memory-efficient tensor-by-tensor processing.
# 2. Full K-Means clustering logic for MoE layers.
# 3. MKL stability checks for SVD.
# 4. The final `.contiguous()` fix to prevent the saving error.
# ====================================================================================
import torch
import torch.fft
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm
import os
import json
import re
import numpy as np
from sklearn.cluster import KMeans
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

# --- 3. LORA RANK CONFIGURATION ---
RANK_MAP = { "self_attn": 1536, "mlp": 1280, "block_sparse_moe": 2048, "default": 1024 }
LORA_ALPHA = 1.0

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_tensors_from_shards(keys_to_find, model_folder, weight_map):
    """Generic function to load a list of tensors from a sharded model directory."""
    shards_to_load = defaultdict(list)
    for key in keys_to_find:
        if key in weight_map:
            shards_to_load[weight_map[key]].append(key)
    
    tensors = {}
    for shard_file, keys_in_shard in shards_to_load.items():
        shard_path = os.path.join(model_folder, shard_file)
        shard_weights = load_file(shard_path, device="cpu")
        for key in keys_in_shard:
            if key in shard_weights:
                tensors[key] = shard_weights[key].to(torch.float32)
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

def project_tensor(teacher_tensor, student_shape):
    student_tensor_placeholder = torch.empty(student_shape)
    if teacher_tensor.dim() == 2 and student_tensor_placeholder.dim() == 2:
        try:
            teacher_mean = teacher_tensor.float().mean(); teacher_std = teacher_tensor.float().std() + 1e-9
            normalized_teacher = (teacher_tensor.float() - teacher_mean) / teacher_std
            U, S, Vh = torch.linalg.svd(normalized_teacher, full_matrices=False)
            target_out, target_in = student_shape; k = min(len(S), target_out, target_in)
            U_p, S_p, Vh_p = U[:, :k], torch.diag(S[:k]), Vh[:k, :]
            proj_norm = U_p @ S_p @ Vh_p
            final_norm = torch.zeros(student_shape, dtype=proj_norm.dtype, device=proj_norm.device)
            copy_out, copy_in = min(proj_norm.shape[0], target_out), min(proj_norm.shape[1], target_in)
            final_norm[:copy_out, :copy_in] = proj_norm[:copy_out, :copy_in]
            return (final_norm * teacher_std) + teacher_mean
        except torch.linalg.LinAlgError: return torch.zeros(student_shape)
    elif teacher_tensor.dim() == 1 and student_tensor_placeholder.dim() == 1:
        return project_tensor_fourier(teacher_tensor, student_shape)
    else: return torch.zeros(student_shape)

def distill_moe_layer(student_layer_idx, cfg, teacher_folder, teacher_weight_map, student_folder, student_weight_map):
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
    
    all_teacher_tensors = get_tensors_from_shards(all_teacher_keys, teacher_folder, teacher_weight_map)

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
            interpolated_fingerprints.append(((1 - interp_weight) * fp_floor + interp_weight * fp_ceil).numpy())

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
            student_tensor_shape = get_tensors_from_shards([student_key], student_folder, student_weight_map)[student_key].shape
            blended_tensor = torch.zeros(student_tensor_shape, dtype=torch.float32)
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
    """Final version with safety checks and the .contiguous() fix."""
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

# ==============================================================================
#                               MAIN EXECUTION
# ==============================================================================

# In your main script block, after loading the indexes:

if __name__ == "__main__":
    print("--- Starting Truly Memory-Efficient MoE Distillation (v7.4) ---")
    cfg = MODEL_ARCHITECTURE_CONFIG

    print("Loading model indexes...")
    with open(os.path.join(STUDENT_BASE_FOLDER, "model.safetensors.index.json"), 'r') as f:
        student_weight_map = json.load(f)['weight_map']
    with open(os.path.join(TEACHER_MODEL_FOLDER, "model.safetensors.index.json"), 'r') as f:
        teacher_weight_map = json.load(f)['weight_map']

    ### ================================================================== ###
    ###                  [NEW] PRE-FLIGHT VALIDATION BLOCK                 ###
    ### ================================================================== ###
    print("\n--- Validating student model architecture against configuration ---")
    
    discovered_max_layer = 0
    discovered_max_expert = 0
    
    for key in student_weight_map.keys():
        layer_match = re.search(r'model\.layers\.(\d+)\.', key)
        if layer_match:
            discovered_max_layer = max(discovered_max_layer, int(layer_match.group(1)))
        
        expert_match = re.search(r'experts\.(\d+)\.', key)
        if expert_match:
            discovered_max_expert = max(discovered_max_expert, int(expert_match.group(1)))

    # The number of layers/experts is max_index + 1
    discovered_layers = discovered_max_layer + 1
    discovered_experts = discovered_max_expert + 1

    print(f"Discovered student architecture: {discovered_layers} layers, {discovered_experts} experts per layer.")
    
    # Compare with configuration and fail fast if they don't match
    if discovered_layers != cfg['student_layers']:
        raise ValueError(f"Architecture mismatch! Config expects student_layers={cfg['student_layers']}, but found {discovered_layers} in the model.")
    
    if discovered_experts != cfg['student_experts_per_layer']:
        raise ValueError(f"Architecture mismatch! Config expects student_experts_per_layer={cfg['student_experts_per_layer']}, but found {discovered_experts} in the model.")
        
    print("✅ Configuration matches discovered architecture.\n")
    ### ================================================================== ###
    ###                     END OF VALIDATION BLOCK                      ###
    ### ================================================================== ###


    final_lora_weights = {}
    processed_moe_layers = set()
    student_keys = sorted(student_weight_map.keys())

    # ... (the rest of the script continues as before)

    print("\n--- Processing model tensor-by-tensor ---")
    for student_key in tqdm(student_keys, desc="Distilling Tensors"):
        if not ('.weight' in student_key and 'norm' not in student_key): continue

        key_match = re.search(r'model\.layers\.(\d+)\.', student_key)

        if key_match and 'block_sparse_moe' in student_key:
            student_layer_idx = int(key_match.group(1))
            if student_layer_idx in processed_moe_layers: continue
            
            print(f"\nDistilling MoE block for layer {student_layer_idx}...")
            moe_weights = distill_moe_layer(student_layer_idx, cfg, TEACHER_MODEL_FOLDER, teacher_weight_map, STUDENT_BASE_FOLDER, student_weight_map)
            for key, synthetic_tensor in moe_weights.items():
                student_tensor = get_tensors_from_shards([key], STUDENT_BASE_FOLDER, student_weight_map)[key]
                diff_tensor = synthetic_tensor - student_tensor
                rank = get_rank_for_key(key, RANK_MAP)
                lora_A, lora_B = extract_lora_from_diff(diff_tensor, rank)
                if lora_A is not None:
                    final_lora_weights[f"base_model.model.{key}.lora_A.weight"] = lora_A
                    final_lora_weights[f"base_model.model.{key}.lora_B.weight"] = lora_B
            processed_moe_layers.add(student_layer_idx)
            print(f"Finished MoE block for layer {student_layer_idx}.")
            continue

        student_tensor_dict = get_tensors_from_shards([student_key], STUDENT_BASE_FOLDER, student_weight_map)
        if student_key not in student_tensor_dict: continue
        student_tensor = student_tensor_dict[student_key]

        synthetic_tensor = None
        if not key_match:
            if student_key in teacher_weight_map:
                teacher_tensor_dict = get_tensors_from_shards([student_key], TEACHER_MODEL_FOLDER, teacher_weight_map)
                if student_key in teacher_tensor_dict:
                    synthetic_tensor = project_tensor(teacher_tensor_dict[student_key], student_tensor.shape)
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
            teacher_tensors = get_tensors_from_shards([key_floor, key_ceil], TEACHER_MODEL_FOLDER, teacher_weight_map)

            if key_floor in teacher_tensors and key_ceil in teacher_tensors:
                blended = (1 - interp_weight) * teacher_tensors[key_floor] + interp_weight * teacher_tensors[key_ceil]
                synthetic_tensor = project_tensor(blended, student_tensor.shape)

        if synthetic_tensor is not None:
            diff_tensor = synthetic_tensor - student_tensor
            rank = get_rank_for_key(student_key, RANK_MAP)
            lora_A, lora_B = extract_lora_from_diff(diff_tensor, rank)
            if lora_A is not None:
                final_lora_weights[f"base_model.model.{student_key}.lora_A.weight"] = lora_A
                final_lora_weights[f"base_model.model.{student_key}.lora_B.weight"] = lora_B

    print(f"\n--- Saving final LoRA weights to {OUTPUT_LORA_PATH} ---")
    save_file(final_lora_weights, OUTPUT_LORA_PATH)
    
    adapter_config = {
        "base_model_name_or_path": STUDENT_BASE_FOLDER,
        "peft_type": "LORA",
        "r": RANK_MAP['default'],
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": 0.0,
        "target_modules": sorted(list(set([ re.search(r'\.([^.]+)\.lora_A', key).group(1) for key in final_lora_weights.keys() ]))),
        "task_type": "CAUSAL_LM", "bias": "none"
    }
    with open(OUTPUT_LORA_CONFIG_PATH, 'w') as f:
        json.dump(adapter_config, f, indent=4)
    print(f"--- Saving LoRA adapter config to {OUTPUT_LORA_CONFIG_PATH} ---")

    print("\n\n✅ TRULY MEMORY-EFFICIENT DISTILLATION COMPLETE!")