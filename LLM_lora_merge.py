# ==============================================================================
# BARE METAL FP32 MERGE SCRIPT (v4.1 - Corrected Loop Logic)
#
# This version fixes the critical FileNotFoundError by correctly iterating
# over the shard FILENAMES instead of the TENSOR names.
# ==============================================================================
import torch
import json
import gc
from safetensors.torch import load_file, save_file
from transformers import AutoConfig
import os

# --- 1. Define Paths ---
BASE_MODEL_PATH = "/home/workstation/Desktop/qwen30b"
LORA_SAFETENSORS_PATH = "/media/workstation/crucial/adapter_model.safetensors"
LORA_CONFIG_PATH = "/media/workstation/crucial"
MERGED_MODEL_OUTPUT_PATH = "/media/workstation/crucial/qwen30bcoderdistill"

if __name__ == "__main__":
    print("--- Starting BARE METAL LoRA merge process in float32 ---")
    os.makedirs(MERGED_MODEL_OUTPUT_PATH, exist_ok=True)

    # --- 2. Load Configs and Copy Necessary Files ---
    print("Loading configs and copying tokenizer...")
    base_model_config = AutoConfig.from_pretrained(BASE_MODEL_PATH)
    lora_config = json.load(open(os.path.join(LORA_CONFIG_PATH, "adapter_config.json")))
    
    for filename in os.listdir(BASE_MODEL_PATH):
        if 'tokenizer' in filename or 'token' in filename or '.tiktoken' in filename:
            print(f"Copying {filename}...")
            os.system(f'cp "{os.path.join(BASE_MODEL_PATH, filename)}" "{MERGED_MODEL_OUTPUT_PATH}"')

    base_model_config.save_pretrained(MERGED_MODEL_OUTPUT_PATH)

    # --- 3. Load the entire high-rank LoRA into memory ---
    print(f"Loading LoRA weights from: {LORA_SAFETENSORS_PATH}")
    lora_state_dict = load_file(LORA_SAFETENSORS_PATH, device="cpu")

    # --- 4. Get LoRA scaling factor ---
    scaling = lora_config["lora_alpha"] / lora_config["r"]
    print(f"LoRA config: r={lora_config['r']}, alpha={lora_config['lora_alpha']}, scaling={scaling}")

    # --- 5. Locate the base model's index file ---
    index_path = os.path.join(BASE_MODEL_PATH, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise ValueError(f"Base model must have an index file at {index_path}")
    with open(index_path, 'r') as f:
        base_model_index = json.load(f)
    
    # --- 6. Process each base model shard individually ---
    # THE FIX IS HERE: Get the unique list of shard filenames from the index's VALUES.
    shard_filenames = sorted(list(set(base_model_index["weight_map"].values())))
    
    new_weight_map = {}
    for shard_filename in shard_filenames: # <-- Now looping over the correct filenames
        print(f"\n--- Processing Shard: {shard_filename} ---")
        
        shard_path = os.path.join(BASE_MODEL_PATH, shard_filename)
        base_shard_state_dict = load_file(shard_path, device="cpu")
        
        merged_shard_state_dict = {}
        
        for tensor_name, tensor in base_shard_state_dict.items():
            tensor = tensor.to(torch.float32)
            
            lora_a_name = f"base_model.model.{tensor_name.replace('.weight', '.lora_A.weight')}"
            lora_b_name = f"base_model.model.{tensor_name.replace('.weight', '.lora_B.weight')}"

            if lora_a_name in lora_state_dict:
                lora_A = lora_state_dict[lora_a_name].to(torch.float32)
                lora_B = lora_state_dict[lora_b_name].to(torch.float32)
                lora_delta = (lora_B @ lora_A) * scaling
                
                merged_shard_state_dict[tensor_name] = tensor + lora_delta
                print(f"  Merged {tensor_name}")
            else:
                merged_shard_state_dict[tensor_name] = tensor
        
        output_shard_path = os.path.join(MERGED_MODEL_OUTPUT_PATH, shard_filename)
        print(f"Saving merged shard to: {output_shard_path}")
        save_file(merged_shard_state_dict, output_shard_path, metadata={'format': 'pt'})
        
        # Update the new index for all tensors that were in this original shard
        for original_tensor_name, original_shard_filename in base_model_index["weight_map"].items():
            if original_shard_filename == shard_filename:
                new_weight_map[original_tensor_name] = shard_filename
            
        print("Cleaning up memory...")
        del base_shard_state_dict, merged_shard_state_dict
        gc.collect()

    # --- 7. Create and save the final index file ---
    print("\n--- Finalizing Merge ---")
    final_index = {"metadata": base_model_index["metadata"], "weight_map": new_weight_map}
    with open(os.path.join(MERGED_MODEL_OUTPUT_PATH, "model.safetensors.index.json"), 'w') as f:
        json.dump(final_index, f, indent=4)

    print("\n\n✅ BARE METAL FP32 Merge Successful! Your masterpiece is ready.")