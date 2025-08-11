# fix_config.py
import json
import re
from safetensors.torch import load_file

# --- CONFIGURE THIS --- --- Change to where your files are stored --- #
LORA_FILE_PATH = "/media/workstation/crucial/distilled_qwen3_v2.safetensors"
OUTPUT_CONFIG_PATH = "/media/workstation/crucial/adapter_config_v2_FIXED.json"
BASE_MODEL_PATH = "/home/workstation/Desktop/qwen30b"
LORA_RANK = 2048
LORA_ALPHA = 2048
# --------------------

print(f"--- Loading LoRA file to extract keys from: {LORA_FILE_PATH} ---")
lora_weights = load_file(LORA_FILE_PATH)

lora_A_keys = [key for key in lora_weights.keys() if key.endswith(".lora_A.weight")]

if not lora_A_keys:
    print("\n\n❌ CRITICAL FAILURE: No LoRA A weights were found in the file.")
else:
    # --- THIS IS THE CORRECTED REGEX ---
    # It correctly finds module names like 'q_proj', 'k_proj', 'up_proj', etc.
    module_names = sorted(list(set(re.search(r'\.([^.]+?)\.weight\.lora_A', key).group(1) for key in lora_A_keys)))
    
    print(f"--- Found {len(module_names)} target modules: ---")
    print(module_names)

    adapter_config = {
        "base_model_name_or_path": BASE_MODEL_PATH,
        "peft_type": "LORA",
        "r": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": 0.0,
        "target_modules": module_names,
        "task_type": "CAUSAL_LM",
        "bias": "none"
    }

    with open(OUTPUT_CONFIG_PATH, 'w') as f:
        json.dump(adapter_config, f, indent=4)
        
    print(f"\n--- ✅ Successfully saved corrected LoRA adapter config to: {OUTPUT_CONFIG_PATH} ---")
