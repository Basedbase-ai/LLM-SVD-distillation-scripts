LOCAL_MODEL_PATH = "/media/workstation/NVME drive/distilled_GLM-4.6-Air"

import os
import subprocess
import sys
  # "q2_K",
  #  "q3_K_S",
   # "q3_K_M",
   # "q4_0",
   # "q4_K_M",
   # "q4_K_S",
   # "q5_K_M",
   # "q5_K_S",
   # "q6_K",
# ==============================================================================
#                                 CONFIGURATION
# ==============================================================================
#    This should be the final model ready for quantization.
LOCAL_MODEL_PATH = "/media/workstation/NVME drive/distilled_GLM-4.6-Air"

# --- 2. Path to your cloned and built llama.cpp repository ---
#    This is the top-level directory of the llama.cpp project.
LLAMA_CPP_PATH = "/home/workstation/Desktop/llama.cpp"

# --- 3. Path to the directory where you want to save the quantized models ---
#    The script will create this directory if it doesn't exist.
OUTPUT_DIRECTORY = "/media/workstation/4tb/quantized_GLM-4.6-Air"

# --- 4. List of the quantization levels you want to create ---
#    These are the standard quantization methods supported by llama.cpp.
#    You can add or remove methods from this list as needed.
QUANTIZATION_LEVELS = [
    "q2_K",
    "q8_0"
]  # <-- This closing bracket was missing

# --- 5. Keep the F16 intermediate file? ---
#    If False, the F16 file will be deleted after all quantizations are complete
#    Set to True if you want to keep it for future quantizations
KEEP_F16_FILE = True

# ==============================================================================
#                           END OF CONFIGURATION
# ==============================================================================


def run_command(command, description):
    """
    Runs a command in a subprocess, prints its status, and handles errors.
    """
    print(f"--- Running: {description} ---")
    print(f"Command: {' '.join(command)}\n")
    try:
        # Using capture_output=True to hide the verbose output of the tools
        # unless an error occurs.
        process = subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"--- Successfully completed: {description} ---\n")
        return True
    except FileNotFoundError as e:
        print(f"--- FATAL ERROR during: {description} ---", file=sys.stderr)
        print(f"Error: The program '{command[0]}' was not found.", file=sys.stderr)
        print("Please ensure your LLAMA_CPP_PATH is correct and the project is built.", file=sys.stderr)
        return False
    except subprocess.CalledProcessError as e:
        print(f"--- FATAL ERROR during: {description} ---", file=sys.stderr)
        print(f"The command returned a non-zero exit status: {e.returncode}", file=sys.stderr)
        print("\n--- STDOUT ---", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        print("\n--- STDERR ---", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        return False

def main():
    """
    Main function to orchestrate the model conversion and quantization process.
    """
    print("--- Starting GGUF Quantization Process ---")

    # Create the output directory if it does not exist
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    # --- Step 0: Define paths to the necessary llama.cpp tools ---
    # The Python script for converting Hugging Face models to GGUF
    convert_script = os.path.join(LLAMA_CPP_PATH, "convert_hf_to_gguf.py")

    # The C++ executable for quantizing GGUF files
    # Note: On newer llama.cpp versions, this is in build/bin/
    quantize_executable = os.path.join(LLAMA_CPP_PATH, "build", "bin", "llama-quantize")
    
    # Path for the intermediate, unquantized F16 GGUF model
    base_gguf_model_path = os.path.join(OUTPUT_DIRECTORY, "model-F16.gguf")

    # --- Step 1: Check if F16 model already exists ---
    if os.path.exists(base_gguf_model_path):
        print(f"--- F16 model already exists at: {base_gguf_model_path} ---")
        print("--- Skipping conversion step ---\n")
    else:
        # --- Step 1: Convert the Hugging Face model to a base F16 GGUF model ---
        # This F16 model will be the source for all subsequent quantizations.
        convert_command = [
            sys.executable,        # Use the current python interpreter
            convert_script,
            LOCAL_MODEL_PATH,
            "--outfile",
            base_gguf_model_path,
            "--outtype",
            "f16"                  # f16 is the standard base for quantization
        ]
        if not run_command(convert_command, "Convert HF model to F16 GGUF"):
            sys.exit(1) # Exit if the crucial first step fails

    # --- Step 2: Loop through the desired levels and quantize the F16 model ---
    for level in QUANTIZATION_LEVELS:
        # Define the output path for the new quantized model
        output_filename_path = os.path.join(OUTPUT_DIRECTORY, f"model-{level.upper()}.gguf")
        
        quantize_command = [
            quantize_executable,
            base_gguf_model_path,
            output_filename_path,
            level
        ]
        
        # If any quantization level fails, stop the script to allow for debugging.
        if not run_command(quantize_command, f"Quantizing to {level.upper()}"):
            print("--- Halting script due to previous error. ---", file=sys.stderr)
            break
    
    # --- Step 3: Optionally delete the F16 intermediate file ---
    if not KEEP_F16_FILE and os.path.exists(base_gguf_model_path):
        print(f"--- Deleting intermediate F16 file: {base_gguf_model_path} ---")
        try:
            os.remove(base_gguf_model_path)
            print("--- F16 file deleted successfully ---\n")
        except Exception as e:
            print(f"--- Warning: Could not delete F16 file: {e} ---\n", file=sys.stderr)
    
    print("--- All quantization tasks are complete! ---")
    print(f"Your quantized models are located in: {OUTPUT_DIRECTORY}")
    if KEEP_F16_FILE:
        print(f"F16 intermediate file is available at: {base_gguf_model_path}")

if __name__ == "__main__":
    main()