import sys
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

def main():
    if len(sys.argv) < 3:
        print("Usage: merge.py <adapter_path> <output_merged_path>")
        sys.exit(1)

    adapter_path = sys.argv[1]
    output_path = sys.argv[2]
    
    print(f"--- Starting Merge ---")
    print(f"Loading Base: {MODEL_ID}")
    
    # 1. Load Base Model (Must be in float16 for merging)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    # 2. Load Adapter
    print(f"Loading Adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # 3. Merge
    print("Merging LoRA into Base Model...")
    merged_model = model.merge_and_unload()
    
    # 4. Save
    print(f"Saving Merged Model to: {output_path}")
    merged_model.save_pretrained(output_path)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(output_path)
    print("--- Merge Complete ---")

if __name__ == "__main__":
    main()