import sys
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# --- Configuration ---
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

def main():
    # 0. Parse Arguments
    if len(sys.argv) < 3:
        data_path = "data/processed/batch_1_account.jsonl"
        adapter_name = "test_adapter"
    else:
        data_path = sys.argv[1]
        adapter_name = sys.argv[2]
        
    dataset_name = os.path.basename(data_path).replace(".jsonl", "")
    # New Output Directory Format: results/result_{adapter_name}_{dataset_name}
    output_dir = f"results/result_{adapter_name}_{dataset_name}"
    
    print(f"--- Starting Training ---")
    print(f"Data: {data_path}")
    print(f"Adapter Output Name: {adapter_name}")
    print(f"Results Directory: {output_dir}")

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded.")
    
    # 2. Configure 4-bit Loading (Windows Safe Mode)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # Force float16
        bnb_4bit_use_double_quant=False,
    )
    
    # 3. Load Base Model
    print("Loading Base Model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print("Base Model loaded.")
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()
    
    # 4. LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        inference_mode=False,
    )
    
    # 5. Apply LoRA
    model = get_peft_model(model, peft_config)
    print("LoRA applied.")
    
    # 6. Windows Compatibility Fix (Float32 cast for trainable params)
    print("Applying Windows Compatibility Fix...")
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.dtype != torch.float32:
                param.data = param.data.to(torch.float32)
                
    for name, buffer in model.named_buffers():
        if buffer.dtype == torch.bfloat16:
            buffer.data = buffer.data.to(torch.float16)
            
    model.config.torch_dtype = torch.float16
    print("Compatibility fix applied.")
    
    # 7. Load and Split Dataset
    print("Loading and splitting dataset...")
    full_dataset = load_dataset("json", data_files=data_path, split="train")
    
    # Split: 80% Train, 10% Val, 10% Test
    # 1. Split Train (80%) vs Temp (20%)
    train_temp = full_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_temp['train']
    temp_dataset = train_temp['test']
    
    # 2. Split Temp (20%) -> Val (10% total) and Test (10% total)
    # Since Temp is 20% of total, we split it 50/50
    val_test = temp_dataset.train_test_split(test_size=0.5, seed=42)
    eval_dataset = val_test['train'] # This is validation
    test_dataset = val_test['test']
    
    print(f"Dataset Splits: Train={len(train_dataset)}, Val={len(eval_dataset)}, Test={len(test_dataset)}")
    
    # Save the Test set to data/processed
    test_file_path = f"data/processed/test_{adapter_name}.jsonl"
    test_dataset.to_json(test_file_path)
    print(f"Saved Test split to: {test_file_path}")
    
    # 8. Training Arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4, # Restored user preference
        
        # Precision
        fp16=False,
        bf16=False,
        
        # Evaluation & Logging
        eval_strategy="steps", # Updated from evaluation_strategy
        eval_steps=10, 
        logging_steps=5,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True, 
        
        # Optimizer
        optim="paged_adamw_8bit", # Restored user preference
        
        # Misc
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        dataset_text_field="messages",
        packing=False,
        report_to="none",
    )
    
    # 9. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, # Pass validation set here
        args=training_args,
        processing_class=tokenizer,
    )
    
    print("Starting training...")
    
    try:
        trainer.train()
    except RuntimeError as e:
        if "BFloat16" in str(e):
            print("\n!!! BFloat16 error detected !!!")
            raise e
        else:
            raise e
    
    # 10. Save Adapter
    final_adapter_path = f"models/adapters/{adapter_name}"
    print(f"\nSaving final adapter to {final_adapter_path}...")
    trainer.model.save_pretrained(final_adapter_path)
    tokenizer.save_pretrained(final_adapter_path)
    
    print("\n--- Training Complete ---")

if __name__ == "__main__":
    main()


# import sys
# import os
# import torch
# from datasets import load_dataset
# from transformers import (
#     AutoTokenizer, 
#     AutoModelForCausalLM, 
#     BitsAndBytesConfig,
# )
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# from trl import SFTTrainer, SFTConfig

# # --- Configuration ---
# MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# def main():
#     # 0. Parse Arguments
#     if len(sys.argv) < 3:
#         data_path = "data/processed/batch_1_account.jsonl"
#         adapter_name = "test_adapter"
#     else:
#         data_path = sys.argv[1]
#         adapter_name = sys.argv[2]
        
#     dataset_name = os.path.basename(data_path).replace(".jsonl", "")
#     output_dir = f"results/result_{adapter_name}_{dataset_name}"
    
#     print(f"--- Starting Training ---")
#     print(f"Data: {data_path}")
#     print(f"Adapter Output Name: {adapter_name}")

#     # 1. Load Tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
    
#     # 2. Configure 4-bit Loading
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.float16, 
#         bnb_4bit_use_double_quant=False,
#     )
    
#     # 3. Load Base Model
#     print("Loading Base Model...")
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_ID,
#         quantization_config=bnb_config,
#         device_map="auto",
#         torch_dtype=torch.float16,
#         low_cpu_mem_usage=True,
#         trust_remote_code=True,
#     )
    
#     model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
#     model.gradient_checkpointing_enable()
    
#     # 4. LoRA Config
#     peft_config = LoraConfig(
#         r=16,
#         lora_alpha=32,
#         lora_dropout=0.05,
#         bias="none",
#         task_type="CAUSAL_LM",
#         target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
#         inference_mode=False,
#     )
    
#     # 5. Apply LoRA
#     model = get_peft_model(model, peft_config)

#     # 6. CRITICAL FIX 1: Force Parameters to GPU ("cuda")
#     print("Applying Windows Compatibility Fix (Float32 + CUDA)...")
#     for name, param in model.named_parameters():
#         if param.requires_grad:
#             # We force them to Float32 AND move them to the GPU
#             param.data = param.data.to(torch.float32).to("cuda") 
            
#     for name, buffer in model.named_buffers():
#         if buffer.dtype == torch.bfloat16:
#             buffer.data = buffer.data.to(torch.float16).to("cuda")
            
#     model.config.torch_dtype = torch.float16
#     print("Compatibility fix applied.")
    
#     # 7. Load and Split Dataset
#     print("Loading and splitting dataset...")
#     full_dataset = load_dataset("json", data_files=data_path, split="train")
    
#     train_temp = full_dataset.train_test_split(test_size=0.2, seed=42)
#     train_dataset = train_temp['train']
#     temp_dataset = train_temp['test']
    
#     val_test = temp_dataset.train_test_split(test_size=0.5, seed=42)
#     eval_dataset = val_test['train'] 
#     test_dataset = val_test['test']
    
#     print(f"Dataset Splits: Train={len(train_dataset)}, Val={len(eval_dataset)}, Test={len(test_dataset)}")
    
#     test_file_path = f"data/processed/test_{adapter_name}.jsonl"
#     test_dataset.to_json(test_file_path)
    
#     # 8. Training Arguments
#     training_args = SFTConfig(
#         output_dir=output_dir,
#         num_train_epochs=1,
#         per_device_train_batch_size=1,
#         gradient_accumulation_steps=4,
#         learning_rate=2e-4, 
        
#         # Precision
#         fp16=True,  # Set to True for standard Mixed Precision
#         bf16=False,
        
#         # Evaluation
#         eval_strategy="steps", 
#         eval_steps=10, 
#         logging_steps=5,
#         logging_first_step=True,
#         save_strategy="steps",
#         save_steps=50,
#         save_total_limit=2,
#         load_best_model_at_end=True, 
        
#         # CRITICAL FIX 2: Use Standard Optimizer
#         optim="adamw_torch", 
        
#         max_grad_norm=0.3,
#         warmup_ratio=0.03,
#         dataset_text_field="messages",
#         packing=False,
#         report_to="none",
#     )
    
#     # 9. Initialize Trainer
#     trainer = SFTTrainer(
#         model=model,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset, 
#         args=training_args,
#         processing_class=tokenizer,
#     )
    
#     print("Starting training...")
#     trainer.train()
    
#     # 10. Save Adapter
#     final_adapter_path = f"models/adapters/{adapter_name}"
#     print(f"\nSaving final adapter to {final_adapter_path}...")
#     trainer.model.save_pretrained(final_adapter_path)
#     tokenizer.save_pretrained(final_adapter_path)
    
#     print("\n--- Training Complete ---")

# if __name__ == "__main__":
#     main()