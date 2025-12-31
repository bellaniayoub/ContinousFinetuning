import sys
import os
import torch
import json
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate import load
from datasets import load_dataset

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

def main():
    if len(sys.argv) < 1:
        print("Usage: run_evaluation.py <adapter_name>")
        sys.exit(1)

    adapter_name = sys.argv[1]
    
    # Define EXACT paths to match train.py
    base_dir = os.getcwd()
    adapter_path = os.path.join(base_dir, "models", "adapters", adapter_name)
    test_data_path = os.path.join(base_dir, "data", "processed", f"test_{adapter_name}.jsonl")
    results_path = os.path.join(adapter_path, "eval_results.txt")

    print(f"--- Starting Evaluation ---")
    print(f"Looking for Adapter at: {adapter_path}")
    
    # Check if adapter exists
    if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        print(f"❌ ERROR: Adapter not found at {adapter_path}")
        print("Did the Training task finish successfully?")
        sys.exit(1)

    # 1. Load Metrics
    bleu = load("bleu")
    bert = load("bertscore")
    
    # 2. Load Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    # 3. Evaluation Loop
    dataset = load_dataset("json", data_files=test_data_path, split="train")
    dataset = dataset.select(range(min(10, len(dataset)))) # Test on 10 samples
    
    predictions = []
    references = []
    
    print("Generating predictions...")
    for item in dataset:
        prompt = item["messages"][0]["content"]
        true_label = item["messages"][1]["content"]
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)
        
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
        predictions.append(pred)
        references.append([true_label]) # BLEU expects list of lists

    # 4. Calc Score
    # BLEU
    bleu_results = bleu.compute(predictions=predictions, references=references)
    bleu_score = bleu_results['bleu']
    print(f"BLEU Score: {bleu_score}")

    # BERTScore
    # references for bert must be list of strings, not list of lists
    flat_references = [r[0] for r in references]
    bert_results = bert.compute(predictions=predictions, references=flat_references, lang="en")
    bert_f1 = sum(bert_results['f1']) / len(bert_results['f1'])
    print(f"BERTScore F1: {bert_f1}")

    # 5. Save Result for Airflow Branching
    with open(results_path, "w") as f:
        f.write(f"BLEU: {bleu_score}\n")
        f.write(f"BERTScore: {bert_f1}")

if __name__ == "__main__":
    main()