import pandas as pd
from datasets import load_dataset
import os
import json

# --- Configuration ---
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def format_chat(row):
    # Convert Bitext format to Chat format (OpenAI style)
    # The dataset uses "instruction" (User) and "response" (Bot)
    return {
        "messages": [
            {"role": "user", "content": row['instruction']},
            {"role": "assistant", "content": row['response']}
        ]
    }

def main():
    print("Downloading dataset 'bitext/Bitext-customer-support-llm-chatbot-training-dataset'...")
    # Load dataset from Hugging Face
    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
    df = ds.to_pandas()
    
    # Define our batches to simulate time/evolution
    batches = {
        "batch_1_account": ["ACCOUNT"],
        "batch_2_logistics": ["SHIPPING", "DELIVERY"], # Combined for a "Logistics" update
        "batch_3_money": ["CANCEL", "REFUND"]          # Combined for a "Policy" update
    }
    
    print(f"Total raw examples: {len(df)}")
    
    for batch_name, categories in batches.items():
        # 1. Filter data
        batch_df = df[df['category'].isin(categories)].copy()
        
        # 2. Format
        formatted_data = batch_df.apply(format_chat, axis=1).tolist()
        
        # 3. Save
        output_path = f"{OUTPUT_DIR}/{batch_name}.jsonl"
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in formatted_data:
                json.dump(entry, f)
                f.write('\n')
                
        print(f"Saved {batch_name}: {len(formatted_data)} examples -> {output_path}")

if __name__ == "__main__":
    main()


