import json
import os

# Path to the merged model
MODEL_DIR = r"c:\Users\AM\Desktop\MLopsProject\models\adapters\merged_adapter_v2_docker"
CONFIG_PATH = os.path.join(MODEL_DIR, "tokenizer_config.json")
TEMPLATE_PATH = os.path.join(MODEL_DIR, "chat_template.jinja")

def fix_chat_template():
    if not os.path.exists(CONFIG_PATH) or not os.path.exists(TEMPLATE_PATH):
        print(f"Error: Files not found in {MODEL_DIR}")
        return

    # Read the Jinja template
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        chat_template = f.read()

    # Read the Tokenizer Config
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Inject the template
    config["chat_template"] = chat_template
    print("Injecting chat_template into tokenizer_config.json...")

    # Save back
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("✅ Successfully updated tokenizer_config.json")

if __name__ == "__main__":
    fix_chat_template()
