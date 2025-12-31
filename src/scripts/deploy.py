import sys
import os
import shutil
import time
from huggingface_hub import HfApi, login

def main():
    if len(sys.argv) < 3:
        print("Usage: deploy.py <adapter_name> <model_path>")
        sys.exit(1)

    adapter_name = sys.argv[1]
    model_path = sys.argv[2] 
    
    print(f"--- Starting Deployment ---")
    print(f"Adapter Name: {adapter_name}")
    print(f"Model Path: {model_path}")

    # 1. Upload to Hugging Face Hub
    try:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print("⚠️ HF_TOKEN not found. Skipping Hugging Face upload.")
        else:
            print("Found HF_TOKEN. Logging in...")
            login(token=hf_token)
            api = HfApi()
            
            # Repo name: username/model_name
            # We'll infer username from the token or use a default if available, 
            # for now let's assume we want to push to a new model repo.
            user_info = api.whoami()
            username = user_info['name']
            repo_id = f"{username}/{adapter_name}_merged"
            
            print(f"Uploading to Hugging Face Hub: {repo_id}...")
            api.create_repo(repo_id=repo_id, exist_ok=True)
            
            api.upload_folder(
                folder_path=model_path,
                repo_id=repo_id,
                repo_type="model"
            )
            print(f"✅ Successfully uploaded to https://huggingface.co/{repo_id}")
            
    except Exception as e:
        print(f"❌ Error uploading to Hugging Face: {e}")
        # We don't stop the pipeline for this, assuming local deploy might still be wanted.

    # 2. Local Deployment (Symlink)
    models_dir = os.path.dirname(model_path)
    production_link = os.path.join(models_dir, "production")

    print(f"Updating Local Production Symlink: {production_link} -> {model_path}")

    try:
        # Check if link exists
        if os.path.exists(production_link) or os.path.islink(production_link):
            # Try to remove it (files/links)
            if os.path.isdir(production_link) and not os.path.islink(production_link):
                shutil.rmtree(production_link)
            else:
                os.remove(production_link) 
        
        os.symlink(model_path, production_link)
        print("✅ Symlink updated successfully.")
        
    except OSError as e:
        print(f"⚠️ Warning: Could not create symlink locally (Permission Issue?): {e}")
        print("Attempting to write current_model_path.txt as fallback...")
        try:
            with open(os.path.join(models_dir, "current_model_path.txt"), "w") as f:
                f.write(model_path)
            print("✅ current_model_path.txt updated.")
        except Exception as e2:
             print(f"⚠️ Warning: Check local permissions. Could not write fallback file either: {e2}")

    print("--- Deployment Complete ---")

if __name__ == "__main__":
    main()
