"""
Script to download all Olmo models to disk for storage.
Uses huggingface_hub to download files without loading them into memory/GPU.
"""
import sys
from pathlib import Path
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: huggingface_hub is not installed.")
    print("Please run: pip install huggingface_hub")
    sys.exit(1)

# Model configuration
MODELS = {
    "base": "allenai/Olmo-3-1025-7B",
    "sft": "allenai/Olmo-3-7B-Instruct-SFT",
    "dpo": "allenai/Olmo-3-7B-Instruct-DPO",
    "rlvr": "allenai/Olmo-3-7B-Instruct",
}

def main():
    # Define root and models directory
    root_dir = Path(__file__).parent.parent
    models_dir = root_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("Starting Olmo models download script...")
    print(f"Target Storage Directory: {models_dir}\n")
    
    for model_key, model_name in MODELS.items():
        print(f"{'='*60}")
        print(f"Downloading {model_key}: {model_name}")
        print(f"{'='*60}")
        
        try:
            # snapshot_download fetches all files in the repo to the cache_dir.
            # It checks if files exist first, so it's safe to re-run.
            local_path = snapshot_download(
                repo_id=model_name,
                cache_dir=models_dir,
                # Optional: local_files_only=False ensures we actually connect to the internet
            )
            print(f"✓ Successfully stored {model_name}")
            print(f"  Location: {local_path}\n")
            
        except Exception as e:
            print(f"✗ Error downloading {model_name}: {e}\n")
            continue
    
    print(f"{'='*60}")
    print("Download process complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()