#!/usr/bin/env python3
"""
Download CrowdHuman dataset from HuggingFace.

The CrowdHuman dataset contains:
- 15,000 training images
- 4,370 validation images  
- 5,000 test images
- ~470K human instances with ~23 persons per image

Reference: https://huggingface.co/datasets/sshao0516/CrowdHuman
Paper: https://arxiv.org/pdf/1805.00123
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm


def download_crowdhuman(output_dir: str = "./datasets/crowdhuman_raw"):
    """Download CrowdHuman dataset from HuggingFace."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    repo_id = "sshao0516/CrowdHuman"
    
    print(f"📦 Downloading CrowdHuman dataset to {output_path.absolute()}")
    print("=" * 60)
    
    # List all files in the repository
    print("🔍 Fetching file list from HuggingFace...")
    try:
        files = list_repo_files(repo_id, repo_type="dataset")
    except Exception as e:
        print(f"❌ Error listing files: {e}")
        return
    
    print(f"📁 Found {len(files)} files in repository")
    
    # Files we need to download
    required_files = [
        "annotation_train.odgt",
        "annotation_val.odgt",
        "CrowdHuman_train01.zip",
        "CrowdHuman_train02.zip", 
        "CrowdHuman_train03.zip",
        "CrowdHuman_val.zip",
    ]
    
    # Filter to only required files that exist
    files_to_download = [f for f in required_files if f in files]
    
    print(f"\n📥 Will download {len(files_to_download)} files:")
    for f in files_to_download:
        print(f"   - {f}")
    
    print("\n⚠️  Note: This is a large dataset (~20GB+). Download may take a while.")
    print("=" * 60)
    
    # Download each file
    for filename in tqdm(files_to_download, desc="Downloading files"):
        local_path = output_path / filename
        
        if local_path.exists():
            print(f"⏭️  Skipping {filename} (already exists)")
            continue
            
        print(f"\n📥 Downloading {filename}...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=output_path,
                local_dir_use_symlinks=False,
            )
            print(f"✅ Downloaded: {downloaded_path}")
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("✅ Download complete!")
    print(f"📁 Files saved to: {output_path.absolute()}")
    print("\n📝 Next steps:")
    print("   1. Extract the zip files")
    print("   2. Run convert_to_yolo.py to convert annotations")
    print("   3. Run train.py to start training")
    
    # Provide extraction commands
    print("\n💡 To extract all zip files, run:")
    print(f"   cd {output_path}")
    print("   unzip '*.zip'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download CrowdHuman dataset from HuggingFace"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./datasets/crowdhuman_raw",
        help="Directory to save downloaded files (default: ./datasets/crowdhuman_raw)"
    )
    
    args = parser.parse_args()
    download_crowdhuman(args.output_dir)

