#!/usr/bin/env python3
"""
Download and extract images from TIGER-Lab/MMEB-eval dataset.
"""

from huggingface_hub import hf_hub_download
import zipfile
import os
from pathlib import Path

def download_and_extract_images(target_dir="/work/aho13/"):
    """
    Download images.zip from TIGER-Lab/MMEB-eval and extract to target directory.
    
    Args:
        target_dir: Directory where images should be extracted
    """
    print(f"Target directory: {target_dir}")
    
    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)
    
    # Download images.zip from HuggingFace
    print("Downloading images.zip from TIGER-Lab/MMEB-eval...")
    try:
        zip_path = hf_hub_download(
            repo_id="TIGER-Lab/MMEB-eval",
            filename="images.zip",
            repo_type="dataset",
            cache_dir=None  # Use default cache
        )
        print(f"Downloaded to: {zip_path}")
    except Exception as e:
        print(f"Error downloading file: {e}")
        return
    
    # Extract the zip file
    print(f"Extracting images to {target_dir}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get total number of files for progress tracking
            total_files = len(zip_ref.namelist())
            print(f"Extracting {total_files} files...")
            
            # Extract with progress
            for i, member in enumerate(zip_ref.namelist(), 1):
                zip_ref.extract(member, target_dir)
                if i % 1000 == 0:
                    print(f"  Extracted {i}/{total_files} files...")
            
            print(f"✓ Successfully extracted all {total_files} files")
    except Exception as e:
        print(f"Error extracting zip file: {e}")
        return
    
    # Verify extraction
    print("\nVerifying extraction...")
    extracted_items = os.listdir(target_dir)
    print(f"Items in {target_dir}:")
    for item in sorted(extracted_items)[:20]:  # Show first 20 items
        item_path = os.path.join(target_dir, item)
        if os.path.isdir(item_path):
            file_count = len(os.listdir(item_path))
            print(f"  {item}/ ({file_count} items)")
        else:
            print(f"  {item}")
    
    if len(extracted_items) > 20:
        print(f"  ... and {len(extracted_items) - 20} more items")
    
    print("\n✓ Download and extraction complete!")

if __name__ == "__main__":
    download_and_extract_images()
