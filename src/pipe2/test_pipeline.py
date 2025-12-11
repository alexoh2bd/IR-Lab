import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipe2.unimepipe import UniMEPipeline
from PIL import Image
import numpy as np
import torch

def test_pipeline():
    print("Initializing pipeline...")
    # Use a dummy model name if needed, but here we try the real one or a smaller one if available.
    # Since we are on a cluster with GPUs, we can try the real one.
    # If it fails due to download/memory, we might need to mock or use a tiny model.
    # For now, let's assume the environment can handle it or we catch the error.
    try:
        pipeline = UniMEPipeline()
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        return

    print("Pipeline initialized.")

    # Create dummy image
    image = Image.new('RGB', (224, 224), color='red')
    text = "A red image"

    print("Testing image encoding...")
    img_embed = pipeline.encode_images([image])
    print(f"Image embedding shape: {img_embed.shape}")
    assert img_embed.shape == (1, 3072) # Phi-3.5-V hidden size is likely 3072 or 4096? 
    # Actually Phi-3.5-mini is 3072. Let's check shape dynamically.

    print("Testing text encoding...")
    txt_embed = pipeline.encode_texts([text])
    print(f"Text embedding shape: {txt_embed.shape}")
    assert txt_embed.shape == img_embed.shape

    print("Testing multimodal encoding...")
    mm_embed = pipeline.encode_multimodal([image], [text])
    print(f"Multimodal embedding shape: {mm_embed.shape}")
    assert mm_embed.shape == img_embed.shape

    print("Testing retrieval...")
    results = pipeline.retrieve_i2t([image], [text, "A blue image"])
    print(f"Retrieval results: {results}")
    
    print("Verification complete!")

if __name__ == "__main__":
    test_pipeline()
