"""
Quick test script to verify Jina model works properly before running full pipeline.
Tests basic functionality: loading model, encoding images, encoding text, and computing similarity.
"""

import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoProcessor
from datetime import datetime
import sys

class Config:
    """Configuration for testing."""
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16

def create_test_image(size=(224, 224), color=(255, 0, 0)):
    """Create a simple test image."""
    return Image.new('RGB', size, color)

def test_jina_model():
    """Test the Jina model with basic operations."""
    
    print("="*60)
    print("JINA MODEL TEST")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {Config.DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("="*60)
    
    model_name = "jinaai/jina-clip-v2"
    
    try:
        # Test 1: Load model
        print("\n[Test 1/5] Loading Jina model...")
        model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            torch_dtype=Config.DTYPE
        ).to(Config.DEVICE)
        processor = AutoProcessor.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        model.eval()
        print("✓ Model loaded successfully")
        
        # Test 2: Create test data
        print("\n[Test 2/5] Creating test data...")
        test_images = [
            create_test_image(color=(255, 0, 0)),    # Red
            create_test_image(color=(0, 255, 0)),    # Green
            create_test_image(color=(0, 0, 255)),    # Blue
        ]
        test_texts = [
            "A red colored image",
            "A green colored image", 
            "A blue colored image",
            "An unrelated text about cats"
        ]
        print(f"✓ Created {len(test_images)} test images and {len(test_texts)} test texts")
        
        # Test 3: Encode images
        print("\n[Test 3/5] Encoding images...")
        with torch.no_grad():
            image_inputs = processor(
                images=test_images,
                return_tensors="pt",
                padding=True
            )
            image_inputs = {k: v.to(Config.DEVICE) for k, v in image_inputs.items() 
                           if isinstance(v, torch.Tensor)}
            
            image_embeds = model.get_image_features(**image_inputs)
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            image_embeds_np = image_embeds.cpu().numpy()
            
        print(f"✓ Image embeddings shape: {image_embeds_np.shape}")
        print(f"  Embedding dimension: {image_embeds_np.shape[1]}")
        print(f"  Norm check (should be ~1.0): {np.linalg.norm(image_embeds_np[0]):.4f}")
        
        # Test 4: Encode texts
        print("\n[Test 4/5] Encoding texts...")
        with torch.no_grad():
            text_inputs = processor(
                text=test_texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            text_inputs = {k: v.to(Config.DEVICE) for k, v in text_inputs.items() 
                          if isinstance(v, torch.Tensor)}
            
            text_embeds = model.get_text_features(**text_inputs)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds_np = text_embeds.cpu().numpy()
            
        print(f"✓ Text embeddings shape: {text_embeds_np.shape}")
        print(f"  Embedding dimension: {text_embeds_np.shape[1]}")
        print(f"  Norm check (should be ~1.0): {np.linalg.norm(text_embeds_np[0]):.4f}")
        
        # Test 5: Compute similarity
        print("\n[Test 5/5] Computing similarity matrix...")
        similarity = np.matmul(image_embeds_np, text_embeds_np.T)
        print(f"✓ Similarity matrix shape: {similarity.shape}")
        print(f"\nSimilarity scores (Image x Text):")
        print("  " + " ".join([f"Text{i}" for i in range(len(test_texts))]))
        for i, row in enumerate(similarity):
            print(f"Img{i}: " + " ".join([f"{score:6.3f}" for score in row]))
        
        # Verify diagonal should have higher scores (matching descriptions)
        print("\n[Verification] Checking if matching pairs have high similarity...")
        for i in range(min(len(test_images), len(test_texts)-1)):
            if similarity[i, i] > 0.2:  # Reasonable threshold
                print(f"✓ Image {i} matches Text {i}: {similarity[i, i]:.3f}")
            else:
                print(f"⚠ Image {i} - Text {i} similarity is low: {similarity[i, i]:.3f}")
        
        # Memory cleanup
        print("\n[Cleanup] Clearing GPU memory...")
        del model, processor, image_embeds, text_embeds, image_inputs, text_inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✓ Cleanup complete")
        
        # Final summary
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED with error:")
        print(f"  {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_jina_model()
    sys.exit(0 if success else 1)
