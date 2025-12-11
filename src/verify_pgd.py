
import torch
from PIL import Image
import numpy as np
from pipe.experiment import perturb
from pipe.pipe import MultimodalRetrievalPipeline

def test_pgd():
    print("Initializing pipeline...")
    # Use a smaller model for faster testing if possible, but let's stick to default to be safe
    pipeline = MultimodalRetrievalPipeline("openai/clip-vit-base-patch32") 
    
    print("Creating dummy image...")
    # Create a random image
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    print("Testing PGD perturbation...")
    try:
        perturbed_img = perturb(img, "pgd", pipeline)
        print("PGD perturbation successful!")
        
        # Check if image is modified
        img_np = np.array(img)
        p_img_np = np.array(perturbed_img)
        
        if np.array_equal(img_np, p_img_np):
            print("WARNING: Perturbed image is identical to original!")
        else:
            diff = np.abs(img_np.astype(int) - p_img_np.astype(int)).mean()
            print(f"Perturbed image differs from original. Mean pixel diff: {diff}")
            
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pgd()
