import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipe2.unimepipe import UniMEPipeline
from pipe2.attacks import pgd_attack_unime
from PIL import Image
import numpy as np
import torch

def test_attack():
    print("Initializing pipeline...")
    try:
        pipeline = UniMEPipeline()
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        return

    print("Pipeline initialized.")

    # Create dummy image
    image = Image.open("worklink/MMEB-eval/MSCOCO_i2t/._COCO_val2014_000000000042.jpg")
    
    print("Testing PGD attack...")
    try:
        image.save("image.jpg")
        adv_image = pgd_attack_unime(image, pipeline, steps=2) # Low steps for speed
        print("Attack finished.")
        adv_image.save("adv_image.jpg")
        # Check if image changed
        diff = np.mean(np.abs(np.array(image) - np.array(adv_image)))
        print(f"Mean pixel difference: {diff}")
        
        if diff > 0:
            print("SUCCESS: Image was perturbed.")
        else:
            print("WARNING: Image was NOT perturbed (diff=0).")
            
    except Exception as e:
        print(f"Attack failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_attack()
