import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipe2.unimepipe import UniMEPipeline
from pipe2.attacks import pgd_attack_to_target
from PIL import Image
import torch

def verify_target_attack():
    print("Initializing pipeline...")
    try:
        pipeline = UniMEPipeline()
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        return

    # Create a dummy image (red square)
    image = Image.new('RGB', (336, 336), color='red')
    
    # Target text: "A blue circle" (something very different from red square)
    target_text = "A blue circle"
    print(f"Target Text: {target_text}")
    
    # Calculate original similarity
    print("Calculating original similarity...")
    with torch.no_grad():
        img_emb = pipeline.encode_images([image])
        target_txt_emb = pipeline.encode_texts([target_text])
        orig_sim = (img_emb @ target_txt_emb.T)[0][0]
        print(f"Original Similarity: {orig_sim:.4f}")

    print("Running Targeted PGD attack...")
    try:
        adv_image = pgd_attack_to_target(image, pipeline, target_text=target_text, steps=10, epsilon=0.05)
        print("Attack finished.")
        
        # Calculate adversarial similarity
        with torch.no_grad():
            adv_emb = pipeline.encode_images([adv_image])
            adv_sim = (adv_emb @ target_txt_emb.T)[0][0]
            print(f"Adversarial Similarity to '{target_text}': {adv_sim:.4f}")
            
            diff = adv_sim - orig_sim
            print(f"Improvement: {diff:.4f}")
            
            if adv_sim > orig_sim:
                print("SUCCESS: Similarity increased.")
            else:
                print("FAILURE: Similarity did not increase.")
            
    except Exception as e:
        print(f"Attack failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_target_attack()
