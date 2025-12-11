import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipe2.unimepipe import UniMEPipeline
from pipe2.attacks import pgd_attack_unime, pgd_attack_to_target
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import torch.nn.functional as F
from pipe.pipe import MultimodalRetrievalPipeline

def visualize_attack():
    print("Initializing pipeline...")
    try:
        pipeline = UniMEPipeline()
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        return

    print("Pipeline initialized.")

    # Load specific image
    image_path = "worklink/MMEB-eval/VisualNews_i2t/usa_today_images_0516_633.jpg"
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"Loaded image from {image_path}")
    except Exception as e:
        print(f"Failed to load image from {image_path}: {e}")
        # Fallback to dummy if file not found (though user said it exists)
        image = Image.new('RGB', (336, 336), color='white')
        draw = ImageDraw.Draw(image)
        draw.rectangle([50, 50, 286, 286], fill='red')
    
    # ds = load_dataset("TIGER-Lab/MMEB-eval", "VisualNews_i2t")

    original_text = "Patriots Football quarterback"
    target_text = "A threat to public safety"
    print(f"Target Text: {target_text}")
    
    # Calculate original similarity
    print("Calculating original similarity...")
    with torch.no_grad():
        img_emb = pipeline.encode_images([image])
        original_txt_emb = pipeline.encode_texts([original_text])
        target_txt_emb = pipeline.encode_texts([target_text])
        orig_sim = (img_emb @ original_txt_emb.T)[0][0]
        print(f"Original Similarity: {orig_sim:.4f}")
        target_sim = (img_emb @ target_txt_emb.T)[0][0]
        print(f"Target Similarity: {target_sim:.4f}")



    print("Running PGD attack...")
    try:
        adv_image = pgd_attack_to_target(image, pipeline, target_text=target_text, steps=20, epsilon=0.05)
        print("Attack finished.")
        
        # Calculate adversarial similarity
        with torch.no_grad():
            adv_emb = pipeline.encode_images([adv_image])
            adv_sim = (adv_emb @ original_txt_emb.T)[0][0]
            print(f"Adversarial Similarity to '{original_text}': {adv_sim:.4f}")
            adv_target_sim = (adv_emb @ target_txt_emb.T)[0][0]
            print(f"Adversarial Similarity to '{target_text}': {adv_target_sim:.4f}")
            print(f"Drop: {orig_sim - adv_sim:.4f}")
        
        # Save comparison
        combined = Image.new('RGB', (336 * 2 + 20, 336 + 40), color='white')
        combined.paste(image, (0, 40))
        combined.paste(adv_image, (336 + 20, 40))
        
        draw = ImageDraw.Draw(combined)
        # Try to load a font, fallback to default
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 20)
        except:
            font = ImageFont.load_default()
            
        draw.text((10, 10), f"Original Similarity: {orig_sim:.2f}. \nTarget Similarity: {target_sim:.2f}", fill='black', font=font)
        draw.text((336 + 30, 10), f"Adversarial Similarity: {adv_sim:.2f}. \nAdversarial Target Similarity: {adv_target_sim:.2f}", fill='black', font=font)
        
        output_path = "tom_bradyattack_visualizationclipbase.png"
        combined.save(output_path)
        print(f"Visualization saved to {output_path}")
            
    except Exception as e:
        print(f"Attack failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    visualize_attack()
