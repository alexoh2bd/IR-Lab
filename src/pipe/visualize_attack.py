import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipe import MultimodalRetrievalPipeline
from attacks import pgd_attack_to_target_clip
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch

def visualize_attack():
    # Configuration
    model_name_base = "openai/clip-vit-base-patch32"
    model_name_large = "openai/clip-vit-large-patch14"
    
    # Load specific image
    image_path = "worklink/MMEB-eval/VisualNews_i2t/usa_today_images_0516_633.jpg"
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"Loaded image from {image_path}")
    except Exception as e:
        print(f"Failed to load image from {image_path}: {e}")
        # Fallback to dummy
        image = Image.new('RGB', (336, 336), color='white')
        draw = ImageDraw.Draw(image)
        draw.rectangle([50, 50, 286, 286], fill='red')
        print("Using dummy image.")

    original_text = "Patriots Football Quarterback"
    target_text = "A Brick"
    print(f"Target Text: {target_text}")
    
    # Initialize pipelines and run attacks
    results = {}
    
    for model_name in [model_name_base, model_name_large]:
        print(f"\nProcessing {model_name}...")
        try:
            pipeline = MultimodalRetrievalPipeline(model_name=model_name)
            
            # Calculate original similarity
            print("Calculating original similarity...")
            with torch.no_grad():
                img_emb = pipeline.encode_images([image])
                original_txt_emb = pipeline.encode_texts([original_text])
                target_txt_emb = pipeline.encode_texts([target_text])
                orig_sim = (img_emb @ original_txt_emb.T)[0][0]
                target_sim = (img_emb @ target_txt_emb.T)[0][0]
                print(f"Original Similarity: {orig_sim:.4f}")
                print(f"Target Similarity: {target_sim:.4f}")
            
            # Run PGD attack
            print("Running PGD attack...")
            adv_image = pgd_attack_to_target_clip(image, pipeline, target_text=target_text, steps=10, epsilon=0.03)
            print("Attack finished.")
            
            # Calculate adversarial similarity
            with torch.no_grad():
                adv_emb = pipeline.encode_images([adv_image])
                adv_sim = (adv_emb @ original_txt_emb.T)[0][0]
                adv_target_sim = (adv_emb @ target_txt_emb.T)[0][0]
                print(f"Adversarial Similarity to '{original_text}': {adv_sim:.4f}")
                print(f"Adversarial Similarity to '{target_text}': {adv_target_sim:.4f}")
            
            results[model_name] = {
                'adv_image': adv_image,
                'orig_sim': orig_sim,
                'target_sim': target_sim,
                'adv_sim': adv_sim,
                'adv_target_sim': adv_target_sim
            }
            
            # Cleanup
            del pipeline
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Failed for {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Visualization
    print("\nGenerating visualization...")
    import matplotlib.pyplot as plt

    # Create figure
    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    plt.suptitle(f"CLIP Attack Comparison\nOriginal Text: '{original_text}'\nTarget Text: '{target_text}'", fontsize=24, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # 1. Original Image
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(image.resize((225, 225), Image.BILINEAR))
    ax1.set_title("Original Image", fontsize=14, pad=10)
    ax1.axis('off')

    # Shared text configuration for consistency
    text_config = dict(
        ha='center', 
        va='top', 
        fontsize=12,    # Readable but not huge
        fontweight='bold',
        bbox=dict(facecolor='#f0f0f0', edgecolor='none', alpha=0.9, pad=6)
    )

    # 2. Base Attack
    ax2 = plt.subplot(1, 3, 2)
    if model_name_base in results:
        res = results[model_name_base]
        ax2.imshow(res['adv_image'])
        ax2.set_title("CLIP Base Attack", fontsize=14, pad=10)
        
        # CHANGED: Arrow format "Start -> End" is easier to scan
        stats_text = (
            f"True Class: {res['orig_sim']:.2f} → {res['adv_sim']:.2f}\n"
            f"Target:     {res['target_sim']:.2f} → {res['adv_target_sim']:.2f}"
        )
        
        # Use relative coordinates (transAxes) so it stays centered
        ax2.text(0.5, -0.05, stats_text, transform=ax2.transAxes, **text_config)
    else:
        ax2.text(0.5, 0.5, "Failed", ha='center', va='center')
    ax2.axis('off')

    # 3. Large Attack
    ax3 = plt.subplot(1, 3, 3)
    if model_name_large in results:
        res = results[model_name_large]
        ax3.imshow(res['adv_image'])
        ax3.set_title("CLIP Large Attack", fontsize=14, pad=10)
        
        stats_text = (
            f"True Class: {res['orig_sim']:.2f} → {res['adv_sim']:.2f}\n"
            f"Target:     {res['target_sim']:.2f} → {res['adv_target_sim']:.2f}"
        )
        
        ax3.text(0.5, -0.05, stats_text, transform=ax3.transAxes, **text_config)
    else:
        ax3.text(0.5, 0.5, "Failed", ha='center', va='center')
    ax3.axis('off')
    
    # Add extra spacing at the bottom so the text isn't cut off
    plt.subplots_adjust(bottom=0.2)

    plt.tight_layout()
    output_path = "violent_clip_attack_comparison2.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    visualize_attack()
