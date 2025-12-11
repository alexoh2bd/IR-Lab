
import torch
from PIL import Image
import numpy as np
from pipe.pipe import MultimodalRetrievalPipeline

def test_multimodal():
    print("Initializing pipeline...")
    pipeline = MultimodalRetrievalPipeline("openai/clip-vit-base-patch32")
    
    print("Creating dummy data...")
    # Create dummy images and texts
    img1 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    img2 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    txt1 = "A photo of a cat"
    txt2 = "A photo of a dog"
    
    print("Testing encode_multimodal...")
    try:
        embeds = pipeline.encode_multimodal([img1, img2], [txt1, txt2])
        print(f"Embeddings shape: {embeds.shape}")
        
        # Check normalization
        norms = np.linalg.norm(embeds, axis=1)
        print(f"Norms: {norms}")
        if not np.allclose(norms, 1.0, atol=1e-5):
            print("WARNING: Embeddings are not normalized!")
        else:
            print("Embeddings are normalized.")
            
        print("Testing retrieve_multimodal...")
        results = pipeline.retrieve_multimodal(
            [img1], [txt1],
            [img1, img2], [txt1, txt2]
        )
        print("Retrieval results:", results)
        
        # Expect img1+txt1 to match itself best (index 0)
        if results['indices'][0][0] == 0:
            print("SUCCESS: Retrieval matched correctly.")
        else:
            print(f"FAILURE: Expected index 0, got {results['indices'][0][0]}")
            
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_multimodal()
