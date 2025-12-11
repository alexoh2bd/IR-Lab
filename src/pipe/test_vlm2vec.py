"""
Test script for VLM2Vec pipeline
Replicates the examples from the user's request
"""

import sys
import os

# Add VLM2Vec to path if needed
# Uncomment and modify if VLM2Vec is not in your PYTHONPATH
# sys.path.insert(0, '/hpc/group/csdept/aho13/labIR/VLM2Vec')

from vlm2vec import VLM2VecRetrievalPipeline
from PIL import Image
import torch


def test_image_text_to_text():
    """Test Image + Text -> Text retrieval (Example from user request)"""
    print("=" * 80)
    print("TEST 1: Image + Text -> Text Retrieval")
    print("=" * 80)
    
    # Initialize pipeline
    pipeline = VLM2VecRetrievalPipeline()
    
    # Load query image
    query_img = Image.open('figures/example.jpg')
    query_text = 'What is in the image'
    
    # Target texts
    target_texts = ['A cat and a dog', 'A cat and a tiger']
    
    # Encode query (image + text)
    print(f"\nQuery: Image + '{query_text}'")
    qry_embeds = pipeline.encode_image_text_queries([query_img], [query_text])
    print(f"Query embedding shape: {qry_embeds.shape}")
    
    # Encode target texts
    tgt_embeds = pipeline.encode_texts(target_texts)
    print(f"Target embeddings shape: {tgt_embeds.shape}")
    
    # Compute similarities
    similarities = pipeline.compute_similarity(qry_embeds, tgt_embeds)
    
    print("\nResults:")
    for text, sim in zip(target_texts, similarities[0]):
        print(f"  '{text}' = {sim:.4f}")
    
    return pipeline


def test_text_to_image(pipeline):
    """Test Text -> Image retrieval (Example from user request)"""
    print("\n" + "=" * 80)
    print("TEST 2: Text -> Image Retrieval")
    print("=" * 80)
    
    # Load target image
    target_img = Image.open('figures/example.jpg')
    
    # Query texts
    query_texts = ['A cat and a dog', 'A cat and a tiger']
    
    # Encode target image
    print("\nEncoding target image...")
    tgt_embeds = pipeline.encode_images([target_img], instruction='Represent the given image.')
    print(f"Target image embedding shape: {tgt_embeds.shape}")
    
    # Encode query texts
    print("\nEncoding query texts...")
    qry_embeds = pipeline.encode_text_queries(
        query_texts,
        instruction_prefix='Find me an everyday image that matches the given caption:'
    )
    print(f"Query embeddings shape: {qry_embeds.shape}")
    
    # Compute similarities
    similarities = pipeline.compute_similarity(qry_embeds, tgt_embeds)
    
    print("\nResults:")
    for text, sim in zip(query_texts, similarities[:, 0]):
        print(f"  'Find me an everyday image that matches the given caption: {text}' = {sim:.4f}")


def test_retrieval_functions(pipeline):
    """Test the high-level retrieval functions"""
    print("\n" + "=" * 80)
    print("TEST 3: High-level Retrieval Functions")
    print("=" * 80)
    
    # Image + Text -> Text retrieval
    print("\n--- Image+Text to Text Retrieval ---")
    query_img = Image.open('figures/example.jpg')
    query_texts = ['What animals are in this image?']
    corpus_texts = [
        'A cat and a dog',
        'A cat and a tiger',
        'Two dogs playing',
        'A bird in a tree',
        'A fish in water'
    ]
    
    results = pipeline.retrieve_i2t(
        query_images=[query_img],
        query_texts=query_texts,
        corpus_texts=corpus_texts,
        top_k=3
    )
    
    print(f"\nTop 3 matches for query image + '{query_texts[0]}':")
    for rank, (idx, score) in enumerate(zip(results['indices'][0], results['scores'][0]), 1):
        print(f"  {rank}. '{corpus_texts[idx]}' (score: {score:.4f})")
    
    # Text -> Image retrieval
    print("\n--- Text to Image Retrieval ---")
    query_texts = ['A cat and a dog']
    corpus_images = [Image.open('figures/example.jpg')]  # In practice, you'd have multiple images
    
    results = pipeline.retrieve_t2i(
        query_texts=query_texts,
        corpus_images=corpus_images,
        top_k=1
    )
    
    print(f"\nTop match for query '{query_texts[0]}':")
    print(f"  Image index: {results['indices'][0][0]} (score: {results['scores'][0][0]:.4f})")


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("VLM2Vec Pipeline Test Suite")
    print("=" * 80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("=" * 80)
    
    try:
        # Run tests
        pipeline = test_image_text_to_text()
        test_text_to_image(pipeline)
        test_retrieval_functions(pipeline)
        
        # Cleanup
        print("\n" + "=" * 80)
        print("Cleaning up...")
        pipeline.cleanup()
        print("All tests completed successfully!")
        print("=" * 80)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Make sure 'figures/example.jpg' exists in your working directory.")
        print("You can download an example image or use your own.")
    except ImportError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure VLM2Vec is properly installed:")
        print("1. Clone: git clone https://github.com/TIGER-AI-Lab/VLM2Vec.git")
        print("2. Install: cd VLM2Vec && pip install -e .")
        print("3. Add to path or modify imports in vlm2vec.py")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
