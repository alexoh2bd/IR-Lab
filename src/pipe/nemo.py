import os
# Set temp directory to avoid NFS issues
import tempfile
os.environ["TMPDIR"] = "/tmp"
tempfile.tempdir = "/tmp"

from transformers import AutoModel, logging as lg
from PIL import Image
import torch
from tqdm import tqdm
os.environ["NVTE_FLASH_ATTN"] = "0"
os.environ["NVTE_FUSED_ATTN"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
from typing import List, Dict, Union, Optional
from datasets import load_dataset
import logging
from datetime import datetime
from sklearn.metrics import ndcg_score

# Assuming these imports from your existing code
from pipe import Config
from experiment import setup_ndcg_logger, perturb


class NemoRetrieverPipeline:
    """
    Unified multimodal retrieval pipeline for NVIDIA Llama NemoRetriever.
    
    Llama NemoRetriever is a multimodal embedding model that supports
    both text and image inputs, creating embeddings in a unified semantic space.
    """
    
    def __init__(self, 
                 model_name: str = "nvidia/llama-nemoretriever-colembed-3b-v1",
                 output_mode: str = "single-vector"):
        """
        Initialize the Llama NemoRetriever pipeline.
        
        Args:
            model_name: HuggingFace model identifier
            output_mode: 'single-vector' for standard embeddings
        """
        print(f"Loading Llama NemoRetriever model on {Config.DEVICE}...")
        
        # Clear GPU memory before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Suppress transformers logging to avoid config.to_dict() bug
        import logging as py_logging
        transformers_logger = py_logging.getLogger("transformers")
        original_level = transformers_logger.level
        transformers_logger.setLevel(py_logging.ERROR)
        
        try:
            self.model = AutoModel.from_pretrained(
                model_name,
                device_map=Config.DEVICE,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                attn_implementation="eager",
                low_cpu_mem_usage=True
            ).eval()
        finally:
            # Restore original logging level
            transformers_logger.setLevel(original_level)
        
        self.output_mode = output_mode
        
        print(f"Model loaded. Output mode: {output_mode}")
    
    def encode_images(self, 
                     images: List[Union[str, Image.Image]], 
                     batch_size: int = 8,
                     prompt_name: Optional[str] = None) -> torch.Tensor:
        """
        Encode images using NemoRetriever's unified architecture.
        
        Args:
            images: List of image paths (URLs/local) or PIL Images
            batch_size: Batch size for encoding
            prompt_name: Optional prompt name ('query' or 'passage' for retrieval)
            
        Returns:
            torch.Tensor of embeddings (multi-vector, variable length per image)
        """
        with torch.no_grad():
            # Process all images at once (model handles batching internally)
            embeddings = self.model.forward_passages(images, batch_size=batch_size)
            
            # Clear cache after encoding
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return embeddings
    
    def encode_texts(self, 
                    texts: List[str], 
                    batch_size: int = 32,
                    prompt_name: Optional[str] = None) -> torch.Tensor:
        """
        Encode texts using NemoRetriever's unified architecture.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            prompt_name: Optional prompt name ('query' or 'passage' for retrieval)
            
        Returns:
            torch.Tensor of embeddings (multi-vector, variable length per text)
        """
        with torch.no_grad():
            # Process all texts at once (model handles batching internally)
            embeddings = self.model.forward_queries(texts, batch_size=batch_size)
            
            # Clear cache after encoding
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return embeddings
    
    def compute_similarity(self, 
                          query_embeds: torch.Tensor, 
                          corpus_embeds: torch.Tensor) -> np.ndarray:
        """
        Compute similarity scores between queries and corpus using NemoRetriever's get_scores.
        
        Args:
            query_embeds: Query embeddings (multi-vector) as torch.Tensor
            corpus_embeds: Corpus embeddings (multi-vector) as torch.Tensor
            
        Returns:
            Similarity matrix [num_queries, num_corpus]
        """
        with torch.no_grad():
            scores = self.model.get_scores(query_embeds, corpus_embeds)
            
        return scores.cpu().numpy()
    
    def retrieve_i2t(self,
                    query_images: List[Union[str, Image.Image]],
                    corpus_texts: List[str],
                    top_k: int = 10) -> Dict:
        """
        Retrieve texts from images (Image-to-Text retrieval).
        
        Args:
            query_images: List of query images
            corpus_texts: List of candidate texts
            top_k: Number of results to return
            
        Returns:
            Dictionary with 'indices' and 'scores'
        """
        # Encode with appropriate prompts for asymmetric retrieval
        query_embeds = self.encode_images(query_images, prompt_name="query")
        corpus_embeds = self.encode_texts(corpus_texts, prompt_name="passage")
        
        # Compute similarities in unified space
        similarities = self.compute_similarity(query_embeds, corpus_embeds)
        
        # Get top-k results
        top_k = min(top_k, similarities.shape[1])
        top_indices = np.argsort(-similarities, axis=1)[:, :top_k]
        top_scores = np.take_along_axis(similarities, top_indices, axis=1)
        
        return {
            'indices': top_indices,
            'scores': top_scores,
        }
    
    def retrieve_t2i(self,
                    query_texts: List[str],
                    corpus_images: List[Union[str, Image.Image]],
                    top_k: int = 10) -> Dict:
        """
        Retrieve images from texts (Text-to-Image retrieval).
        
        Args:
            query_texts: List of query texts
            corpus_images: List of candidate images
            top_k: Number of results to return
            
        Returns:
            Dictionary with 'indices' and 'scores'
        """
        # Encode with appropriate prompts
        query_embeds = self.encode_texts(query_texts, prompt_name="query")
        corpus_embeds = self.encode_images(corpus_images, prompt_name="passage")
        
        # Compute similarities in unified space
        similarities = self.compute_similarity(query_embeds, corpus_embeds)
        
        # Get top-k results
        top_k = min(top_k, similarities.shape[1])
        top_indices = np.argsort(-similarities, axis=1)[:, :top_k]
        top_scores = np.take_along_axis(similarities, top_indices, axis=1)
        
        return {
            'indices': top_indices,
            'scores': top_scores,
        }
    
    def cleanup(self):
        """Clean up GPU memory."""
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Pipeline cleaned up.")


def evaluate_i2t(test_data, pipeline, base_img_url: str, perturbation: str = None, img_embed_cache: Dict = None) -> tuple:
    """
    Evaluate image-to-text retrieval with embedding caching.
    
    Args:
        test_data: Test dataset
        pipeline: NemoRetrieverPipeline instance
        base_img_url: Base URL for images
        perturbation: Type of perturbation to apply
        img_embed_cache: Dictionary to cache image embeddings
        
    Returns:
        Tuple of (scores, perfect_count, retrieved_count)
    """
    if img_embed_cache is None:
        img_embed_cache = {}
    
    retrieved_count = 0
    perfect_count = 0
    scores = []
    hit_count = 0
    
    for row in tqdm(test_data, desc=f"Evaluating I2T ({perturbation or 'clean'})"):
        # Check cache for query image
        img_path = base_img_url + row['qry_img_path']
        cache_key = f"{img_path}_{perturbation or 'clean'}"
        
        if cache_key in img_embed_cache:
            query_embeds = img_embed_cache[cache_key]
            hit_count += 1
        else:
            # Load and optionally perturb image
            image = Image.open(img_path)
            if perturbation:
                image = perturb(image, perturbation, pipeline)
            
            # Encode and cache (returns tensor for single image)
            query_embeds = pipeline.encode_images([image], prompt_name="query")
            img_embed_cache[cache_key] = query_embeds  # Cache the embedding
            
            # Cleanup
            if hasattr(image, 'close'):
                image.close()
        
        # Encode corpus texts
        corpus_embeds = pipeline.encode_texts(row['tgt_text'], prompt_name="passage")
        
        # Compute similarities
        similarities = pipeline.compute_similarity(query_embeds, corpus_embeds)
        
        # Get top-k results
        top_k = min(10, similarities.shape[1])
        top_indices = np.argsort(-similarities, axis=1)[0, :top_k]
        top_scores = similarities[0, top_indices]
        
        # Ground truth: first text is relevant
        truth = [1] + [0] * (len(row['tgt_text']) - 1)
        retrieved_truth = [truth[i] for i in top_indices]
        
        # Calculate nDCG
        score = ndcg_score([retrieved_truth], [top_scores.tolist()], k=10)
        
        if score > 0.0:
            retrieved_count += 1
        if score == 1.0:
            perfect_count += 1
        
        scores.append(score)
    
    return scores, perfect_count, retrieved_count


def evaluate_t2i(test_data, pipeline, base_img_url: str, perturbation: str = None, img_embed_cache: Dict = None) -> tuple:
    """
    Evaluate text-to-image retrieval with embedding caching.
    
    Args:
        test_data: Test dataset
        pipeline: NemoRetrieverPipeline instance
        base_img_url: Base URL for images
        perturbation: Type of perturbation to apply
        img_embed_cache: Dictionary to cache image embeddings
        
    Returns:
        Tuple of (scores, perfect_count, retrieved_count)
    """
    if img_embed_cache is None:
        img_embed_cache = {}
    
    retrieved_count = 0
    perfect_count = 0
    scores = []
    hit_count = 0
    
    for row in tqdm(test_data, desc=f"Evaluating T2I ({perturbation or 'clean'})"):
        # Separate cached and non-cached images
        cached_embeddings = []
        images_to_encode = []
        urls_to_cache = []
        
        for j, img_path in enumerate(row['tgt_img_path']):
            img_url = base_img_url + img_path
            cache_key = f"{img_url}_{perturbation or 'clean'}"
            
            if cache_key in img_embed_cache:
                cached_embeddings.append((j, img_embed_cache[cache_key]))
                hit_count += 1
            else:
                # Load and optionally perturb image
                img = Image.open(img_url)
                if perturbation:
                    img = perturb(img, perturbation, pipeline)
                images_to_encode.append((j, img))
                urls_to_cache.append(cache_key)
        
        # Batch encode all non-cached images
        if images_to_encode:
            imgs_only = [img for _, img in images_to_encode]
            batch_embeds = pipeline.encode_images(imgs_only)
            
            # Cache the new embeddings
            for i, cache_key in enumerate(urls_to_cache):
                img_embed_cache[cache_key] = batch_embeds[i]
            
            # Combine cached and newly encoded embeddings in correct order
            newly_encoded = [(images_to_encode[i][0], batch_embeds[i]) for i in range(len(images_to_encode))]
            all_embeddings = cached_embeddings + newly_encoded
            all_embeddings.sort(key=lambda x: x[0])
            
            # Stack tensors along batch dimension
            corpus_embeds = torch.stack([emb for _, emb in all_embeddings], dim=0)
            
            # Cleanup images
            for _, img in images_to_encode:
                if hasattr(img, 'close'):
                    img.close()
        else:
            # All cached - stack tensors
            corpus_embeds = torch.stack([emb for _, emb in cached_embeddings], dim=0)
        
        # Ground truth
        truth = [1] + [0] * (len(corpus_embeds) - 1)
        
        # Encode query and compute similarities
        query_embeds = pipeline.encode_texts([row['qry_text']], prompt_name="query")
        similarities = pipeline.compute_similarity(query_embeds, corpus_embeds)
        
        # Get top-k results
        top_k = min(10, similarities.shape[1])
        top_indices = np.argsort(-similarities, axis=1)[0, :top_k]
        top_scores = similarities[0, top_indices]
        
        # Calculate nDCG
        retrieved_truth = [truth[i] for i in top_indices]
        score = ndcg_score([retrieved_truth], [top_scores.tolist()], k=10)
        
        if score > 0.0:
            retrieved_count += 1
        if score == 1.0:
            perfect_count += 1
        
        scores.append(score)
    
    return scores, perfect_count, retrieved_count


def run_evaluation(datasets: List[str], 
                   perturbations: List[str],
                   model_name: str = "nvidia/llama-nemoretriever-colembed-3b-v1",
                   output_mode: str = "single-vector"):
    """
    Run complete evaluation pipeline.
    """
    pipeline = NemoRetrieverPipeline(model_name=model_name, output_mode=output_mode)
    
    # Re-enable tqdm now that the model is loaded (for outer progress bars)
    import sys
    if 'tqdm' in sys.modules:
        del sys.modules['tqdm']
    os.environ.pop('TQDM_DISABLE', None)
    from tqdm.auto import tqdm  # Re-import with progress enabled
    
    base_img_url = '/work/aho13/MMEB-eval/'
    k = 10
    
    i2t_datasets = {"MSCOCO_i2t", "VisualNews_i2t"}
    t2i_datasets = {"MSCOCO_t2i", "VisualNews_t2i", "VisDial", "Wiki-SS-NQ"}
    
    all_results = {}
    
    for ds_name in datasets:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {ds_name}")
        print(f"{'='*60}")
        
        ds = load_dataset("TIGER-Lab/MMEB-eval", ds_name)
        
        # Setup logging
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"logs/nemo_retriever/{ds_name}_{timestamp}.log"
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logger = setup_ndcg_logger(log_file)
        
        # Create image embedding cache for this dataset
        img_embed_cache = {}
        
        dataset_results = {}
        
        for p in perturbations:
            logger.info(f"\nEvaluating perturbation: {p}")
            
            # Choose evaluation function based on dataset type
            if ds_name in i2t_datasets:
                scores, perfect, retrieved = evaluate_i2t(
                    list(ds['test']), pipeline, base_img_url, p, img_embed_cache
                )
            elif ds_name in t2i_datasets:
                scores, perfect, retrieved = evaluate_t2i(
                    list(ds['test']), pipeline, base_img_url, p, img_embed_cache
                )
            else:
                print(f"Unknown dataset type: {ds_name}")
                continue
            
            # Calculate statistics
            avg_ndcg = np.mean(scores)
            median_ndcg = np.median(scores)
            std_ndcg = np.std(scores)
            
            dataset_results[p] = {
                'scores': scores,
                'mean': avg_ndcg,
                'median': median_ndcg,
                'std': std_ndcg,
                'perfect_count': perfect,
                'retrieved_count': retrieved,
                'total': len(scores)
            }
            
            # Print and log results
            result_str = [
                "=" * 60,
                f"NDCG@{k} - {ds_name} - {p}",
                f"  Mean NDCG: {avg_ndcg:.4f}",
                f"  Median NDCG: {median_ndcg:.4f}",
                f"  Std Dev: {std_ndcg:.4f}",
                f"  Perfect retrievals: {perfect}/{len(scores)} ({100*perfect/len(scores):.1f}%)",
                f"  Relevant in top-{k}: {retrieved}/{len(scores)} ({100*retrieved/len(scores):.1f}%)",
                "=" * 60
            ]
            
            # Log to file
            if logger:
                for line in result_str:
                    logger.info(line)
        
        all_results[ds_name] = dataset_results
    
    pipeline.cleanup()
    return all_results


def main():
    """Main execution function."""
    datasets = ["VisualNews_t2i","VisualNews_i2t"]
    perturbations = ["clear", "gauss2"]
    
    # Run with single-vector mode
    results = run_evaluation(
        datasets=datasets,
        perturbations=perturbations,
        model_name="nvidia/llama-nemoretriever-colembed-3b-v1",
        output_mode="single-vector"
    )
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
