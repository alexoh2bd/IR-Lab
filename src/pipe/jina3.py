import os
# os.environ['TQDM_DISABLE'] = '1'  # Disable tqdm globally at the start (will be re-enabled later)

from transformers import AutoModel, logging as lg
# lg.set_verbosity_error()
from PIL import Image
import torch
from tqdm import tqdm

import numpy as np
from typing import List, Dict, Union, Optional
# from tqdm.auto import tqdm  # Removed - will import later after re-enabling
from datasets import load_dataset
import logging
from datetime import datetime
# import os  # Already imported
from sklearn.metrics import ndcg_score

# Assuming these imports from your existing code
from pipe import Config
from experiment import setup_ndcg_logger, perturb


class JinaV4Pipeline:
    """
    Unified multimodal retrieval pipeline for Jina CLIP v2.
    
    Jina CLIP v2 is a bilingual, multimodal embedding model that supports
    both text and image inputs, creating embeddings in a unified semantic space.
    """
    
    def __init__(self, 
                 model_name: str = "jinaai/jina-clip-v2",
                #  task: str = "retrieval",
                 output_mode: str = "single-vector"):
        """
        Initialize the Jina CLIP v2 pipeline.
        
        Args:
            model_name: HuggingFace model identifier
            task: One of 'retrieval', 'text-matching', or 'code'
            output_mode: 'single-vector' (2048-d, truncatable) or 
                        'multi-vector' (128-d per token, late interaction)
        """
        print(f"Loading Jina CLIP v2 model on {Config.DEVICE}...")
        
        self.model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        self.model = self.model.to(Config.DEVICE)
        self.model.eval()
        
        self.output_mode = output_mode
        
        # print(f"Model loaded. Task: {task}, Output mode: {output_mode}")
    # 
    def encode_images(self, 
                     images: List[Union[str, Image.Image]], 
                     batch_size: int = 128,
                     prompt_name: Optional[str] = None) -> np.ndarray:
        """
        Encode images using Jina v4's unified architecture.
        
        Args:
            images: List of image paths (URLs/local) or PIL Images
            batch_size: Batch size for encoding
            prompt_name: Optional prompt name ('query' or 'passage' for retrieval)
            
        Returns:
            np.ndarray of embeddings [num_images, embedding_dim]
        """
        all_embeddings = []
        
        with torch.no_grad(), torch.amp.autocast(device_type=Config.DEVICE, dtype=torch.bfloat16, enabled=(Config.DEVICE == "cuda")):
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                
                # Use the model's encode_image method
                # Note: Jina CLIP v2 doesn't support task or prompt_name parameters
                embeddings = self.model.encode_image(
                    images=batch
                )
                
                # Convert to numpy array if needed
                if isinstance(embeddings, torch.Tensor):
                    embeddings = embeddings.cpu().numpy()
                elif isinstance(embeddings, list):
                    # Handle list of tensors
                    embeddings = [e.cpu().numpy() if isinstance(e, torch.Tensor) else e for e in embeddings]
                    embeddings = np.array(embeddings)
                elif not isinstance(embeddings, np.ndarray):
                    # Try to move to CPU if it has the method
                    if hasattr(embeddings, 'cpu'):
                        embeddings = embeddings.cpu()
                    embeddings = np.array(embeddings)
                
                # Handle both single-vector and multi-vector outputs
                if self.output_mode == "single-vector":
                    # Normalize for similarity search
                    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                
                all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)
    
    def encode_texts(self, 
                    texts: List[str], 
                    batch_size: int = 1000,
                    prompt_name: Optional[str] = None) -> np.ndarray:
        """
        Encode texts using Jina v4's unified architecture.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            prompt_name: Optional prompt name ('query' or 'passage' for retrieval)
            
        Returns:
            np.ndarray of embeddings [num_texts, embedding_dim]
        """
        all_embeddings = []
        
        with torch.no_grad(), torch.amp.autocast(device_type=Config.DEVICE, dtype=torch.bfloat16, enabled=(Config.DEVICE == "cuda")):
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Jina CLIP v2 uses 'sentences' parameter instead of 'texts'
                # and doesn't support task or prompt_name parameters
                embeddings = self.model.encode_text(
                    sentences=batch
                )
                
                # Convert to numpy array if needed
                if isinstance(embeddings, torch.Tensor):
                    embeddings = embeddings.cpu().numpy()
                elif isinstance(embeddings, list):
                    # Handle list of tensors
                    embeddings = [e.cpu().numpy() if isinstance(e, torch.Tensor) else e for e in embeddings]
                    embeddings = np.array(embeddings)
                elif not isinstance(embeddings, np.ndarray):
                    # Try to move to CPU if it has the method
                    if hasattr(embeddings, 'cpu'):
                        embeddings = embeddings.cpu()
                    embeddings = np.array(embeddings)
                
                # Normalize for similarity search
                if self.output_mode == "single-vector":
                    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                
                all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)
    
    def compute_similarity(self, 
                          query_embeds: np.ndarray, 
                          corpus_embeds: np.ndarray) -> np.ndarray:
        """
        Compute similarity scores between queries and corpus.
        
        For single-vector: Uses cosine similarity (dot product after normalization)
        For multi-vector: Uses late interaction (MaxSim)
        
        Args:
            query_embeds: Query embeddings [num_queries, dim] or [num_queries, num_tokens, dim]
            corpus_embeds: Corpus embeddings [num_corpus, dim] or [num_corpus, num_tokens, dim]
            
        Returns:
            Similarity matrix [num_queries, num_corpus]
        """
        if self.output_mode == "single-vector":
            # Simple dot product for normalized vectors (cosine similarity)
            return query_embeds @ corpus_embeds.T
        
        else:  # multi-vector (late interaction)
            # Check if embeddings are actually multi-vector (3D)
            if query_embeds.ndim == 2 or corpus_embeds.ndim == 2:
                # Fallback to single-vector similarity if embeddings are 2D
                # (Jina v4 returns 2D embeddings even in multi-vector mode)
                return query_embeds @ corpus_embeds.T
            
            # MaxSim: For each query token, find max similarity with any corpus token
            # Then average across query tokens
            num_queries = query_embeds.shape[0]
            num_corpus = corpus_embeds.shape[0]
            similarities = np.zeros((num_queries, num_corpus))
            
            for i in range(num_queries):
                for j in range(num_corpus):
                    # Compute token-level similarities
                    token_sims = query_embeds[i] @ corpus_embeds[j].T
                    # MaxSim: max over corpus tokens, then mean over query tokens
                    similarities[i, j] = np.mean(np.max(token_sims, axis=1))
            
            return similarities
    
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
        pipeline: JinaV4Pipeline instance
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
            query_embeds = img_embed_cache[cache_key].reshape(1, -1)
            hit_count += 1
        else:
            # Load and optionally perturb image
            image = Image.open(img_path)
            if perturbation:
                image = perturb(image, perturbation, pipeline)
            
            # Encode and cache
            query_embeds = pipeline.encode_images([image], prompt_name="query")
            img_embed_cache[cache_key] = query_embeds[0]
            
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
        pipeline: JinaV4Pipeline instance
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
            
            corpus_embeds = np.array([emb for _, emb in all_embeddings])
            
            # Cleanup images
            for _, img in images_to_encode:
                if hasattr(img, 'close'):
                    img.close()
        else:
            # All cached
            corpus_embeds = np.array([emb for _, emb in cached_embeddings])
        
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
                   model_name: str = "jinaai/jina-clip-v2",
                   output_mode: str = "single-vector"):
    """
    Run complete evaluation pipeline.
    """
    pipeline = JinaV4Pipeline(model_name=model_name, output_mode=output_mode)
    
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
        log_file = f"logs/jina_clip_v2/{ds_name}_{timestamp}.log"
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
    # datasets = ["VisualNews_t2i", "MSCOCO_t2i", "VisDial", "Wiki-SS-NQ"]
    # datasets = ["MSCOCO_t2i", "MSCOCO_i2t", "VisualNews_t2i", "VisualNews_i2t", "VisDial", "Wiki-SS-NQ"]
    datasets = [ "VisualNews_t2i", "VisualNews_i2t", "MSCOCO_t2i","VisDial","Wiki-SS-NQ"]
    perturbations = ["clear", "grayscale", "gauss2", "bright" ]
    
    # Run with single-vector mode (efficient)
    # results_single = run_evaluation(
    #     datasets=datasets,
    #     perturbations=perturbations,
    #     model_name="jinaai/jina-clip-v2",
    #     output_mode="single-vector"
    # )
    
    # Optionally run with multi-vector mode (more precise)
    results_multi = run_evaluation(
        datasets=datasets,
        perturbations=perturbations,
        model_name="jinaai/jina-clip-v2",
        output_mode="multi-vector"
    )
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
