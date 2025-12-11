"""
VLM2Vec Multimodal Retrieval Pipeline
Adapted from pipe.py for VLM2Vec-Full model
"""

import torch
import gc
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Union
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
import requests
from io import BytesIO
import sys
import os
from datetime import datetime
from datasets import load_dataset
from sklearn.metrics import ndcg_score
import logging

# Add VLM2Vec to path
sys.path.insert(0, '/hpc/group/csdept/aho13/labIR/VLM2Vec')

# VLM2Vec specific imports - these need to be installed from TIGER-Lab/VLM2Vec
try:
    from src.model.model import MMEBModel
    from src.arguments import ModelArguments
    from src.model.processor import load_processor
except ImportError as e:
    print(f"Warning: VLM2Vec modules not found: {e}")
    print("Please ensure VLM2Vec is cloned at /hpc/group/csdept/aho13/labIR/VLM2Vec")
    print("Run: cd /hpc/group/csdept/aho13/labIR && git clone https://github.com/TIGER-AI-Lab/VLM2Vec.git")
    raise

# Import perturb from experiment module
try:
    from experiment import perturb
except ImportError:
    # Fallback if experiment module not available
    def perturb(image, perturbation_type, pipeline):
        """Fallback perturb function - returns image unchanged."""
        return image


class Config:
    """Configuration for the VLM2Vec retrieval pipeline."""
    MODEL_NAME = 'TIGER-Lab/VLM2Vec-Full'
    BATCH_SIZE = 32  # Adjust based on your GPU memory
    IMAGE_BATCH_SIZE = 16  # Smaller batch for images (more memory intensive)
    TEXT_BATCH_SIZE = 64  # Larger batch for text-only processing
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16  # VLM2Vec uses bfloat16
    NUM_WORKERS = 0  # For DataLoader
    PIN_MEMORY = True
    POOLING = 'last'
    NORMALIZE = True
    MODEL_BACKBONE = 'phi3_v'
    NUM_CROPS = 16


class VLM2VecDataset(Dataset):
    """Dataset for image-text pairs for VLM2Vec."""

    def __init__(self, data: List[Dict], mode: str = "query"):
        """
        Args:
            data: List of dicts with 'image' and/or 'text' keys
            mode: 'query' or 'corpus' to handle different data structures
        """
        self.data = data
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Handle image loading
        if 'image' in item and item['image'] is not None:
            img = item['image']
            if isinstance(img, str):
                # Load from URL or path
                if img.startswith('http'):
                    response = requests.get(img, stream=True)
                    img = Image.open(BytesIO(response.content)).convert('RGB')
                else:
                    img = Image.open(img).convert('RGB')
            elif not isinstance(img, Image.Image):
                img = Image.fromarray(img).convert('RGB')
        else:
            img = None

        # Handle text
        text = item.get('text', '')
        if isinstance(text, list):
            text = text[0] if text else ""

        return {'image': img, 'text': text, 'idx': idx}


class VLM2VecRetrievalPipeline:
    """Efficient multimodal retrieval pipeline for VLM2Vec."""

    def __init__(self, 
                 model_name: str = Config.MODEL_NAME,
                 pooling: str = Config.POOLING,
                 normalize: bool = Config.NORMALIZE,
                 model_backbone: str = Config.MODEL_BACKBONE,
                 num_crops: int = Config.NUM_CROPS):
        """Initialize the pipeline with VLM2Vec model."""
        print(f"Loading VLM2Vec model on {Config.DEVICE}...")

        # Initialize model arguments
        self.model_args = ModelArguments(
            model_name=model_name,
            pooling=pooling,
            normalize=normalize,
            model_backbone=model_backbone,
            num_crops=num_crops
        )

        # Load processor and model
        self.processor = load_processor(self.model_args)
        self.model = MMEBModel.load(self.model_args)
        self.model.eval()
        self.model = self.model.to(Config.DEVICE, dtype=Config.DTYPE)

        print(f"VLM2Vec model loaded successfully. Using dtype: {Config.DTYPE}")

    @torch.no_grad()
    def encode_image_text_queries(self, 
                                   images: List[Image.Image], 
                                   texts: List[str]) -> np.ndarray:
        """
        Encode image+text queries to embeddings.
        Format: '<|image_1|> Represent the given image with the following question: {text}'

        Args:
            images: List of PIL Images
            texts: List of text queries

        Returns:
            numpy array of shape (n_queries, embedding_dim)
        """
        embeddings = []
        # print("images len", len(images))
        
        for i in range(0, len(images), Config.IMAGE_BATCH_SIZE):
            # batch_images = images[i:i + Config.IMAGE_BATCH_SIZE]
            batch_texts = texts[i:i + Config.IMAGE_BATCH_SIZE]
            
            batch_embeds = []
            for text in batch_texts:
                # Format: image + text query
                query_text = f'<|image_1|> {text}'
                inputs = self.processor(query_text, images)
                
                # Convert pixel_values to list format expected by model
                if 'pixel_values' in inputs and isinstance(inputs['pixel_values'], torch.Tensor):
                    pv = inputs['pixel_values']
                    if pv.ndim == 5:
                        # Convert from [batch, num_crops, 3, 336, 336] to list of [num_crops, 3, 336, 336]
                        inputs['pixel_values'] = [pv[i] for i in range(pv.shape[0])]
                
                # Move to device
                inputs = {key: value.to(Config.DEVICE) if isinstance(value, torch.Tensor) else 
                         [v.to(Config.DEVICE) if isinstance(v, torch.Tensor) else v for v in value] if isinstance(value, list) else value
                         for key, value in inputs.items()}
                
                qry_output = self.model(qry=inputs)["qry_reps"]
                batch_embeds.append(qry_output.cpu())
                
                del inputs, qry_output
            
            embeddings.append(torch.cat(batch_embeds, dim=0))
            torch.cuda.empty_cache()
        
        result = torch.cat(embeddings, dim=0).numpy()
        return result

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image], instruction: str = None) -> np.ndarray:
        """
        Encode images to embeddings.
        Format: '<|image_1|> Represent the given image.'

        Args:
            images: List of PIL Images
            instruction: Optional instruction text (default: 'Represent the given image.')

        Returns:
            numpy array of shape (n_images, embedding_dim)
        """
        if instruction is None:
            instruction = 'Represent the given image.'
        
        embeddings = []
        
        for i in range(0, len(images), Config.IMAGE_BATCH_SIZE):
            batch_images = images[i:i + Config.IMAGE_BATCH_SIZE]
            
            batch_embeds = []
            for img in batch_images:
                query_text = f'<|image_1|> {instruction}'
                inputs = self.processor(query_text, [img])
                
                # Convert pixel_values to list format expected by model
                # if 'pixel_values' in inputs and isinstance(inputs['pixel_values'], torch.Tensor):
                #     pv = inputs['pixel_values']
                #     if pv.ndim == 5:
                #         # Convert from [batch, num_crops, 3, 336, 336] to list of [num_crops, 3, 336, 336]
                #         inputs['pixel_values'] = [pv[i] for i in range(pv.shape[0])]
                
                # Move to device
                inputs = {key: value.to(Config.DEVICE) if isinstance(value, torch.Tensor) else 
                         [v.to(Config.DEVICE) if isinstance(v, torch.Tensor) else v for v in value] if isinstance(value, list) else value
                         for key, value in inputs.items()}
                
                tgt_output = self.model(tgt=inputs)["tgt_reps"]
                batch_embeds.append(tgt_output.cpu())
                
                del inputs, tgt_output
            
            embeddings.append(torch.cat(batch_embeds, dim=0))
            torch.cuda.empty_cache()
        
        result = torch.cat(embeddings, dim=0).numpy()
        return result

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings (as targets).

        Args:
            texts: List of strings

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        embeddings = []
        
        for i in range(0, len(texts), Config.TEXT_BATCH_SIZE):
            batch_texts = texts[i:i + Config.TEXT_BATCH_SIZE]
            
            batch_embeds = []
            for text in batch_texts:
                inputs = self.processor(text)
                inputs = {key: value.to(Config.DEVICE) for key, value in inputs.items()}
                
                tgt_output = self.model(tgt=inputs)["tgt_reps"]
                batch_embeds.append(tgt_output.cpu())
                
                del inputs, tgt_output
            
            embeddings.append(torch.cat(batch_embeds, dim=0))
            torch.cuda.empty_cache()
        
        result = torch.cat(embeddings, dim=0).numpy()
        return result

    @torch.no_grad()
    def encode_text_queries(self, texts: List[str], instruction_prefix: str = None) -> np.ndarray:
        """
        Encode text queries to embeddings.
        Format: 'Find me an everyday image that matches the given caption: {text}'

        Args:
            texts: List of text queries
            instruction_prefix: Optional prefix (default: 'Find me an everyday image that matches the given caption:')

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if instruction_prefix is None:
            instruction_prefix = 'Find me an everyday image that matches the given caption:'
        
        embeddings = []
        
        for i in range(0, len(texts), Config.TEXT_BATCH_SIZE):
            batch_texts = texts[i:i + Config.TEXT_BATCH_SIZE]
            
            batch_embeds = []
            for text in batch_texts:
                query_text = f'{instruction_prefix} {text}'
                inputs = self.processor(query_text)
                inputs = {key: value.to(Config.DEVICE) for key, value in inputs.items()}
                
                qry_output = self.model(qry=inputs)["qry_reps"]
                batch_embeds.append(qry_output.cpu())
                
                del inputs, qry_output
            
            embeddings.append(torch.cat(batch_embeds, dim=0))
            torch.cuda.empty_cache()
        
        result = torch.cat(embeddings, dim=0).numpy()
        return result

    @torch.no_grad()
    def compute_similarity(self,
                          query_embeds: np.ndarray,
                          corpus_embeds: np.ndarray,
                          batch_size: int = 1000) -> np.ndarray:
        """
        Compute similarity scores between queries and corpus using the model's similarity function.

        Args:
            query_embeds: Query embeddings (n_queries, dim)
            corpus_embeds: Corpus embeddings (n_corpus, dim)
            batch_size: Batch size for similarity computation

        Returns:
            Similarity matrix (n_queries, n_corpus)
        """
        n_queries = query_embeds.shape[0]
        n_corpus = corpus_embeds.shape[0]

        # Convert to torch tensors
        query_tensor = torch.from_numpy(query_embeds).to(Config.DEVICE)
        corpus_tensor = torch.from_numpy(corpus_embeds).to(Config.DEVICE)

        similarities = []

        for i in range(0, n_queries, batch_size):
            batch_queries = query_tensor[i:i+batch_size]
            
            batch_sims = []
            for j in range(0, n_corpus, batch_size):
                batch_corpus = corpus_tensor[j:j+batch_size]
                
                # Compute similarity for each query-corpus pair
                sim_batch = []
                for q in batch_queries:
                    q_expanded = q.unsqueeze(0).expand(batch_corpus.shape[0], -1)
                    sim = self.model.compute_similarity(q_expanded, batch_corpus)
                    sim_batch.append(sim.squeeze())
                
                batch_sims.append(torch.stack(sim_batch).cpu())
            
            similarities.append(torch.cat(batch_sims, dim=1))

        del query_tensor, corpus_tensor
        torch.cuda.empty_cache()

        return torch.cat(similarities, dim=0).numpy()

    def retrieve_i2t(self,
                     query_images: List[Image.Image],
                     query_texts: List[str],
                     corpus_texts: List[str],
                     top_k: int = 10) -> Dict:
        """
        Perform Image+Text to Text retrieval.

        Args:
            query_images: List of query images
            query_texts: List of query texts (questions about images)
            corpus_texts: List of corpus texts
            top_k: Number of top results to return

        Returns:
            Dictionary with retrieval results
        """
        print("Encoding image+text queries...")
        query_embeds = self.encode_image_text_queries(query_images, query_texts)

        print("Encoding corpus texts...")
        corpus_embeds = self.encode_texts(corpus_texts)

        print("Computing similarities...")
        similarities = self.compute_similarity(query_embeds, corpus_embeds)

        # Get top-k results
        top_k = min(top_k, similarities.shape[1])
        top_indices = np.argsort(-similarities, axis=1)[:, :top_k]
        top_scores = np.take_along_axis(similarities, top_indices, axis=1)

        results = {
            'indices': top_indices,
            'scores': top_scores,
        }

        return results

    def retrieve_t2i(self,
                     query_texts: List[str],
                     corpus_images: List[Image.Image],
                     top_k: int = 10,
                     corpus_embeds: np.ndarray = None) -> Dict:
        """
        Perform Text to Image retrieval.

        Args:
            query_texts: List of query texts
            corpus_images: List of corpus images (only needed if corpus_embeds is None)
            top_k: Number of top results to return
            corpus_embeds: Pre-computed corpus embeddings (optional)

        Returns:
            Dictionary with retrieval results
        """
        print("Encoding text queries...")
        query_embeds = self.encode_text_queries(query_texts)

        if corpus_embeds is None:
            print("Encoding corpus images...")
            corpus_embeds = self.encode_images(corpus_images)

        print("Computing similarities...")
        similarities = self.compute_similarity(query_embeds, corpus_embeds)

        # Get top-k results
        top_k = min(top_k, similarities.shape[1])
        top_indices = np.argsort(-similarities, axis=1)[:, :top_k]
        top_scores = np.take_along_axis(similarities, top_indices, axis=1)

        results = {
            'indices': top_indices,
            'scores': top_scores,
        }

        return results

    def cleanup(self):
        """Clean up GPU memory."""
        del self.model
        del self.processor
        torch.cuda.empty_cache()
        gc.collect()
        print("Pipeline cleaned up.")
def i2t_loop(test, pipeline, base_img_url, p):
    retrieved_count = 0
    perfect_count = 0
    scores = []
    for row in tqdm(test):
        # Load example image
        imgURL = base_img_url+ row['qry_img_path']
        image = Image.open(imgURL)
        
        p_img = perturb(image, p, pipeline)
            
        truth = [1] + [0]*999
        results = pipeline.retrieve_i2t(
            query_images = [p_img],
            query_texts = row['qry_text'],
            corpus_texts= row['tgt_text'],
        )
        
        # Clean up image after encoding
        if hasattr(p_img, 'close'):
            p_img.close()
        if hasattr(image, 'close'):
            image.close()
        
        # Calculate nDCG
        t = [truth[i] for i in results['indices'][0][:10]]
        score = ndcg_score([t],[results['scores'][0][:10].tolist()], k=10)
        if score > 0.0: 
            retrieved_count +=1
        if score == 1.0:
            perfect_count +=1

        scores.append(score)
    return scores, perfect_count, retrieved_count
def t2i_loop(test, pipeline, base_img_url, p):
    retrieved_count = 0
    perfect_count = 0
    scores = []
    img_embed_dict = {}
    for row in tqdm(test):
        # Load example image
        # textURL = base_img_url+ row['qry_text']
        idx = 0 
        imgbatch_len = 64  # Match GPU batch size for optimal performance
        img_embeddings=[]
        hit_count = 0
        # encode images in batches
        num_batches = (len(row['tgt_img_path']) + imgbatch_len - 1) // imgbatch_len  # Ceiling division
        for batch_idx in range(num_batches):
            # Separate cached and non-cached images
            cached_embeddings = []
            images_to_encode = []
            urls_to_cache = []
            
            # Handle last batch which may be smaller
            batch_size = min(imgbatch_len, len(row['tgt_img_path']) - batch_idx * imgbatch_len)
            
            for j in range(batch_size):
                idx = batch_idx * imgbatch_len + j
                imgURL = base_img_url+ row['tgt_img_path'][idx] 

                if imgURL in img_embed_dict:
                    cached_embeddings.append((j, img_embed_dict[imgURL]))
                    hit_count += 1
                else:
                    # Load and perturb image (don't use context manager to avoid closing before encoding)
                    image = Image.open(imgURL)
                    p_img = perturb(image, p, pipeline)
                    images_to_encode.append((j, p_img))
                    urls_to_cache.append(imgURL)
            
            # Batch encode all non-cached images at once (single GPU transfer)
            if images_to_encode:
                imgs_only = [img for _, img in images_to_encode]
                batch_embeds = pipeline.encode_images(imgs_only)
                
                # Cache the new embeddings
                for i, imgURL in enumerate(urls_to_cache):
                    img_embed_dict[imgURL] = batch_embeds[i]
                
                # Combine cached and newly encoded embeddings in correct order
                newly_encoded = [(images_to_encode[i][0], batch_embeds[i]) for i in range(len(images_to_encode))]
                all_embeddings = cached_embeddings + newly_encoded
                all_embeddings.sort(key=lambda x: x[0])  # Sort by original position
                
                img_embeddings.extend([emb for _, emb in all_embeddings])
                
                # Clean up images and temporary data
                for img in imgs_only:
                    if hasattr(img, 'close'):
                        img.close()
                del batch_embeds, imgs_only, newly_encoded, all_embeddings, images_to_encode
            else:
                # All cached - just add them in order
                img_embeddings.extend([emb for _, emb in cached_embeddings])
                del images_to_encode
            
            # Clear lists for next batch
            del cached_embeddings, urls_to_cache
            
        tqdm.write(f"Embedding Dict Hit count {hit_count}/{len(row['tgt_img_path'])}")
        
        # Stack embeddings efficiently - embeddings are already numpy arrays
        corpus_embeds = np.vstack(img_embeddings) if img_embeddings else np.array([])
        
        truth = [1] + [0]*999
        results = pipeline.retrieve_t2i(
            query_texts=[row['qry_text']],
            corpus_images=None,
            corpus_embeds=corpus_embeds,
        )

        # Calculate nDCG
        t = [truth[i] for i in results['indices'][0][:10]]
        #print(results['scores'].shape, len(t), results['scores'])
        score = ndcg_score([t],[results['scores'][0][:10].tolist()], k=10)
        if score > 0.0: 
            retrieved_count +=1
        if score == 1.0:
            perfect_count +=1

        scores.append(score)
    return scores, perfect_count, retrieved_count

def example_usage():
    """Example usage demonstrating the pipeline."""
    
    # Initialize pipeline
    pipeline = VLM2VecRetrievalPipeline()
    
    # Example 1: Image + Text -> Text
    print("\n=== Example 1: Image + Text -> Text ===")
    ds = load_dataset("TIGER-Lab/MMEB-eval", "VisualNews_i2t")

    query_img = Image.open('/work/aho13/MMEB-eval/VisualNews_i2t/washington_post_images_0374_729.jpg')
    query_text = 'Find a caption for the news in the given photo'
    
    # Encode query
    qry_output = pipeline.encode_image_text_queries([query_img], [query_text])
    
    # Encode target texts
    target_texts = ['Indian National Congress Vice President Rahul Gandhi addresses the special plenary session of Confederation of Indian Industr in New Delhi on April 4 2013.',\
        'Forensic pathologist Shawn Parcells demonstrates how Michael Brown might have been standing.',\
        "Former prostitute Tamieka Gamble was convicted of tying up and slashing a client with a knife The victim later died."]
    tgt_outputs = pipeline.encode_texts(target_texts)
    
    # Compute similarities
    similarities = pipeline.compute_similarity(qry_output, tgt_outputs)
    
    for text, sim in zip(target_texts, similarities[0]):
        print(f"{text} = {sim:.4f}")
    
    # Example 2: Text -> Image
    print("\n=== Example 2: Text -> Image ===")
    query_texts = ['Find me an everyday image that matches the given caption: A cat and a dog.',
                   'Find me an everyday image that matches the given caption: A cat and a tiger.']
    
    # Encode queries
    qry_outputs = pipeline.encode_text_queries(list(target_texts[0]), instruction_prefix='')
    
    # Encode target image
    tgt_output = pipeline.encode_images([query_img], instruction='<|image_1|>\nRepresent the given image.')
    
    # Compute similarities
    similarities = pipeline.compute_similarity(qry_outputs, tgt_output)
    
    for query, sim in zip(query_texts, similarities[:, 0]):
        print(f"{query} = {sim:.4f}")
    
    # Cleanup
    pipeline.cleanup()


def setup_ndcg_logger(log_file):
    """Setup logger for NDCG scores."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def deliverthegoods(datasets, perturbations, model_name):
    pipeline = VLM2VecRetrievalPipeline(model_name)
    ndcg_scores = {}
    avg_ndcg_scores = {}
    base_img_url = '/work/aho13/MMEB-eval/'
    k=10

    i2tds = set(["MSCOCO_i2t","VisualNews_i2t"])
    t2ids = set(["MSCOCO_t2i","VisualNews_t2i", "VisDial", "Wiki-SS-NQ"])


    with torch.autocast(device_type='cuda', dtype=torch.float16):
        # Loops for each dataset
        for ds_name in datasets:
            #df = pd.read_csv(base_url+f"{ds_name}/test-00000-of-00001.parquet", encoding="utf-16")
            ds = load_dataset("TIGER-Lab/MMEB-eval", ds_name)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f"{os.getcwd()}/logs/{ds_name}_{timestamp}.log"
    
            logger = setup_ndcg_logger(log_file)

            for p in perturbations:
                scores=[]
                perfect_count = 0
                retrieved_count = 0
                if ds_name in i2tds:
                    scores, perfect_count, retrieved_count = i2t_loop(list(ds['test']), pipeline, base_img_url, p)
                
                elif ds_name in t2ids:
                    scores, perfect_count, retrieved_count = t2i_loop(list(ds['test']), pipeline, base_img_url, p)
                
                # Calculate average nDCG
                avg_ndcg = np.mean(scores)
                # Log summary
                if logger:
                    logger.info("="*60)
                    logger.info(f"NDCG@{k} {ds_name} {p} Results:")
                    logger.info(f"  Mean NDCG: {avg_ndcg:.4f}")
                    logger.info(f"  Median NDCG: {np.median(scores):.4f}")
                    logger.info(f"  Std Dev: {np.std(scores):.4f}")
                    logger.info(f"  Perfect retrievals (rank 1): {perfect_count}/{len(scores)} ({100*perfect_count/len(scores):.1f}%)")
                    logger.info(f"  Relevant in top-{k}: {retrieved_count}/{len(scores)} ({100*retrieved_count/len(scores):.1f}%)")
                    logger.info("="*60)
    pipeline.cleanup()
    return ndcg_scores

if __name__ == "__main__":
    # example_usage()
    datasets = ['MSCOCO_i2t', 'MSCOCO_t2i', 'VisualNews_i2t', 'VisualNews_t2i']
    perturbations = ['none', 'pgd']
    deliverthegoods(datasets, perturbations, 'TIGER-Lab/VLM2Vec-Full')
