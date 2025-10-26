from huggingface_hub import hf_hub_download
from datasets import load_dataset
import zipfile, os
from PIL import Image, ImageFilter, ImageEnhance
import requests
import torch

import gc
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Union
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from sklearn.metrics import ndcg_score
import cv2

from transformers import CLIPProcessor, CLIPModel, AutoModelForCausalLM

# model = AutoModelForCausalLM.from_pretrained("TIGER-Lab/VLM2Vec-Full", trust_remote_code=True, torch_dtype="auto")
torch.cuda.is_available()
class Config:
    """Configuration for the retrieval pipeline."""
    MODEL_NAME = "openai/clip-vit-base-patch32" #"TIGER-Lab/VLM2Vec-Full" #
    BATCH_SIZE = 256  # Adjust based on your GPU memory
    IMAGE_BATCH_SIZE = 64  # Smaller batch for images (more memory intensive)
    TEXT_BATCH_SIZE = 256  # Larger batch for text-only processing
    MAX_TEXT_LENGTH = 77  # CLIP's maximum text length
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16  # Use mixed precision for memory efficiency
    NUM_WORKERS = 0  # For DataLoader
    PIN_MEMORY = True

# =============================================================================
# Custom Dataset Classes
# =============================================================================

class MMEBImageTextDataset(Dataset):
    """Dataset for image-text pairs from MMEB."""

    def __init__(self, data: List[Dict], processor: CLIPProcessor, mode: str = "query"):
        """
        Args:
            data: List of dicts with 'image' and 'text' keys
            processor: CLIP processor
            mode: 'query' or 'corpus' to handle different data structures
        """
        self.data = data
        self.processor = processor
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Handle image loading
        if 'image' in item:
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
        if 'text' in item:
            text = item['text']
            if isinstance(text, list):
                text = text[0] if text else ""
            text = text[:Config.MAX_TEXT_LENGTH]
        elif 'tgt_text' in item:
            text = item['tgt_text']
            if isinstance(text, list):
                text = [t[:Config.MAX_TEXT_LENGTH] for t in text[:1]]
                text = text[0] if text else ""
        else:
            text = ""

        return {'image': img, 'text': text, 'idx': idx}
# =============================================================================
# Custom Dataset Classes
# =============================================================================

class MMEBImageTextDataset(Dataset):
    """Dataset for image-text pairs from MMEB."""

    def __init__(self, data: List[Dict], processor: CLIPProcessor, mode: str = "query"):
        """
        Args:
            data: List of dicts with 'image' and 'text' keys
            processor: CLIP processor
            mode: 'query' or 'corpus' to handle different data structures
        """
        self.data = data
        self.processor = processor
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Handle image loading
        if 'image' in item:
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
        if 'text' in item:
            text = item['text']
            if isinstance(text, list):
                text = text[0] if text else ""
            text = text[:Config.MAX_TEXT_LENGTH]
        elif 'tgt_text' in item:
            text = item['tgt_text']
            if isinstance(text, list):
                text = [t[:Config.MAX_TEXT_LENGTH] for t in text[:1]]
                text = text[0] if text else ""
        else:
            text = ""

        return {'image': img, 'text': text, 'idx': idx}

class TextOnlyDataset(Dataset):
    """Dataset for text-only processing (corpus)."""

    def __init__(self, texts: List[str], processor: CLIPProcessor):
        self.texts = [t[:Config.MAX_TEXT_LENGTH] for t in texts]
        self.processor = processor

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {'text': self.texts[idx], 'idx': idx}

# =============================================================================
# Custom Collate Functions
# =============================================================================

def collate_image_text(batch, processor):
    """Collate function for image-text batches."""
    images = [item['image'] for item in batch if item['image'] is not None]
    texts = [item['text'] for item in batch]
    indices = [item['idx'] for item in batch]

    # Process with CLIP processor
    if images:
        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=Config.MAX_TEXT_LENGTH
        )
    else:
        inputs = processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=Config.MAX_TEXT_LENGTH
        )

    inputs['indices'] = torch.tensor(indices)
    return inputs

def collate_text_only(batch, processor):
    """Collate function for text-only batches."""
    texts = [item['text'] for item in batch]
    indices = [item['idx'] for item in batch]

    inputs = processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=Config.MAX_TEXT_LENGTH
    )

    inputs['indices'] = torch.tensor(indices)
    return inputs

# =============================================================================
# Retrieval Pipeline
# =============================================================================

class MultimodalRetrievalPipeline:
    """Efficient multimodal retrieval pipeline for MMEB."""

    def __init__(self, model_name: str = Config.MODEL_NAME):
        """Initialize the pipeline with a CLIP model."""
        print(f"Loading model on {Config.DEVICE}...")


        if model_name == "openai/clip-vit-base-patch32":
            # Load model with memory optimizations
            self.model = CLIPModel.from_pretrained(
                model_name,
                dtype=Config.DTYPE
            ).to(Config.DEVICE)
        else:
            self.model = AutoModelForCausalLM.from_pretrained("TIGER-Lab/VLM2Vec-Full", trust_remote_code=True, dtype="auto")

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode

        print(f"Model loaded successfully. Using dtype: {Config.DTYPE}")

    @torch.no_grad()
    def encode_images(self, images: Union[List[Image.Image], DataLoader]) -> np.ndarray:
        """
        Encode images to embeddings.

        Args:
            images: List of PIL Images or DataLoader

        Returns:
            numpy array of shape (n_images, embedding_dim)
        """
        if isinstance(images, list):
            # Create dataset and dataloader
            data = [{'image': img, 'text': ''} for img in images]
            dataset = MMEBImageTextDataset(data, self.processor)
            dataloader = DataLoader(
                dataset,
                batch_size=Config.IMAGE_BATCH_SIZE,
                collate_fn=lambda x: collate_image_text(x, self.processor),
                num_workers=Config.NUM_WORKERS,
                pin_memory=Config.PIN_MEMORY
            )
        else:
            dataloader = images

        embeddings = []

        for batch in dataloader:
            # Move to device
            inputs = {k: v.to(Config.DEVICE) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items() if k != 'indices'}

            # Get image embeddings
            if 'pixel_values' in inputs:
                image_embeds = self.model.get_image_features(pixel_values=inputs['pixel_values'])
            else:
                continue

            # Normalize embeddings
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

            embeddings.append(image_embeds.cpu().numpy())

            # Clear GPU memory
            del inputs, image_embeds
            torch.cuda.empty_cache()

        return np.vstack(embeddings)

    @torch.no_grad()
    def encode_texts(self, texts: Union[List[str], DataLoader]) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of strings or DataLoader

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, list):
            # Create dataset and dataloader
            dataset = TextOnlyDataset(texts, self.processor)
            dataloader = DataLoader(
                dataset,
                batch_size=Config.TEXT_BATCH_SIZE,
                collate_fn=lambda x: collate_text_only(x, self.processor),
                num_workers=Config.NUM_WORKERS,
                pin_memory=Config.PIN_MEMORY
            )
        else:
            dataloader = texts

        embeddings = []

        for batch in dataloader:
            # Move to device
            inputs = {k: v.to(Config.DEVICE) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items() if k != 'indices'}

            # Get text embeddings
            text_embeds = self.model.get_text_features(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )

            # Normalize embeddings
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

            embeddings.append(text_embeds.cpu().numpy())

            # Clear GPU memory
            del inputs, text_embeds
            torch.cuda.empty_cache()

        return np.vstack(embeddings)

    @torch.no_grad()
    def compute_similarity(self,
                          query_embeds: np.ndarray,
                          corpus_embeds: np.ndarray,
                          batch_size: int = 1000) -> np.ndarray:
        """
        Compute similarity scores between queries and corpus.
        Processes in batches to avoid memory issues.

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
        corpus_tensor = torch.from_numpy(corpus_embeds).to(Config.DEVICE)

        similarities = []

        for i in range(0, n_queries, batch_size):
            batch_queries = query_embeds[i:i+batch_size]
            query_tensor = torch.from_numpy(batch_queries).to(Config.DEVICE)

            # Compute cosine similarity: (batch_size, dim) @ (dim, n_corpus) = (batch_size, n_corpus)
            dotprod = torch.matmul(query_tensor, corpus_tensor.T)
            sim = dotprod # * denom
            similarities.append(sim.cpu().numpy())

            del query_tensor, sim
            torch.cuda.empty_cache()

        del corpus_tensor
        torch.cuda.empty_cache()

        return np.vstack(similarities)

    def retrieve(self,
                 query_text: Union[List[str], Dict],
                 query_img: List[Image],
                 corpus: Union[List[str], List[Image.Image], Dict],
                 top_k: int = 10,
                 query_type: str = 'text',
                 corpus_type: str = 'text') -> Dict:
        """
        Perform retrieval from queries to corpus.

        Args:
            queries: Query data
            corpus: Corpus data
            top_k: Number of top results to return
            query_type: 'text', 'image', or 'multimodal'
            corpus_type: 'text', 'image', or 'multimodal'

        Returns:
            Dictionary with retrieval results
        """
        # Encode queries
        # if query_type == 'text':
        # elif query_type == 'image':
        qry_emb_text = self.encode_texts(query_text)
        qry_emb_img = self.encode_images(query_img)

        print("emb text shape", qry_emb_text.shape)
        print("emb img shape", qry_emb_img.shape)

        # Encode corpus
        if corpus_type == 'text':
            corpus_embeds = self.encode_texts(corpus)
        elif corpus_type == 'image':
            corpus_embeds = self.encode_images(corpus)
        else:
            raise ValueError(f"Unknown corpus_type: {corpus_type}")
        # query_embed=  self.compute_similarity(qry_emb_img, qry_emb_text)
        # Compute similarities
        similarities = self.compute_similarity(qry_emb_text, corpus_embeds)

        # Get top-k results
        top_k = min(top_k, similarities.shape[1])
        top_indices = np.argsort(-similarities, axis=1)[:, :top_k]
        top_scores = np.take_along_axis(similarities, top_indices, axis=1)

        results = {
            'indices': top_indices,
            'scores': top_scores,
            # 'query_embeddings': query_embeds,
            # 'corpus_embeddings': corpus_embeds
        }

        return results

    def cleanup(self):
        """Clean up GPU memory."""
        del self.model
        torch.cuda.empty_cache()
        gc.collect()
        print("Pipeline cleaned up.")

