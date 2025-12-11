import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Dict, Union, Optional
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
import requests
from io import BytesIO
from transformers import cache_utils

# Monkey patch for DynamicCache issue in newer transformers versions
if hasattr(cache_utils, "DynamicCache") and not hasattr(cache_utils.DynamicCache, "get_usable_length"):
    def get_usable_length(self, input_length, layer_idx=None):
        if layer_idx is None:
            layer_idx = 0
        return self.get_seq_length(layer_idx) if hasattr(self, "get_seq_length") else input_length
    cache_utils.DynamicCache.get_usable_length = get_usable_length

class Config:
    """Configuration for the UNiME retrieval pipeline."""
    MODEL_NAME = "DeepGlint-AI/UniME-Phi3.5-V-4.2B"
    BATCH_SIZE = 32
    IMAGE_BATCH_SIZE = 8
    TEXT_BATCH_SIZE = 64
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    NUM_WORKERS = 2
    PIN_MEMORY = True

class UniMEPipeline:
    """
    UNiME retrieval pipeline.
    Adapts the interface of MultimodalRetrievalPipeline for UNiME model.
    """

    def __init__(self, model_name: str = Config.MODEL_NAME):
        """Initialize the pipeline with UNiME model."""
        print(f"Loading model {model_name} on {Config.DEVICE}...")
        
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=Config.DTYPE,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Enable gradient checkpointing for memory efficiency during attacks
        self.model.gradient_checkpointing_enable()

        # Match official setup from unime.py
        self.processor.tokenizer.padding_side = "left"
        self.processor.tokenizer.padding = True

        # Official prompts
        self.img_prompt = "<|user|>\n<|image_1|>\nSummary above image in one word: <|end|>\n<|assistant|>\n"
        self.text_prompt = "<|user|>\n<sent>\nSummary above sentence in one word: <|end|>\n<|assistant|>\n"
        # Constructed multimodal prompt
        self.multimodal_prompt = "<|user|>\n<|image_1|>\n<sent>\nSummary above image and sentence in one word: <|end|>\n<|assistant|>\n"

        self.model.eval()
        # Freeze model parameters to prevent gradient computation during attacks
        for param in self.model.parameters():
            param.requires_grad = False
            
        print(f"Model loaded successfully.")

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image], img_prompt: str = None) -> np.ndarray:
        embeds = []
        if img_prompt is None:
            img_prompt = self.img_prompt
        for img in images:
            inputs = self.processor(
                text=img_prompt,
                images=[img],
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(Config.DEVICE) for k, v in inputs.items()}

            out = self.model(**inputs, output_hidden_states=True, return_dict=True)
            vec = out.hidden_states[-1][:, -1, :]
            vec = F.normalize(vec, dim=-1)
            embeds.append(vec.cpu())

        if not embeds:
            return np.array([])
            
        return torch.cat(embeds, dim=0).to(torch.float32).numpy()


    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings.
        """
        embeds = []
        for t in texts:
            prompts = [self.text_prompt.replace("<sent>", t)]

            inputs = self.processor(
                text=prompts,
                images=None,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(Config.DEVICE) for k, v in inputs.items()}

            out = self.model(**inputs, output_hidden_states=True, return_dict=True)
            vec = out.hidden_states[-1][:, -1, :]
            vec = F.normalize(vec, dim=-1)
            embeds.append(vec.cpu())

        if not embeds:
            return np.array([])

        return torch.cat(embeds, dim=0).to(torch.float32).numpy()

    @torch.no_grad()
    def encode_multimodal(self, images: List[Image.Image], texts: List[str]) -> np.ndarray:
        """
        Encode image-text pairs into a single embedding.
        """
        if len(images) != len(texts):
            raise ValueError("Number of images and texts must match")
            
        embeds = []
        for img, t in zip(images, texts):
            prompt = self.multimodal_prompt.replace("<sent>", t)
            inputs = self.processor(
                text=prompt,
                images=[img],
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(Config.DEVICE) for k, v in inputs.items()}

            out = self.model(**inputs, output_hidden_states=True, return_dict=True)
            vec = out.hidden_states[-1][:, -1, :]
            vec = F.normalize(vec, dim=-1)
            embeds.append(vec.cpu())
            
        if not embeds:
            return np.array([])

        return torch.cat(embeds, dim=0).to(torch.float32).numpy()

    @torch.no_grad()
    def compute_similarity(self,
                          query_embeds: np.ndarray,
                          corpus_embeds: np.ndarray,
                          batch_size: int = 1000) -> np.ndarray:
        """
        Compute similarity scores between queries and corpus.
        """
        n_queries = query_embeds.shape[0]
        
        # Convert to torch tensors
        corpus_tensor = torch.from_numpy(corpus_embeds).to(Config.DEVICE)
        
        similarities = []

        for i in range(0, n_queries, batch_size):
            batch_queries = query_embeds[i:i+batch_size]
            query_tensor = torch.from_numpy(batch_queries).to(Config.DEVICE)

            # Compute cosine similarity
            sim = torch.matmul(query_tensor, corpus_tensor.T)
            similarities.append(sim.float().cpu().numpy())
            
            del query_tensor, sim

        del corpus_tensor
        
        if not similarities:
            return np.array([])

        return np.vstack(similarities)

    def retrieve_i2t(self,
                 query_img: List[Image.Image],
                 corpus_text: Union[List[str], Dict]) -> Dict:
        """
        Perform retrieval from image queries to text corpus.
        """
        top_k = 10

        # Encode query images
        qry_emb = self.encode_images(query_img)

        # Encode corpus
        # Handle case where corpus_text might be a list of strings or something else
        # Assuming list of strings for now based on usage
        if isinstance(corpus_text, list):
             corpus_embeds = self.encode_texts(corpus_text)
        else:
            # Fallback or error if needed, but for now assume list
             corpus_embeds = self.encode_texts(list(corpus_text))

        similarities = self.compute_similarity(qry_emb, corpus_embeds)
        
        top_k = min(top_k, similarities.shape[1])
        top_indices = np.argsort(-similarities, axis=1)[:, :top_k]
        top_scores = np.take_along_axis(similarities, top_indices, axis=1)

        results = {
            'indices': top_indices,
            'scores': top_scores,
        }

        return results

    def retrieve_t2i(self, query_text: List[str], corpus_img_embeds: np.ndarray) -> Dict:
        """
        Perform retrieval from text queries to image corpus embeddings.
        """
        top_k = 10
        query_txt_embeds = self.encode_texts(query_text)
        similarities = self.compute_similarity(query_txt_embeds, corpus_img_embeds)

        top_k = min(top_k, similarities.shape[1])
        top_indices = np.argsort(-similarities, axis=1)[:, :top_k]
        top_scores = np.take_along_axis(similarities, top_indices, axis=1)

        results = {
            'indices': top_indices,
            'scores': top_scores,
        }

        return results

    def retrieve_multimodal(self, 
                          query_images: List[Image.Image], 
                          query_texts: List[str],
                          corpus_images: List[Image.Image],
                          corpus_texts: List[str]) -> Dict:
        """
        Perform retrieval using multimodal queries and corpus.
        """
        top_k = 10
        
        # Encode queries
        query_embeds = self.encode_multimodal(query_images, query_texts)
        
        # Encode corpus
        corpus_embeds = self.encode_multimodal(corpus_images, corpus_texts)
        
        # Compute similarities
        similarities = self.compute_similarity(query_embeds, corpus_embeds)
        
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
        print("Pipeline cleaned up.")

