from huggingface_hub import hf_hub_download
from datasets import load_dataset
import zipfile, os
from PIL import Image, ImageFilter, ImageEnhance
import requests
import torch
import torch.nn.functional as F
import torchvision.transforms as TF
import logging
import gc
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Union
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from sklearn.metrics import ndcg_score
import cv2
from pipe import MultimodalRetrievalPipeline
from datetime import datetime
from transformers import AutoModel, AutoProcessor
from pipe import Config
from experiment import setup_ndcg_logger, fgsm_attack_clip, perturb, i2t, t2i


class jinaPipeline:
    """Efficient multimodal retrieval pipeline for MMEB."""
    def __init__(self, model_name: str = Config.MODEL_NAME):
        """Initialize the pipeline with a CLIP model."""
        print(f"Loading model on {Config.DEVICE}...")

        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype="auto")
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval() 

        print(f"Model loaded successfully. Using dtype: {Config.DTYPE}")

    def process(self, images, texts):

        dataset = mmebDataset(images, texts)
        dataloader = DataLoader(
            dataset,
            batch_size=Config.IMAGE_BATCH_SIZE,
            collate_fn=lambda x: collate_image_text(x, self.processor),
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY
        )

        embeddings = []
        for batch in dataloader:
            # Move to device
            inputs = {k: v.to(Config.DEVICE, non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items() if k != 'indices'}
            # Get image embeddings
            image_embeds = self.model.get_image_features(pixel_values=inputs['pixel_values'])

            # Normalize embeddings
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            result = image_embeds.cpu().numpy()

            # Clear GPU memory
            del inputs, image_embeds
            embeddings.append(result)

        return np.vstack(embeddings)
    def retrieve_i2t(self,
                 query_img: List[Image],
                 corpus_text: Union[List[str], List[Image.Image], Dict]  ) -> Dict:
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

        top_k = 10

        # Encode query images
        qry_emb_img = self.encode_images(query_img)

        # Encode corpus
        inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True)

        corpus_embeds = self.encode_texts(corpus_text)
        similarities=  self.compute_similarity(qry_emb_img, corpus_embeds)
        
        # Compute similarities
        # Get top-k results
        top_k = min(top_k, similarities.shape[1])
        top_indices = np.argsort(-similarities, axis=1)[:, :top_k]
        top_scores = np.take_along_axis(similarities, top_indices, axis=1)

        results = {
            'indices': top_indices,
            'scores': top_scores,
        }

        return results


    def retrieve_t2i(self, query_text, corpus_img_embeds) -> Dict:
        top_k = 10
        query_txt_embeds = self.encode_texts(query_text)
        similarities=  self.compute_similarity(query_txt_embeds, corpus_img_embeds)

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
        torch.cuda.empty_cache()
        gc.collect()
        print("Pipeline cleaned up.")

def i2t(test, pipeline, base_img_url, p):
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
            query_img = [p_img],
            corpus_text= row['tgt_text'],
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



def deliverTheGoods2Jina(datasets, perturbations, model_name):
    pipeline = MultimodalRetrievalPipeline(model_name)
    ndcg_scores = {}
    avg_ndcg_scores = {}
    base_img_url = '/work/aho13/work/aho13/'
    k=10

    i2tds = set(["MSCOCO_i2t","VisualNews_i2t"])
    t2ids = set(["MSCOCO_t2i","VisualNews_t2i", "VisDial", "Wiki-SS-NQ"])


    with torch.autocast(device_type='cuda', dtype=torch.float16):
        # Loops for each dataset
        for ds_name in datasets:
            ds = load_dataset("TIGER-Lab/MMEB-eval", ds_name)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f"{os.getcwd()}/logs/jina/{ds_name}_{timestamp}.log"
    
            logger = setup_ndcg_logger(log_file)
            scores, perfect_count, retrieved_count = i2t(list(ds['test']), pipeline, base_img_url, p)


            # for p in perturbations:
            scores=[]
            perfect_count = 0
            retrieved_count = 0

            image_embeddings = pipeline.process(ds['query_img'], ds['query_text'])
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


def main():

    # Visual News_i2t dataset and pipeline
    datasets=["VisualNews_t2i","MSCOCO_t2i","VisDial","WIKI-SS-NQ"]
    perturbations = ["flip", "gauss1","bright","grayscale",'gauss2','compress', 'flip']
    model_name = "jinaai/jina-embeddings-v4"

    ndcg_scores = deliverTheGoods2Jina(datasets, perturbations, model_name)


if __name__ == "__main__":
    main()



