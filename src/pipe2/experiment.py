from datasets import load_dataset
from unimepipe import UniMEPipeline
import os
from PIL import Image, ImageFilter, ImageEnhance
import requests
import torch
import torch.nn.functional as F
import logging
import gc
from typing import List, Dict, Optional, Union
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from sklearn.metrics import ndcg_score
from datetime import datetime
from pathlib import Path
from attacks import pgd_attack_unime


def setup_ndcg_logger(log_file: str = None):
    """
    Setup a simple logger for NDCG results.
    """
    if log_file is None:
        log_dir = Path("logs/unimelogs")
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"unime_{timestamp}.log"
    else:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger('UNiME_Eval')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers
    
    # File handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info(f"Logging to: {log_file}")
    return logger


def perturb(img, ptype, pipeline=None):
    if ptype == "gauss1":
        return img.filter(ImageFilter.GaussianBlur(radius=1))
    if ptype == "gauss2":
        return img.filter(ImageFilter.GaussianBlur(radius=2))
    elif ptype== "grayscale":
        return img.convert('L').convert('RGB') # Convert back to RGB for model compatibility
    # increase brightness
    elif ptype=="bright":
        enhancer = ImageEnhance.Brightness(img)
        brightness_factor = 1.5
        brightened_image = enhancer.enhance(brightness_factor)
        return brightened_image
    elif ptype=="flip":
        flip_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return flip_img
    elif ptype=="compress":
        new_width, new_height = img.size[0] // 5 * 4, img.size[1] //5 *4
        return img.resize((new_width, new_height))
    # Placeholder for attacks if needed later
    elif ptype == "fgsm":
        # assert pipeline is not None, "Pipeline required for FGSM attack"
        # return fgsm_attack_clip(img, pipeline, epsilon=0.03)
        print("FGSM not implemented for UNiME yet, returning original")
        return img
    elif ptype == "pgd":
        assert pipeline is not None, "pipeline required for pgd"
        return pgd_attack_unime(img, pipeline, epsilon=0.03)
    else:
        return img

def i2t(test, pipeline, base_img_url, p):
    retrieved_count = 0
    perfect_count = 0
    scores = []
    text_cache = {}
    
    for row in tqdm(test):
        # Load example image
        img_path = row['qry_img_path']
        if img_path.startswith('http'):
            imgURL = img_path
        else:
            imgURL = os.path.join(base_img_url, img_path)
            
        try:
            if imgURL.startswith('http'):
                response = requests.get(imgURL, stream=True)
                image = Image.open(response.raw).convert('RGB')
            else:
                image = Image.open(imgURL).convert('RGB')
        except Exception as e:
            print(f"Error loading image {imgURL}: {e}")
            continue
        
        p_img = perturb(image, p, pipeline)
        
        # Clear cache after perturbation (especially for PGD)
        if p == "pgd":
            torch.cuda.empty_cache()
            gc.collect()
            
        truth = [1] + [0]*999
        
        # Ensure target text is a list
        tgt_text = row['tgt_text']
        if not isinstance(tgt_text, list):
            tgt_text = [tgt_text]
            
        # Encode image
        img_prompt = row.get('qry_text')
        qry_emb = pipeline.encode_images([p_img], img_prompt=img_prompt)
        
        # Encode targets with cache
        target_embeds_list = []
        texts_to_encode = []
        indices_to_fill = []
        
        for idx, t in enumerate(tgt_text):
            if t in text_cache:
                target_embeds_list.append(text_cache[t])
            else:
                texts_to_encode.append(t)
                target_embeds_list.append(None) # Placeholder
                indices_to_fill.append(idx)
        
        if texts_to_encode:
            # Batch encode new texts
            new_embeds = pipeline.encode_texts(texts_to_encode)
            for i, txt in enumerate(texts_to_encode):
                emb = new_embeds[i]
                text_cache[txt] = emb
                # Fill placeholder
                original_idx = indices_to_fill[i]
                target_embeds_list[original_idx] = emb
                
        target_embeds = np.stack(target_embeds_list)
        
        # Compute similarity
        similarities = pipeline.compute_similarity(qry_emb, target_embeds)[0]
        
        # Clean up image after encoding
        if hasattr(p_img, 'close'):
            p_img.close()
        if hasattr(image, 'close'):
            image.close()
        
        # Calculate nDCG
        # Check if we have enough results
        num_results = len(similarities)
        limit = min(10, num_results)
        
        top_indices = np.argsort(-similarities)[:limit]
        top_scores = similarities[top_indices]
        
        t = [truth[i] for i in top_indices]
        # Pad if less than 10
        if len(t) < 10:
            t += [0] * (10 - len(t))
            
        score = ndcg_score([t],[top_scores.tolist() + [0]*(10-limit)], k=10)
        
        if score > 0.0: 
            retrieved_count +=1
        if score == 1.0:
            perfect_count +=1

        scores.append(score)
    return scores, perfect_count, retrieved_count


def i2t_batch(test, pipeline, base_img_url, p, batch_size=16):
    """Optimized I2T evaluation with batching."""
    retrieved_count = 0
    perfect_count = 0
    scores = []
    
    # Collect all queries first for batch processing
    all_images = []
    all_targets = []
    all_prompts = [] # Collect prompts
    valid_indices = []
    
    print("Loading images...")
    for idx, row in enumerate(tqdm(test)):
        img_path = row['qry_img_path']
        image, imgURL = load_image(img_path, base_img_url)
        
        if image is None:
            continue
            
        p_img = perturb(image, p, pipeline)
        if p == "pgd":
            torch.cuda.empty_cache()
            gc.collect()
        all_images.append(p_img)
        
        tgt_text = row['tgt_text']
        if not isinstance(tgt_text, list):
            tgt_text = [tgt_text]
        all_targets.append(tgt_text)
        
        # Collect prompt if available
        if 'qry_text' in row:
             all_prompts.append(row['qry_text'])
        else:
             all_prompts.append(None)

        valid_indices.append(idx)
        
        # Close original if different from perturbed
        if hasattr(image, 'close') and image != p_img:
            image.close()
    
    print(f"Processing {len(all_images)} queries in batches of {batch_size}...")
    
    # Process in batches
    text_cache = {}
    
    for i in tqdm(range(0, len(all_images), batch_size)):
        batch_images = all_images[i:i+batch_size]
        batch_targets = all_targets[i:i+batch_size]
        batch_prompts = all_prompts[i:i+batch_size]
        
        # Encode all queries in batch
        # If prompts are available, pass them. If any is None, fallback to default (handled by pipeline if we pass None?)
        # unimepipe.encode_images expects img_prompt to be str or list.
        # If all_prompts contains None, we should probably handle it.
        # But for now let's assume qry_text is present if user tried to use it.
        
        if all(p is not None for p in batch_prompts):
             query_embeds = pipeline.encode_images(batch_images, img_prompt=batch_prompts)
        else:
             query_embeds = pipeline.encode_images(batch_images)
        
        # Collect all unique texts needed for this batch
        unique_texts_needed = set()
        for targets in batch_targets:
            for t in targets:
                if t not in text_cache:
                    unique_texts_needed.add(t)
        
        # Encode new texts in sub-batches
        if unique_texts_needed:
            texts_to_encode = list(unique_texts_needed)
            text_batch_size = 64 # Adjust as needed
            
            for k in range(0, len(texts_to_encode), text_batch_size):
                sub_batch = texts_to_encode[k:k+text_batch_size]
                sub_embeds = pipeline.encode_texts(sub_batch)
                
                for txt, emb in zip(sub_batch, sub_embeds):
                    text_cache[txt] = emb
                    
        # Process each query's targets using cache
        for j, (qry_emb, targets) in enumerate(zip(query_embeds, batch_targets)):
            # Retrieve embeddings from cache
            target_embeds_list = [text_cache[t] for t in targets]
            target_embeds = np.stack(target_embeds_list)
            
            # Compute similarity
            qry_emb_expanded = np.expand_dims(qry_emb, 0)
            similarities = pipeline.compute_similarity(qry_emb_expanded, target_embeds)[0]
            
            truth = [1] + [0]*999
            
            # Get top k
            num_results = len(similarities)
            limit = min(10, num_results)
            top_indices = np.argsort(-similarities)[:limit]
            top_scores = similarities[top_indices]
            
            t = [truth[idx] for idx in top_indices]
            if len(t) < 10:
                t += [0] * (10 - len(t))
            
            score = ndcg_score([t], [top_scores.tolist() + [0]*(10-limit)], k=10)
            
            if score > 0.0: 
                retrieved_count += 1
            if score == 1.0:
                perfect_count += 1
            
            scores.append(score)
        
        # Clean up batch images
        for img in batch_images:
            if hasattr(img, 'close'):
                img.close()
        
        # Periodic GPU cleanup
        if i % (batch_size * 4) == 0:
            torch.cuda.empty_cache()
    
    return scores, perfect_count, retrieved_count


def t2i(test, pipeline, base_img_url, p):
    retrieved_count = 0
    perfect_count = 0
    scores = []
    img_embed_dict = {}
    
    for row in tqdm(test):
        # row['tgt_img_path'] is a list of image paths
        tgt_img_paths = row['tgt_img_path']
        
        imgbatch_len = 8 # Smaller batch for VLM
        img_embeddings=[]
        
        # encode images in batches
        num_batches = (len(tgt_img_paths) + imgbatch_len - 1) // imgbatch_len
        
        for batch_idx in range(num_batches):
            images_to_encode = []
            urls_to_cache = []
            
            batch_start = batch_idx * imgbatch_len
            batch_end = min((batch_idx + 1) * imgbatch_len, len(tgt_img_paths))
            
            for j in range(batch_start, batch_end):
                img_path = tgt_img_paths[j]
                if img_path.startswith('http'):
                    imgURL = img_path
                else:
                    imgURL = os.path.join(base_img_url, img_path)

                if imgURL in img_embed_dict:
                    img_embeddings.append(img_embed_dict[imgURL])
                else:
                    try:
                        if imgURL.startswith('http'):
                            response = requests.get(imgURL, stream=True)
                            image = Image.open(response.raw).convert('RGB')
                        else:
                            image = Image.open(imgURL).convert('RGB')
                            
                        p_img = perturb(image, p, pipeline)
                        if p == "pgd":
                            torch.cuda.empty_cache()
                            gc.collect()
                        images_to_encode.append(p_img)
                        urls_to_cache.append(imgURL)
                    except Exception as e:
                        print(f"Error loading image {imgURL}: {e}")
                        # Append zero embedding or handle error? 
                        # For now, let's skip or append a dummy if we want to keep alignment
                        # But skipping might break index alignment with truth.
                        # Let's try to be robust: create a black image
                        image = Image.new('RGB', (224, 224))
                        images_to_encode.append(image)
                        urls_to_cache.append(imgURL)

            if images_to_encode:
                batch_embeds = pipeline.encode_images(images_to_encode)
                
                for k, imgURL in enumerate(urls_to_cache):
                    if k < len(batch_embeds):
                         img_embed_dict[imgURL] = batch_embeds[k]
                         img_embeddings.append(batch_embeds[k])
                
                # Cleanup
                for img in images_to_encode:
                    if hasattr(img, 'close'):
                        img.close()
        
        if not img_embeddings:
            continue
            
        corpus_embeds = np.vstack(img_embeddings)
        
        truth = [1] + [0]*999
        # Ensure truth length matches corpus size if corpus is smaller (e.g. download errors)
        if len(truth) > len(corpus_embeds):
            truth = truth[:len(corpus_embeds)]
        elif len(truth) < len(corpus_embeds):
             truth += [0] * (len(corpus_embeds) - len(truth))

        results = pipeline.retrieve_t2i(
            query_text =[row['qry_text']],
            corpus_img_embeds= corpus_embeds,
        )

        # Calculate nDCG
        num_results = len(results['indices'][0])
        limit = min(10, num_results)
        
        t = [truth[i] for i in results['indices'][0][:limit]]
        if len(t) < 10:
            t += [0] * (10 - len(t))
            
        score = ndcg_score([t],[results['scores'][0][:limit].tolist() + [0]*(10-limit)], k=10)
        
        if score > 0.0: 
            retrieved_count +=1
        if score == 1.0:
            perfect_count +=1

        scores.append(score)
    return scores, perfect_count, retrieved_count


def load_image(img_path, base_img_url):
    """Load a single image from path or URL."""
    if img_path.startswith('http'):
        imgURL = img_path
    else:
        imgURL = os.path.join(base_img_url, img_path)
    
    try:
        if imgURL.startswith('http'):
            response = requests.get(imgURL, stream=True, timeout=10)
            image = Image.open(response.raw).convert('RGB')
        else:
            image = Image.open(imgURL).convert('RGB')
        return image, imgURL
    except Exception as e:
        print(f"Error loading image {imgURL}: {e}")
        return None, imgURL

def t2i_optimized(test, pipeline, base_img_url, p, img_batch_size=32):
    """Optimized T2I evaluation with better batching and caching."""
    retrieved_count = 0
    perfect_count = 0
    scores = []
    img_embed_cache = {}
    
    for row in tqdm(test):
        tgt_img_paths = row['tgt_img_path']
        
        # Collect images for this query
        images_to_encode = []
        img_urls = []
        cached_embeds = []
        
        for img_path in tgt_img_paths:
            if img_path.startswith('http'):
                imgURL = img_path
            else:
                imgURL = os.path.join(base_img_url, img_path)
            
            if imgURL in img_embed_cache:
                cached_embeds.append(img_embed_cache[imgURL])
            else:
                image, url = load_image(img_path, base_img_url)
                if image is not None:
                    p_img = perturb(image, p, pipeline)
                    if p == "pgd":
                        torch.cuda.empty_cache()
                        gc.collect()
                    images_to_encode.append(p_img)
                    img_urls.append(imgURL)
                    if hasattr(image, 'close') and image != p_img:
                        image.close()
                else:
                    # Use dummy image
                    dummy = Image.new('RGB', (224, 224))
                    images_to_encode.append(dummy)
                    img_urls.append(imgURL)
        
        # Encode all new images in batches
        new_embeds = []
        if images_to_encode:
            # Process in batches
            for i in range(0, len(images_to_encode), img_batch_size):
                batch = images_to_encode[i:i+img_batch_size]
                batch_embeds = pipeline.encode_images(batch)
                new_embeds.extend(batch_embeds)
            
            # Cache new embeddings
            for url, emb in zip(img_urls, new_embeds):
                img_embed_cache[url] = emb
            
            # Clean up images
            for img in images_to_encode:
                if hasattr(img, 'close'):
                    img.close()
        
        # Combine cached and new embeddings
        all_embeds = cached_embeds + new_embeds
        
        if not all_embeds:
            continue
        
        corpus_embeds = np.vstack(all_embeds)
        
        truth = [1] + [0]*999
        if len(truth) > len(corpus_embeds):
            truth = truth[:len(corpus_embeds)]
        elif len(truth) < len(corpus_embeds):
            truth += [0] * (len(corpus_embeds) - len(truth))
        
        # Encode query text
        query_emb = pipeline.encode_texts([row['qry_text']])
        
        # Compute similarity
        similarities = pipeline.compute_similarity(query_emb, corpus_embeds)[0]
        
        # Get top k
        num_results = len(similarities)
        limit = min(10, num_results)
        top_indices = np.argsort(-similarities)[:limit]
        top_scores = similarities[top_indices]
        
        t = [truth[idx] for idx in top_indices]
        if len(t) < 10:
            t += [0] * (10 - len(t))
        
        score = ndcg_score([t], [top_scores.tolist() + [0]*(10-limit)], k=10)
        
        if score > 0.0: 
            retrieved_count += 1
        if score == 1.0:
            perfect_count += 1
        
        scores.append(score)
        
        # Periodic cleanup
        if len(scores) % 50 == 0:
            torch.cuda.empty_cache()
    
    return scores, perfect_count, retrieved_count

def deliverthegoods(datasets, perturbations, model_name):
    pipeline = UniMEPipeline(model_name)
    ndcg_scores = {}
    base_img_url = '/work/aho13/MMEB-eval/'
    k=10
    vqads = set(["A-OKVQA","ChartQA", "DocVQA","InfographicsVQA", "Visual7W", "ScienceQA", "VizWiz"])
    i2tds = set(["MSCOCO_i2t","VisualNews_i2t"])
    t2ids = set(["MSCOCO_t2i","VisualNews_t2i", "VisDial", "Wiki-SS-NQ"])

    # Use autocast for mixed precision
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for ds_name in datasets:
            ds = load_dataset("TIGER-Lab/MMEB-eval", ds_name)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f"{os.getcwd()}/logs/{ds_name}_{timestamp}.log"
    
            logger = setup_ndcg_logger(log_file)

            for p in perturbations:
                scores=[]
                perfect_count = 0
                retrieved_count = 0
                
                logger.info(f"Starting {ds_name} with perturbation {p}")
                
                if ds_name in i2tds or ds_name in vqads:
                    scores, perfect_count, retrieved_count = i2t(list(ds['test']), pipeline, base_img_url, p)
                
                elif ds_name in t2ids:
                    scores, perfect_count, retrieved_count = t2i(list(ds['test']), pipeline, base_img_url, p)
                
                # if ds_name in i2tds:
                #     # Use batched version
                #     scores, perfect_count, retrieved_count = i2t_batch(
                #         list(ds['test']), pipeline, base_img_url, p, batch_size=16
                #     )
                
                # elif ds_name in t2ids:
                #     # Use optimized version with better caching
                #     scores, perfect_count, retrieved_count = t2i_optimized(
                #         list(ds['test']), pipeline, base_img_url, p, img_batch_size=32
                #     )
                
                if scores:
                    avg_ndcg = np.mean(scores)
                    if logger:
                        logger.info("="*60)
                        logger.info(f"NDCG@{k} {ds_name} {p} Results:")
                        logger.info(f"  Mean NDCG: {avg_ndcg:.4f}")
                        logger.info(f"  Median NDCG: {np.median(scores):.4f}")
                        logger.info(f"  Std Dev: {np.std(scores):.4f}")
                        logger.info(f"  Perfect retrievals (rank 1): {perfect_count}/{len(scores)} ({100*perfect_count/len(scores):.1f}%)")
                        logger.info(f"  Relevant in top-{k}: {retrieved_count}/{len(scores)} ({100*retrieved_count/len(scores):.1f}%)")
                        logger.info("="*60)
                else:
                    logger.info(f"No scores computed for {ds_name} {p}")

    pipeline.cleanup()
    return ndcg_scores


def main():
    # Example usage
    # datasets=["MSCOCO_i2t", "MSCOCO_t2i"] # Start with one dataset
    # perturbations = ['pgd'] # Start with no perturbation
    perturbations = ['ctrl']
    #VQA 
    datasets = ["ChartQA", "DocVQA","InfographicsVQA", "Visual7W", "ScienceQA", "VizWiz"]
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    deliverthegoods(datasets, perturbations, "DeepGlint-AI/UniME-Phi3.5-V-4.2B")

if __name__ == "__main__":
    main()
