from huggingface_hub import hf_hub_download
from datasets import load_dataset
import zipfile, os
from PIL import Image, ImageFilter, ImageEnhance
import requests
import torch
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
from transformers import CLIPProcessor, CLIPModel, AutoModelForCausalLM

# model = AutoModelForCausalLM.from_pretrained("TIGER-Lab/VLM2Vec-Full", trust_remote_code=True, torch_dtype="auto")
torch.cuda.is_available()
def setup_ndcg_logger(log_file: str = None):
    """
    Setup a simple logger for NDCG results.
    
    Args:
        log_file: Path to log file (default: logs/ndcg_TIMESTAMP.log)
    
    Returns:
        logger instance
    """
    if log_file is None:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"ndcg_{timestamp}.log"
    
    logger = logging.getLogger('NDCG_Eval')
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


# Perturb images function
def perturb(img, ptype):
    if ptype == "gauss1":
        return img.filter(ImageFilter.GaussianBlur(radius=1)), True
    if ptype == "gauss2":
        return img.filter(ImageFilter.GaussianBlur(radius=2)), True
    elif ptype== "grayscale":
        return img.convert('L'), True
    # increase brightness
    elif ptype=="bright":
        enhancer = ImageEnhance.Brightness(img)
        brightness_factor = 1.5
        brightened_image = enhancer.enhance(brightness_factor)
        return brightened_image, True
    elif ptype=="flip":
        flip_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return flip_img, True
    elif ptype=="compress":
        new_width, new_height = img.size[0] // 5 * 4, img.size[1] //5 *4
        return img.resize((new_width, new_height)), True

    # base
    else:
        return img, False

def deliverthegoods(datasets, perturbations):
    pipeline = MultimodalRetrievalPipeline()
    ndcg_scores = {}
    avg_ndcg_scores = {}
    base_img_url = '/work/aho13/work/aho13/'
    k=10

    i2t= set(["MSCOCO_i2t","VisualNews_i2t"])
    


    with torch.autocast(device_type='cuda', dtype=torch.float16):
        # Loops for each dataset
        for ds_name in datasets:
            #df = pd.read_csv(base_url+f"{ds_name}/test-00000-of-00001.parquet", encoding="utf-16")
            ds = load_dataset("TIGER-Lab/MMEB-eval", ds_name)

            log_file = f"{os.getcwd()}/logs/{ds_name}.log"
            logger = setup_ndcg_logger(log_file)
    
            # determine type of the retrieval 
            if ds_name in i2t:
                retrieval_type = 'i2t'
            else:
                retrieval_type = 't2i'
            for p in perturbations:
                scores=[]
                perfect_count = 0
                retrieved_count = 0
                for row in tqdm(list(ds['test'])):
                    # Load example image
                    # print(ds)
                    imgURL = base_img_url+ row['qry_img_path']
                    # print(imgURL)
                    image = Image.open(imgURL)
                    
                    p_img, ifpert = perturb(image, p)
                    assert (p == "none" and ifpert==False) or ifpert
                    
                    truth = [1] + [0]*999
                    results = pipeline.retrieve_i2t(
                      query_img = [p_img],
                      corpus= row['tgt_text'],
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
                #print(f"\nAverage nDCG for {ds_name} dataset, {p} : {avg_ndcg:.4f}")
    pipeline.cleanup()
    return ndcg_scores


def main():

    # Visual News_i2t dataset and pipeline
    datasets= ["VisualNews_i2t"]
    perturbations = ["none", "compress","flip", "gauss1","gauss2","bright","grayscale"]

    ndcg_scores = deliverthegoods(datasets, perturbations)









if __name__ == "__main__":
    main()



