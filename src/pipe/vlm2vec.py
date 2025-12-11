import torch
from PIL import Image, ImageFilter, ImageEnhance
import requests
from io import BytesIO
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset
from sklearn.metrics import ndcg_score
import logging
import os
import sys
from datetime import datetime

# Add VLM2Vec to path
sys.path.insert(0, '/hpc/group/csdept/aho13/labIR/VLM2Vec')

try:
    from src.model.model import MMEBModel
    from src.arguments import ModelArguments
    from src.model.processor import load_processor
except ImportError:
    # Fallback or error if path is wrong
    print("Error: Could not import VLM2Vec modules. Checking path...")
    sys.path.append('/hpc/group/csdept/aho13/labIR/VLM2Vec')
    from src.model.model import MMEBModel
    from src.arguments import ModelArguments
    from src.model.processor import load_processor

class VLM2VecPipeline:
    def __init__(self, model_id="TIGER-Lab/VLM2Vec-Full"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        
        print(f"Loading {model_id} on {self.device}...")
        
        self.model_args = ModelArguments(
            model_name=model_id,
            pooling='last',
            normalize=True,
            model_backbone='phi3_v',
            num_crops=16
        )
        
        self.processor = load_processor(self.model_args)
        self.model = MMEBModel.load(self.model_args)
        self.model.eval()
        self.model = self.model.to(self.device, dtype=self.dtype)

    def get_embedding(self, text, image=None, is_query=True):
        """
        Generates normalized vector embeddings for text or text+image inputs.
        
        Args:
            text: str or List[str]
            image: PIL.Image or List[PIL.Image], optional
            is_query: bool, True if this is a query (uses qry_reps), False if target (uses tgt_reps)
        """
        # Ensure inputs are lists
        if isinstance(text, str):
            text = [text]
        if image is not None and not isinstance(image, list):
            image = [image]
            
        # Processor expects: text, images (optional)
        # Based on user snippet: processor(text, images)
        # And legacy code: processor(text, images)
        
        # Note: The processor might handle batching differently. 
        # If image is provided, it must match text length or be a single image broadcasted?
        # User snippet: inputs = processor(string, [image])
        
        if image:
            inputs = self.processor(text, image)
        else:
            inputs = self.processor(text)
            
        # Move inputs to device
        # inputs is a dict of tensors or lists of tensors
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        # Handle nested lists/tensors if necessary (legacy code had complex handling)
        # But user snippet just does {key: value.to('cuda') ...}
        
        with torch.no_grad():
            if is_query:
                outputs = self.model(qry=inputs)["qry_reps"]
            else:
                outputs = self.model(tgt=inputs)["tgt_reps"]
                
        return outputs.float().cpu().numpy()

    def compute_similarity(self, query_vecs, target_vecs):
        # query_vecs: (N, D)
        # target_vecs: (M, D)
        # Result: (N, M)
        return np.dot(query_vecs, target_vecs.T)

# --- Helper Functions ---

def setup_ndcg_logger(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def perturb(img, ptype, pipeline=None):
    if ptype == "none":
        return img
    if ptype == "gauss1":
        return img.filter(ImageFilter.GaussianBlur(radius=1))
    if ptype == "gauss2":
        return img.filter(ImageFilter.GaussianBlur(radius=2))
    elif ptype== "grayscale":
        return img.convert('L').convert('RGB')
    elif ptype=="bright":
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(1.5)
    elif ptype=="flip":
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    elif ptype=="compress":
        new_width, new_height = img.size[0] // 5 * 4, img.size[1] //5 *4
        return img.resize((new_width, new_height))
    else:
        return img

# --- Evaluation Loops ---

def i2t_loop(test_data, pipeline, base_img_url, ptype):
    retrieved_count = 0
    perfect_count = 0
    scores = []
    
    for row in tqdm(test_data, desc=f"I2T {ptype}"):
        try:
            # Load and perturb image
            img_url = base_img_url + row['qry_img_path']
            image = Image.open(img_url).convert('RGB')
            p_img = perturb(image, ptype, pipeline)
            
            # Query: Image + Instruction
            # User snippet: '<|image_1|> Represent the given image with the following question: ...'
            # But legacy code used: '<|image_1|> {text}'
            # I'll stick to the format that seems appropriate for retrieval
            query_text = f"<|image_1|> {row['qry_text']}"
            
            # is_query=True
            q_emb = pipeline.get_embedding([query_text], [p_img], is_query=True)
            
            # Targets: Texts
            # is_query=False (Targets)
            t_embs = pipeline.get_embedding(row['tgt_text'], is_query=False)
            
            # Similarity
            sims = pipeline.compute_similarity(q_emb, t_embs)
            
            # Metrics
            truth = [1] + [0] * (len(row['tgt_text']) - 1)
            if len(truth) < sims.shape[1]:
                truth += [0] * (sims.shape[1] - len(truth))
            elif len(truth) > sims.shape[1]:
                truth = truth[:sims.shape[1]]
                
            score = ndcg_score([truth], sims, k=10)
            
            if score > 0.0: retrieved_count += 1
            if score == 1.0: perfect_count += 1
            scores.append(score)
            
        except Exception as e:
            print(f"Error processing row: {e}")
            continue
            
    return scores, perfect_count, retrieved_count

def t2i_loop(test_data, pipeline, base_img_url, ptype):
    retrieved_count = 0
    perfect_count = 0
    scores = []
    
    img_embed_cache = {}
    
    for row in tqdm(test_data, desc=f"T2I {ptype}"):
        try:
            # Query: Text
            # User snippet: 'Find me an everyday image that matches the given caption: ...'
            query_text = f"Find me an everyday image that matches the given caption: {row['qry_text']}"
            # is_query=True
            q_emb = pipeline.get_embedding([query_text], is_query=True)
            
            # Targets: Images
            target_embs_list = []
            target_paths = row['tgt_img_path']
            
            # We need to process images. 
            # Target format: '<|image_1|> Represent the given image.'
            # is_query=False
            
            # Collect images to process
            images_to_process = []
            indices_to_fill = []
            
            for idx, path in enumerate(target_paths):
                full_url = base_img_url + path
                if full_url in img_embed_cache:
                    target_embs_list.append(img_embed_cache[full_url])
                else:
                    try:
                        img = Image.open(full_url).convert('RGB')
                        p_img = perturb(img, ptype, pipeline)
                        images_to_process.append(p_img)
                        indices_to_fill.append((idx, full_url))
                        target_embs_list.append(None) # Placeholder
                    except:
                        target_embs_list.append(np.zeros((1, 4096))) # Placeholder for failed load
            
            # Batch process new images
            if images_to_process:
                batch_size = 16
                for i in range(0, len(images_to_process), batch_size):
                    b_imgs = images_to_process[i:i+batch_size]
                    b_texts = ["<|image_1|> Represent the given image."] * len(b_imgs)
                    
                    # is_query=False for targets
                    b_embs = pipeline.get_embedding(b_texts, b_imgs, is_query=False)
                    
                    for j, emb in enumerate(b_embs):
                        global_idx, url = indices_to_fill[i+j]
                        target_embs_list[global_idx] = emb.reshape(1, -1)
                        img_embed_cache[url] = emb.reshape(1, -1)
            
            # Stack
            target_embs_list = [e for e in target_embs_list if e is not None]
            if not target_embs_list:
                continue
                
            target_embs_matrix = np.vstack(target_embs_list)
            
            # Similarity
            sims = pipeline.compute_similarity(q_emb, target_embs_matrix)
            
            # Metrics
            truth = [1] + [0] * (len(target_paths) - 1)
            if len(truth) < sims.shape[1]:
                truth += [0] * (sims.shape[1] - len(truth))
            elif len(truth) > sims.shape[1]:
                truth = truth[:sims.shape[1]]
                
            score = ndcg_score([truth], sims, k=10)
            
            if score > 0.0: retrieved_count += 1
            if score == 1.0: perfect_count += 1
            scores.append(score)
            
        except Exception as e:
            print(f"Error processing row: {e}")
            continue
            
    return scores, perfect_count, retrieved_count

def deliverthegoods(datasets, perturbations, model_name):
    pipeline = VLM2VecPipeline(model_name)
    base_img_url = '/work/aho13/MMEB-eval/'
    k=10

    i2tds = set(["MSCOCO_i2t","VisualNews_i2t"])
    t2ids = set(["MSCOCO_t2i","VisualNews_t2i", "VisDial", "Wiki-SS-NQ"])

    for ds_name in datasets:
        print(f"Loading dataset {ds_name}...")
        try:
            ds = load_dataset("TIGER-Lab/MMEB-eval", ds_name)
        except Exception as e:
            print(f"Failed to load dataset {ds_name}: {e}")
            continue
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"{os.getcwd()}/logs/{ds_name}_{timestamp}.log"
        logger = setup_ndcg_logger(log_file)

        for p in perturbations:
            print(f"Running {ds_name} with perturbation {p}...")
            scores = []
            perfect_count = 0
            retrieved_count = 0
            
            test_data = list(ds['test'])
            
            if ds_name in i2tds:
                scores, perfect_count, retrieved_count = i2t_loop(test_data, pipeline, base_img_url, p)
            elif ds_name in t2ids:
                scores, perfect_count, retrieved_count = t2i_loop(test_data, pipeline, base_img_url, p)
            
            if scores:
                avg_ndcg = np.mean(scores)
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

if __name__ == "__main__":
    datasets = ['VisualNews_i2t'] 
    perturbations = ['none']
    deliverthegoods(datasets, perturbations, 'TIGER-Lab/VLM2Vec-Full')
