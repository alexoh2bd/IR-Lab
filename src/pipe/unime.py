"""
UniME Multimodal Retrieval Pipeline
Cleaned and fixed version (adapted from your original).
"""

import os
from typing import List, Dict
from PIL import Image
from io import BytesIO
import requests
import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm
from datasets import load_dataset
from sklearn.metrics import ndcg_score
import logging
from transformers import AutoProcessor, AutoModelForCausalLM

# Fallback perturb (keeps original behavior you had)
# try:
from experiment import perturb
# except Exception:
#     def perturb(image, perturbation_type, pipeline):
#         return image


class Config:
    MODEL_NAME = "DeepGlint-AI/UniME-Phi3.5-V-4.2B"
    BATCH_SIZE = 32
    IMAGE_BATCH_SIZE = 8
    TEXT_BATCH_SIZE = 64
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
    NUM_WORKERS = 0
    PIN_MEMORY = True

class UniMERetrievalPipeline:
    """Correct UniME retrieval pipeline using hidden_states[-1][:, -1, :]."""

    def __init__(self, model_name="DeepGlint-AI/UniME-Phi3.5-V-4.2B"):
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda",
        )


        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # match official setup
        self.processor.tokenizer.padding_side = "left"
        self.processor.tokenizer.padding = True

        # official prompts
        self.img_prompt = "<|user|>\n<|image_1|>\nSummary above image in one word: <|end|>\n<|assistant|>\n"
        self.text_prompt = "<|user|>\n<sent>\nSummary above sentence in one word: <|end|>\n<|assistant|>\n"

        self.model.eval()

    @torch.no_grad()
    def encode_texts(self, texts):
        embeds = []
        for t in texts:
            prompt = self.text_prompt.replace("<sent>", t)
            inputs = self.processor(text=prompt, images=None, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            out = self.model(**inputs, output_hidden_states=True, return_dict=True)
            vec = out.hidden_states[-1][:, -1, :]
            vec = F.normalize(vec, dim=-1)
            embeds.append(vec.cpu())

        return torch.cat(embeds, dim=0).numpy()

    @torch.no_grad()
    def encode_image_text_queries(self, images, texts):
        embeds = []
        for img, txt in zip(images, texts):
            prompt = self.img_prompt
            inputs = self.processor(text=prompt, images=[img], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            out = self.model(**inputs, output_hidden_states=True, return_dict=True)
            vec = out.hidden_states[-1][:, -1, :]
            vec = F.normalize(vec, dim=-1)
            embeds.append(vec.cpu())

        return torch.cat(embeds, dim=0).numpy()

    def compute_similarity(self, q, c):
        q = torch.tensor(q)
        c = torch.tensor(c)
        return (q @ c.T).numpy()

    def retrieve_i2t(self, query_images: List[Image.Image], query_texts: List[str], corpus_texts: List[str], top_k: int = 10):
        """
        Image+Text -> Text retrieval wrapper returning top indices & scores.
        """
        q_emb = self.encode_image_text_queries(query_images, query_texts)
        c_emb = self.encode_texts(corpus_texts)
        sims = self.compute_similarity(q_emb, c_emb)

        top_k = min(top_k, sims.shape[1])
        top_idx = np.argsort(-sims, axis=1)[:, :top_k]
        top_scores = np.take_along_axis(sims, top_idx, axis=1)
        return {"indices": top_idx, "scores": top_scores}

    def cleanup(self):
        """Free model & GPU memory when done."""
        try:
            del self.model
        except Exception:
            pass
        torch.cuda.empty_cache()


# ---------- Helpers & evaluation loop (fixed) ----------

def i2t_loop(test, pipeline, base_img_url, p):
    retrieved_count = 0
    perfect_count = 0
    scores = []

    for row in tqdm(test):
        # load image either from URL or local path
        img_path = os.path.join(base_img_url, row["qry_img_path"]) if not row["qry_img_path"].startswith("http") else row["qry_img_path"]
        # robustly open image
        if img_path.startswith("http"):
            resp = requests.get(img_path, stream=True)
            image = Image.open(BytesIO(resp.content)).convert("RGB")
        else:
            image = Image.open(img_path).convert("RGB")

        p_img = perturb(image, p, pipeline)

        # ensure row['qry_text'] is a list of length 1 (pipeline expects single-element per image)
        q_text = row["qry_text"]
        if isinstance(q_text, list):
            q_texts = [q_text[0]]
        else:
            q_texts = [q_text]

        results = pipeline.retrieve_i2t(query_images=[p_img], query_texts=q_texts, corpus_texts=row["tgt_text"], top_k=10)

        # cleanup images
        for im in (p_img, image):
            try:
                im.close()
            except Exception:
                pass

        # ground truth: assume first element is relevant (same as your original)
        truth = [1] + [0] * (len(row["tgt_text"]) - 1)
        # convert indices from results for top-10 and compute ndcg
        top_indices = results["indices"][0][:10]
        top_scores = results["scores"][0][:10]
        t = [truth[idx] for idx in top_indices]
        score = ndcg_score([t], [top_scores.tolist()], k=10)
        if score > 0.0:
            retrieved_count += 1
        if score == 1.0:
            perfect_count += 1
        scores.append(score)
    return scores, perfect_count, retrieved_count


def setup_ndcg_logger(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def deliverthegoods(datasets, perturbations, model_name):
    pipeline = UniMERetrievalPipeline(model_name)
    base_img_url = "/work/aho13/MMEB-eval/"
    k = 10

    i2tds = set(["MSCOCO_i2t", "VisualNews_i2t"])
    t2ids = set(["MSCOCO_t2i", "VisualNews_t2i", "VisDial", "Wiki-SS-NQ"])

    ndcg_scores = {}
    with torch.autocast(device_type="cuda", dtype=torch.float16) if torch.cuda.is_available() else torch.cpu.amp.autocast():
        for ds_name in datasets:
            ds = load_dataset("TIGER-Lab/MMEB-eval", ds_name)
            timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"{os.getcwd()}/logs/{ds_name}_{timestamp}.log"
            logger = setup_ndcg_logger(log_file)

            for p in perturbations:
                scores = []
                perfect_count = 0
                retrieved_count = 0
                if ds_name in i2tds:
                    scores, perfect_count, retrieved_count = i2t_loop(list(ds["test"]), pipeline, base_img_url, p)
                elif ds_name in t2ids:
                    # t2i_loop not included in original snippet; skip or implement analogously.
                    # For now we skip t2i datasets to avoid NameError.
                    logger.info(f"Skipping {ds_name} (t2i not implemented in this script).")
                    continue

                avg_ndcg = float(np.mean(scores)) if len(scores) else 0.0
                if logger:
                    logger.info("=" * 60)
                    logger.info(f"NDCG@{k} {ds_name} {p} Results:")
                    logger.info(f"  Mean NDCG: {avg_ndcg:.4f}")
                    logger.info(f"  Median NDCG: {np.median(scores) if len(scores) else 0.0:.4f}")
                    logger.info(f"  Std Dev: {np.std(scores) if len(scores) else 0.0:.4f}")
                    logger.info(f"  Perfect retrievals (rank 1): {perfect_count}/{len(scores)} ({100*perfect_count/len(scores) if len(scores) else 0.0:.1f}%)")
                    logger.info(f"  Relevant in top-{k}: {retrieved_count}/{len(scores)} ({100*retrieved_count/len(scores) if len(scores) else 0.0:.1f}%)")
                    logger.info("=" * 60)
    pipeline.cleanup()
    return ndcg_scores


if __name__ == "__main__":
    datasets = ["MSCOCO_i2t", "VisualNews_i2t"]
    perturbations = ["none", "gauss2"]
    deliverthegoods(datasets, perturbations, Config.MODEL_NAME)
