
import torch
import torch.nn.functional as F
import torchvision.transforms as TF
from PIL import Image
import numpy as np
from tqdm import tqdm

def pgd_attack_to_target_clip(image, pipeline, target_text, epsilon=0.0314, alpha=0.0078, steps=10):
    """
    Targeted PGD attack for CLIP model.
    Optimizes the image to MAXIMIZE similarity with the target text.
    
    Args:
        image: PIL Image.
        pipeline: MultimodalRetrievalPipeline instance.
        target_text: Target text to maximize similarity with.
        epsilon: Perturbation magnitude.
        alpha: Step size.
        steps: Number of steps.
        
    Returns:
        Perturbed PIL Image.
    """
    device = pipeline.model.device
    
    # 1. Prepare Image Tensor [0, 1]
    # Resize to model input size (224x224 for CLIP usually, but let's check processor)
    try:
        height = pipeline.processor.image_processor.crop_size['height']
        width = pipeline.processor.image_processor.crop_size['width']
    except:
        height, width = 224, 224
        
    img_resized = image.resize((width, height), Image.BILINEAR)
    img_np = np.array(img_resized.convert("RGB")).astype(np.float32) / 255.0
    x0 = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device) # (1, 3, H, W)
    
    # 2. Prepare Normalization
    try:
        mean = torch.tensor(pipeline.processor.image_processor.image_mean).view(1, 3, 1, 1).to(device)
        std = torch.tensor(pipeline.processor.image_processor.image_std).view(1, 3, 1, 1).to(device)
    except:
        # Fallback to CLIP mean/std
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
        
    def normalize_fn(x):
        return (x - mean) / std
    
    # 3. Prepare Target Text Embedding
    with torch.no_grad():
        text_inputs = pipeline.processor(
            text=[target_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        text_embeds = pipeline.model.get_text_features(**text_inputs)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
    # 4. Optimization Loop
    x_adv = x0.clone().detach()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0, 1)
    
    print(f"Preparing Targeted PGD attack (steps={steps})...")
    
    for step in range(steps):
        x_adv.requires_grad = True
        
        # Normalize
        pixel_values = normalize_fn(x_adv)
        
        # Forward
        img_embeds = pipeline.model.get_image_features(pixel_values=pixel_values)
        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
        
        # Loss: Maximize similarity to text_embeds
        sim = img_embeds @ text_embeds.T
        loss = sim.mean() # We want to MAXIMIZE this.
        
        pipeline.model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            grad = x_adv.grad
            # Gradient Ascent: x = x + alpha * grad
            x_adv = x_adv + alpha * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x0 + epsilon), x0 - epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)
            
        if (step + 1) % 5 == 0:
            print(f"Targeted PGD Step {step+1}/{steps}, Similarity: {loss.item():.4f}")
            
    # Convert back to PIL
    perturbed_np = x_adv.squeeze(0).detach().cpu().numpy() # (3, H, W)
    perturbed_np = (perturbed_np.transpose(1, 2, 0) * 255).astype(np.uint8)
    perturbed_image = Image.fromarray(perturbed_np)
    
    # Do NOT resize back to original size. 
    # Resizing destroys the high-frequency adversarial perturbations.
    # We return the image at the model's input resolution (e.g., 224x224).
    # if perturbed_image.size != image.size:
    #     perturbed_image = perturbed_image.resize(image.size, Image.LANCZOS)
        
    return perturbed_image

def pgd_attack(model, image_tensor_01, text_tokens, normalize_fn, epsilon=0.0314, alpha=0.0078, steps=20, device='cuda'):
    """
    PGD attack - exact copy from attack_clip.py
    """
    x0 = image_tensor_01.clone().detach()
    
    # Random initialization
    x_adv = x0 + torch.empty_like(x0).uniform_(-epsilon, epsilon)
    x_adv = x_adv.clamp(0, 1)
    
    for _ in range(steps):
        x_adv.requires_grad_(True)
        
        # Normalize for CLIP
        x_input = normalize_fn(x_adv)
        
        # Get embeddings
        img_emb = model.encode_image(x_input)
        img_emb = F.normalize(img_emb, dim=-1)
        
        text_emb = model.encode_text(text_tokens)
        text_emb = F.normalize(text_emb, dim=-1)
        
        # Similarity
        sim = img_emb @ text_emb.T  # (B, L)
        
        # Loss: NEGATIVE similarity (we want to minimize it)
        loss = -sim.mean()
        
        # Backward
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            grad = x_adv.grad
            # PGD step
            x_adv = x_adv + alpha * grad.sign()
            # Project to epsilon ball
            x_adv = torch.max(torch.min(x_adv, x0 + epsilon), x0 - epsilon)
            x_adv = x_adv.clamp(0, 1)
        
        x_adv.requires_grad_(True)
    
    return x_adv.detach()

# This attack is based on the FGSM attack on CLIP to reduce image-text similarity.
def fgsm_attack_clip(image, pipeline, epsilon=0.03, target_text=None):
    """
    FGSM attack on CLIP to reduce image-text similarity.
    
    Args:
        image: PIL Image
        pipeline: MultimodalRetrievalPipeline with CLIP model
        epsilon: Perturbation magnitude (default 0.03)
        target_text: Optional specific text to attack against. If None, uses generic text.
    
    Returns:
        Perturbed PIL Image
    """
    # Use a generic text if none provided
    # This genereci text is aimed to reduce image-text similarity
    if target_text is None:
        target_text = "A photo of an object"
    
    # Convert image to tensor with CLIP preprocessing
    img_rgb = image.convert("RGB")
    transform = TF.Compose([
        TF.Resize((224, 224)),
        TF.ToTensor(),
        TF.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                     std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    img_tensor = transform(img_rgb).unsqueeze(0).to(pipeline.model.device)
    img_tensor.requires_grad = True
    
    # Encode text (target to reduce similarity with)
    text_inputs = pipeline.processor(text=[target_text], return_tensors="pt", padding=True)
    text_inputs = {k: v.to(pipeline.model.device) for k, v in text_inputs.items()}
    
    with torch.enable_grad():
        # Get embeddings
        text_embeds = pipeline.model.get_text_features(**text_inputs)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        image_embeds = pipeline.model.get_image_features(pixel_values=img_tensor)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        
        # Loss: maximize distance (minimize similarity)
        # We want to REDUCE relevance, so we minimize the negative similarity
        similarity = F.cosine_similarity(image_embeds, text_embeds)
        loss = similarity.mean()  # We'll add gradient to reduce this
        
        # Backprop
        pipeline.model.zero_grad()
        loss.backward()
        
        # FGSM: perturb in direction that INCREASES loss (reduces similarity)
        # Since we want to reduce similarity, we add epsilon * sign(gradient)
        sign_grad = img_tensor.grad.sign()
        perturbed_tensor = img_tensor + epsilon * sign_grad
        
        # Denormalize back to [0, 1] range
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(pipeline.model.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(pipeline.model.device)
        
        perturbed_tensor = perturbed_tensor * std + mean
        perturbed_tensor = torch.clamp(perturbed_tensor, 0, 1)
    
    # Convert back to PIL Image
    perturbed_np = perturbed_tensor.squeeze(0).detach().cpu().numpy()
    perturbed_np = (perturbed_np.transpose(1, 2, 0) * 255).astype(np.uint8)
    perturbed_image = Image.fromarray(perturbed_np)
    
    # Do NOT resize back to original size.
    # if perturbed_image.size != image.size:
    #     perturbed_image = perturbed_image.resize(image.size, Image.LANCZOS)
    
    return perturbed_image