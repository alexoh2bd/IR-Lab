import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import Optional

def pgd_attack_unime(image: Image.Image, 
                     pipeline, 
                     target_text: str = "A photo of an object", 
                     epsilon: float = 0.0314, 
                     alpha: float = 0.0078, 
                     steps: int = 5):
    """
    PGD attack for UNiME model.
    Optimizes the image to minimize similarity with the target text (or maximize distance).
    
    Args:
        image: Original PIL Image.
        pipeline: UniMEPipeline instance.
        target_text: Target text to minimize similarity with (or generic text).
        epsilon: Perturbation magnitude.
        alpha: Step size.
        steps: Number of steps.
        
    Returns:
        Perturbed PIL Image.
    """
    device = pipeline.model.device
    
    # 1. Prepare Image Tensor [0, 1]
    # Resize to model input size (likely 336x336 or similar, let processor handle it? 
    # No, we need fixed size for tensor optimization. 
    # UNiME/Phi-3.5-V usually handles dynamic resolutions, but for attack we should probably fix it.
    # Let's see what the processor does. It likely resizes.
    # To be safe and simple, we can resize to a standard size like 336x336 or keep original if model handles it.
    # However, to optimize a tensor, it must have a fixed shape.
    # Let's use the processor to get the expected input size or just resize to 336x336.
    # For now, let's resize to 336x336 which is common for LLaVA/Phi-V.
    # Actually, let's check processor.image_processor.crop_size if available.
    
    try:
        height = pipeline.processor.image_processor.crop_size['height']
        width = pipeline.processor.image_processor.crop_size['width']
    except:
        height, width = 336, 336
        
    img_resized = image.resize((width, height), Image.BILINEAR)
    img_np = np.array(img_resized.convert("RGB")).astype(np.float32) / 255.0
    x0 = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device) # (1, 3, H, W)
    
    # 2. Prepare Normalization
    try:
        mean = torch.tensor(pipeline.processor.image_processor.image_mean).view(1, 3, 1, 1).to(device)
        std = torch.tensor(pipeline.processor.image_processor.image_std).view(1, 3, 1, 1).to(device)
    except:
        # Fallback to CLIP/ImageNet mean/std
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
        
    def normalize_fn(x):
        return (x - mean) / std
    
    # 3. Prepare Target Text Embedding
    # We want to minimize similarity to this text.
    # Or if it's "ground truth", we minimize. If it's "target for attack", we maximize?
    # The user said "reduce image-text similarity" in the reference code (loss = -sim).
    # Reference: "Loss: NEGATIVE similarity (we want to minimize it)" -> This means we want to MAXIMIZE similarity?
    # Wait, "Loss: NEGATIVE similarity" usually means `loss = - similarity`. Minimizing `-sim` is Maximizing `sim`.
    # BUT the comment says "PGD attack on CLIP to reduce image-text similarity (maximize similarity to generic text)".
    # In `attack_mmeb_train.py`:
    #   sim = img_emb @ text_emb.T
    #   loss = -sim.mean()
    #   loss.backward()
    #   x_adv = x_adv + alpha * grad.sign()
    # This MAXIMIZES similarity to `text_tokens`.
    # If `text_tokens` is the GROUND TRUTH, this makes the image MORE like the ground truth? That's not an attack.
    # Ah, let's re-read `attack_mmeb_train.py`.
    # text = item.get('pos_text', item.get('tgt_text'))
    # It uses the positive text.
    # And it maximizes similarity?
    # "if adv_sim < clean_sim * 0.5: Consider it successful"
    # If we maximize similarity, adv_sim should be HIGHER.
    # Wait, `x_adv = x_adv + alpha * grad.sign()`. This is Gradient ASCENT on the loss.
    # If loss = -sim, then we are increasing -sim, i.e., decreasing sim.
    # Correct. Gradient ascent on `-sim` makes `-sim` larger, so `sim` smaller.
    # So this MINIMIZES similarity to the target text.
    
    # We need the text prompt format from pipeline
    text_prompt = pipeline.text_prompt.replace("<sent>", target_text)
    
    # Encode text once (it's constant)
    # We can't use pipeline.encode_texts because we need the raw tensors for the model, 
    # but actually we just need the embedding to compute similarity against.
    # Wait, UNiME is a Causal LM. The image embedding is part of the forward pass.
    # We can't easily decouple "image embedding" and "text embedding" like CLIP unless we use the specific extraction logic.
    # In `unimepipe.py`:
    #   encode_texts: returns last hidden state.
    #   encode_images: returns last hidden state given image + img_prompt.
    # So we can compute the target text embedding once.
    
    with torch.no_grad():
        # Use pipeline's method but we need the tensor, not numpy
        # Actually pipeline.encode_texts returns numpy.
        # Let's manually do it to keep it in torch
        inputs = pipeline.processor(
            text=text_prompt,
            images=None,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = pipeline.model(**inputs, output_hidden_states=True, return_dict=True)
        text_emb = out.hidden_states[-1][:, -1, :]
        text_emb = F.normalize(text_emb, dim=-1)
        
    # 4. Optimization Loop
    x_adv = x0.clone().detach()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0, 1)
    
    # Pre-compute constant inputs for the loop
    print(f"Preparing PGD attack (steps={steps})...")
    
    # Get input_ids for the image prompt
    temp_inputs = pipeline.processor(
        text=pipeline.img_prompt,
        images=None, 
        return_tensors="pt",
        padding=True
    )
    img_input_ids = temp_inputs['input_ids'].to(device)
    img_attention_mask = temp_inputs['attention_mask'].to(device)
    
    # Create dummy image inputs to get structure
    dummy_img = Image.fromarray((np.random.rand(height, width, 3) * 255).astype(np.uint8))
    base_inputs = pipeline.processor(
        text=pipeline.img_prompt,
        images=[dummy_img],
        return_tensors="pt",
        padding=True
    )
    base_inputs = {k: v.to(device) for k, v in base_inputs.items()}
    
    # Check target shape from dummy input
    target_shape = base_inputs['pixel_values'].shape
    
    for step in range(steps):
        x_adv.requires_grad = True
        
        # Normalize
        pixel_values = normalize_fn(x_adv)
        
        # Reshape pixel_values to match model expectation
        if pixel_values.shape == target_shape:
            current_pixel_values = pixel_values
        elif len(target_shape) == 5 and target_shape[1] == 5:
            # Global + 2x2 Grid strategy
            x_global = pixel_values
            x_up = F.interpolate(pixel_values, size=(672, 672), mode='bilinear', align_corners=False)
            x_tl = x_up[:, :, 0:336, 0:336]
            x_tr = x_up[:, :, 0:336, 336:672]
            x_bl = x_up[:, :, 336:672, 0:336]
            x_br = x_up[:, :, 336:672, 336:672]
            current_pixel_values = torch.stack([x_global, x_tl, x_tr, x_bl, x_br], dim=1)
        elif len(target_shape) == 5 and target_shape[1] == 1:
             current_pixel_values = pixel_values.unsqueeze(1)
        else:
            repeats = [1] * len(target_shape)
            repeats[1] = target_shape[1]
            current_pixel_values = pixel_values.unsqueeze(1).repeat(repeats)
                 
        # Forward
        forward_kwargs = {k: v for k, v in base_inputs.items() if k != 'pixel_values'}
        forward_kwargs['pixel_values'] = current_pixel_values
        
        out = pipeline.model(**forward_kwargs, output_hidden_states=True, return_dict=True)
        img_emb = out.hidden_states[-1][:, -1, :]
        img_emb = F.normalize(img_emb, dim=-1)
        
        # Loss: Minimize similarity to text_emb (maximize distance)
        sim = img_emb @ text_emb.T
        loss = -sim.mean()
        
        pipeline.model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            grad = x_adv.grad
            x_adv = x_adv + alpha * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x0 + epsilon), x0 - epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)
            
        if (step + 1) % 5 == 0:
            print(f"PGD Step {step+1}/{steps}, Loss: {loss.item():.4f}")
            
    # Convert back to PIL
    perturbed_np = x_adv.squeeze(0).detach().cpu().numpy() # (3, H, W)
    perturbed_np = (perturbed_np.transpose(1, 2, 0) * 255).astype(np.uint8)
    perturbed_image = Image.fromarray(perturbed_np)
    
    # Resize back to original if needed?
    # The original function `pgd_attack_clip` did this.
    if perturbed_image.size != image.size:
        perturbed_image = perturbed_image.resize(image.size, Image.LANCZOS)
        
    return perturbed_image

def pgd_attack_to_target(image: Image.Image, 
                     pipeline, 
                     target_text: str, 
                     epsilon: float = 0.0314, 
                     alpha: float = 0.0078, 
                     steps: int = 5):
    """
    Targeted PGD attack for UNiME model.
    Optimizes the image to MAXIMIZE similarity with the target text.
    
    Args:
        image: Original PIL Image.
        pipeline: UniMEPipeline instance.
        target_text: Target text to maximize similarity with.
        epsilon: Perturbation magnitude.
        alpha: Step size.
        steps: Number of steps.
        
    Returns:
        Perturbed PIL Image.
    """
    device = pipeline.model.device
    
    # 1. Prepare Image Tensor [0, 1]
    try:
        height = pipeline.processor.image_processor.crop_size['height']
        width = pipeline.processor.image_processor.crop_size['width']
    except:
        height, width = 336, 336
        
    img_resized = image.resize((width, height), Image.BILINEAR)
    img_np = np.array(img_resized.convert("RGB")).astype(np.float32) / 255.0
    x0 = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device) # (1, 3, H, W)
    
    # 2. Prepare Normalization
    try:
        mean = torch.tensor(pipeline.processor.image_processor.image_mean).view(1, 3, 1, 1).to(device)
        std = torch.tensor(pipeline.processor.image_processor.image_std).view(1, 3, 1, 1).to(device)
    except:
        # Fallback to CLIP/ImageNet mean/std
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
        
    def normalize_fn(x):
        return (x - mean) / std
    
    # 3. Prepare Target Text Embedding
    # We want to MAXIMIZE similarity to this text.
    
    # We need the text prompt format from pipeline
    if hasattr(pipeline, 'text_prompt') and pipeline.text_prompt:
        text_prompt = pipeline.text_prompt.replace("<sent>", target_text)
    else:
        text_prompt = target_text
    
    with torch.no_grad():
        inputs = pipeline.processor(
            text=text_prompt,
            images=None,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = pipeline.model(**inputs, output_hidden_states=True, return_dict=True)
        text_emb = out.hidden_states[-1][:, -1, :]
        text_emb = F.normalize(text_emb, dim=-1)
        
    # 4. Optimization Loop
    x_adv = x0.clone().detach()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0, 1)
    
    # Pre-compute constant inputs for the loop
    print(f"Preparing Targeted PGD attack (steps={steps})...")
    
    # Create dummy image inputs to get structure
    dummy_img = Image.fromarray((np.random.rand(height, width, 3) * 255).astype(np.uint8))
    base_inputs = pipeline.processor(
        text=pipeline.img_prompt,
        images=[dummy_img],
        return_tensors="pt",
        padding=True
    )
    base_inputs = {k: v.to(device) for k, v in base_inputs.items()}
    
    # Check target shape from dummy input
    target_shape = base_inputs['pixel_values'].shape
    
    for step in range(steps):
        x_adv.requires_grad = True
        
        # Normalize
        pixel_values = normalize_fn(x_adv)
        
        # Reshape pixel_values to match model expectation
        if pixel_values.shape == target_shape:
            current_pixel_values = pixel_values
        elif len(target_shape) == 5 and target_shape[1] == 5:
            # Global + 2x2 Grid strategy
            x_global = pixel_values
            x_up = F.interpolate(pixel_values, size=(672, 672), mode='bilinear', align_corners=False)
            x_tl = x_up[:, :, 0:336, 0:336]
            x_tr = x_up[:, :, 0:336, 336:672]
            x_bl = x_up[:, :, 336:672, 0:336]
            x_br = x_up[:, :, 336:672, 336:672]
            current_pixel_values = torch.stack([x_global, x_tl, x_tr, x_bl, x_br], dim=1)
        elif len(target_shape) == 5 and target_shape[1] == 1:
             current_pixel_values = pixel_values.unsqueeze(1)
        else:
            repeats = [1] * len(target_shape)
            repeats[1] = target_shape[1]
            current_pixel_values = pixel_values.unsqueeze(1).repeat(repeats)
                 
        # Forward
        forward_kwargs = {k: v for k, v in base_inputs.items() if k != 'pixel_values'}
        forward_kwargs['pixel_values'] = current_pixel_values
        
        out = pipeline.model(**forward_kwargs, output_hidden_states=True, return_dict=True)
        img_emb = out.hidden_states[-1][:, -1, :]
        img_emb = F.normalize(img_emb, dim=-1)
        
        # Loss: Maximize similarity to text_emb
        sim = img_emb @ text_emb.T
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
    
    if perturbed_image.size != image.size:
        perturbed_image = perturbed_image.resize(image.size, Image.LANCZOS)
        
    return perturbed_image
