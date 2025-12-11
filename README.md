# labIR: Multimodal Information Retrieval & Robustness

This project explores the robustness of Multimodal Information Retrieval (MIR) systems against various perturbations and adversarial attacks. It contains pipelines for evaluating CLIP-based models (`src/pipe`) and UNiME-based models (`src/pipe2`).

## Perturbations

The evaluation pipelines support a variety of image perturbations to test model robustness. These are defined in `experiment.py` within each pipeline folder.

| Perturbation | Description |
| :--- | :--- |
| **gauss1** | Gaussian Blur with radius 1. |
| **gauss2** | Gaussian Blur with radius 2. |
| **grayscale** | Converts the image to grayscale. |
| **bright** | Increases image brightness by a factor of 1.5. |
| **flip** | Horizontally flips the image. |
| **compress** | Resizes the image to 80% of its original dimensions (simulating compression/downsampling). |

## Adversarial Attacks

The project implements gradient-based adversarial attacks to evaluate the security of the models. These attacks are located in `src/pipe/attacks.py` (for CLIP) and `src/pipe2/attacks.py` (for UNiME).

### CLIP Attacks (`src/pipe`)

Attacks are implemented for CLIP models (e.g., `openai/clip-vit-base-patch32`).

*   **FGSM (Fast Gradient Sign Method)**:
    *   **Function**: `fgsm_attack_clip`
    *   **Goal**: Reduce image-text similarity.
    *   **Mechanism**: Computes the gradient of the cosine similarity between the image and a target text (default: "A photo of an object") and perturbs the image in the direction that *minimizes* this similarity (or maximizes distance).
    *   **Parameters**: `epsilon` (perturbation magnitude, default 0.03).

*   **PGD (Projected Gradient Descent)**:
    *   **Untargeted / Similarity Reduction**: `pgd_attack`
        *   **Goal**: Minimize similarity between the image and a given text.
        *   **Mechanism**: Iteratively perturbs the image to minimize the cosine similarity with the text.
    *   **Targeted**: `pgd_attack_to_target_clip`
        *   **Goal**: Maximize similarity to a specific target text (e.g., making an image of a dog look like "A photo of a cat" to the model).
        *   **Mechanism**: Iteratively perturbs the image to *maximize* the cosine similarity with the `target_text`.
        *   **Parameters**: `epsilon` (0.0314), `alpha` (step size, 0.0078), `steps` (20).

### UNiME Attacks (`src/pipe2`)

Attacks are implemented for UNiME models (e.g., `DeepGlint-AI/UniME-Phi3.5-V-4.2B`).

*   **PGD (Projected Gradient Descent)**:
    *   **Untargeted**: `pgd_attack_unime`
        *   **Goal**: Minimize similarity to a target text (or generic text).
        *   **Mechanism**: Optimizes the image tensor to minimize the similarity score computed by the UNiME model. Handles UNiME's specific input processing (dynamic resolution strategies like Global+2x2 Grid).
    *   **Targeted**: `pgd_attack_to_target`
        *   **Goal**: Maximize similarity to a specific target text.
        *   **Mechanism**: Optimizes the image to maximize the similarity score with the `target_text`.

## Usage

### Running Evaluations
The `experiment.py` scripts in `src/pipe` and `src/pipe2` are used to run evaluations on datasets like MMEB-eval. They support passing a list of perturbations to evaluate.

```python
# Example in src/pipe/experiment.py
perturbations = ['gauss1', 'flip', 'pgd']
deliverthegoods(datasets, perturbations, "openai/clip-vit-large-patch14")
```

### Visualizing Attacks
You can visualize the effect of attacks using `visualize_attack.py`.

```bash
python src/pipe/visualize_attack.py
```
This script generates a comparison of attacks on different models (e.g., CLIP Base vs. Large), showing the original image, adversarial images, and similarity scores for original and target texts.
