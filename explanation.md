# Image Generation with Stable Diffusion — Complete Explanation

---

## 1. What is Stable Diffusion?

Stable Diffusion is a **deep learning, text-to-image generation model** released in 2022 by Stability AI.

- It is a **generative model** — a model that creates new content (images) from scratch
- It was pre-trained on **billions of image-text pairs** from the internet (LAION-5B dataset)
- Its job is simple: **given a text description, generate a matching image**
- It uses a **Latent Diffusion Model (LDM)** architecture — which works in compressed latent space instead of pixel space, making it fast and memory-efficient

### Stable Diffusion Variants

| Model | Release | Quality | Notes |
|---|---|---|---|
| SD v1.4 | 2022 | Good | First public release |
| SD v1.5 | 2022 | Better | Most widely used version |
| SD v2.0 | 2022 | Better | Higher resolution (768px) |
| SD XL | 2023 | Best | Larger model, more detailed |

> We use **Stable Diffusion v1.5** (`runwayml/stable-diffusion-v1-5`) in this task — it is stable, well-supported, and runs efficiently on a T4 GPU.

---

## 2. What is Image Generation?

Image generation means **creating an entirely new image from a text prompt** — there is no existing image being edited or filtered.

### Analogy
> Think of Stable Diffusion as an extremely skilled artist. You describe what you want in words — "a sunset over the ocean, golden hour, photorealistic" — and the artist paints it from scratch, having studied billions of paintings, photographs, and artworks.

### Why Use Pre-trained Models?
- Training a model like Stable Diffusion from scratch requires **billions of images** and months of compute on thousands of GPUs
- Using a **pre-trained model** means all that learning is already done
- We just load the model and use it directly — no training required
- HuggingFace's `diffusers` library makes this as simple as a few lines of Python

---

## 3. How Does Stable Diffusion Work? — The Pipeline

Stable Diffusion follows a **3-stage pipeline** internally:

```
Text Prompt
      ↓
[ Text Encoder — CLIP ]
      ↓  (text embeddings)
[ Diffusion Model — U-Net ]
      ↓  (denoised latents)
[ Image Decoder — VAE ]
      ↓
Generated Image ✅
```

### Stage 1 — Text Encoder (CLIP)
- The text prompt (e.g., `"a futuristic city at night"`) is passed into a **CLIP text encoder**
- CLIP (**Contrastive Language-Image Pretraining**) was trained to understand the relationship between text and images
- It converts the text into a list of numbers called **embeddings** that capture the meaning and content of the prompt
- These embeddings tell the model *what* the final image should look like

### Stage 2 — Diffusion Model (U-Net)
- The model starts with a **random noise tensor** — like static on a TV screen, but in compressed form
- It **iteratively denoises** this noise over multiple steps, guided by the text embeddings
- At each step, the model asks: *"Given this noisy image and this text, what should I remove to get closer to the prompt?"*
- After all steps, the noise has been refined into a meaningful latent representation
- This is the **inference steps** process — more steps = cleaner, more refined result

### Stage 3 — Image Decoder (VAE)
- The final denoised result exists in **latent space** — a compact, compressed representation
- The **VAE (Variational Autoencoder) decoder** converts this latent tensor back into a full pixel image (e.g., 512×512 pixels)
- This is the final image you see and save

### Why "Latent" Diffusion?
Older diffusion models worked directly in pixel space — which is enormous (512×512×3 = 786,432 numbers). Stable Diffusion compresses images into a **64×64 latent space** (8× smaller), making the whole process much faster and memory-efficient.

---

## 4. How Does Text Guide the Image? — Classifier-Free Guidance (CFG)

This is the mechanism behind the **guidance scale** parameter.

### The Process
1. The model runs the denoising step **twice** at each inference step:
   - Once **with** the text prompt (conditioned)
   - Once **without** any prompt (unconditioned)
2. The two outputs are compared, and the difference is amplified by the **guidance scale**
3. The final update pushes strongly in the direction of the prompt

### Formula
```
output = unconditioned + guidance_scale × (conditioned − unconditioned)
```

| Guidance Scale | Effect |
|---|---|
| 1.0 | Ignores the prompt completely |
| 3.0 | Loosely follows the prompt — more creative, unpredictable |
| 7.5 | Balanced — good quality and prompt adherence (recommended) |
| 12.0+ | Very strictly follows the prompt — can look over-saturated |

---

## 5. The Code — Step by Step

### 🔧 Setup and Imports

```python
import torch
import matplotlib.pyplot as plt
from PIL import Image
import time
import warnings
warnings.filterwarnings("ignore")
```

| Library | Purpose |
|---|---|
| `torch` | PyTorch — deep learning framework for GPU computations |
| `matplotlib.pyplot` | Used to display and save images |
| `PIL (Pillow)` | Python Imaging Library — handles image objects |
| `time` | Measures how long each image takes to generate |
| `warnings` | Suppresses unnecessary warning messages |

---

### 🖥️ Device Setup

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

- Checks if a **GPU (CUDA)** is available
- If yes → uses GPU (very fast, ~5 sec per image)
- If no → falls back to CPU (very slow, ~5–10 min per image)
- In Google Colab with **T4 GPU**, this prints `Using device: cuda`

---

### 📦 Loading the Model

```python
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(DEVICE)
pipe.enable_attention_slicing()
```

| Line | What it does |
|---|---|
| `from_pretrained()` | Downloads and loads the full model from HuggingFace Hub |
| `torch_dtype=torch.float16` | Uses half-precision (FP16) — halves memory usage, works on T4 GPU |
| `DPMSolverMultistepScheduler` | Replaces default scheduler with a faster, higher-quality one |
| `.to(DEVICE)` | Moves all model weights to GPU memory |
| `enable_attention_slicing()` | Processes attention in chunks — prevents out-of-memory errors |

---

### 🖼️ The `generate_image()` Function

```python
def generate_image(
    prompt,
    negative_prompt="blurry, bad quality, distorted, ugly, low resolution",
    num_inference_steps=25,
    guidance_scale=7.5,
    width=512,
    height=512,
    seed=42,
):
```

This is the core function used in every experiment. Each parameter controls a different aspect of generation:

| Parameter | Default | Role |
|---|---|---|
| `prompt` | — | What you WANT in the image |
| `negative_prompt` | "blurry, bad quality..." | What you DON'T want |
| `num_inference_steps` | 25 | Denoising steps — more = better quality, slower |
| `guidance_scale` | 7.5 | How closely to follow the prompt |
| `width` / `height` | 512 | Output image size in pixels |
| `seed` | 42 | Fixes randomness — same seed = same image every time |

```python
generator = torch.Generator(device=DEVICE).manual_seed(seed)
```
- Creates a **random number generator** with a fixed seed
- Ensures **reproducibility** — running the same code with the same seed always produces the identical image

---

### 📊 Display Functions

```python
def show_single(image, title):   # Displays one image
def show_grid(images, titles, main_title, cols):  # Displays multiple images in a grid
```

- Both use `matplotlib` to render images
- Both **save** the result as a `.png` file automatically
- `show_grid()` calculates rows automatically based on number of images and columns

---

## 6. Experiments Explained

### Experiment 1 — Text to Image Generation

**Goal:** Test how well the model understands diverse text prompts

```python
prompts = [
    "a beautiful sunset over the ocean, golden hour, photorealistic, high quality",
    "a futuristic city at night with neon lights, cyberpunk style, highly detailed",
    "a magical forest with glowing mushrooms and fireflies, fantasy art, ethereal",
    "a cute robot reading a book in a cozy library, digital art, warm lighting",
]
```

- 4 very different prompts — realistic, sci-fi, fantasy, and digital art
- Each uses a different `seed` (`42+i`) so each image is unique
- **Key learning:** Style keywords like `photorealistic`, `highly detailed`, `warm lighting` greatly improve output quality

---

### Experiment 2 — Effect of Guidance Scale

**Goal:** Understand how guidance scale affects the balance between creativity and accuracy

```python
guidance_scales = [3.0, 7.5, 12.0]
```

- Same prompt, same seed — only guidance scale changes
- Lower value → model is more creative and may deviate from the prompt
- Higher value → model strictly follows the prompt but images can look oversaturated
- **Key learning:** Guidance scale **7–9** is the recommended sweet spot for most use cases

---

### Experiment 3 — Effect of Inference Steps

**Goal:** Understand the quality vs speed trade-off of denoising steps

```python
steps_list = [10, 20, 30]
```

| Steps | Quality | Speed |
|---|---|---|
| 10 | Rough, incomplete | Very fast |
| 20 | Good quality | Moderate |
| 30 | High quality, refined | Slower |

- **Key learning:** With the DPM-Solver++ scheduler, **20–30 steps** gives excellent results. Beyond 50 steps, improvement becomes minimal.

---

### Experiment 4 — Effect of Negative Prompts

**Goal:** Show how negative prompts clean up and improve generated images

```python
neg_prompts = [
    "",                                    # No negative prompt
    "blurry, bad quality, distorted, ugly, watermark",
    "blurry, bad quality, distorted, ugly, watermark, dark, sad, gloomy, low resolution",
]
```

- **No negative prompt** — model is free; may produce artifacts or poor quality
- **Basic negative prompt** — removes most common defects
- **Detailed negative prompt** — produces the cleanest, most refined result
- **Key learning:** Negative prompts work through CFG — the model is pushed *away* from these descriptions, just as it is pushed *toward* the positive prompt

---

### Experiment 5 — Different Art Styles

**Goal:** Show that Stable Diffusion understands a wide range of artistic styles

```python
subject = "a cat sitting on a windowsill"
styles = [
    f"{subject}, oil painting, impressionist style, Vincent van Gogh",
    f"{subject}, watercolor painting, soft colors, artistic",
    f"{subject}, anime style, Studio Ghibli, highly detailed",
    f"{subject}, photorealistic, professional photography, 4K",
    f"{subject}, pixel art, retro game style, colorful",
    f"{subject}, pencil sketch, detailed drawing, black and white",
]
```

- Same subject, 6 completely different artistic styles
- The model has learned these styles from its training on billions of images
- **Key learning:** Simply adding style keywords to a prompt is enough to completely transform the visual rendering of the same subject

---

### Custom Prompt — Personal Image

**Goal:** Apply everything learned to generate a personal, fine-tuned image

```python
my_prompt = "a magical treehouse in an enchanted forest, fantasy art, glowing lights, highly detailed, 4K"
my_negative = "blurry, bad quality, dark, ugly, distorted"
# num_inference_steps=30, guidance_scale=8.0
```

- Uses optimal parameters: 30 steps, guidance scale 8.0
- Combines detailed positive prompt + strong negative prompt
- **Key learning:** Fine-tuning parameters + prompt quality makes the biggest difference in final output

---

## 7. Prompt Writing Tips

| Tip | Example |
|---|---|
| Be descriptive | `"a golden retriever playing in autumn leaves, warm sunlight, bokeh background"` |
| Add quality keywords | `photorealistic`, `highly detailed`, `4K`, `cinematic`, `sharp focus` |
| Add lighting keywords | `golden hour`, `soft lighting`, `dramatic lighting`, `warm glow` |
| Specify art style | `oil painting`, `watercolor`, `anime style`, `pixel art`, `pencil sketch` |
| Reference artists/studios | `in the style of Van Gogh`, `Studio Ghibli`, `Pixar style` |
| Always use negative prompt | `"blurry, bad quality, distorted, ugly, watermark, low resolution"` |

---

## 8. End-to-End Flow Summary

```
1. Install libraries        →  diffusers, transformers, accelerate, torch
2. Set up device            →  GPU (CUDA) or CPU
3. Load the model           →  StableDiffusionPipeline from HuggingFace
4. Set the scheduler        →  DPMSolverMultistepScheduler (faster, better)
5. Write a prompt           →  Positive + Negative
6. Set parameters           →  steps, guidance_scale, seed, size
7. Generate image           →  pipe(prompt, ...)
8. Display and save         →  matplotlib + PIL
9. Run experiments          →  Vary parameters to understand their effects
```

---

## 9. Tips for Better Results

| Tip | Details |
|---|---|
| **Always use GPU** | CPU generation takes 5–10 minutes per image; GPU takes ~5 seconds |
| **Use 20–30 steps** | Good quality without wasting time |
| **Keep guidance scale 7–9** | Best balance between creativity and accuracy |
| **Always use a negative prompt** | Significantly improves image quality and removes artifacts |
| **Use a fixed seed for comparison** | Isolates the effect of one parameter at a time |
| **Be descriptive in prompts** | More detail = more control over the output |
| **Enable attention slicing** | Prevents GPU out-of-memory errors on limited hardware |

---

## 10. Glossary

| Term | Definition |
|---|---|
| **Stable Diffusion** | A text-to-image generative model using latent diffusion |
| **Latent Space** | A compressed mathematical representation of images |
| **Diffusion** | The process of iteratively removing noise to generate an image |
| **Denoising** | Removing noise from a latent tensor step by step |
| **Inference Steps** | The number of denoising steps — more steps = better quality |
| **Guidance Scale** | Controls how strictly the model follows the text prompt |
| **CLIP** | Contrastive Language-Image Pretraining — converts text to embeddings |
| **CFG** | Classifier-Free Guidance — mechanism behind guidance scale |
| **VAE** | Variational Autoencoder — encodes/decodes images to/from latent space |
| **U-Net** | The neural network that performs denoising in latent space |
| **Prompt** | A text description of the image you want to generate |
| **Negative Prompt** | A text description of what you do NOT want in the image |
| **Seed** | A number that controls randomness — same seed = same image |
| **Scheduler** | Controls the denoising algorithm (e.g., DPM-Solver, PNDM) |
| **FP16** | Half-precision floating point — reduces memory usage |
| **HuggingFace** | Python library providing pre-trained models and pipelines |
| **Perplexity** | Not applicable here — used for language models, not image models |
| **Pipeline** | A pre-built workflow that chains model components together |

---

*This explanation was written as part of the Generative AI Internship at Prodigy InfoTech — Task 02.*
