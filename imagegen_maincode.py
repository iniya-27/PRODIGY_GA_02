# =============================================================
#   Task 02 — Image Generation with Stable Diffusion
#   Run this in Google Colab (Runtime → GPU recommended)
# =============================================================

import torch
import matplotlib.pyplot as plt
from PIL import Image
import time
import warnings
warnings.filterwarnings("ignore")

# ── DEVICE SETUP ──────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ── LOAD MODEL ────────────────────────────────────────────────
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

MODEL_ID = "runwayml/stable-diffusion-v1-5"
print(f"\nLoading Stable Diffusion model: {MODEL_ID}")
print("This may take a few minutes on first run...")

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(DEVICE)
pipe.enable_attention_slicing()
print("Model loaded successfully!")


# ── GENERATION FUNCTION ───────────────────────────────────────
def generate_image(
    prompt,
    negative_prompt="blurry, bad quality, distorted, ugly, low resolution",
    num_inference_steps=25,
    guidance_scale=7.5,
    width=512,
    height=512,
    seed=42,
):
    """
    Generate an image from a text prompt using Stable Diffusion.

    Args:
        prompt            : Text description of the image to generate
        negative_prompt   : What you DON'T want in the image
        num_inference_steps: More steps = better quality but slower (20-50)
        guidance_scale    : How closely to follow the prompt (7-12 recommended)
        width, height     : Image dimensions
        seed              : Random seed for reproducibility
    """
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    start = time.time()

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
    )

    elapsed = time.time() - start
    print(f"  Generated in {elapsed:.1f} seconds")
    return result.images[0]


def show_single(image, title="Generated Image"):
    """Display a single image."""
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=100)
    plt.show()
    print(f"Saved as '{title.replace(' ', '_')}.png'")


def show_grid(images, titles, main_title="Generated Images", cols=3):
    """Display multiple images in a grid."""
    n = len(images)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    fig.suptitle(main_title, fontsize=16, fontweight="bold", y=1.02)

    # Flatten axes for easy indexing
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx < n:
                axes[r][c].imshow(images[idx])
                axes[r][c].set_title(titles[idx], fontsize=11, fontweight="bold")
                axes[r][c].axis("off")
            else:
                axes[r][c].axis("off")
            idx += 1

    plt.tight_layout()
    plt.savefig(f"{main_title.replace(' ', '_')}.png", dpi=100, bbox_inches="tight")
    plt.show()
    print(f"Saved as '{main_title.replace(' ', '_')}.png'")


# ── EXPERIMENT 1: TEXT TO IMAGE ───────────────────────────────
print("\n" + "="*55)
print("   EXPERIMENT 1: Text to Image Generation")
print("="*55)

# ✏️ Change these prompts to anything you like!
prompts = [
    "a beautiful sunset over the ocean, golden hour, photorealistic, high quality",
    "a futuristic city at night with neon lights, cyberpunk style, highly detailed",
    "a magical forest with glowing mushrooms and fireflies, fantasy art, ethereal",
    "a cute robot reading a book in a cozy library, digital art, warm lighting",
]

generated_images = []
for i, prompt in enumerate(prompts, 1):
    print(f"\n[{i}/{len(prompts)}] Prompt: '{prompt[:55]}...'")
    img = generate_image(prompt, seed=42 + i)
    img.save(f"text_to_image_{i}.png")
    generated_images.append(img)
    print(f"  Saved as 'text_to_image_{i}.png'")

show_grid(
    generated_images,
    [f"Prompt {i+1}" for i in range(len(prompts))],
    main_title="Text to Image Generation",
    cols=2
)


# ── EXPERIMENT 2: GUIDANCE SCALE ──────────────────────────────
print("\n" + "="*55)
print("   EXPERIMENT 2: Effect of Guidance Scale")
print("="*55)
print("Low value = more creative | High value = follows prompt strictly\n")

prompt = "a majestic lion in a savanna at sunset, professional photography"
guidance_scales = [3.0, 7.5, 12.0]
gs_images = []

for gs in guidance_scales:
    print(f"Generating with guidance_scale = {gs}...")
    img = generate_image(prompt, guidance_scale=gs, seed=42)
    gs_images.append(img)

show_grid(
    gs_images,
    [f"Guidance Scale = {gs}" for gs in guidance_scales],
    main_title="Effect of Guidance Scale",
    cols=3
)


# ── EXPERIMENT 3: INFERENCE STEPS ─────────────────────────────
print("\n" + "="*55)
print("   EXPERIMENT 3: Effect of Inference Steps")
print("="*55)
print("More steps = better quality but slower\n")

prompt = "a detailed portrait of an astronaut on Mars, realistic, high quality"
steps_list = [10, 20, 30]
steps_images = []

for steps in steps_list:
    print(f"Generating with {steps} inference steps...")
    img = generate_image(prompt, num_inference_steps=steps, seed=42)
    steps_images.append(img)

show_grid(
    steps_images,
    [f"{steps} Steps" for steps in steps_list],
    main_title="Effect of Inference Steps",
    cols=3
)


# ── EXPERIMENT 4: NEGATIVE PROMPTS ────────────────────────────
print("\n" + "="*55)
print("   EXPERIMENT 4: Effect of Negative Prompts")
print("="*55)
print("Negative prompts tell the model what to AVOID\n")

prompt = "a portrait of a woman in a garden, flowers, sunlight"
neg_prompts = [
    "",
    "blurry, bad quality, distorted, ugly, watermark",
    "blurry, bad quality, distorted, ugly, watermark, dark, sad, gloomy, low resolution",
]
neg_labels = [
    "No Negative Prompt",
    "Basic Negative Prompt",
    "Detailed Negative Prompt"
]
neg_images = []

for neg, label in zip(neg_prompts, neg_labels):
    print(f"Generating: {label}...")
    img = generate_image(prompt, negative_prompt=neg, seed=42)
    neg_images.append(img)

show_grid(
    neg_images,
    neg_labels,
    main_title="Effect of Negative Prompts",
    cols=3
)


# ── EXPERIMENT 5: ART STYLES ──────────────────────────────────
print("\n" + "="*55)
print("   EXPERIMENT 5: Different Art Styles")
print("="*55)
print("Same subject rendered in 6 completely different styles\n")

subject = "a cat sitting on a windowsill"
styles = [
    f"{subject}, oil painting, impressionist style, Vincent van Gogh",
    f"{subject}, watercolor painting, soft colors, artistic",
    f"{subject}, anime style, Studio Ghibli, highly detailed",
    f"{subject}, photorealistic, professional photography, 4K",
    f"{subject}, pixel art, retro game style, colorful",
    f"{subject}, pencil sketch, detailed drawing, black and white",
]
style_labels = [
    "Oil Painting",
    "Watercolor",
    "Anime Style",
    "Photorealistic",
    "Pixel Art",
    "Pencil Sketch",
]
style_images = []

for style_prompt, label in zip(styles, style_labels):
    print(f"Generating: {label}...")
    img = generate_image(style_prompt, seed=42)
    style_images.append(img)

show_grid(
    style_images,
    style_labels,
    main_title="Same Subject Different Art Styles",
    cols=3
)


# ── CUSTOM PROMPT ─────────────────────────────────────────────
print("\n" + "="*55)
print("   YOUR CUSTOM IMAGE")
print("="*55)

# ✏️ CHANGE THIS TO YOUR OWN PROMPT!
my_prompt = "a magical treehouse in an enchanted forest, fantasy art, glowing lights, highly detailed, 4K"
my_negative = "blurry, bad quality, dark, ugly, distorted"

print(f"Prompt: '{my_prompt}'")
my_image = generate_image(
    prompt=my_prompt,
    negative_prompt=my_negative,
    num_inference_steps=30,
    guidance_scale=8.0,
    seed=123,
)
my_image.save("my_custom_image.png")
show_single(my_image, title="My Custom Generated Image")


# ── SUMMARY ───────────────────────────────────────────────────
total = len(generated_images) + len(gs_images) + len(steps_images) + len(neg_images) + len(style_images) + 1

print("\n" + "="*55)
print("   SUMMARY")
print("="*55)
print(f"  Model          : Stable Diffusion v1.5")
print(f"  Device         : {DEVICE}")
print(f"  Total images   : {total}")
print(f"  Image size     : 512 x 512")
print(f"\n  Experiments completed:")
print(f"  1. Text to Image Generation   — {len(generated_images)} images")
print(f"  2. Effect of Guidance Scale   — {len(gs_images)} images")
print(f"  3. Effect of Inference Steps  — {len(steps_images)} images")
print(f"  4. Effect of Negative Prompts — {len(neg_images)} images")
print(f"  5. Different Art Styles       — {len(style_images)} images")
print(f"  6. Custom Prompt              — 1 image")
print("="*55)
print("  Done! Image generation complete.")
print("="*55)
