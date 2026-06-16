# PRODIGY_GA_02 — Image Generation with Stable Diffusion

## 📌 Task
Utilize pre-trained generative models to create images from text prompts.

## 📋 Description
This project is part of my Generative AI Internship at Prodigy InfoTech. I have used 
Stable Diffusion v1.5, a powerful pre-trained text-to-image model, to generate images 
from text prompts. The project covers multiple experiments to understand how different 
parameters affect the quality and style of generated images.

## 🛠️ Tech Stack
- Python 3.12
- HuggingFace Diffusers
- PyTorch
- Stable Diffusion v1.5
- Google Colab (GPU)
- Matplotlib

## 📂 Files
| File | Description |
|---|---|
| `stable_diffusion.py` | Complete Python code for image generation |
| `requirements.txt` | All required Python libraries with versions |

## 🔬 Experiments Conducted
| Experiment | Description |
|---|---|
| Text to Image Generation | Generated images from 4 different text prompts |
| Effect of Guidance Scale | Tested values 3.0, 7.5, 12.0 on same prompt |
| Effect of Inference Steps | Compared 10, 20, 30 steps on same prompt |
| Effect of Negative Prompts | Compared no prompt vs basic vs detailed negative prompt |
| Different Art Styles | Same subject in 6 styles — Oil, Watercolor, Anime, Photorealistic, Pixel Art, Sketch |
| Custom Prompt | Generated a personal custom image |

## 💡 Key Concepts Learned
- How Stable Diffusion works (Text Encoder → Diffusion Model → Decoder)
- How text prompts are converted to images using latent diffusion
- Effect of guidance scale on image creativity vs accuracy
- Effect of inference steps on image quality
- How negative prompts improve image quality
- Generating images in different artistic styles

## 📊 Results
| Parameter | Value |
|---|---|
| Model | Stable Diffusion v1.5 |
| Image Size | 512 x 512 pixels |
| Total Images Generated | 17 |
| Default Inference Steps | 25 |
| Default Guidance Scale | 7.5 |
| Device | CUDA (T4 GPU) |

## ▶️ How to Run
1. Open Google Colab
2. Set Runtime to GPU — Runtime → Change runtime type → T4 GPU
3. Run Cell 1 to install dependencies:
```
!pip uninstall diffusers huggingface_hub transformers accelerate peft -y
!pip install -q diffusers==0.25.0 huggingface_hub==0.23.4 transformers==4.38.2 accelerate==0.30.0 safetensors
!pip uninstall peft -y
```
4. Restart runtime after installation
5. Run Cell 2 with the main code

## 📚 References
- [Stable Diffusion in TensorFlow/KerasCV](https://www.tensorflow.org/tutorials/generative/generate_images_with_stable_diffusion)
- [DALL-E Mini Image Generator](https://colab.research.google.com/github/robgon-art/e-dall-e/blob/main/DALL_E_Mini_Image_Generator.ipynb)
- [HuggingFace Diffusers](https://huggingface.co/docs/diffusers)
