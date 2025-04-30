# Running Stable Diffusion with CoreML on Apple M1/M2/M4

This guide walks you through setting up your `poster` project to generate images using Apple’s optimized CoreML Stable Diffusion pipeline, fully replacing AUTOMATIC1111.

## Prerequisites

- **macOS** with Apple Silicon (M1/M2/M4)
- **Homebrew** installed
- **Git LFS** installed and initialized:
  ```bash
  brew install git-lfs
  git lfs install
  ```

## 1. Create and Activate a Python 3.11 Virtual Environment

```bash
cd /Users/felipe/workspace/poster
python3.11 -m venv .venv
source .venv/bin/activate
```

> If you don’t have Python 3.11:  
> `brew install python@3.11`

## 2. Install CoreML SD Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install diffusers coremltools transformers accelerate pillow scipy
```

## 3. Clone and Install Apple’s CoreML Stable Diffusion Repo

```bash
git clone https://github.com/apple/ml-stable-diffusion.git
cd ml-stable-diffusion
pip install -e .
cd ..
```

## 4. Download the CoreML Model Bundles

Within your `poster` directory:

```bash
git clone https://huggingface.co/apple/coreml-stable-diffusion-v1-5 coreml-stable-diffusion-v1-5
cd coreml-stable-diffusion-v1-5
git lfs pull
cd ..
```

This creates `coreml-stable-diffusion-v1-5/original/packages` containing:
- `Stable_Diffusion_version_runwayml_stable-diffusion-v1-5_text_encoder.mlpackage`
- `..._unet.mlpackage`
- `..._vae_decoder.mlpackage`
- `..._safety_checker.mlpackage`
- etc.

## 5. Patch Your `post.py` Code

Ensure you’ve applied the patches to `post.py`:

1. **Imports** at the top:
   ```python
   from python_coreml_stable_diffusion.pipeline import CoreMLStableDiffusionPipeline
   from python_coreml_stable_diffusion.coreml_model import CoreMLModel
   from transformers import CLIPTokenizer, CLIPFeatureExtractor
   from diffusers import PNDMScheduler
   from pathlib import Path
   ```

2. **Singleton and Generate Function**:
   ```python
   _coreml_pipe = None

   def generate_image(prompt):
       global _coreml_pipe
       if _coreml_pipe is None:
           packages_dir = Path(__file__).parent / "coreml-stable-diffusion-v1-5" / "original" / "packages"

           # Load CoreML models
           text_encoder_model = CoreMLModel(str(packages_dir / "Stable_Diffusion_version_runwayml_stable-diffusion-v1-5_text_encoder.mlpackage"), compute_unit="ALL")
           unet_model         = CoreMLModel(str(packages_dir / "Stable_Diffusion_version_runwayml_stable-diffusion-v1-5_unet.mlpackage"), compute_unit="ALL")
           vae_decoder_model  = CoreMLModel(str(packages_dir / "Stable_Diffusion_version_runwayml_stable-diffusion-v1-5_vae_decoder.mlpackage"), compute_unit="ALL")
           safety_checker_model = CoreMLModel(str(packages_dir / "Stable_Diffusion_version_runwayml_stable-diffusion-v1-5_safety_checker.mlpackage"), compute_unit="ALL")

           # Load scheduler and tokenizer
           scheduler = PNDMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
           tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
           feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")

           # Initialize CoreML pipeline
           _coreml_pipe = CoreMLStableDiffusionPipeline(
               text_encoder_model,
               unet_model,
               vae_decoder_model,
               scheduler,
               tokenizer,
               controlnet=None,
               xl=False,
               force_zeros_for_empty_prompt=True,
               feature_extractor=feature_extractor,
               safety_checker=safety_checker_model,
               text_encoder_2=None,
               tokenizer_2=None
           )

       prompt = prompt.strip()
       cache_key = hashlib.sha256(prompt.encode()).hexdigest()
       os.makedirs(".cache/images", exist_ok=True)
       output_path = os.path.join(".cache/images", f"{cache_key}.png")
       if os.path.exists(output_path):
           return output_path

       result = _coreml_pipe(prompt=prompt, num_inference_steps=30, guidance_scale=7.5)
       image = result.images[0]
       image.save(output_path)
       return output_path
   ```

## 6. Run Your Script

```bash
python3 post.py --idea "Your blog post idea here"
```

🎉 You should see CoreML-powered image generation (~8s per image) and successful WordPress uploads.