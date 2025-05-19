"""
Apple-specific utilities for image generation using MLX Core and CoreML.
This module provides functions to detect Apple Silicon hardware and
generate images using the Neural Processing Unit (NPU) when available.
"""

import os
import sys
import platform
import hashlib
import base64
from io import BytesIO
from typing import Optional, Union, Dict, Any
from PIL import Image


# Check if we're running on macOS with Apple Silicon
def is_apple_silicon() -> bool:
    """
    Detect if the code is running on Apple Silicon hardware.

    Returns:
        bool: True if running on macOS with Apple Silicon (arm64), False otherwise.
    """
    try:
        return platform.system() == "Darwin" and platform.machine() == "arm64"
    except Exception:
        return False


# Import MLX-related modules only if we're on Apple Silicon
if is_apple_silicon():
    try:
        import mlx.core as mx
        import coremltools as ct

        HAS_MLX = True
    except ImportError:
        print("⚠️ Running on Apple Silicon but MLX or CoreMLTools not installed.")
        HAS_MLX = False
else:
    HAS_MLX = False


def get_coreml_model_path(model_type: str = "split_einsum") -> str:
    """
    Get the path to the CoreML model based on the specified type.

    Args:
        model_type (str): The type of CoreML model to use ('original' or 'split_einsum').

    Returns:
        str: Path to the CoreML model directory.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if model_type == "original":
        return os.path.join(
            base_dir, "coreml-stable-diffusion-v1-5", "original", "packages"
        )
    else:
        return os.path.join(
            base_dir, "coreml-stable-diffusion-v1-5", "split_einsum", "packages"
        )


def generate_image_with_mlx(
    prompt: str,
    negative_prompt: str = "blurry, lowres, artifacts, jpeg artifacts",
    width: int = 800,
    height: int = 600,
    steps: int = 30,
    guidance_scale: float = 9.0,
    seed: Optional[int] = None,
) -> Optional[str]:
    """
    Generate an image using MLX Core and CoreML on Apple Silicon.

    Args:
        prompt (str): The text prompt for image generation.
        negative_prompt (str): Negative prompt to guide what to avoid in generation.
        width (int): Width of the generated image.
        height (int): Height of the generated image.
        steps (int): Number of diffusion steps.
        guidance_scale (float): Guidance scale for classifier-free guidance.
        seed (Optional[int]): Random seed for reproducibility.

    Returns:
        Optional[str]: Path to the generated image, or None if generation failed.
    """
    if not is_apple_silicon() or not HAS_MLX:
        print(
            "⚠️ Cannot use MLX Core: not running on Apple Silicon or MLX not installed."
        )
        return None

    try:
        # Clean up and enhance the prompt
        base_prompt = prompt.strip()
        if not base_prompt.lower().startswith("a "):
            base_prompt = "A " + base_prompt
        # Add tech-focused style modifiers
        enhanced_prompt = (
            base_prompt
            + ", ultra-realistic, high-tech, cyberpunk, neon, holographic, futuristic, digital, neural, quantum, AI, robotic, synthetic, ultra-modern, technological, innovative, cutting-edge, 8k, photorealistic, detailed, sharp focus"
        )
        print(f"🎨 Final tech prompt for MLX Core: {enhanced_prompt}")

        # Create cache directory and check for cached image
        os.makedirs(".cache/images", exist_ok=True)
        cache_key = hashlib.sha256(enhanced_prompt.encode()).hexdigest()
        webp_path = f".cache/images/{cache_key}.webp"

        if os.path.exists(webp_path):
            print("🖼️ Cached WebP image used")
            return webp_path

        # Import the python_coreml_stable_diffusion module from the ml-stable-diffusion directory
        sys.path.append(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "ml-stable-diffusion"
            )
        )
        from python_coreml_stable_diffusion.pipeline import (
            CoreMLStableDiffusionPipeline,
        )
        from python_coreml_stable_diffusion.coreml_model import (
            CoreMLModel,
            get_available_compute_units,
        )
        from diffusers import PNDMScheduler
        from transformers import CLIPTokenizer, CLIPFeatureExtractor

        # Get the actual packages directory (already includes split_einsum/packages)
        packages_dir = get_coreml_model_path()
        print(f"🔍 Using CoreML model packages from: {packages_dir}")

        # Load individual models with compute_unit parameter
        text_encoder_path = os.path.join(
            packages_dir,
            "Stable_Diffusion_version_runwayml_stable-diffusion-v1-5_text_encoder.mlpackage",
        )
        unet_path = os.path.join(
            packages_dir,
            "Stable_Diffusion_version_runwayml_stable-diffusion-v1-5_unet.mlpackage",
        )
        vae_decoder_path = os.path.join(
            packages_dir,
            "Stable_Diffusion_version_runwayml_stable-diffusion-v1-5_vae_decoder.mlpackage",
        )
        safety_checker_path = os.path.join(
            packages_dir,
            "Stable_Diffusion_version_runwayml_stable-diffusion-v1-5_safety_checker.mlpackage",
        )

        # Verify model files exist
        if (
            not os.path.exists(text_encoder_path)
            or not os.path.exists(unet_path)
            or not os.path.exists(vae_decoder_path)
        ):
            print(f"❌ Required CoreML model files not found in {model_path}")
            return None

        try:
            # Load CoreML models
            text_encoder = CoreMLModel(text_encoder_path, compute_unit="ALL")
            unet = CoreMLModel(unet_path, compute_unit="ALL")
            vae_decoder = CoreMLModel(vae_decoder_path, compute_unit="ALL")
            safety_checker = (
                CoreMLModel(safety_checker_path, compute_unit="ALL")
                if os.path.exists(safety_checker_path)
                else None
            )

            # Load scheduler and tokenizer from Hugging Face
            scheduler = PNDMScheduler.from_pretrained(
                "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
            )
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            feature_extractor = CLIPFeatureExtractor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )

            # Truncate prompts if they exceed the tokenizer's maximum length
            max_len = tokenizer.model_max_length
            prompt_tokens = tokenizer.encode(enhanced_prompt)
            if len(prompt_tokens) > max_len:
                prompt_tokens = prompt_tokens[:max_len]
                enhanced_prompt = tokenizer.decode(
                    prompt_tokens, skip_special_tokens=True
                )
                print("✂️ Prompt truncated for MLX Core")

            neg_tokens = tokenizer.encode(negative_prompt)
            if len(neg_tokens) > max_len:
                neg_tokens = neg_tokens[:max_len]
                negative_prompt = tokenizer.decode(neg_tokens, skip_special_tokens=True)

            # Ensure prompt and negative prompt have matching lengths
            if len(prompt_tokens) != len(neg_tokens):
                pad_token = tokenizer.pad_token_id or tokenizer.eos_token_id
                if len(prompt_tokens) > len(neg_tokens):
                    neg_tokens += [pad_token] * (len(prompt_tokens) - len(neg_tokens))
                else:
                    prompt_tokens += [pad_token] * (
                        len(neg_tokens) - len(prompt_tokens)
                    )
                enhanced_prompt = tokenizer.decode(
                    prompt_tokens, skip_special_tokens=True
                )
                negative_prompt = tokenizer.decode(neg_tokens, skip_special_tokens=True)

            # Initialize the pipeline with the correct parameters
            pipeline = CoreMLStableDiffusionPipeline(
                text_encoder=text_encoder,
                unet=unet,
                vae_decoder=vae_decoder,
                scheduler=scheduler,
                tokenizer=tokenizer,
                controlnet=None,
                xl=False,
                force_zeros_for_empty_prompt=True,
                feature_extractor=feature_extractor,
                safety_checker=safety_checker,
                text_encoder_2=None,
                tokenizer_2=None,
            )

            # Generate the image
            print(f"🖌️ Generating image with MLX Core on Apple Silicon NPU...")
            result = pipeline(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
            )

            # Get the generated image
            image = result.images[0]

            # Save the image
            png_path = f".cache/images/{cache_key}.png"
            image.save(png_path)

            # Convert to WebP for better compression
            webp_image = image.resize((384, 384), Image.Resampling.LANCZOS)
            webp_image.save(webp_path, format="WEBP", quality=85)

            return webp_path
        except Exception as e:
            print(f"❌ Failed to generate image with MLX Core: {e}")
            return None

    except Exception as e:
        print(f"❌ Failed to generate image with MLX Core: {e}")
        import traceback

        traceback.print_exc()
        return None
