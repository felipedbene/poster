import inspect
from coremltools.models import MLModel
from transformers import CLIPTokenizer, CLIPFeatureExtractor
from diffusers import PNDMScheduler
from python_coreml_stable_diffusion.pipeline import CoreMLStableDiffusionPipeline
from pathlib import Path

print(inspect.signature(CoreMLStableDiffusionPipeline.__init__))

_coreml_pipe = None

def generate_image(prompt):
    global _coreml_pipe
    if _coreml_pipe is None:
        # Directory containing CoreML .mlpackage bundles
        packages_dir = Path(__file__).parent / "coreml-stable-diffusion-v1-5" / "original" / "packages"

        # Load CoreML models
        text_encoder_model = MLModel(str(packages_dir / "Stable_Diffusion_version_runwayml_stable-diffusion-v1-5_text_encoder.mlpackage"))
        unet_model = MLModel(str(packages_dir / "Stable_Diffusion_version_runwayml_stable-diffusion-v1-5_unet.mlpackage"))
        vae_decoder_model = MLModel(str(packages_dir / "Stable_Diffusion_version_runwayml_stable-diffusion-v1-5_vae_decoder.mlpackage"))
        safety_checker_model = MLModel(str(packages_dir / "Stable_Diffusion_version_runwayml_stable-diffusion-v1-5_safety_checker.mlpackage"))

        # Load scheduler and tokenizer
        scheduler = PNDMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")

        # Initialize CoreML SD pipeline with correct arg order
        _coreml_pipe = CoreMLStableDiffusionPipeline(
            text_encoder_model,
            unet_model,
            vae_decoder_model,
            scheduler,
            tokenizer,
            controlnet=None,
            feature_extractor=feature_extractor,
            safety_checker=safety_checker_model
        )