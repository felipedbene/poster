#!/usr/bin/env python3
"""
Test script for image generation with Apple NPU support and local Ollama.
This script tests the enhanced image generation functionality that uses
Apple's Neural Processing Unit (NPU) via MLX Core when running on Apple hardware,
local Ollama if available, or falls back to AUTOMATIC1111 API.
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Import the generate_image function from post.py
from post import generate_image
from apple_utils import is_apple_silicon
from ollama_utils import is_local_ollama_available, list_available_models

def main():
    """
    Test the image generation functionality.
    """
    parser = argparse.ArgumentParser(description="Test image generation with Apple NPU support and local Ollama")
    parser.add_argument("--prompt", type=str, default="A futuristic cityscape with flying cars and neon lights",
                        help="Text prompt for image generation")
    parser.add_argument("--force-automatic", action="store_true",
                        help="Force using AUTOMATIC1111 API even on Apple Silicon or with local Ollama")
    parser.add_argument("--force-ollama", action="store_true",
                        help="Force using local Ollama even on Apple Silicon")
    args = parser.parse_args()
    
    # Print platform information
    if is_apple_silicon():
        print("🍎 Running on Apple Silicon - MLX Core with NPU acceleration is available")
    else:
        print("💻 Not running on Apple Silicon - will use other methods")
    
    # Check for local Ollama
    if is_local_ollama_available():
        print("🦙 Local Ollama is available")
        models = list_available_models()
        if models:
            print(f"📋 Available Ollama models: {', '.join(models)}")
        else:
            print("⚠️ No models found in local Ollama")
    else:
        print("⚠️ Local Ollama is not available")
    
    # If force_automatic is set, temporarily modify the is_apple_silicon function
    if args.force_automatic:
        print("⚠️ Forcing use of AUTOMATIC1111 API as requested")
        import apple_utils
        apple_utils.is_apple_silicon = lambda: False
        # Also modify the is_local_ollama_available function
        import ollama_utils
        ollama_utils.is_local_ollama_available = lambda: False
    
    # If force_ollama is set, temporarily modify the is_apple_silicon function
    if args.force_ollama and is_local_ollama_available():
        print("⚠️ Forcing use of local Ollama as requested")
        import apple_utils
        apple_utils.is_apple_silicon = lambda: False
    
    # Generate the image
    print(f"🖌️ Generating image with prompt: '{args.prompt}'")
    try:
        image_path = generate_image(args.prompt)
        if image_path:
            print(f"✅ Image generated successfully: {image_path}")
            # Try to display the image if running in an environment that supports it
            try:
                from PIL import Image
                img = Image.open(image_path)
                img.show()
                print("🖼️ Image displayed")
            except Exception as e:
                print(f"⚠️ Could not display image: {e}")
        else:
            print("❌ Image generation failed")
    except Exception as e:
        print(f"❌ Error during image generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()