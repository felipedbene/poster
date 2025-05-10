"""
Ollama utilities for local and remote Ollama interactions.
This module provides functions to detect and use local Ollama installations
or fall back to remote Ollama servers when configured.
"""

import os
import sys
import requests
import platform
import socket
import time
import hashlib
from io import BytesIO
from PIL import Image
import base64
from typing import Optional, Dict, Any, Union, List, Tuple

# Default local Ollama endpoint
DEFAULT_LOCAL_OLLAMA = "http://localhost:11434"

def is_local_ollama_available() -> bool:
    """
    Check if a local Ollama instance is available by attempting to connect to the API.
    
    Returns:
        bool: True if local Ollama is available, False otherwise.
    """
    try:
        # Try to connect to the local Ollama API
        response = requests.get(f"{DEFAULT_LOCAL_OLLAMA}/api/tags", timeout=2)
        # Check if the response is valid JSON and contains model information
        if response.status_code == 200:
            try:
                data = response.json()
                # Verify that the response contains the expected structure
                if "models" in data or isinstance(data, list):
                    return True
            except:
                pass
        return False
    except (requests.RequestException, socket.error):
        return False

def get_ollama_endpoint() -> str:
    """
    Determine the Ollama endpoint to use, preferring local if available.
    
    Returns:
        str: The Ollama API endpoint URL to use.
    """
    # Always try local Ollama first
    if is_local_ollama_available():
        print("🔍 Using local Ollama instance")
        return DEFAULT_LOCAL_OLLAMA
    
    # If local is not available, check if OLLAMA_SERVER is set in environment
    env_server = os.getenv("OLLAMA_SERVER")
    if env_server:
        # Ensure the URL has a protocol
        if not env_server.startswith(("http://", "https://")):
            env_server = f"http://{env_server}"
        print(f"🔍 Using remote Ollama server: {env_server}")
        return env_server
    
    # If we get here, no Ollama is available, but return the local endpoint
    # as a fallback (it will fail gracefully when used)
    print("⚠️ No Ollama server available (neither local nor remote)")
    return DEFAULT_LOCAL_OLLAMA  # Return default even though it's not available

def list_available_models() -> List[str]:
    """
    List available models from the Ollama endpoint.
    
    Returns:
        List[str]: List of available model names.
    """
    endpoint = get_ollama_endpoint()
    try:
        response = requests.get(f"{endpoint}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()
        models = [model["name"] for model in data.get("models", [])]
        return models
    except Exception as e:
        print(f"⚠️ Failed to list Ollama models: {e}")
        return []

def generate_text_with_ollama(
    prompt: str,
    model: str = "llama3:8b",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    stream: bool = False,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Generate text using Ollama.
    
    Args:
        prompt (str): The prompt to generate text from.
        model (str): The model to use for generation.
        temperature (float): The temperature for generation.
        max_tokens (int): Maximum number of tokens to generate.
        stream (bool): Whether to stream the response.
        device (str): Device to use for inference.
        
    Returns:
        Dict[str, Any]: The response from Ollama.
    """
    endpoint = get_ollama_endpoint()
    
    try:
        # Check if the model exists
        available_models = list_available_models()
        if model not in available_models and available_models:
            # Try to find a similar model
            if "llama3" in [m for m in available_models if "llama3" in m.lower()]:
                model = next(m for m in available_models if "llama3" in m.lower())
            elif "llama" in [m for m in available_models if "llama" in m.lower()]:
                model = next(m for m in available_models if "llama" in m.lower())
            elif len(available_models) > 0:
                model = available_models[0]
            
            print(f"⚠️ Model {model} not available, using {model} instead")
        
        response = requests.post(
            f"{endpoint}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
                "device": device,
            },
            timeout=300
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Failed to generate text with Ollama: {e}")
        raise

def generate_image_with_ollama(
    prompt: str,
    model: str = "sdxl:latest",
    width: int = 1024,
    height: int = 1024,
    steps: int = 30,
    seed: Optional[int] = None
) -> Optional[str]:
    """
    Generate an image using Ollama's diffusion models.
    
    Args:
        prompt: The text prompt to generate an image from
        model: The diffusion model to use
        width: Width of the generated image
        height: Height of the generated image
        steps: Number of inference steps
        seed: Random seed for reproducibility
        
    Returns:
        Path to the generated image or None if generation failed
    """
    endpoint = get_ollama_endpoint()
    
    try:
        # Check if the model exists
        available_models = list_available_models()
        if model not in available_models:
            # Try to find a similar model
            if "sdxl" in [m for m in available_models if "sdxl" in m.lower()]:
                model = next(m for m in available_models if "sdxl" in m.lower())
            elif "sd" in [m for m in available_models if "sd" in m.lower()]:
                model = next(m for m in available_models if "sd" in m.lower())
            else:
                print(f"⚠️ Model {model} not available in Ollama. Available models: {available_models}")
                return None
        
        # Create cache directory
        os.makedirs(".cache/images", exist_ok=True)
        cache_key = hashlib.sha256(prompt.encode()).hexdigest()
        webp_path = f".cache/images/{cache_key}_ollama.webp"
        
        if os.path.exists(webp_path):
            print("🖼️ Cached WebP image used")
            return webp_path
        
        # Generate the image
        print(f"🖌️ Generating image with Ollama using {model}...")
        
        # Prepare the request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": steps,
        }
        
        if seed is not None:
            payload["seed"] = seed
        
        response = requests.post(
            f"{endpoint}/api/generate",
            json=payload,
            timeout=300
        )
        response.raise_for_status()
        
        # Process the response
        data = response.json()
        if "image" in data:
            # Decode the base64 image
            image_data = base64.b64decode(data["image"])
            image = Image.open(BytesIO(image_data))
            
            # Save the image
            png_path = f".cache/images/{cache_key}_ollama.png"
            image.save(png_path)
            
            # Convert to WebP for better compression
            webp_image = image.resize((384, 384), Image.Resampling.LANCZOS)
            webp_image.save(webp_path, format="WEBP", quality=85)
            
            return webp_path
        else:
            print("❌ No image data in Ollama response")
            return None
    except Exception as e:
        print(f"❌ Failed to generate image with Ollama: {e}")
        return None
