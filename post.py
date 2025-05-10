import os
import sys
import argparse
import requests
import hashlib
import json
import datetime
import random
import re
import logging
import time
import platform
from dotenv import load_dotenv
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.codehilite import CodeHiliteExtension
import yaml
from typing import Tuple, Dict, Optional
import frontmatter
from PIL import Image
from io import BytesIO
import base64

# Import Apple-specific utilities
from apple_utils import is_apple_silicon, generate_image_with_mlx
# Import Ollama utilities
from ollama_utils import get_ollama_endpoint, generate_text_with_ollama, generate_image_with_ollama

# Load .env, overriding existing environment variables
load_dotenv(override=True)

# Define global variables from environment
WP_USER = os.getenv("WORDPRESS_USERNAME")
WP_PASS = os.getenv("WORDPRESS_APP_PASSWORD")
WP_URL  = os.getenv("WORDPRESS_URL")
HC_APIKEY     = os.getenv("HC_APIKEY")
SD_API_BASE   = os.getenv("SD_API_URL")
OLLAMA_SERVER = os.getenv("OLLAMA_SERVER")
NEWSAPI_KEY   = os.getenv("NEWSAPI_KEY")

def _strip_frontmatter(text):
    """
    Strip YAML front matter from text.
    """
    pattern = r'^\s*---\s*\n.*?\n---\s*\n'
    return re.sub(pattern, '', text, flags=re.DOTALL)

def _parse_yaml_safely(yaml_content, topic):
    """
    Parse YAML content safely, handling special characters and replacing placeholders.
    """
    # Replace placeholders in the YAML content
    yaml_content = yaml_content.replace("{topic}", topic)
    
    # Handle colons in titles by quoting them
    lines = yaml_content.split('\n')
    for i, line in enumerate(lines):
        if ':' in line and not line.strip().endswith(':'):
            key = line.split(':', 1)[0].strip()
            if key in ['title', 'meta_title', 'meta_desc', 'slug', 'keyphrase', 'alt_text', 'hero_image_prompt']:
                # Check if the value is already quoted
                if not ('"' in line or "'" in line):
                    # Add quotes around the value
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        lines[i] = f"{parts[0]}: \"{parts[1].strip()}\""
    
    # Rejoin the lines
    yaml_content = '\n'.join(lines)
    
    try:
        # Parse the YAML
        return yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        print(f"❌ YAML parsing error: {e}")
        # Create a minimal valid YAML as fallback
        return {
            "title": f"Article about {topic}",
            "meta_title": f"Article about {topic}",
            "meta_desc": f"Learn about {topic} in this comprehensive guide",
            "slug": topic.lower().replace(' ', '-'),
            "keyphrase": topic,
            "sections": ["Introduction", "Main Content", "Conclusion"]
        }

def clean_llm_output(text, topic=None):
    """
    Clean up LLM-generated text by removing common artifacts and response phrases.
    """
    # Remove any triple backticks
    text = re.sub(r'```(?:yaml|)\s*', '', text)
    text = re.sub(r'```', '', text)
    
    # Remove common LLM response phrases
    phrases_to_remove = [
        "I hope this helps!",
        "Let me know if you need any further assistance.",
        "Feel free to ask if you have any questions.",
        "Hope this helps!",
        "Let me know if you need anything else.",
        "Here is the YAML",
        "Here's the YAML",
        "Here is a YAML",
        "Here's a YAML",
        "I hope this meets your requirements!",
        "front-matter with thoughtful field values:",
        "Here is the YAML front-matter",
        "Here's the YAML front-matter"
    ]
    
    for phrase in phrases_to_remove:
        text = text.replace(phrase, '')
    
    # Remove any extra newlines that might have been created
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Replace placeholders with actual topic if provided
    if topic:
        text = text.replace("[Topic]", topic)
        text = text.replace("[topic]", topic)
    
    return text

def generate_blog_components(topic):
    """
    Generate blog components including metadata and content.
    """
    content = ""
    comeco = False
    #Create a cache directory if it doesn't exist
    os.makedirs(".cache/posts", exist_ok=True)
    cache_key = hashlib.sha256(topic.encode()).hexdigest()
    cache_path = f".cache/posts/{cache_key}.yaml"
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cached = f.read()
        if "IMAGE_PROMPT:" not in cached:
            print("CACHE MISS - Re-generating Image")
        else:
            print(f"💾 Cached blog retrieved for: {topic}")
            return cached

    # Generate metadata front matter AND outline via a single LLM call
    meta_prompt = f"""
You are a witty and conversational tech blogger crafting a tutorial on "{topic}." Using up to 500 tokens, output valid YAML front-matter fenced with triple dashes. Fill in each field thoughtfully—no placeholders. Also:
- Suggest a `hero_image_prompt` for the article's header.
- Include a list field `inline_image_prompts` for images placed within sections.
- Include a list field `sections` with EXACTLY 3 witty, sarcastic section headings for a parody article on this topic.

---
title: ""
meta_title: ""
meta_desc: ""
slug: ""
keyphrase: ""
synonyms: []
categories: []
tags: []
hero_image_prompt: ""
inline_image_prompts: []
alt_text: ""
sections:
  - "Ridiculous Introduction"
  - "Absurd Claims"
  - "Conclusion Full of Regret"
---
"""
    try:
        # Use the generate_text_with_ollama function instead of direct HTTP request
        meta_data = generate_text_with_ollama(
            prompt=meta_prompt,
            model="llama3:8b",
            temperature=0.5,
            max_tokens=500,
            stream=False,
            device="cuda"
        )
        meta_text = meta_data.get("response", "").strip()
        print(f"🔍 [DEBUG] meta_data from LLM (first 300 chars):\n{meta_text[:300]}")
    except Exception as e:
        print(f"❌ Failed to generate metadata: {e}")
        meta_text = f"""---
title: "Article about {topic}"
meta_title: "Article about {topic}"
meta_desc: "Learn about {topic} in this comprehensive guide"
slug: "{topic.lower().replace(' ', '-')}"
keyphrase: "{topic}"
synonyms: []
categories: []
tags: []
hero_image_prompt: "A beautiful illustration of {topic}"
inline_image_prompts: []
alt_text: "Illustration of {topic}"
sections:
  - "Introduction"
  - "Main Content"
  - "Conclusion"
---
"""
    
    # Parse the metadata to extract front matter and sections
    try:
        # Extract YAML content between triple dashes
        match = re.search(r'^\s*---\s*\n(.*?)(?:\n---\s*|$)', meta_text, flags=re.S | re.MULTILINE)
        if match:
            yaml_content = match.group(1)
            # Parse YAML safely to handle special characters
            parsed_yaml = _parse_yaml_safely(yaml_content, topic)
            outline = parsed_yaml.get('sections', [])
            # Ensure we have exactly 3 sections
            outline = outline[:3]
            if len(outline) < 3:
                # Add default sections if needed
                default_sections = ["Introduction", "Main Content", "Conclusion"]
                outline.extend(default_sections[len(outline):3])
            print(f"🔍 Extracted sections: {outline}")
            
            # Extract the content after the front matter
            content_after_frontmatter = _strip_frontmatter(meta_text)
            # Clean the content
            content_after_frontmatter = clean_llm_output(content_after_frontmatter, topic)
            
            # Start full_raw with just the YAML front matter
            full_raw = f"---\n{yaml_content}\n---\n"
        else:
            # Fallback if no YAML found
            outline = ["Introduction", "Main Content", "Conclusion"]
            content_after_frontmatter = ""
            full_raw = meta_text
    except Exception as e:
        print(f"⚠️ Failed to parse sections from YAML: {e}")
        outline = ["Introduction", "Main Content", "Conclusion"]
        content_after_frontmatter = ""
        full_raw = meta_text
    
    # Initialize chaining context
    context_accum = full_raw
    
    # For each section heading, generate its content
    for section in outline:
        section_prompt = f"""
Write the next section titled "{section}" in a friendly, engaging style—imagine you're explaining to a curious friend. 
Use smooth transitions, a bit of humor, and emphasize clarity.

Your output should include:
- At least one **specific comparison, benchmark, stat, or quantified insight** (real or plausible) relevant to the topic.
- A **real-world use case or anecdote** that illustrates the core point or claim.
- Avoid vague or generic claims—ground the section in reality with a concrete example, data point, or mini-case study.
- It's okay to be witty or over-the-top, but never at the expense of clarity or informativeness.

When it fits naturally(don't over use it), insert image placeholders like [IMAGE: description of scene]. Only output the section content.

The topic is: {topic}
"""     
        # Minimal feedback for section generation
        print(f"🔨 Generating section content: {section}")
        try:
            section_data = generate_text_with_ollama(
                prompt=section_prompt,
                model="llama3:8b",
                temperature=0.7,
                max_tokens=1200,
                stream=False,
                device="cuda"
            )
            section_text = section_data.get("response", "").strip()
        except Exception as e:
            print(f"❌ Failed to generate section content: {e}")
            section_text = f"# {section}\n\nThis section content could not be generated due to an error."
            
        # Sanitize markdown headings or bold lines that could break YAML
        section_text = re.sub(r'^\s*\*\*(.*?)\*\*', r'\1', section_text, flags=re.MULTILINE)
        section_text = re.sub(r'^#+\s*(.*)', r'\1', section_text, flags=re.MULTILINE)
        
        # Clean the section text
        section_text = clean_llm_output(section_text, topic)
        
        # Remove a repeated section heading if present as first line
        lines = section_text.splitlines()
        if lines and lines[0].strip().startswith(section):
            section_text = "\n".join(lines[1:]).strip()
            
        # Append each section to full_raw (metadata remains at top)
        full_raw += f"\n## {section}\n\n{section_text}\n"
    
    # Write full_raw to cache and use as the response_text
    with open(cache_path, "w") as f:
        f.write(full_raw)
    
    # Return the cleaned content (with [IMAGE: ...] placeholders intact)
    return full_raw

def generate_image(prompt, negative_prompt="", width=512, height=512, steps=30, seed=None):
    """
    Generate an image based on a text prompt.
    
    Uses the following methods in order of preference:
    1. If running on Apple Silicon, uses MLX Core with the Neural Processing Unit (NPU).
    2. If local Ollama is available with a diffusion model, uses that.
    3. Otherwise, falls back to the AUTOMATIC1111 API.
    """
    # Create cache directory if it doesn't exist
    os.makedirs(".cache/images", exist_ok=True)
    
    # Create a cache key based on the prompt and parameters
    cache_key = hashlib.sha256(f"{prompt}_{negative_prompt}_{width}_{height}_{steps}_{seed}".encode()).hexdigest()
    
    # Check if the cache file exists
    png_path = f".cache/images/{cache_key}.png"
    webp_path = f".cache/images/{cache_key}.webp"
    if os.path.exists(webp_path):
        print("🖼️ Cached WebP image used")
        return webp_path
    elif os.path.exists(png_path):
        print("🖼️ Cached PNG image used")
        return png_path
    
    # Try Apple Silicon first if available
    if is_apple_silicon():
        try:
            print("🍎 Using Apple Silicon NPU for image generation")
            image_path = generate_image_with_mlx(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                seed=seed
            )
            if image_path:
                return image_path
            print("⚠️ Apple Silicon generation failed, trying Ollama")
        except Exception as e:
            print(f"⚠️ Error with Apple Silicon generation: {e}")
    
    # Try Ollama next if available
    try:
        print("🦙 Using Ollama for image generation")
        image_path = generate_image_with_ollama(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            seed=seed
        )
        if image_path:
            return image_path
        print("⚠️ Ollama generation failed, falling back to AUTOMATIC1111")
    except Exception as e:
        print(f"⚠️ Error with Ollama generation: {e}")
    
    # Fall back to AUTOMATIC1111 API
    if SD_API_BASE:
        try:
            print("🖌️ Using AUTOMATIC1111 API for image generation")
            # Check if the cache file exists
            webp_path = f".cache/images/{cache_key}.webp"
            if os.path.exists(webp_path):
                print("🖼️ Cached WebP image used")
                return webp_path
                
            # Prepare the payload for AUTOMATIC1111 API
            payload = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "steps": steps,
                "cfg_scale": 7.5,
                "sampler_name": "DPM++ 2M Karras",
            }
            
            if seed is not None:
                payload["seed"] = seed
                
            # Make the API request
            response = requests.post(
                f"{SD_API_BASE}/sdapi/v1/txt2img",
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            
            # Process the response
            data = response.json()
            if "images" in data and data["images"]:
                # Decode the base64 image
                image_data = base64.b64decode(data["images"][0])
                image = Image.open(BytesIO(image_data))
                
                # Save the image
                png_path = f".cache/images/{cache_key}.png"
                image.save(png_path)
                
                # Convert to WebP for better compression
                webp_image = image.resize((384, 384), Image.Resampling.LANCZOS)
                webp_image.save(webp_path, format="WEBP", quality=85)
                
                return webp_path
            else:
                print("❌ No image data in AUTOMATIC1111 response")
                return None
        except Exception as e:
            print(f"❌ Failed to generate image with AUTOMATIC1111: {e}")
            return None
    else:
        print("❌ No image generation method available")
        return None

def upload_image_to_wordpress(image_path):
    """
    Upload an image to WordPress and return the media ID.
    """
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return None
    
    try:
        # Prepare the image data
        with open(image_path, 'rb') as img:
            image_data = img.read()
        
        # Set up the request
        headers = {
            'Authorization': f'Basic {base64.b64encode(f"{WP_USER}:{WP_PASS}".encode()).decode()}'
        }
        
        # Upload the image
        response = requests.post(
            f"{WP_URL}/wp-json/wp/v2/media",
            headers=headers,
            files={
                'file': (os.path.basename(image_path), image_data)
            }
        )
        
        if response.status_code in (201, 200):
            return response.json().get('id')
        else:
            print(f"❌ Failed to upload image: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error uploading image: {e}")
        return None

def create_or_update_post(post_data):
    """
    Create or update a WordPress post and return the post ID.
    """
    try:
        # Set up the request
        headers = {
            'Authorization': f'Basic {base64.b64encode(f"{WP_USER}:{WP_PASS}".encode()).decode()}',
            'Content-Type': 'application/json'
        }
        
        # Check if post with the same slug exists
        slug = post_data.get('slug')
        response = requests.get(
            f"{WP_URL}/wp-json/wp/v2/posts?slug={slug}",
            headers=headers
        )
        
        if response.status_code == 200 and response.json():
            # Update existing post
            post_id = response.json()[0]['id']
            response = requests.post(
                f"{WP_URL}/wp-json/wp/v2/posts/{post_id}",
                headers=headers,
                json=post_data
            )
        else:
            # Create new post
            response = requests.post(
                f"{WP_URL}/wp-json/wp/v2/posts",
                headers=headers,
                json=post_data
            )
        
        if response.status_code in (201, 200):
            return response.json().get('id')
        else:
            print(f"❌ Failed to create/update post: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error creating/updating post: {e}")
        return None

def update_seo_metadata(post_id, seo_data):
    """
    Update SEO metadata for a WordPress post using Yoast SEO REST API.
    """
    try:
        # Set up the request
        headers = {
            'Authorization': f'Basic {base64.b64encode(f"{WP_USER}:{WP_PASS}".encode()).decode()}',
            'Content-Type': 'application/json'
        }
        
        # Update SEO metadata
        response = requests.post(
            f"{WP_URL}/wp-json/yoast/v1/update_meta",
            headers=headers,
            json={
                'post_id': post_id,
                'data': seo_data
            }
        )
        
        if response.status_code != 200:
            print(f"⚠️ Failed to update SEO metadata: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"⚠️ Error updating SEO metadata: {e}")

def main():
    """
    Main function to generate a blog post based on command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate a blog post with optional image generation")
    parser.add_argument("--idea", type=str, help="Topic or idea for the blog post")
    parser.add_argument("--keyphrase", type=str, help="SEO keyphrase for the blog post")
    parser.add_argument("--days", type=int, default=0, help="Days to schedule the post in the future")
    parser.add_argument("--no-images", action="store_true", help="Skip image generation")
    args = parser.parse_args()
    
    if not args.idea:
        print("❌ Please provide a topic or idea using the --idea argument")
        sys.exit(1)
    
    # Generate blog components
    print(f"🖋️ Generating blog post for topic: {args.idea}")
    blog_content = generate_blog_components(args.idea)
    
    # Extract YAML front matter
    try:
        post = frontmatter.loads(blog_content)
        metadata = post.metadata
        print(f"✅ Generated blog post with title: {metadata.get('title', 'Untitled')}")
        
        # Generate hero image if not disabled
        if not args.no_images and 'hero_image_prompt' in metadata:
            hero_prompt = metadata['hero_image_prompt']
            print(f"🖼️ Generating hero image with prompt: {hero_prompt}")
            image_path = generate_image(hero_prompt)
            if image_path:
                print(f"✅ Hero image generated: {image_path}")
            else:
                print("⚠️ Failed to generate hero image")
        
        # Upload to WordPress if credentials are available
        if WP_URL and WP_USER and WP_PASS:
            print("📝 Uploading post to WordPress...")
            
            # Process content for WordPress
            content = post.content
            
            # Upload any inline images if needed
            if not args.no_images and 'inline_image_prompts' in metadata:
                for i, prompt in enumerate(metadata.get('inline_image_prompts', [])):
                    image_path = generate_image(prompt)
                    if image_path:
                        # Upload image to WordPress
                        image_id = upload_image_to_wordpress(image_path)
                        if image_id:
                            # Replace placeholder with image
                            placeholder = f"[IMAGE: {prompt}]"
                            if placeholder in content:
                                content = content.replace(placeholder, f'<figure><img src="{WP_URL}/wp-content/uploads/{os.path.basename(image_path)}" alt="{metadata.get("alt_text", "Alternative text descriptions for the images in this tutorial")}"/><figcaption>{prompt}</figcaption></figure>')
            
            # Add footer
            content += '\n<p><strong>Ready to dive deeper?</strong> Check out <a href="https://github.com/felipedbene" target="_blank">my GitHub</a> for more code examples and in-depth tutorials!</p>'
            
            # Prepare post data
            post_data = {
                'title': metadata.get('title', f'Article about {args.idea}'),
                'slug': metadata.get('slug', args.idea.lower().replace(' ', '-')),
                'status': 'draft',
                'content': content,
                'excerpt': metadata.get('meta_desc', f'Learn about {args.idea} in this comprehensive guide'),
                'featured_media': None,  # Will be set if hero image is uploaded
                'categories': [2],  # Default category ID
                'tags': [3]  # Default tag ID
            }
            
            # Upload hero image if available
            if not args.no_images and 'hero_image_prompt' in metadata and image_path:
                image_id = upload_image_to_wordpress(image_path)
                if image_id:
                    post_data['featured_media'] = image_id
            
            # Create or update post
            post_id = create_or_update_post(post_data)
            
            if post_id:
                # Update SEO metadata if needed
                if 'meta_title' in metadata or 'meta_desc' in metadata:
                    seo_data = {
                        'title': metadata.get('meta_title', metadata.get('title', '')),
                        'metadesc': metadata.get('meta_desc', ''),
                        'focuskw': args.keyphrase or metadata.get('keyphrase', '')
                    }
                    update_seo_metadata(post_id, seo_data)
                    print("🔧 SEO metadata patched successfully.")
                
                # Get the post URL
                post_url = f"{WP_URL}/?p={post_id}"
                print(f"✅ Published: {post_url}")
            else:
                print("❌ Failed to publish post to WordPress")
        else:
            print("⚠️ WordPress credentials not found. Skipping upload.")
    except Exception as e:
        print(f"❌ Error processing blog content: {e}")
    
    print("✅ Blog post generation complete")

if __name__ == "__main__":
    main()
