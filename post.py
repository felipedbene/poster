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
        # Skip empty lines or lines that are just indentation
        if not line.strip() or line.strip().startswith('-') or line.strip().startswith('#'):
            continue
            
        # Skip lines that don't have a key-value structure
        if ':' not in line:
            continue
            
        # Get the key part (before the first colon)
        key_part = line.split(':', 1)[0].strip()
        
        # Check if this is a key we want to process
        if key_part in ['title', 'meta_title', 'meta_desc', 'slug', 'keyphrase', 'alt_text', 'hero_image_prompt']:
            # Count the number of colons in the line
            colon_count = line.count(':')
            
            # If there's more than one colon or the line doesn't end with a colon, we need to quote the value
            if colon_count > 1 or not line.strip().endswith(':'):
                # Check if the value is already quoted (with either single or double quotes)
                value_part = line.split(':', 1)[1].strip()
                if not (value_part.startswith('"') and value_part.endswith('"')) and not (value_part.startswith("'") and value_part.endswith("'")):
                    # Add quotes around the entire value part
                    lines[i] = f"{key_part}: \"{value_part}\""
    
    # Rejoin the lines
    yaml_content = '\n'.join(lines)
    
    try:
        # Parse the YAML
        parsed_yaml = yaml.safe_load(yaml_content)
        
        # Ensure categories and tags are lists
        if 'categories' in parsed_yaml and not isinstance(parsed_yaml['categories'], list):
            if parsed_yaml['categories'] is None:
                parsed_yaml['categories'] = []
            else:
                parsed_yaml['categories'] = [parsed_yaml['categories']]
                
        if 'tags' in parsed_yaml and not isinstance(parsed_yaml['tags'], list):
            if parsed_yaml['tags'] is None:
                parsed_yaml['tags'] = []
            else:
                parsed_yaml['tags'] = [parsed_yaml['tags']]
                
        return parsed_yaml
    except yaml.YAMLError as e:
        print(f"❌ YAML parsing error: {e}")
        # Create a minimal valid YAML as fallback
        return {
            "title": f"Article about {topic}",
            "meta_title": f"Article about {topic}",
            "meta_desc": f"Learn about {topic} in this comprehensive guide",
            "slug": topic.lower().replace(' ', '-'),
            "keyphrase": topic,
            "categories": ["Technology", "articles"],
            "tags": [topic, "guide", "article", "how-to"],
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
    Generate blog components including metadata and content in a single LLM call.
    """
    # Create a cache directory if it doesn't exist
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

    # Generate complete blog post with a single LLM call
    blog_prompt = f"""
You are a senior technical writer crafting an in-depth, comprehensive article about "{topic}". Your goal is to create a detailed, well-researched piece that provides real value to readers. Follow these guidelines:

1. Start with YAML front-matter between triple dashes (---) containing:
   - title: A compelling, technical title that captures the essence
   - meta_title: SEO-optimized title with technical keywords
   - meta_desc: Detailed meta description that summarizes the article's value
   - slug: URL-friendly version of the title
   - keyphrase: The main topic "{topic}" as the primary keyphrase, followed by 2-3 related technical terms
   - synonyms: List of related technical terms and variations
   - categories: Choose from these exact categories: ["Technology", "Programming", "AI & ML", "Web Development", "Data Science", "Cloud Computing", "Cybersecurity", "DevOps", "Mobile Development", "Blockchain"]
   - tags: 5-7 specific technical tags related to the topic, using standard tech terminology
   - hero_image_prompt: A detailed description for a high-tech header image that directly relates to the topic. Focus on technical visualization, architecture diagrams, or conceptual representations.
   - inline_image_prompts: List of 3-4 specific prompts for images that illustrate key concepts in the article. Each prompt should be detailed and directly related to the content it accompanies.
   - alt_text: Detailed descriptions for images
   - sections: List of 4-5 comprehensive section headings that cover the topic in depth

2. After the front-matter, write a complete article that includes:
   - A compelling introduction that establishes context and importance
   - 4-5 detailed sections, each 300-500 words
   - Technical depth with specific examples, code snippets, or architecture details
   - Real-world use cases and practical applications
   - Performance considerations and best practices
   - Troubleshooting tips or common pitfalls
   - Future trends or upcoming developments
   - Clear transitions between sections
   - Image placeholders using [IMAGE: description] format, with each image directly supporting the surrounding content
   - Technical diagrams or visualizations where appropriate
   - A strong conclusion that summarizes key points and next steps

3. Content Requirements:
   - Minimum 2000 words total
   - Each section should be substantial (300-500 words)
   - Include specific technical details, not just general information
   - Use real-world examples and case studies
   - Provide actionable insights and practical advice
   - Include relevant technical comparisons or benchmarks
   - Address common challenges and their solutions
   - Maintain a professional but engaging tone

4. Image Guidelines:
   - Each image must directly support and illustrate the surrounding content
   - Hero image should represent the main concept or architecture
   - Inline images should visualize specific technical concepts
   - Use [IMAGE: description] format with detailed, technical descriptions
   - Ensure images flow naturally with the content
   - Each image should have a clear purpose in the article

Example format:
---
title: "Advanced Topic Implementation Guide"
meta_title: "Comprehensive Guide to Topic: Architecture, Best Practices, and Implementation"
meta_desc: "Learn how to implement Topic effectively with detailed architecture, performance considerations, and real-world examples"
slug: "advanced-topic-implementation"
keyphrase: "topic implementation architecture best practices"
synonyms: ["implementation", "architecture", "deployment", "optimization"]
categories: ["Technology", "Programming"]
tags: ["implementation", "architecture", "performance", "optimization", "best practices"]
hero_image_prompt: "A detailed technical architecture diagram showing the components and data flow of the system, with clear labels and connections"
inline_image_prompts: [
    "A sequence diagram showing the interaction between system components",
    "A performance comparison chart showing different implementation approaches",
    "A troubleshooting flowchart for common issues"
]
alt_text: "Technical architecture and implementation diagrams"
sections:
  - "Understanding the Core Concepts"
  - "Architecture and Implementation"
  - "Performance Optimization"
  - "Common Challenges and Solutions"
  - "Best Practices and Future Trends"
---

## Understanding the Core Concepts
[Detailed content with specific examples and [IMAGE: description] placeholders]

## Architecture and Implementation
[Technical details with code examples and [IMAGE: description] placeholders]

## Performance Optimization
[Performance considerations with benchmarks and [IMAGE: description] placeholders]

## Common Challenges and Solutions
[Troubleshooting guide with real-world examples and [IMAGE: description] placeholders]

## Best Practices and Future Trends
[Actionable insights and forward-looking analysis with [IMAGE: description] placeholders]
"""
    try:
        # Use the generate_text_with_ollama function for a single call
        blog_data = generate_text_with_ollama(
            prompt=blog_prompt,
            model="llama3:8b",
            temperature=0.7,
            max_tokens=4000,  # Increased token limit for more detailed content
            stream=False,
            device="cuda"
        )
        blog_text = blog_data.get("response", "").strip()
        print(f"🔍 [DEBUG] blog_data from LLM (first 300 chars):\n{blog_text[:300]}")
        
        # Clean up the content
        # Remove any markdown image syntax
        blog_text = re.sub(r'!\[.*?\]\(.*?\)', '', blog_text)
        
        # Fix YAML formatting
        # Remove any leading/trailing whitespace around the YAML section
        blog_text = re.sub(r'^\s*---\s*\n', '---\n', blog_text)
        blog_text = re.sub(r'\n\s*---\s*\n', '\n---\n', blog_text)
        
        # Ensure proper YAML indentation
        lines = blog_text.split('\n')
        in_yaml = False
        yaml_lines = []
        content_lines = []
        
        for line in lines:
            if line.strip() == '---':
                in_yaml = not in_yaml
                yaml_lines.append(line)
                continue
                
            if in_yaml:
                # Ensure proper YAML indentation
                if line.strip() and not line.startswith(' '):
                    line = ' ' + line
                yaml_lines.append(line)
            else:
                content_lines.append(line)
        
        # Reconstruct the blog text with proper formatting
        blog_text = '\n'.join(yaml_lines) + '\n\n' + '\n'.join(content_lines)
        
        # Validate and fix YAML if needed
        try:
            # Extract YAML content between triple dashes
            match = re.search(r'^\s*---\s*\n(.*?)(?:\n---\s*|$)', blog_text, flags=re.S | re.MULTILINE)
            if match:
                yaml_content = match.group(1)
                # Parse YAML safely to handle special characters
                parsed_yaml = _parse_yaml_safely(yaml_content, topic)
                
                # Ensure all required fields are present
                required_fields = ['title', 'meta_title', 'meta_desc', 'slug', 'keyphrase', 
                                 'categories', 'tags', 'hero_image_prompt', 'sections']
                for field in required_fields:
                    if field not in parsed_yaml:
                        if field == 'categories':
                            parsed_yaml[field] = ["Technology", "articles"]
                        elif field == 'tags':
                            parsed_yaml[field] = [topic, "guide", "article", "how-to"]
                        elif field == 'sections':
                            parsed_yaml[field] = ["Introduction", "Main Content", "Conclusion"]
                        else:
                            parsed_yaml[field] = f"Default {field}"
                
                # Reconstruct the blog text with valid YAML
                yaml_str = yaml.dump(parsed_yaml, default_flow_style=False, sort_keys=False)
                content_after_yaml = _strip_frontmatter(blog_text)
                blog_text = f"---\n{yaml_str}\n---\n{content_after_yaml}"
                
        except Exception as e:
            print(f"⚠️ Error parsing YAML, using fallback: {e}")
            # Create a minimal valid blog post as fallback
            blog_text = f"""---
title: "Article about {topic}"
meta_title: "Article about {topic}"
meta_desc: "Learn about {topic} in this comprehensive guide"
slug: "{topic.lower().replace(' ', '-')}"
keyphrase: "{topic}"
synonyms: []
categories: ["Technology", "articles"]
tags: ["{topic}", "guide", "article", "how-to"]
hero_image_prompt: "A beautiful illustration of {topic}"
inline_image_prompts: []
alt_text: "Illustration of {topic}"
sections:
  - "Introduction"
  - "Main Content"
  - "Conclusion"
---

## Introduction
This is an introduction to {topic}.

## Main Content
Here's the main content about {topic}.

## Conclusion
In conclusion, {topic} is an important topic to understand.
"""
    except Exception as e:
        print(f"❌ Failed to generate blog content: {e}")
        # Create a minimal valid blog post as fallback
        blog_text = f"""---
title: "Article about {topic}"
meta_title: "Article about {topic}"
meta_desc: "Learn about {topic} in this comprehensive guide"
slug: "{topic.lower().replace(' ', '-')}"
keyphrase: "{topic}"
synonyms: []
categories: ["Technology", "articles"]
tags: ["{topic}", "guide", "article", "how-to"]
hero_image_prompt: "A beautiful illustration of {topic}"
inline_image_prompts: []
alt_text: "Illustration of {topic}"
sections:
  - "Introduction"
  - "Main Content"
  - "Conclusion"
---

## Introduction
This is an introduction to {topic}.

## Main Content
Here's the main content about {topic}.

## Conclusion
In conclusion, {topic} is an important topic to understand.
"""

    # Write the generated content to cache
    with open(cache_path, "w") as f:
        f.write(blog_text)
    
    return blog_text

def generate_image(prompt, negative_prompt="", width=512, height=512, steps=30, seed=None):
    """
    Generate an image based on a text prompt.
    
    Uses the following methods in order of preference:
    1. If running on Apple Silicon, uses MLX Core with the Neural Processing Unit (NPU).
    2. Otherwise, falls back to the AUTOMATIC1111 API.
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
            # Add tech-focused style modifiers to the prompt
            tech_prompt = f"{prompt}, ultra-realistic, high-tech, cyberpunk, neon, holographic, futuristic, digital, neural, quantum, AI, robotic, synthetic, ultra-modern, technological, innovative, cutting-edge, 8k, photorealistic, detailed, sharp focus"
            # Add negative prompt to avoid artisan style
            tech_negative_prompt = f"{negative_prompt}, artisan, hand-crafted, watercolor, painting, artistic, sketchy, blurry, low quality, low resolution"
            image_path = generate_image_with_mlx(
                prompt=tech_prompt,
                negative_prompt=tech_negative_prompt,
                width=width,
                height=height,
                steps=steps,
                seed=seed
            )
            if image_path:
                return image_path
            print("⚠️ Apple Silicon generation failed, falling back to AUTOMATIC1111")
        except Exception as e:
            print(f"⚠️ Error with Apple Silicon generation: {e}")
    
    # Fall back to AUTOMATIC1111 API
    if SD_API_BASE:
        try:
            print("🖌️ Using AUTOMATIC1111 API for image generation")
            # Check if the cache file exists
            webp_path = f".cache/images/{cache_key}.webp"
            if os.path.exists(webp_path):
                print("🖼️ Cached WebP image used")
                return webp_path
                
            # Add tech-focused style modifiers to the prompt
            tech_prompt = f"{prompt}, ultra-realistic, high-tech, cyberpunk, neon, holographic, futuristic, digital, neural, quantum, AI, robotic, synthetic, ultra-modern, technological, innovative, cutting-edge, 8k, photorealistic, detailed, sharp focus"
            # Add negative prompt to avoid artisan style
            tech_negative_prompt = f"{negative_prompt}, artisan, hand-crafted, watercolor, painting, artistic, sketchy, blurry, low quality, low resolution"
            
            # Prepare the payload for AUTOMATIC1111 API
            payload = {
                "prompt": tech_prompt,
                "negative_prompt": tech_negative_prompt,
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

def get_or_create_taxonomy_term(taxonomy, term_name):
    """
    Get the ID of an existing taxonomy term or create a new one.
    
    Args:
        taxonomy (str): The taxonomy type ('category' or 'post_tag')
        term_name (str): The name of the term to get or create
        
    Returns:
        int: The ID of the term, or None if failed
    """
    try:
        # Set up the request
        headers = {
            'Authorization': f'Basic {base64.b64encode(f"{WP_USER}:{WP_PASS}".encode()).decode()}',
            'Content-Type': 'application/json'
        }
        
        # For categories, ensure we're using the predefined list
        if taxonomy == 'category':
            valid_categories = [
                "Technology", "Programming", "AI & ML", "Web Development", 
                "Data Science", "Cloud Computing", "Cybersecurity", "DevOps", 
                "Mobile Development", "Blockchain"
            ]
            if term_name not in valid_categories:
                print(f"⚠️ Invalid category: {term_name}. Using default category.")
                return 2  # Default category ID
        
        # First, try to get existing terms
        response = requests.get(
            f"{WP_URL}/wp-json/wp/v2/{taxonomy}",
            headers=headers,
            params={'search': term_name}
        )
        
        if response.status_code == 200:
            terms = response.json()
            # Find exact match
            for term in terms:
                if term['name'].lower() == term_name.lower():
                    print(f"✅ Found existing {taxonomy}: {term_name} (ID: {term['id']})")
                    return term['id']
        
        # If no exact match found, create new term
        print(f"🔄 Creating new {taxonomy}: {term_name}")
        response = requests.post(
            f"{WP_URL}/wp-json/wp/v2/{taxonomy}",
            headers=headers,
            json={
                'name': term_name,
                'slug': term_name.lower().replace(' ', '-')
            }
        )
        
        if response.status_code in (201, 200):
            term_id = response.json().get('id')
            print(f"✅ Created new {taxonomy}: {term_name} (ID: {term_id})")
            return term_id
        else:
            print(f"❌ Failed to create {taxonomy} term: {response.status_code} - {response.text}")
            # Return default IDs for known taxonomies
            if taxonomy == 'category':
                return 2  # Default category ID
            elif taxonomy == 'post_tag':
                return 3  # Default tag ID
            return None
            
    except Exception as e:
        print(f"❌ Error getting/creating {taxonomy} term: {e}")
        # Return default IDs for known taxonomies
        if taxonomy == 'category':
            return 2  # Default category ID
        elif taxonomy == 'post_tag':
            return 3  # Default tag ID
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
            existing_post = response.json()[0]
            
            # Merge categories: combine existing and new categories
            existing_categories = existing_post.get('categories', [])
            new_categories = post_data.get('categories', [])
            
            # If we have new categories, merge them with existing ones (avoiding duplicates)
            if new_categories:
                merged_categories = list(set(existing_categories + new_categories))
                post_data['categories'] = merged_categories
            # If no new categories but existing ones, preserve them
            elif existing_categories:
                post_data['categories'] = existing_categories
            
            # Merge tags: combine existing and new tags
            existing_tags = existing_post.get('tags', [])
            new_tags = post_data.get('tags', [])
            
            # If we have new tags, merge them with existing ones (avoiding duplicates)
            if new_tags:
                merged_tags = list(set(existing_tags + new_tags))
                post_data['tags'] = merged_tags
            # If no new tags but existing ones, preserve them
            elif existing_tags:
                post_data['tags'] = existing_tags
                    
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
        hero_image_path = None
        if not args.no_images and 'hero_image_prompt' in metadata:
            hero_prompt = metadata['hero_image_prompt']
            print(f"🖼️ Generating hero image with prompt: {hero_prompt}")
            hero_image_path = generate_image(hero_prompt)
            if hero_image_path:
                print(f"✅ Hero image generated: {hero_image_path}")
            else:
                print("⚠️ Failed to generate hero image")
        
        # Upload to WordPress if credentials are available
        if WP_URL and WP_USER and WP_PASS:
            print("📝 Uploading post to WordPress...")
            
            # Process content for WordPress
            content = post.content
            # Remove any YAML front matter that snuck into the body
            content = _strip_frontmatter(content)
            # Remove any YAML front matter that snuck into the body
            content = _strip_frontmatter(content)
            
            # First, upload all images that will be needed
            image_paths = {}
            image_ids = {}
            
            # Process inline image prompts from metadata
            if not args.no_images and 'inline_image_prompts' in metadata and metadata['inline_image_prompts']:
                for prompt in metadata.get('inline_image_prompts', []):
                    if prompt not in image_paths:
                        image_path = generate_image(prompt)
                        if image_path:
                            image_paths[prompt] = image_path
                            # Upload image to WordPress
                            image_id = upload_image_to_wordpress(image_path)
                            if image_id:
                                image_ids[prompt] = image_id
            
            # Find all image placeholders in content
            if not args.no_images:
                image_placeholders = re.findall(r'\[IMAGE:\s*(.*?)\]', content)
                for placeholder_text in image_placeholders:
                    if placeholder_text not in image_paths:
                        image_path = generate_image(placeholder_text)
                        if image_path:
                            image_paths[placeholder_text] = image_path
                            # Upload image to WordPress
                            image_id = upload_image_to_wordpress(image_path)
                            if image_id:
                                image_ids[placeholder_text] = image_id
            
            # Now replace all image placeholders with actual images
            for prompt, image_path in image_paths.items():
                if image_path:
                    placeholder = f"[IMAGE: {prompt}]"
                    # Try to get the image ID if available
                    image_id = image_ids.get(prompt)
                    img_url = None
                    # If uploaded to WordPress, get the URL from the media endpoint
                    if image_id:
                        # Query WordPress for the image URL
                        try:
                            headers = {
                                'Authorization': f'Basic {base64.b64encode(f"{WP_USER}:{WP_PASS}".encode()).decode()}'
                            }
                            resp = requests.get(f"{WP_URL}/wp-json/wp/v2/media/{image_id}", headers=headers)
                            if resp.status_code == 200:
                                img_url = resp.json().get('source_url')
                        except Exception as e:
                            print(f"⚠️ Error fetching media URL: {e}")
                    # Fallback: guess the upload path
                    if not img_url:
                        img_url = f"{WP_URL}/wp-content/uploads/{os.path.basename(image_path)}"
                    if placeholder in content:
                        content = content.replace(
                            placeholder,
                            f'<figure><img src="{img_url}" alt="{metadata.get("alt_text", "Alternative text descriptions for the images in this article")}"/><figcaption>{prompt}</figcaption></figure>'
                        )
            
            # Convert markdown to HTML for proper rendering in WordPress
            # First, handle section headings (## Heading) to proper HTML
            content = re.sub(r'##\s+(.*?)(?=\n|$)', r'<h2>\1</h2>', content)
            content = re.sub(r'###\s+(.*?)(?=\n|$)', r'<h3>\1</h3>', content)
            
            # Handle other markdown elements like bold, italic, etc.
            content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
            content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', content)
            
            # Convert paragraphs (double newlines)
            paragraphs = content.split('\n\n')
            content = ''
            for p in paragraphs:
                if p.strip() and not p.strip().startswith('<'):
                    content += f'<p>{p.strip()}</p>\n\n'
                else:
                    content += f'{p}\n\n'
            
            # Add footer
            content += '\n<p><strong>Ready to dive deeper?</strong> Check out <a href="https://github.com/felipedbene" target="_blank">my GitHub</a> for more code examples and in-depth articles!</p>'
            
            # Process categories and tags
            categories = []
            tags = []
            
            # Handle categories from metadata
            if 'categories' in metadata and metadata['categories']:
                # Get category IDs or create new categories
                for category_name in metadata['categories']:
                    category_id = get_or_create_taxonomy_term('category', category_name)
                    if category_id:
                        categories.append(category_id)
                print(f"📂 Categories: {metadata['categories']} -> IDs: {categories}")
            
            # Handle tags from metadata
            if 'tags' in metadata and metadata['tags']:
                # Get tag IDs or create new tags
                for tag_name in metadata['tags']:
                    tag_id = get_or_create_taxonomy_term('post_tag', tag_name)
                    if tag_id:
                        tags.append(tag_id)
                print(f"🏷️ Tags: {metadata['tags']} -> IDs: {tags}")
            
            # Use default category if no categories were found
            if not categories:
                categories = [2]  # Default category ID
                print("⚠️ Using default category (ID: 2)")
                
            # Use default tag if no tags were found
            if not tags:
                tags = [3]  # Default tag ID
                print("⚠️ Using default tag (ID: 3)")
            
            # Prepare post data
            post_data = {
                'title': metadata.get('title', f'Article about {args.idea}'),
                'slug': metadata.get('slug', args.idea.lower().replace(' ', '-')),
                'status': 'draft',
                'content': content,
                'excerpt': metadata.get('meta_desc', f'Learn about {args.idea} in this comprehensive guide'),
                'featured_media': None,  # Will be set if hero image is uploaded
                'categories': categories,
                'tags': tags
            }
            
            # Upload hero image if available
            if not args.no_images and 'hero_image_prompt' in metadata and hero_image_path:
                image_id = upload_image_to_wordpress(hero_image_path)
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