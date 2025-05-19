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
import inspect
from dotenv import load_dotenv
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.codehilite import CodeHiliteExtension
from markdown import markdown
import yaml
from typing import Tuple, Dict, Optional
import frontmatter
from PIL import Image
from io import BytesIO
import base64

# Import Apple-specific utilities
from apple_utils import is_apple_silicon, generate_image_with_mlx

# mlx.llm for local model inference
try:
    from mlx_lm import load as load_model, generate as generate_llm
except ImportError:  # pragma: no cover - runtime environment may not have mlx_lm
    load_model = generate_llm = None

# Load .env, overriding existing environment variables
load_dotenv(override=True)

# Define global variables from environment
WP_USER = os.getenv("WORDPRESS_USERNAME")
WP_PASS = os.getenv("WORDPRESS_APP_PASSWORD")
WP_URL = os.getenv("WORDPRESS_URL")
HC_APIKEY = os.getenv("HC_APIKEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
LLM_MODEL = os.getenv("MLX_LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")


def _strip_frontmatter(text):
    """
    Strip YAML front matter from text.
    """
    pattern = r"^\s*---\s*\n.*?\n---\s*\n"
    return re.sub(pattern, "", text, flags=re.DOTALL)


def _parse_yaml_safely(yaml_content, topic):
    """
    Parse YAML content safely, handling special characters and replacing placeholders.
    """
    # Replace placeholders in the YAML content
    yaml_content = yaml_content.replace("{topic}", topic)

    # Handle colons in titles by quoting them
    lines = yaml_content.split("\n")
    for i, line in enumerate(lines):
        if ":" in line and not line.strip().endswith(":"):
            key = line.split(":", 1)[0].strip()
            if key in [
                "title",
                "meta_title",
                "meta_desc",
                "slug",
                "keyphrase",
                "alt_text",
                "hero_image_prompt",
            ]:
                # Check if the value is already quoted
                if not ('"' in line or "'" in line):
                    # Add quotes around the value
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        lines[i] = f'{parts[0]}: "{parts[1].strip()}"'

    # Rejoin the lines
    yaml_content = "\n".join(lines)

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
            "slug": topic.lower().replace(" ", "-"),
            "keyphrase": topic,
            "sections": ["Introduction", "Main Content", "Conclusion"],
        }


def clean_llm_output(text, topic=None):
    """
    Clean up LLM-generated text by removing common artifacts and response phrases.
    """
    # Remove any triple backticks
    text = re.sub(r"```(?:yaml|)\s*", "", text)
    text = re.sub(r"```", "", text)

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
        "Here's the YAML front-matter",
    ]

    for phrase in phrases_to_remove:
        text = text.replace(phrase, "")

    # Remove any extra newlines that might have been created
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Replace placeholders with actual topic if provided
    if topic:
        text = text.replace("[Topic]", topic)
        text = text.replace("[topic]", topic)

    return text


def generate_blog_components(topic):
    """Generate an entire blog post in a single LLM call."""
    os.makedirs(".cache/posts", exist_ok=True)
    cache_key = hashlib.sha256(topic.encode()).hexdigest()
    cache_path = f".cache/posts/{cache_key}.yaml"

    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cached = f.read()
        print(f"💾 Cached blog retrieved for: {topic}")
        return cached

    prompt = f"""
Write a comprehensive, informative blog post about "{topic}" with a professional yet engaging tone. Output the result as a single Markdown file.

Start with a Hugo/Jekyll-style frontmatter block using `---`. Include the following fields:
- `title`: an attention-grabbing, SEO-friendly title related to the topic
- `meta_title`: a concise and click-worthy meta title (under 60 characters)
- `meta_desc`: a compelling meta description that encourages clicks (under 155 characters)
- `slug`: a URL-safe version of the title
- `keyphrase`: the main SEO keyword or phrase
- `hero_image_prompt`: a detailed prompt for an image that captures the essence of the article
- `inline_image_prompts`: a list of 2 prompts for supporting images that illustrate key points
- `alt_text`: accessible alt text describing the hero image

After the frontmatter, write a well-structured blog post with at least 1000 words. Include:
1. An engaging introduction that hooks the reader
2. At least three main sections with descriptive headings
3. Practical examples, tips, or actionable advice
4. A conclusion that summarizes key points and provides next steps

Use a professional but conversational tone throughout. Include relevant statistics, examples, and insights where appropriate. Format the content with proper Markdown, including lists, bold text for emphasis, and code blocks if relevant.

Return the entire output as a single Markdown file, and nothing else."""

    if load_model is None or generate_llm is None:
        raise RuntimeError("mlx.lm is required for text generation")

    model, tokenizer = load_model(LLM_MODEL)

    kwargs = {
        "max_tokens": 1800,
        "verbose": False,
    }

    # Newer versions of mlx_lm expect a "temperature" argument while older
    # versions use "temp". Detect the supported name at runtime to avoid
    # TypeError: generate_step() got an unexpected keyword argument
    sig = inspect.signature(generate_llm)
    if "temperature" in sig.parameters:
        kwargs["temperature"] = 0.6
    elif "temp" in sig.parameters:
        kwargs["temp"] = 0.6

    text = generate_llm(model, tokenizer, prompt, **kwargs).strip()
    with open(cache_path, "w") as f:
        f.write(text)
    return text


def generate_image(
    prompt, negative_prompt="", width=512, height=512, steps=30, seed=None
):
    """
    Generate an image based on a text prompt.

    Generate the image exclusively using the Apple NPU via MLX.
    """
    # Create cache directory if it doesn't exist
    os.makedirs(".cache/images", exist_ok=True)

    # Create a cache key based on the prompt and parameters
    cache_key = hashlib.sha256(
        f"{prompt}_{negative_prompt}_{width}_{height}_{steps}_{seed}".encode()
    ).hexdigest()

    # Check if the cache file exists
    png_path = f".cache/images/{cache_key}.png"
    webp_path = f".cache/images/{cache_key}.webp"
    if os.path.exists(webp_path):
        print("🖼️ Cached WebP image used")
        return webp_path
    elif os.path.exists(png_path):
        print("🖼️ Cached PNG image used")
        return png_path

    if is_apple_silicon():
        try:
            print("🍎 Using Apple Silicon NPU for image generation")
            image_path = generate_image_with_mlx(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                seed=seed,
            )
            if image_path:
                return image_path
        except Exception as e:
            print(f"❌ Apple NPU image generation failed: {e}")
    else:
        print("❌ Apple NPU not available")

    return None


def upload_image_to_wordpress(image_path):
    """Upload an image to WordPress and return its ID and URL."""
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return None

    try:
        # Prepare the image data
        with open(image_path, "rb") as img:
            image_data = img.read()

        # Set up the request
        headers = {
            "Authorization": f'Basic {base64.b64encode(f"{WP_USER}:{WP_PASS}".encode()).decode()}'
        }

        # Upload the image
        response = requests.post(
            f"{WP_URL}/wp-json/wp/v2/media",
            headers=headers,
            files={"file": (os.path.basename(image_path), image_data)},
        )

        if response.status_code in (201, 200):
            data = response.json()
            return data.get("id"), data.get("source_url")
        else:
            print(
                f"❌ Failed to upload image: {response.status_code} - {response.text}"
            )
            return None, None
    except Exception as e:
        print(f"❌ Error uploading image: {e}")
        return None, None


def create_or_update_post(post_data):
    """
    Create or update a WordPress post and return the post ID.
    """
    try:
        # Set up the request
        headers = {
            "Authorization": f'Basic {base64.b64encode(f"{WP_USER}:{WP_PASS}".encode()).decode()}',
            "Content-Type": "application/json",
        }

        # Check if post with the same slug exists
        slug = post_data.get("slug")
        response = requests.get(
            f"{WP_URL}/wp-json/wp/v2/posts?slug={slug}", headers=headers
        )

        if response.status_code == 200 and response.json():
            # Update existing post
            post_id = response.json()[0]["id"]
            response = requests.post(
                f"{WP_URL}/wp-json/wp/v2/posts/{post_id}",
                headers=headers,
                json=post_data,
            )
        else:
            # Create new post
            response = requests.post(
                f"{WP_URL}/wp-json/wp/v2/posts", headers=headers, json=post_data
            )

        if response.status_code in (201, 200):
            return response.json().get("id")
        else:
            print(
                f"❌ Failed to create/update post: {response.status_code} - {response.text}"
            )
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
            "Authorization": f'Basic {base64.b64encode(f"{WP_USER}:{WP_PASS}".encode()).decode()}',
            "Content-Type": "application/json",
        }

        # Update SEO metadata
        endpoint = f"{WP_URL}/wp-json/yoast/v1/update_meta"
        response = requests.post(
            endpoint,
            headers=headers,
            json={"post_id": post_id, "data": seo_data},
        )

        if response.status_code == 404:
            # Some Yoast SEO setups use a hyphen instead of an underscore
            alt_endpoint = f"{WP_URL}/wp-json/yoast/v1/update-meta"
            alt_response = requests.post(
                alt_endpoint,
                headers=headers,
                json={"post_id": post_id, "data": seo_data},
            )

            if alt_response.status_code != 200:
                print(
                    f"⚠️ Failed to update SEO metadata: {alt_response.status_code} - {alt_response.text}"
                )
        elif response.status_code != 200:
            print(
                f"⚠️ Failed to update SEO metadata: {response.status_code} - {response.text}"
            )
    except Exception as e:
        print(f"⚠️ Error updating SEO metadata: {e}")


def main():
    """
    Main function to generate a blog post based on command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate a blog post with optional image generation"
    )
    parser.add_argument("--idea", type=str, help="Topic or idea for the blog post")
    parser.add_argument("--keyphrase", type=str, help="SEO keyphrase for the blog post")
    parser.add_argument(
        "--days", type=int, default=0, help="Days to schedule the post in the future"
    )
    parser.add_argument(
        "--no-images", action="store_true", help="Skip image generation"
    )
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
        if not args.no_images and "hero_image_prompt" in metadata:
            hero_prompt = metadata["hero_image_prompt"]
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
            if not args.no_images and "inline_image_prompts" in metadata:
                for i, prompt in enumerate(metadata.get("inline_image_prompts", [])):
                    image_path = generate_image(prompt)
                    if image_path:
                        image_id, image_url = upload_image_to_wordpress(image_path)
                        if image_id and image_url:
                            placeholder = f"[IMAGE: {prompt}]"
                            if placeholder in content:
                                content = content.replace(
                                    placeholder,
                                    f'<figure><img src="{image_url}" alt="{metadata.get("alt_text", "Alternative text descriptions for the images in this tutorial")}"/><figcaption>{prompt}</figcaption></figure>',
                                )

            # Add footer
            content += '\n<p><strong>Ready to dive deeper?</strong> Check out <a href="https://github.com/felipedbene" target="_blank">my GitHub</a> for more code examples and in-depth tutorials!</p>'

            # Convert Markdown to HTML for WordPress
            content = markdown(
                content,
                extensions=[FencedCodeExtension(), CodeHiliteExtension()],
            )

            # Prepare post data
            post_data = {
                "title": metadata.get("title", f"Article about {args.idea}"),
                "slug": metadata.get("slug", args.idea.lower().replace(" ", "-")),
                "status": "draft",
                "content": content,
                "excerpt": metadata.get(
                    "meta_desc", f"Learn about {args.idea} in this comprehensive guide"
                ),
                "featured_media": None,  # Will be set if hero image is uploaded
                "categories": [2],  # Default category ID
                "tags": [3],  # Default tag ID
            }

            # Upload hero image if available
            if not args.no_images and "hero_image_prompt" in metadata and image_path:
                image_id, _ = upload_image_to_wordpress(image_path)
                if image_id:
                    post_data["featured_media"] = image_id

            # Create or update post
            post_id = create_or_update_post(post_data)

            if post_id:
                # Update SEO metadata if needed
                if "meta_title" in metadata or "meta_desc" in metadata:
                    seo_data = {
                        "title": metadata.get("meta_title", metadata.get("title", "")),
                        "metadesc": metadata.get("meta_desc", ""),
                        "focuskw": args.keyphrase or metadata.get("keyphrase", ""),
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
