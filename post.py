# Standard library imports
import os
import sys
import argparse
#
import markdown
import requests
import hashlib
import json
import datetime
import random
import re
import logging
import time
import urllib.parse
import datetime
from dotenv import load_dotenv
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.codehilite import CodeHiliteExtension
import yaml
from typing import Tuple, Dict
import frontmatter
from PIL import Image
from io import BytesIO
import base64
# CoreML Stable Diffusion pipeline singleton for Apple Silicon
from python_coreml_stable_diffusion.pipeline import CoreMLStableDiffusionPipeline
from PIL import Image
from pathlib import Path
# Import CoreMLModel wrapper for CoreML model loading
from python_coreml_stable_diffusion.coreml_model import CoreMLModel
# Add missing imports for pipeline initialization
from transformers import CLIPTokenizer, CLIPFeatureExtractor
from diffusers import PNDMScheduler
# CoreML Stable Diffusion pipeline singleton for Apple Silicon
_coreml_pipe = None

# Needed for reading system events from multiple sources
import subprocess



# --- Helper: Fetch or create taxonomy terms by name ---
def get_term_ids(names: list[str], taxonomy: str) -> list[int]:
    """
    Fetch existing terms by name or create them if they don't exist.
    Returns a list of term IDs.
    """
    ids = []
    for name in names:
        # Search for existing term
        resp = requests.get(f"{WP_URL}/wp-json/wp/v2/{taxonomy}", auth=(WP_USER, WP_PASS), params={"search": name})
        resp.raise_for_status()
        terms = resp.json()
        if terms:
            term_id = terms[0]["id"]
        else:
            # Create new term
            create = requests.post(f"{WP_URL}/wp-json/wp/v2/{taxonomy}", auth=(WP_USER, WP_PASS), json={"name": name})
            create.raise_for_status()
            term_id = create.json()["id"]
        ids.append(term_id)
    return ids

# Helper to convert "* " list lines to HTML lists
def auto_convert_lists(html: str) -> str:
    """Convert lines starting with '* ' into proper <ul><li>...</li></ul> blocks."""
    lines = html.splitlines()
    out = []
    in_list = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith('* '):
            if not in_list:
                out.append('<ul>')
                in_list = True
            out.append(f'<li>{stripped[2:].strip()}</li>')
        else:
            if in_list:
                out.append('</ul>')
                in_list = False
            out.append(line)
    if in_list:
        out.append('</ul>')
    return "\n".join(out)

# Helper to convert inline markdown (bold, links) into HTML
def convert_markdown_inline(html: str) -> str:
    """Convert inline markdown (bold, links) into HTML."""
    # bold
    html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
    # links
    html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', html)
    return html

# --- Helper: Strip front-matter fences and nested headings ---
def _strip_frontmatter(text: str) -> str:
    """Remove YAML front-matter fences and any nested headings."""
    # Remove any top-level YAML fences
    text = re.sub(r'(?s)^---.*?---\s*', '', text)
    return text

# --- Helper: Generate a detailed outline for a blog topic using LLM ---
def generate_blog_outline(topic):
    """
    Generate a detailed outline (list of section headings) for the given topic.
    """
    outline_prompt = f"""
You have up to 700 tokens—generate a detailed JSON list of section headings (strings) for a **parody blog post** on “{topic}”.
Imagine it as a witty, sarcastic, or over-the-top humorous take.
Only output a valid JSON array of strings, e.g. ["Ridiculous Introduction", "Absurd Claims", "Conclusion Full of Regret"].
Limit the JSON array to exactly 4 section headings.
"""
    resp = requests.post(
        f"http://{OLLAMA_SERVER}/api/generate",
        json={
            "model": "llama3.2:latest",
            "prompt": outline_prompt,
            "temperature": 0.6,
            "max_tokens": 700,
            "stream": False,
            "device": "cuda",
        },  
        timeout=300
    )
    data = resp.json()
    # Log raw outline response for debugging
    raw_outline = data.get("response", "")
    print(f"🔍 Raw outline response: {raw_outline}")
    # Extract JSON array substring if the model wrapped it in extra text
    import re
    match = re.search(r'\[.*\]', raw_outline, flags=re.DOTALL)
    outline_json = match.group(0) if match else raw_outline
    try:
        arr = json.loads(outline_json)
        # Normalize outline entries to strings
        headings = []
        for item in arr:
            if isinstance(item, str):
                headings.append(item)
            elif isinstance(item, dict):
                # Prefer 'title' or 'Title' keys, otherwise take first value
                val = item.get('title') or item.get('Title') or next(iter(item.values()))
                headings.append(str(val))
        return headings
    except Exception as e:
        print(f"⚠️ Failed to parse outline JSON: {e}")
        return []


# Load .env, overriding existing environment variables
load_dotenv(override=True)

WP_USER = os.getenv("WORDPRESS_USERNAME")
WP_PASS = os.getenv("WORDPRESS_APP_PASSWORD")
WP_URL  = os.getenv("WORDPRESS_URL")
HC_APIKEY     = os.getenv("HC_APIKEY")
SD_API_BASE   = os.getenv("SD_API_URL")
OLLAMA_SERVER = os.getenv("OLLAMA_SERVER")
NEWSAPI_KEY   = os.getenv("NEWSAPI_KEY")



# Only validate required environment variables for this script
required_env = {
    "WORDPRESS_USERNAME": WP_USER,
    "WORDPRESS_APP_PASSWORD": WP_PASS,
    "WORDPRESS_URL": WP_URL,
    "HC_APIKEY": HC_APIKEY,
    "SD_API_URL": SD_API_BASE,
    "OLLAMA_SERVER": OLLAMA_SERVER,
    "NEWSAPI_KEY": NEWSAPI_KEY,
}
missing = [k for k, v in required_env.items() if not v]
if missing:
    raise EnvironmentError(f"❌ Missing environment variables: {', '.join(missing)}")

# Default Stable Diffusion parameters (override via environment)
STEPS_DEFAULT = int(os.getenv("SD_STEPS", "30"))
SCALE_DEFAULT = float(os.getenv("SD_SCALE", "7.5"))


def generate_blog_components(topic):
    # One-shot full article generation
    prompt = f"""
You are a seasoned pop-culture journalist known for crafting fluid, engaging, and witty narratives. Write a 1000–1500 word feature on “{topic}” with these requirements:
- A compelling introduction that sets the scene and hooks the reader.
- Five numbered sections with descriptive, evocative headings; each should flow seamlessly into the next.
- A polished tone—humorous and slightly sarcastic, but never juvenile or overly cutesy.
- Rich, authentic anecdotes and vivid imagery (e.g., describe Blue Ivy’s stage presence with concrete details, not just “cute”).
- A concise conclusion that ties the themes together.
- YAML front-matter (fenced with '---') containing title, slug, meta_title, meta_desc, keyphrase, categories, tags, hero_image_prompt, inline_image_prompts, and alt_text.
- No meta commentary like “In the next section” or wink-nudge asides.
- Integrate parenthetical citations where appropriate.
Output only the front-matter followed by the HTML body with inline placeholders like [IMAGE: ...].
"""
    os.makedirs(".cache/posts", exist_ok=True)
    cache_key = hashlib.sha256(topic.encode()).hexdigest()
    cache_path = f".cache/posts/{cache_key}.yaml"
    print(f"🔨 Generating full post for topic: {topic}")
    resp = requests.post(
        f"http://{OLLAMA_SERVER}/api/generate",
        json={
            "model": "llama3.2:latest",
            "prompt": prompt,
            "temperature": 0.7,
            "max_tokens": 2000,
            "stream": False,
            "device": "cuda",
        },
        timeout=600
    )
    full_raw = resp.json().get("response", "").strip()
    with open(cache_path, "w") as f:
        f.write(full_raw)
    return full_raw

def generate_image(prompt, steps=STEPS_DEFAULT, scale=SCALE_DEFAULT):
    # Support lists of prompts by concatenating into one string
    if isinstance(prompt, list):
        prompt = " ".join(str(p) for p in prompt)
    global _coreml_pipe

    # Initialize CoreML pipeline on first use
    if _coreml_pipe is None:
        # Directory containing CoreML .mlpackage bundles
        packages_dir = Path(__file__).parent / "coreml-stable-diffusion-v1-5" / "original" / "packages"

        # Load CoreML models using CoreMLModel wrapper, specifying compute_unit="ALL"
        text_encoder_model = CoreMLModel(str(packages_dir / "Stable_Diffusion_version_runwayml_stable-diffusion-v1-5_text_encoder.mlpackage"), compute_unit="ALL")
        unet_model         = CoreMLModel(str(packages_dir / "Stable_Diffusion_version_runwayml_stable-diffusion-v1-5_unet.mlpackage"), compute_unit="ALL")
        vae_decoder_model  = CoreMLModel(str(packages_dir / "Stable_Diffusion_version_runwayml_stable-diffusion-v1-5_vae_decoder.mlpackage"), compute_unit="ALL")
        safety_checker_model = CoreMLModel(str(packages_dir / "Stable_Diffusion_version_runwayml_stable-diffusion-v1-5_safety_checker.mlpackage"), compute_unit="ALL")

        # Load scheduler and tokenizer from Hugging Face
        scheduler = PNDMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")

        # Initialize CoreML SD pipeline with the correct constructor signature:
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

    # Normalize prompt
    prompt = prompt.strip()
    # Compute cache key
    cache_key = hashlib.sha256(prompt.encode()).hexdigest()
    os.makedirs(".cache/images", exist_ok=True)
    output_path = os.path.join(".cache/images", f"{cache_key}.webp")

    if os.path.exists(output_path):
        print("🖼️ Cached image used:", output_path)
        return output_path

    # Generate image
    print(f"🎨 Generating image via CoreML for prompt: {prompt}")
    result = _coreml_pipe(prompt=prompt, num_inference_steps=30, guidance_scale=7.5)
    image = result.images[0]
    # Resize to smaller dimensions (e.g., 384x384)
    image = image.resize((384, 384), Image.Resampling.LANCZOS)
    # Save as WEBP
    image.save(output_path, format="WEBP", quality=85)
    return output_path

def upload_page(title, slug, content, media_id=None, status="publish", categories=None, tags=None, meta_desc=None):
    url = f"{WP_URL}/wp-json/wp/v2/pages"
    payload = {
        "title": title,
        "slug": slug,
        "status": status,
        "content": content,
    }
    if media_id:
        payload["featured_media"] = media_id
    if categories is not None:
        payload["categories"] = categories
    if tags is not None:
        payload["tags"] = tags
    if meta_desc is not None:
        payload["excerpt"] = meta_desc

    r = requests.post(url, auth=(WP_USER, WP_PASS), json=payload)
    r.raise_for_status()
    return r.json()

def preprocess_system_events(raw_events: str) -> str:
    """
    Summarize system logs for better prompt quality by:
    - Grouping repeated log types
    - Deduplicating lines
    - Adding source markers
    """
    lines = raw_events.splitlines()
    grouped = {
        "macOS Unified Log": [],
        "Kernel/dmesg": [],
        "Authentication": [],
        "Kubernetes": [],
        "Syslog": [],
        "Other": []
    }

    current_group = "Other"
    seen = set()
    for line in lines:
        line = line.strip()
        if not line or line in seen:
            continue
        seen.add(line)
        if "Unified macOS log" in line:
            current_group = "macOS Unified Log"
        elif "dmesg" in line:
            current_group = "Kernel/dmesg"
        elif "Authentication logs" in line:
            current_group = "Authentication"
        elif "Kubernetes events" in line:
            current_group = "Kubernetes"
        elif "syslog" in line:
            current_group = "Syslog"
        grouped.setdefault(current_group, []).append(line)

    summary = []
    for group, logs in grouped.items():
        if logs:
            summary.append(f"### {group} ({len(logs)} entries)")
            summary.extend(logs[-20:])  # Show last 20 per group max

    return "\n".join(summary[-200:])  # Max 200 lines overall

def upload_image_to_wp(image_path, alt_text):
    with open(image_path, 'rb') as img:
        filename = os.path.splitext(os.path.basename(image_path))[0] + ".webp"
        headers = {
            "Content-Disposition": f"attachment; filename={filename}"
        }
        r = requests.post(
            f"{WP_URL}/wp-json/wp/v2/media",
            auth=(WP_USER, WP_PASS),
            headers=headers,
            files={'file': img},
            data={'alt_text': alt_text}
        )
        r.raise_for_status()
        media_json = r.json()
        media_id = media_json.get("id")
        media_url = media_json.get("source_url")
        return media_id, media_url

def upload_post(title, slug, content, meta_title, meta_desc, keyphrase, media_id, categories, tags, publish_date=None, publish_date_gmt=None, status="draft", post_id=None ):
    payload = {
        "title": title,
        "slug": slug,
        "status": status,
        "content": content,
        "excerpt": meta_desc,
        "featured_media": media_id,
        "categories": categories,
        "tags": tags
    }
    # Include Yoast SEO meta in the initial creation payload
    payload["meta"] = {
        "yoast_wpseo_focuskw": keyphrase,
        "yoast_wpseo_metadesc": meta_desc,
        "yoast_wpseo_title": meta_title
    }
    print(f"Payload: {payload}")
    if publish_date:
        payload["date"] = publish_date
    if publish_date_gmt:
        payload["date_gmt"] = publish_date_gmt

    if post_id:
        url = f"{WP_URL}/wp-json/wp/v2/posts/{post_id}"
        method = requests.patch
    else:
        url = f"{WP_URL}/wp-json/wp/v2/posts"
        method = requests.post

    r = method(url, auth=(WP_USER, WP_PASS), json=payload)
    try:
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        print("❌ Failed to upload post:", r.text)
        raise e

def update_seo_meta(post_id, meta_title, meta_desc, keyphrase):
    """
    Update Yoast SEO meta fields for an existing post.
    """
    url = f"{WP_URL}/wp-json/wp/v2/posts/{post_id}"
    payload = {
        "meta": {
            "yoast_wpseo_focuskw": keyphrase,
            "yoast_wpseo_metadesc": meta_desc,
            "yoast_wpseo_title": meta_title
        }
    }
    r = requests.patch(url, auth=(WP_USER, WP_PASS), json=payload)
    if r.ok:
        print("🔧 SEO metadata patched successfully.")
    else:
        print("⚠️ Failed to update SEO meta:", r.text)

def parse_generated_text(text: str) -> Tuple[Dict, str]:
    """
    Extract YAML front-matter (between '---' markers) if valid, else skip it.
    Returns a tuple of (front_matter_dict, body_text).
    """
    front_matter: Dict = {}
    body = text

    print("🔍 [DEBUG] parse_generated_text received text (first 500 chars):")
    print(text[:500].replace("\n", "\\n"))

    # Flexible front-matter: allow missing closing fence
    match = re.search(r'^\s*---\s*\n(.*?)(?:\n---\s*|$)', text, flags=re.S | re.MULTILINE)
    if not match:
        print("⚠️ [DEBUG] No front-matter fences found in text.")
        return {}, text
    fm_text = match.group(1)
    tail = text[match.end():]
    if not tail.lstrip().startswith('---'):
        print("⚠️ [DEBUG] Missing closing '---'; force-wrapping front matter")
        stripped = text.lstrip()[3:]
        parts = stripped.split('\n\n', 1)
        fm_text = parts[0]
        fm_text = '---\n' + fm_text.strip() + '\n---'
        body = parts[1] if len(parts) > 1 else ""
    else:
        print("🔍 [DEBUG] YAML front-matter block detected:")
        print(fm_text)
        body = text[match.end():]
    def _quote_value(line: str) -> str:
        if ':' not in line or line.strip().startswith('- '):
            return line
        key, val = line.split(':', 1)
        val = val.strip()
        # Skip quoting if there is no value after the colon
        if not val:
            return line
        # Skip if already quoted or is a list
        if (val.startswith('"') and val.endswith('"')) or val.startswith('['):
            return line
        # Quote the value, escaping any existing quotes
        safe_val = val.replace('"', '\\"')
        return f"{key}: \"{safe_val}\""
    fm_lines = fm_text.splitlines()
    fm_text = "\n".join(_quote_value(l) for l in fm_lines)
    try:
        # Parse possibly multiple YAML documents and take the first as front-matter
        docs = list(yaml.safe_load_all(fm_text))
        if docs and isinstance(docs[0], dict):
            fm = docs[0]
            # Normalize list fields if returned as JSON-formatted strings
            for list_field in ("categories", "tags", "synonyms", "inline_image_prompts"):
                val = fm.get(list_field)
                if isinstance(val, str):
                    try:
                        fm[list_field] = json.loads(val)
                    except Exception:
                        pass
            front_matter = fm
        else:
            raise ValueError(f"Invalid front-matter structure: {docs}")
    except Exception as e:
        print(f"❌ [DEBUG] YAML front-matter parsing failed: {e}")
        raise
    return front_matter, body

# --- Front matter rendering helper ---
def render_front_matter(data):
    """
    Render YAML front-matter manually and return a markdown string
    containing front-matter followed by the HTML body.
    """
    metadata = {
        "title": data.get("title", ""),
        "meta_title": data.get("meta_title", ""),
        "meta_desc": data.get("meta_desc", ""),
        "slug": data.get("slug", ""),
        "keyphrase": data.get("keyphrase", ""),
        "synonyms": data.get("synonyms", []),
        "image_prompt": data.get("image_prompt", ""),
        "alt_text": data.get("alt_text", ""),
    }

    # Build YAML front-matter
    front_matter_lines = ["---"]
    for key, val in metadata.items():
        if isinstance(val, list):
            front_matter_lines.append(f"{key}:")
            for item in val:
                front_matter_lines.append(f"  - {item}")
        else:
            # Escape double quotes in the value
            safe_val = val.replace('"', '\\"')
            front_matter_lines.append(f'{key}: "{safe_val}"')
    front_matter_lines.append("---")
    front_matter = "\n".join(front_matter_lines)

    # Combine front-matter and HTML body
    body = data.get("body_html", "")
    return f"{front_matter}\n{body}"

def scan_broken_links():
    """
    Fetch all posts, scan their content for internal links,
    and write any broken links to broken_pages.txt.
    """
    output_file = "broken_pages.txt"
    base = WP_URL.rstrip('/')
    broken = []

    # Paginate through all posts
    page = 1
    while True:
        resp = requests.get(f"{base}/wp-json/wp/v2/posts", auth=(WP_USER, WP_PASS), params={"per_page": 100, "page": page})
        resp.raise_for_status()
        posts = resp.json()
        if not posts:
            break
        for post in posts:
            content = post.get('content', {}).get('rendered', '')
            # Find all hrefs
            links = re.findall(r'href="([^"]+)"', content)
            for link in links:
                # Check only internal links: start with '/' or the base URL
                if link.startswith('/') or link.startswith(base):
                    url = link if link.startswith('http') else f"{base}{link}"
                    try:
                        r = requests.head(url, timeout=5, allow_redirects=True)
                        status = r.status_code
                    except Exception as e:
                        status = None
                    if status != 200:
                        broken.append(f"Post ID {post['id']}: {url} -> {status}\n")
        page += 1

    # Write out broken links
    with open(output_file, 'w') as f:
        f.writelines(broken)
    print(f"🔍 Scan complete. {len(broken)} broken links written to {output_file}")


# --- Helper: Read last 2 hours of system logs ---
def read_syslog_last_two_hours() -> str:
    """
    Read the last 2 hours of system logs from /var/log/syslog or /var/log/system.log.
    Returns the joined log lines as a single string.
    """
    import os, datetime
    # Choose the first existing log path
    for log_path in ('/var/log/syslog', '/var/log/system.log'):
        if os.path.exists(log_path):
            break
    else:
        print("⚠️ No syslog file found at /var/log/syslog or /var/log/system.log")
        return ""

    cutoff = datetime.datetime.now() - datetime.timedelta(hours=48)
    entries = []
    with open(log_path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            ts_str = " ".join(parts[:3])
            try:
                ts = datetime.datetime.strptime(ts_str, '%b %d %H:%M:%S')
                ts = ts.replace(year=cutoff.year)
            except Exception:
                continue
            if ts >= cutoff:
                entries.append(line.rstrip())
    # Return the last 500 lines to avoid overly long prompts
    return "\n".join(entries[-500:])


# --- Helper: Collect system events from multiple sources ---
def read_system_events() -> str:
    """
    Gather recent system events from multiple sources and return them as a single string.
    """
    entries = []

    # 1) Last 2 hours of unified logs (macOS)
    try:
        entries.append("=== Unified macOS log (last 2h) ===")
        unified = subprocess.check_output(
            ["log", "show", "--style", "syslog", "--last", "2h"],
            text=True, stderr=subprocess.DEVNULL
        )
        entries.append(unified)
    except Exception:
        pass

    # 2) Last 100 dmesg lines
    try:
        entries.append("=== dmesg (tail 100) ===")
        dmesg_out = subprocess.check_output(
            ["dmesg", "-T", "--time-format", "iso8601"],
            text=True, stderr=subprocess.DEVNULL
        ).splitlines()[-100:]
        entries.append("\n".join(dmesg_out))
    except Exception:
        pass

    # 3) Recent login/logout activity
    try:
        entries.append("=== Authentication logs (last 2h) ===")
        last_out = subprocess.check_output(
            ["last", "-2h"],
            text=True, stderr=subprocess.DEVNULL
        )
        entries.append(last_out)
    except Exception:
        pass

    # 4) Kubernetes events (if kubectl available)
    try:
        entries.append("=== Kubernetes events (last 2h) ===")
        kube = subprocess.check_output(
            ["kubectl", "get", "events", "--all-namespaces", "--since=2h"],
            text=True, stderr=subprocess.DEVNULL
        )
        entries.append(kube)
    except Exception:
        pass

    # Fallback to syslog for any remaining events
    try:
        entries.append("=== syslog (last 2h) ===")
        entries.append(read_syslog_last_two_hours())
    except Exception:
        pass

    # Limit output size
    combined = "\n".join(entries)
    return "\n".join(combined.splitlines()[-500:])

def inject_adsense_snippet(html):
    ad_html = """
<!-- Google AdSense Ad -->
<div style="margin: 2em 0; padding: 1em; border-top: 1px dashed #ccc; font-family: 'Lora', Georgia, serif;">
  <small style="display: block; text-align: center; color: #888; margin-bottom: 0.5em;">
    🪙 Supports Me
  </small>
  <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-1479106809372421"
     crossorigin="anonymous"></script>
  <ins class="adsbygoogle"
       style="display:block; text-align:center;"
       data-ad-layout="in-article"
       data-ad-format="fluid"
       data-ad-client="ca-pub-1479106809372421"
       data-ad-slot="6648621056"></ins>
  <script>
       (adsbygoogle = window.adsbygoogle || []).push({});
  </script>
</div>
"""
    paragraphs = html.split("</p>")
    if len(paragraphs) > 1:
        paragraphs.insert(1, ad_html)
    return "</p>".join(paragraphs)

# --- Helper: Extract image prompts from body ---
def extract_image_prompts_from_body(html: str) -> list[str]:
    """
    Scan the HTML body for figcaption tags or "Figure X:" lines and return a list of captions as prompts.
    """
    prompts = []
    # Match <figcaption>...</figcaption>
    for match in re.finditer(r'<figcaption>(.*?)</figcaption>', html, re.IGNORECASE):
        caption = match.group(1).strip()
        if caption:
            prompts.append(caption)
    # Match plain text "Figure N: caption"
    for match in re.finditer(r'Figure\s*\d+:\s*(.+)', html):
        caption = match.group(1).strip()
        if caption and caption not in prompts:
            prompts.append(caption)
    # Match [IMAGE: ...] and [inline_image_prompt: ...] placeholders
    for match in re.finditer(r'\[(?:IMAGE|inline_image_prompt):\s*(.*?)\]', html, re.IGNORECASE):        
        caption = match.group(1).strip()
        if caption and caption not in prompts:
            prompts.append(caption)
    return prompts

# --- Helper: Inject inline images for figcaptions ---
import re

def inject_inline_images(html: str, prompts: list[str], alt_text: str) -> str:
    """
    For each caption prompt, generate an image, upload it, and wrap the existing <figcaption>
    in a <figure> with the new <img>. Also replaces [IMAGE: description] placeholders.
    """
    def prompt_to_str(prompt):
        if isinstance(prompt, str):
            return prompt
        elif isinstance(prompt, dict):
            # Try to get the first value of the dict
            if len(prompt) == 1:
                return str(next(iter(prompt.values())))
            # If multiple keys, join values
            return " ".join(str(v) for v in prompt.values())
        elif isinstance(prompt, list):
            return " ".join(str(p) for p in prompt)
        else:
            return str(prompt)

    # Always use default steps/scale
    steps = STEPS_DEFAULT
    scale = SCALE_DEFAULT

    for prompt in prompts:
        prompt_str = prompt_to_str(prompt)
        img_path = generate_image(prompt_str)
        mid, url = upload_image_to_wp(img_path, alt_text)
        # Wrap the figcaption with a figure/html
        pattern = fr'(<figcaption>\s*{re.escape(prompt_str)}\s*</figcaption>)'
        replacement = rf'<figure><img src="{url}" alt="{alt_text}"/>\1</figure>'
        html = re.sub(pattern, replacement, html, flags=re.IGNORECASE)
    # Replace any leftover [IMAGE: description] placeholders
    for prompt in prompts:
        prompt_str = prompt_to_str(prompt)
        img_path = generate_image(prompt_str)
        mid, url = upload_image_to_wp(img_path, alt_text)
        replacement = (
            f'<figure>'
            f'<img src="{url}" alt="{alt_text}"/>'
            f'<figcaption>{prompt_str}</figcaption>'
            f'</figure>'
        )
        placeholder_pattern = rf'\[(?:IMAGE|inline_image_prompt):\s*{re.escape(prompt_str)}\s*\]'
        html = re.sub(placeholder_pattern, replacement, html, flags=re.IGNORECASE)
    return html

# --- Helper: Generate NewsAPI code snippet for a query ---
def generate_newsapi_code_snippet(query: str) -> str:
    """
    Return a markdown code snippet showing how to call NewsAPI /everything for the given query.
    """
    today = datetime.date.today().isoformat()
    encoded = urllib.parse.quote(query)
    return f"""```http
GET https://newsapi.org/v2/everything?q={encoded}&from={today}&sortBy=popularity&apiKey=YOUR_API_KEY
```
```bash
curl https://newsapi.org/v2/everything -G \\
  -d q="{query}" \\
  -d from={today} \\
  -d sortBy=popularity \\
  -d apiKey=$NEWSAPI_KEY
```
"""

def enrich_with_internal_links(parsed_body, all_posts):
    """
    Insert internal links to other posts with overlapping words near the end of the body.
    """
    # Extract words from parsed_body (lowercase, simple split)
    body_words = set(re.findall(r'\b\w+\b', parsed_body.lower()))
    links_to_add = []
    for post in all_posts:
        # Skip if title or slug missing
        if not post.get("title") or not post.get("slug"):
            continue
        post_title = post["title"]
        post_slug = post["slug"]
        # Extract words from post title
        title_words = set(re.findall(r'\b\w+\b', post_title.lower()))
        # Check for overlap (at least one common word)
        if body_words.intersection(title_words):
            if len(links_to_add) >= 5:
                break
            # Avoid linking to self if link present in body (approximate)
            if post.get("link") and post["link"] in parsed_body:
                continue
            if not post.get("link"):
                continue
            links_to_add.append(f'<p>See also: <a href="{post["link"]}">{post_title}</a></p>')
    if links_to_add:
        # Insert links near the end, before closing </body> or at end if no tag found
        insert_pos = parsed_body.rfind("</body>")
        if insert_pos == -1:
            # Append at end
            enriched_body = parsed_body + "\n" + "\n".join(links_to_add)
        else:
            enriched_body = parsed_body[:insert_pos] + "\n" + "\n".join(links_to_add) + "\n" + parsed_body[insert_pos:]
        return enriched_body
    else:
        return parsed_body

def fetch_trending_topics(count=5, category="technology", query=None):
    """
    Fetch trending topics using NewsAPI.org API only.
    """
    newsapi_key = os.getenv("NEWSAPI_KEY")
    if not newsapi_key:
        print("❌ NEWSAPI_KEY is not set in environment.")
        return []

    topics = []
    # Always use top-headlines so queries return results on the free plan
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "apiKey": newsapi_key,
        "pageSize": count,
        "language": "en",
    }
    if query:
        params["q"] = query
    else:
        params["country"] = "us"
        params["category"] = category

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        articles = data.get("articles", [])
        if not articles and query:
            # Fallback to 'everything' endpoint for more complete coverage
            print("⚠️ No top-headlines; falling back to /everything endpoint.")
            url = "https://newsapi.org/v2/everything"
            params = {
                "apiKey": newsapi_key,
                "pageSize": count,
                "language": "en",
                "q": query,
                "sortBy": "popularity",
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            articles = data.get("articles", [])
        if not articles:
            print("⚠️ Still no articles found after fallback.")
            return []

        for article in articles:
            title = (article.get("title") or "").strip()
            description = (article.get("description") or "").strip()
            parts = [title]
            if description:
                parts.append(description)
            enriched = " | ".join(parts)
            topics.append(enriched)

        return topics
    except Exception as e:
        print(f"❌ Failed to fetch trending topics from NewsAPI.org: {e}")
        return []

def fetch_all_posts_metadata():
    results = []
    page = 1
    while True:
        try:
            resp = requests.get(
                f"{WP_URL}/wp-json/wp/v2/posts",
                auth=(WP_USER, WP_PASS),
                params={"per_page": 100, "page": page}
            )
            if resp.status_code == 400:
                break
            resp.raise_for_status()
            posts = resp.json()
            #print(posts)
            if not posts:
                break
            for p in posts:
                results.append({
                    "title": p["title"]["rendered"],
                    "slug": p["slug"],
                    "id": p["id"],
                    "link": p.get("link")
                })
            page += 1
        except requests.exceptions.RequestException as e:
            logging.warning(f"⚠️ Failed to fetch page {page}: {e}")
            break
    return results

def send_healthcheck_ping():
    """
    Send a healthcheck ping to the configured endpoint.
    """
    try:
        r = requests.get("https://hc-ping.com/b6d51ccc-3470-479d-8ba9-a793f65ad02b", timeout=5)
        logging.info(f"✅ Healthcheck ping sent with status {r.status_code}")
    except Exception as e:
        logging.warning(f"⚠️ Failed to send healthcheck ping: {e}")

    # Helper function to process and publish a post/page from an idea and keyphrase
def process_and_publish(idea, keyphrase, args):
    """
    Generate, process, and publish a post or page from the given idea and keyphrase.
    Returns the published link.
    """
    # Always generate fresh content and parse it directly
    # Retry up to 3 times if parsing or metadata is missing
    for attempt in range(1, 4):
        raw = generate_blog_components(idea)
        print(f"🔄 Attempt {attempt} raw preview:")
        print(raw[:200].replace("\n","\\n"))
        try:
            front_matter, body_html = parse_generated_text(raw)
        except Exception as exc:
            print(f"⚠️ Attempt {attempt} failed parsing YAML: {exc}")
            time.sleep(2 ** (attempt - 1))
            continue
        if front_matter.get("title") and front_matter.get("slug") and front_matter.get("keyphrase"):
            break
        print(f"⚠️ Attempt {attempt} missing essential metadata, retrying…")
        time.sleep(2 ** (attempt - 1))
    else:
        raise RuntimeError("❌ Failed to generate valid metadata after 3 attempts")
    # Use metadata directly without fallbacks
    title = front_matter["title"]
    slug_source = front_matter["slug"]
    slug = slug_source.lower().replace(" ", "-")
    keyphrase_final = front_matter["keyphrase"]

    # Safe fallback for meta_title and meta_desc
    raw_meta_title = front_matter.get("meta_title", title)
    meta_title = raw_meta_title[:60]

    raw_meta_desc = front_matter.get("meta_desc", "")[:155]
    # If still empty, fallback to content below after it's set
    meta_desc = raw_meta_desc[:155]

    synonyms = front_matter.get("synonyms", [])
    image_prompt = front_matter.get("hero_image_prompt", title)
    alt_text = front_matter.get("alt_text", "")
    # Use only the clean HTML body as content
    content = body_html.lstrip("|").lstrip()

    # Inject AdSense snippet before internal links
    content = inject_adsense_snippet(content)
    #content = enrich_with_internal_links(content, all_posts)

    # (Image injection is now handled after all cleanup below)

    # Final HTML transformations on full content
    # Skip markdown-to-HTML conversion since content is already HTML
    content = auto_convert_lists(content)
    content = convert_markdown_inline(content)

    # If a custom query was specified, insert the NewsAPI example snippet
    if args.query:
        snippet = generate_newsapi_code_snippet(args.query)
        content = snippet + "\n\n" + content

    # Append a call-to-action to excite readers
    cta_html = (
        '<p><strong>Ready to dive deeper?</strong> '
        'Check out <a href="https://github.com/felipedbene" target="_blank">my GitHub</a> '
        'for more code examples and in-depth tutorials!</p>'
    )
    content += "\n" + cta_html

    # --- Remove any trailing key: value lines after content ---
    content = re.sub(r'(\n[a-z_]+:.*)+$', '', content, flags=re.MULTILINE).strip()

    # --- Inline image injection and featured-image logic ---
    # Combine front-matter inline prompts with extracted figcaptions/placeholders
    meta_prompts = front_matter.get("inline_image_prompts", []) or []
    extracted_prompts = extract_image_prompts_from_body(content)
    prompts = meta_prompts + extracted_prompts

    if prompts:
        # Generate and inject images for every prompt
        content = inject_inline_images(content, prompts, alt_text)
    # Clean any leftover placeholders
    content = re.sub(r'\[IMAGE:.*?\]', '', content)

    # Always generate a hero image from hero_image_prompt, then skip featured fallback
    media_id = None
    hero_prompt = front_matter.get("hero_image_prompt", "")
    if hero_prompt:
        print("🎨 Generating hero image…")
        img_path = generate_image(hero_prompt)
        media_id, _ = upload_image_to_wp(img_path, alt_text)
    elif '<img ' not in content:
        # Fallback if no inline or hero image present
        print("🎨 Generating contextual image…")
        img_path = generate_image(image_prompt)
        media_id, _ = upload_image_to_wp(img_path, alt_text)

    # Determine randomized publish date if requested
    pub_date = pub_date_gmt = None
    if args.days > 0:
        delta_days = random.uniform(0, args.days)
        delta_secs = random.uniform(0, 86400)
        dt = datetime.datetime.now() - datetime.timedelta(days=delta_days, seconds=delta_secs)
        pub_date = dt.strftime("%Y-%m-%dT%H:%M:%S")
        pub_date_gmt = dt.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        status = "publish"
    else:
        status = "draft"

    # Normalize AI-generated taxonomy lists to flat lists of strings
    raw_categories = front_matter.get("categories", []) or ["default-category"]
    category_names = []
    for item in raw_categories:
        if isinstance(item, list):
            for sub in item:
                category_names.append(str(sub))
        else:
            category_names.append(str(item))

    raw_tags = front_matter.get("tags", []) or [keyphrase]
    tag_names = []
    for item in raw_tags:
        if isinstance(item, list):
            for sub in item:
                tag_names.append(str(sub))
        else:
            tag_names.append(str(item))

    category_ids = get_term_ids(category_names, "categories")
    tag_ids = get_term_ids(tag_names, "tags")

    # Upload as page or post based on flag
    print(f"📝 Uploading {'page' if args.page else 'post'} to WordPress...")
    if args.page:
        if media_id is not None:
            result = upload_page(
                title, slug, content,
                media_id=media_id,
                status=status,
                categories=category_ids,
                tags=tag_ids,
                meta_desc=meta_desc
            )
        else:
            result = upload_page(
                title, slug, content,
                status=status,
                categories=category_ids,
                tags=tag_ids,
                meta_desc=meta_desc
            )
        post_link = result.get('link')
    else:
        if media_id is not None:
            result = upload_post(
                title, slug, content,
                meta_title, meta_desc,
                keyphrase_final, media_id,
                category_ids, tag_ids,
                publish_date=pub_date,
                publish_date_gmt=pub_date_gmt,
                status=status
            )
        else:
            result = upload_post(
                title, slug, content,
                meta_title, meta_desc,
                keyphrase_final, None,
                category_ids, tag_ids,
                publish_date=pub_date,
                publish_date_gmt=pub_date_gmt,
                status=status
            )
        # update_seo_meta(result.get('id'), meta_title, meta_desc, keyphrase_final)  # now a no-op, handled in initial payload
        post_link = result.get('link')
    return post_link


# Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--news', action='store_true', help='Fetch trending topics from News API')
    parser.add_argument('--system', action='store_true', help='Generate a post from recent system events')
    parser.add_argument('--days', type=int, default=0, help='Backdate publish date by up to N random days')
    parser.add_argument('--page', action='store_true', help='Upload as WordPress page instead of post')
    parser.add_argument('--query', type=str, help='Optional query term to insert into NewsAPI example')
    parser.add_argument('--idea', type=str, help='Idea or prompt for generating a blog post')
    parser.add_argument('--keyphrase', type=str, default=None, help='Focus keyphrase for the blog post')

    args = parser.parse_args()
    tasks = []

    if args.system:
        print("✅ --system flag detected, gathering logs...")
        system_log_text = read_system_events()
        idea = preprocess_system_events(system_log_text)
        print(f"🧠 Preprocessed idea: {idea}")
        keyphrase = "system debugging"
        link = process_and_publish(idea, keyphrase, args)
        print(f"✅ Post published at: {link}")
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler("poster.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    
    if args.news:
        logging.info("📥 Fetching new trending topics from News API...")
        all_trends = fetch_trending_topics(count=10, category=args.category, query=args.query)
        logging.info(f"🌐 Retrieved {len(all_trends)} headlines.")
        # Add all trends without Redis deduplication
        for trend in all_trends:
            tasks.append((trend, "default-keyphrase"))
    elif args.idea:
        tasks.append((args.idea, args.keyphrase))
    else:
        parser.error("You must provide --idea, --news, or --system")

    for idea, keyphrase in tasks:
        link = process_and_publish(idea, keyphrase, args)
        print(f"✅ Published: {link}")

    logging.info("📡 Attempting to send final healthcheck ping...")
    send_healthcheck_ping()
    logging.info("📡 Healthcheck ping execution finished.")


if __name__ == '__main__':
    main()