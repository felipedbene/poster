import re

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
#!/usr/bin/env python3
import os
import argparse
import requests
import hashlib
import json
import datetime
import random
import re
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import base64
import redis
import logging
import time
import sys
import yaml
import frontmatter

try:
    import markdown
except ImportError:
    markdown = None


# Load .env
load_dotenv()

# Healthcheck API key (optional)
HC_APIKEY = os.getenv("HC_APIKEY")

# Config
WP_USER = os.getenv("WORDPRESS_USERNAME")
WP_PASS = os.getenv("WORDPRESS_APP_PASSWORD")
WP_URL = os.getenv("WORDPRESS_URL")

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
# Ollama server address (env var): host:port for local or remote Ollama
OLLAMA_SERVER = os.getenv("OLLAMA_SERVER", "localhost:11434")

# Redis config
REDIS_HOST = os.getenv("REDIS_HOST", "redis-master.wp.svc.cluster.local")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

# Grouped environment variable validation
REQUIRED_CREDENTIALS = {
    "WORDPRESS_USERNAME": WP_USER,
    "WORDPRESS_APP_PASSWORD": WP_PASS,
    "WORDPRESS_URL": WP_URL
}

REQUIRED_API_KEYS = {
    "OPENAI_API_KEY": OPENAI_KEY,
    "GNEWS_API_KEY": os.getenv("GNEWS_API_KEY")
}

missing_creds = [key for key, value in REQUIRED_CREDENTIALS.items() if not value]
missing_keys = [key for key, value in REQUIRED_API_KEYS.items() if not value]

if missing_creds or missing_keys:
    problems = []
    if missing_creds:
        problems.append(f"Missing credentials: {', '.join(missing_creds)}")
    if missing_keys:
        problems.append(f"Missing API keys: {', '.join(missing_keys)}")
    raise EnvironmentError("❌ " + " | ".join(problems))

GNEWS_API_KEY = REQUIRED_API_KEYS["GNEWS_API_KEY"]

#openai.api_key = OPENAI_KEY

def load_cached_post(idea, keyphrase):
    cache_key = hashlib.sha256(f"{idea}:{keyphrase}".encode()).hexdigest()
    cache_path = f".cache/posts/{cache_key}.json"
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, "r") as f:
        cached = f.read()
    if "IMAGE_PROMPT:" not in cached:
        return None
    return cached


def generate_blog_components(topic):
    content = ""
    comeco = False
    #Create a cache directory if it doesn't exist
    os.makedirs(".cache/posts", exist_ok=True)
    cache_key = hashlib.sha256(topic.encode()).hexdigest()
    cache_path = f".cache/posts/{cache_key}.json"
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cached = f.read()
        if "IMAGE_PROMPT:" not in cached:
            print("CACHE MISS - Re-generating Image")
        else:
            print(f"💾 Cached blog retrieved for: {topic}")
            return cached

    prompt = fr"""
Always ONLY output YAML front-matter with these keys, even if empty:

[Output starts here]
title: ""
meta_title: ""
meta_desc: ""
slug: ""
keyphrase: ""
synonyms: []
image_prompt: ""
alt_text: ""
body_html: |
  [Your article content here]
[Output ends here without any other text]

Write an engaging, SEO-friendly article on “{topic}” in ~2,000 words. Structure naturally with an introduction, context, architecture deep-dive, use-case examples, and conclusion. Do NOT invent details you don’t know—leave values blank or as “TBD”. Use `<pre><code>…</code></pre>` for code samples, sprinkle 2–3 image captions.
If you have a context-specific GitHub repository URL, use it in a bold call-to-action; if not, always link to my GitHub profile: https://github.com/felipedbene — do NOT output any placeholders like "TBD" or "Your Repo URL".
Do not wrap any content in code fences and do not include any trailing notes or analysis sections in the article body.
"""
    models = ["deepseek-r1:14b","deepseek-r1:7b","mistral:7b-instruct", "llama3.2:latest"]
    for model_name in models:
        for attempt in range(3):
            try:
                print(f"🦙 Calling {model_name} (Ollama) for: {topic} (attempt {attempt + 1})")
                response = requests.post(f"http://{OLLAMA_SERVER}/api/generate", json={
                    "model": model_name,
                    "prompt": prompt,
                    "temperature": 0.9,
                    "stream": False
                }, timeout=600)

                if response.status_code != 200:
                    print(f"❌ Ollama returned status {response.status_code}")
                    print(f"Response body:\n{response.text}")
                    response.raise_for_status()
                # Parse JSON response
                try:
                    data = response.json()
                except ValueError as e:
                    print(f"❌ Ollama returned invalid JSON: {e}")
                    print(f"Response body:\n{response.text}")
                    raise Exception("Invalid JSON response from Ollama")
                raw_response = data.get('response', '')
                if not raw_response:
                    print(f"⚠️ Empty response received from {model_name} for {topic}")
                # Cache the raw response text
                # with open(cache_path, "w") as f:
                #     f.write(raw_response)
                # Use raw_response for further processing
                response_text = raw_response

                # Extract YAML content starting at the title line
                for line in response_text.splitlines():
                    if comeco:
                        content += line + "\n"
                    elif line.startswith("title: "):
                        comeco = True
                        content += line + "\n"
                return content
            
            except Exception as e:
                wait = 2 ** attempt
                print(f"⚠️ {model_name} call failed for '{topic}' (attempt {attempt + 1}) — retrying in {wait}s: {e}")
                time.sleep(wait)

        print(f"⚡ Switching to fallback model after failures: {model_name}")

    print(f"❌ Failed to generate blog for: {topic} after trying all models")
    return ""

def generate_image(prompt):

    # --- Clean up and enhance the prompt for better image quality ---
    base_prompt = prompt.strip()
    if not base_prompt.lower().startswith("a "):
        base_prompt = "A " + base_prompt
    enhanced_prompt = base_prompt + ", digital art, trending on artstation, cinematic lighting, 4k resolution"
    print(f"🎨 Final prompt sent to AUTOMATIC1111: {enhanced_prompt}")
    prompt = enhanced_prompt
    # ---------------------------------------------------------------

    os.makedirs(".cache/images", exist_ok=True)
    cache_key = hashlib.sha256(prompt.encode()).hexdigest()
    webp_path = f".cache/images/{cache_key}.webp"

    if os.path.exists(webp_path):
        print("🖼️ Cached WebP image used")
        return webp_path

    try:
        print(f"🖌️ Sending prompt to AUTOMATIC1111: {prompt}")
        response = requests.post("http://127.0.0.1:7860/sdapi/v1/txt2img", json={
            "prompt": prompt,
            "width": 768,
            "height": 512,
            "steps": 20,
            "cfg_scale": 7,
            "sampler_name": "DPM++ 2M",
        }, timeout=300)
        r = response.json()
        image_data = r['images'][0]
    except Exception as e:
        print(f"❌ Failed to generate image via AUTOMATIC1111: {e}")
        raise

    # Decode and save the image
    img = Image.open(BytesIO(base64.b64decode(image_data)))
    png_path = f".cache/images/{cache_key}.png"
    img.save(png_path)

    # Also save WebP version (optimized)
    webp_path = os.path.splitext(png_path)[0] + ".webp"
    try:
        img.save(webp_path, format="WEBP", quality=80)
        return webp_path
    except Exception as e:
        print(f"⚠️ Failed to convert image to WebP, using PNG instead: {e}")
        return png_path

def upload_page(title, slug, content, media_id=None, status="publish"):
    url = f"{WP_URL}/wp-json/wp/v2/pages"
    payload = {
        "title": title,
        "slug": slug,
        "status": status,
        "content": content,
    }
    if media_id:
        payload["featured_media"] = media_id

    r = requests.post(url, auth=(WP_USER, WP_PASS), json=payload)
    r.raise_for_status()
    return r.json()

def upload_image_to_wp(image_path, alt_text):
    with open(image_path, 'rb') as img:
        filename = os.path.basename(image_path)
        headers = {
            "Content-Disposition": f"attachment; filename={filename}"
        }
        r = requests.post(f"{WP_URL}/wp-json/wp/v2/media", auth=(WP_USER, WP_PASS), headers=headers, files={'file': img})
        r.raise_for_status()
        media_json = r.json()
        media_id = media_json.get("id")
        media_url = media_json.get("source_url")

        # Retry patch for alt text update with exponential backoff
        import time
        for attempt in range(5):
            try:
                r2 = requests.post(f"{WP_URL}/wp-json/wp/v2/media/{media_id}",
                                   auth=(WP_USER, WP_PASS),
                                   json={"alt_text": alt_text})
                r2.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                wait_time = 2 ** attempt
                print(f"⚠️ Alt text update failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
        return media_id, media_url

def upload_post(title, slug, content, meta_title, meta_desc, keyphrase, media_id, publish_date=None, publish_date_gmt=None, status="draft", post_id=None ):
    payload = {
        "title": title,
        "slug": slug,
        "status": status,
        "content": content,
        "excerpt": meta_desc,
        "featured_media": media_id,
        "categories": [6],  # Replace with real IDs
        "tags": [12, 34]    # Replace with real IDs
    }
    print(f"Payload: {payload}")
    if publish_date:
        payload["date"] = publish_date
    if publish_date_gmt:
        payload["date_gmt"] = publish_date_gmt

    if post_id:
        url = f"{WP_URL}/wp-json/wp/v2/posts/{post_id}"
        method = requests.post  # Use requests.patch if partial update is preferred
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

def parse_generated_text(raw: str) -> dict:
    """
    Parse a Markdown document with YAML front-matter or plain metadata and render the body as HTML.
    """
    import re
    # Detect plain metadata without fences (raw begins with "title:")
    if raw.lstrip().startswith("title:"):
        # Split header and body on the literal 'body_html:' marker
        parts = re.split(r'\nbody_html:\s*\|\s*', raw, maxsplit=1)
        header_text = parts[0]
        body_md = parts[1] if len(parts) > 1 else ""
        # Parse metadata using YAML
        try:
            metadata = yaml.safe_load(header_text) or {}
        except Exception:
            metadata = {}
        # Strip leading indentation on body lines
        body_lines = [line.lstrip() for line in body_md.splitlines()]
        md_content = "\n".join(body_lines)
        # Render markdown to HTML
        html_body = markdown.markdown(md_content, extensions=['extra','sane_lists']) if markdown else md_content
        data = metadata.copy()
        data['body_html'] = html_body
        print(f"📄 Parsed data from plain metadata: {data}")
        return data

    # Otherwise, assume proper front-matter fences and use frontmatter.loads
    post = frontmatter.loads(raw)
    metadata = getattr(post, 'metadata', {}) or {}
    content_md = getattr(post, 'content', '') or ""
    html_body = markdown.markdown(content_md, extensions=['extra','sane_lists']) if markdown else content_md
    data = metadata.copy()
    data['body_html'] = html_body
    print(f"📄 Parsed data from fenced frontmatter: {data}")
    return data

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

def inject_adsense_snippet(html):
    ad_html = """
<!-- Google AdSense Ad -->
<div style="margin: 2em 0; padding: 1em; border-top: 1px dashed #ccc; font-family: 'Lora', Georgia, serif;">
  <small style="display: block; text-align: center; color: #888; margin-bottom: 0.5em;">
    🪙 Supports Lipe Land
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
    return prompts

# --- Helper: Inject inline images for figcaptions ---
import re

def inject_inline_images(html: str, prompts: list[str], alt_text: str) -> str:
    """
    For each caption prompt, generate an image, upload it, and wrap the existing <figcaption>
    in a <figure> with the new <img>.
    """
    for prompt in prompts:
        img_path = generate_image(prompt)
        mid, url = upload_image_to_wp(img_path, alt_text)
        # Wrap the figcaption with a figure/html
        pattern = fr'(<figcaption>\s*{re.escape(prompt)}\s*</figcaption>)'
        replacement = rf'<figure><img src="{url}" alt="{alt_text}"/>\1</figure>'
        html = re.sub(pattern, replacement, html, flags=re.IGNORECASE)
    return html

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

def fetch_trending_topics(count=5):
    api_key = GNEWS_API_KEY
    url = f"https://gnews.io/api/v4/top-headlines?token={api_key}&lang=en&max={count}"
    res = requests.get(url)
    articles = res.json().get("articles", [])
    return [article["title"] for article in articles]

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
    # Try loading cache
    cached = load_cached_post(idea, keyphrase)
    if cached:
        print(f"💾 Using pre-existing cached content for: {idea}")
        raw = cached
        parsed = parse_generated_text(raw)
        # Retry generation if critical fields are missing or contain placeholders
        max_retries = 5
        retries = 0
        while (not parsed.get("title") or not parsed.get("body_html") or parsed.get("body_html", "").startswith("[Your article content here]")) and retries < max_retries:
            print(f"⚠️ Incomplete parse detected (title or body missing). Regenerating attempt {retries+1}...")
            raw = generate_blog_components(idea)
            parsed = parse_generated_text(raw)
            retries += 1
        if not parsed.get("title") or not parsed.get("body_html") or parsed.get("body_html", "").startswith("[Your article content here]"):
            raise Exception("❌ Failed to generate valid content after retries.")
    else:
        raw = generate_blog_components(idea)
        parsed = parse_generated_text(raw)
        # Retry generation if critical fields are missing or contain placeholders
        max_retries = 5
        retries = 0
        while (not parsed.get("title") or not parsed.get("body_html") or parsed.get("body_html", "").startswith("[Your article content here]")) and retries < max_retries:
            print(f"⚠️ Incomplete parse detected (title or body missing). Regenerating attempt {retries+1}...")
            raw = generate_blog_components(idea)
            parsed = parse_generated_text(raw)
            retries += 1
        if not parsed.get("title") or not parsed.get("body_html") or parsed.get("body_html", "").startswith("[Your article content here]"):
            raise Exception("❌ Failed to generate valid content after retries.")

    # Build title, slug, etc.
    title = parsed["title"]
    slug = parsed["slug"].lower().replace(" ", "-")
    keyphrase_final = parsed["keyphrase"] or keyphrase
    meta_title = parsed["meta_title"][:60]
    meta_desc = parsed["meta_desc"][:155]
    synonyms = parsed.get("synonyms", [])
    image_prompt = parsed["image_prompt"] or title
    alt_text = parsed.get("alt_text", "")
    # Use only the clean HTML body as content
    content = parsed["body_html"].lstrip("|").lstrip()

    # Fetch all post metadata for internal linking
    all_posts = fetch_all_posts_metadata()
    # Inject AdSense snippet before internal links
    content = inject_adsense_snippet(content)
    content = enrich_with_internal_links(content, all_posts)

    # --- Generate and upload images based on detected captions ---
    prompts = extract_image_prompts_from_body(content)
    if prompts:
        # Inline images for any LLM-generated figcaptions
        content = inject_inline_images(content, prompts, alt_text)
        media_id = None
    else:
        media_id = None
        # Only add a contextual image if none exists already
        if '<img ' not in content:
            print(f"🎨 Generating contextual image...")
            img_path = generate_image(image_prompt)
            mid, url = upload_image_to_wp(img_path, alt_text)
            # Do not prepend the image to the content
            media_id = mid

    # Final HTML transformations on full content
    # (Do all markdown/list/inline parsing after image insertion)
    if markdown:
        content = markdown.markdown(content, extensions=['extra', 'sane_lists'])
    content = auto_convert_lists(content)
    content = convert_markdown_inline(content)

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

    # Upload as page or post based on flag
    print(f"📝 Uploading {'page' if args.page else 'post'} to WordPress...")
    if args.page:
        if media_id is not None:
            result = upload_page(
                title, slug, content,
                media_id=media_id,
                status=status
            )
        else:
            result = upload_page(
                title, slug, content,
                status=status
            )
        post_link = result.get('link')
    else:
        if media_id is not None:
            result = upload_post(
                title, slug, content,
                meta_title, meta_desc,
                keyphrase_final, media_id,
                publish_date=pub_date,
                publish_date_gmt=pub_date_gmt,
                status=status
            )
        else:
            result = upload_post(
                title, slug, content,
                meta_title, meta_desc,
                keyphrase_final, None,
                publish_date=pub_date,
                publish_date_gmt=pub_date_gmt,
                status=status
            )
        # Patch SEO meta only for posts
        update_seo_meta(result.get('id'), meta_title, meta_desc, keyphrase_final)
        post_link = result.get('link')
    return post_link


# Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gnews', action='store_true', help='Fetch trending topics from GNews')
    parser.add_argument('--interval', type=int, default=900, help='Interval in seconds between fetches (default 900s)')
    parser.add_argument('--days', type=int, default=0, help='Range of past days for randomized publication date')
    parser.add_argument('--page', action='store_true', help='Create pages instead of posts')
    parser.add_argument('--scan-broken', action='store_true', help='Scan internal post links for 404s and write to broken_pages.txt')
    parser.add_argument('--idea', type=str, help='Provide a manual idea for a blog post')
    parser.add_argument('--keyphrase', type=str, help='Optional keyphrase to pass for the post')
    args, unknown = parser.parse_known_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler("poster.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    tasks = []
    if args.gnews:
        logging.info("📥 Fetching new trending topics from GNews...")
        all_trends = fetch_trending_topics(count=10)
        logging.info(f"🌐 Retrieved {len(all_trends)} headlines.")
        # Redis deduplication
        r = redis.Redis(
            host=REDIS_HOST,
            port=6379,
            password=REDIS_PASSWORD,
            decode_responses=True
        )
        for trend in all_trends:
            trend_key = f"trend:{hashlib.sha1(trend.encode()).hexdigest()}"
            if r.exists(trend_key):
                logging.info(f"⏩ Skipping already-processed trend: {trend}")
                continue
            r.setex(trend_key, 86400, "seen")  # 24h TTL
            tasks.append((trend, "default-keyphrase"))
    elif args.idea:
        tasks.append((args.idea, args.keyphrase or "default-keyphrase"))
    else:
        parser.error("You must provide --idea or --gnews")

    for idea, keyphrase in tasks:
        link = process_and_publish(idea, keyphrase, args)
        print(f"✅ Published: {link}")

    logging.info("📡 Attempting to send final healthcheck ping...")
    send_healthcheck_ping()
    logging.info("📡 Healthcheck ping execution finished.")

# TODO: Implement internal link enrichment — scan past posts and inject semantic internal links into parsed['body'] based on relevance

if __name__ == '__main__':
    main()