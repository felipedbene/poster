#!/usr/bin/env python3
import os
import requests
import io
import json
import re
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI
import json

# path to file that tracks processed item IDs
PROCESSED_FILE = Path(__file__).resolve().parent / "processed.json"

# load processed IDs or initialize empty set
if PROCESSED_FILE.exists():
    with open(PROCESSED_FILE, "r") as f:
        processed_ids = set(json.load(f))
else:
    processed_ids = set()

def save_processed():
    with open(PROCESSED_FILE, "w") as f:
        json.dump(list(processed_ids), f)

# load .env from the script's directory
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

WP_URL = os.getenv("WORDPRESS_URL")
WP_USER = os.getenv("WORDPRESS_USERNAME")
WP_PASS = os.getenv("WORDPRESS_APP_PASSWORD")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# User’s favorite topics for tone and style guidance
USER_TOPICS = [
    "Euclid: The Elements",
    "Clarice Lispector",
    "Schopenhauer",
    "Nietzsche",
    "Mario Bortoloto",
    "Beatniks",
    "Linus Torvalds",
    "Open source",
    "Cloud computing",
    "Mathematics"
]

client = OpenAI(api_key=OPENAI_KEY)
AUTH = (WP_USER, WP_PASS)

def generate_image(prompt: str) -> bytes:
    """Generate an image via OpenAI and return the raw PNG bytes."""
    resp = client.images.generate(
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    image_url = resp.data[0].url
    img_resp = requests.get(image_url)
    img_resp.raise_for_status()
    return img_resp.content

def upload_image_to_wp(image_bytes: bytes, filename: str) -> int:
    """Upload image bytes to WordPress media library and return the attachment ID."""
    files = {
        'file': (filename, image_bytes, 'image/png')
    }
    resp = requests.post(
        f"{WP_URL}/wp-json/wp/v2/media",
        auth=AUTH,
        files=files
    )
    resp.raise_for_status()
    return resp.json()['id']

def get_all_items(item_type):
    items = []
    page = 1
    while True:
        r = requests.get(f"{WP_URL}/wp-json/wp/v2/{item_type}", params={"per_page": 100, "page": page}, auth=AUTH)
        if r.status_code != 200:
            break
        data = r.json()
        if not data:
            break
        items.extend(data)
        page += 1
    return items

def post_needs_keyphrase(post):
    return not post.get("meta", {}).get("yoast_wpseo_focuskw")

def generate_keyphrase(title, content):
    prompt = f"""You're a seasoned SEO assistant. Suggest a short, relevant SEO keyphrase based on this blog post:\n\nTitle: {title}\n\nContent:\n{content[:1000]}\n\nRespond only with the phrase."""
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
    )
    return response.choices[0].message.content.strip().strip('"')

def update_post_seo(post_id, keyphrase):
    url = f"{WP_URL}/wp-json/wp/v2/posts/{post_id}"
    payload = {
        "meta": {
            "yoast_wpseo_focuskw": keyphrase
        }
    }
    r = requests.patch(url, auth=AUTH, json=payload)
    if r.ok:
        print(f"✅ Updated post {post_id} with keyphrase: {keyphrase}")
    else:
        print(f"❌ Failed to update post {post_id}: {r.text}")

def ai_meta(title, content):
    prompt = (
        "You are an SEO specialist who writes with influences from "
        + ", ".join(USER_TOPICS[:-1])
        + ", and " + USER_TOPICS[-1]
        + ".\n"
        "Given this title and content, propose:\n"
        "1) An optimized HTML <title> (<=60 chars)\n"
        "2) A compelling meta description (<=155 chars)\n"
        "Respond in JSON: { 'title': '...', 'description': '...' }."
    )
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
    )
    raw = response.choices[0].message.content.strip()
    # Extract the JSON object from the raw response
    match = re.search(r'(\{.*\})', raw, re.DOTALL)
    json_str = match.group(1) if match else raw
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Fallback: convert single quotes to double quotes
        return json.loads(json_str.replace("'", '"'))

def rewrite_content(article_body):
    prompt = (
        "Rewrite the following article body for clarity, style, and SEO. "
        "Respond only with valid WP-compatible HTML—use <p>, <h2>, <pre><code>, <img>, and similar HTML tags. "
        "Do NOT use any Markdown syntax (no ### or ```). "
        "Infuse the style and philosophical depth of " + ", ".join(USER_TOPICS[:-1]) + ", and " + USER_TOPICS[-1] + ". "
        "Preserve code blocks, images, and structure, and include any necessary SEO-specific tags."
    )
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
    )
    return response.choices[0].message.content.strip()

def run():
    for content_type in ("posts", "pages"):
        items = get_all_items(content_type)
        for item in items:
            post_id = item["id"]
            if post_id in processed_ids:
                print(f"⏭️ Already processed {content_type[:-1]} {post_id}, skipping.")
                continue
            title = item.get("title", {}).get("rendered", "").strip()
            if post_needs_keyphrase(item):
                body = item.get("content", {}).get("rendered", "")
                if not body.strip():
                    print(f"⚠️ Skipping empty post {post_id}: '{title}'")
                    continue
                print(f"🔍 Processing post {post_id}: '{title}'")
                # Generate full SEO metadata and rewritten content
                meta = ai_meta(title, body)
                rewritten = rewrite_content(body)
                # If item has no featured image, generate and upload one
                if not item.get("featured_media"):
                    img_prompt = f"An evocative illustration for blog post titled: '{title}'."
                    print(f"🖼️ Generating image for post {post_id}")
                    img_bytes = generate_image(img_prompt)
                    img_name = f"post-{post_id}-featured.png"
                    media_id = upload_image_to_wp(img_bytes, img_name)
                    print(f"🆙 Uploaded image as media ID {media_id}")
                else:
                    media_id = item.get("featured_media")
                # Prepare payload to update title, meta description, focus keyword, content, and featured image
                full_payload = {
                    "title": meta.get("title"),
                    "meta": {
                        "yoast_wpseo_metadesc": meta.get("description"),
                        "yoast_wpseo_focuskw": meta.get("title")
                    },
                    "content": rewritten,
                    "featured_media": media_id
                }
                endpoint = f"{WP_URL}/wp-json/wp/v2/{content_type}/{post_id}"
                resp = requests.patch(endpoint, auth=AUTH, json=full_payload)
                if resp.ok:
                    print(f"✅ Fully updated post {post_id}: '{meta.get('title')}'")
                    updated = requests.get(endpoint, auth=AUTH).json()
                    print(f"🔗 URL: {updated.get('link')}")
                    processed_ids.add(post_id)
                    save_processed()
                else:
                    print(f"❌ Failed to fully update post {post_id}: {resp.text}")
            else:
                print(f"⏭️ Skipped post {post_id}: '{title}' (focus keyphrase exists)")

if __name__ == "__main__":
    run()