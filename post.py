import os
import argparse
import openai
import requests
import hashlib
import json
import csv
import datetime
import random
import re
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Config
WP_USER = os.getenv("WORDPRESS_USERNAME")
WP_PASS = os.getenv("WORDPRESS_APP_PASSWORD")
WP_URL = os.getenv("WORDPRESS_URL")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_KEY

def generate_blog_components(idea, keyphrase):
    os.makedirs(".cache/posts", exist_ok=True)
    cache_key = hashlib.sha256(f"{idea}:{keyphrase}".encode()).hexdigest()
    cache_path = f".cache/posts/{cache_key}.json"
    
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cached = f.read()
        if "IMAGE_PROMPT:" not in cached:
            print("⚠️ Cached version outdated — regenerating...")
        else:
            print(f"💾 Using cached blog content for: \"{idea}\" + \"{keyphrase}\"")
            return cached

    prompt = f"""
    You are a witty, experienced DevOps engineer and storyteller writing for a WordPress audience. Based on this inspiration:
    Idea: "{idea}"
 
    Craft a single unified narrative (no bullets, no numbered or disconnected sections) of at least 600 words that:
      - Starts with an engaging hook that mentions the keyphrase "{keyphrase}".
      - Weaves together your migration from Google Photos to Immich, Kubernetes debugging, and the WordPress rebuild into a cohesive weekend story.
      - Reflects on your experiences and ties back to the concept of "{keyphrase}" with a strong conclusion.
      - Naturally sprinkles 2–4 synonyms or related phrases of "{keyphrase}".
      - Includes at least one outbound link (<a href="https://">) and one internal link (<a href="/") within the narrative.
      - Does NOT repeat the title anywhere in the body.
 
    Additionally generate:
      - A Yoast-style SEO meta title
      - A meta description (under 160 characters)
      - A suggested slug
      - An image description prompt for DALL·E
      - Alt text for the image
 
    Output format exactly:
    ---
    TITLE: <title>
    META_TITLE: <seo title>
    META_DESC: <meta description>
    SLUG: <slug>
    KEYPHRASE: <keyphrase>
    SYNONYMS: <comma-separated synonyms>
    IMAGE_PROMPT: <image prompt>
    IMAGE_ALT: <alt text>
    BODY:
    <full blog body in HTML without repeating title>
    ---
    """
    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )
    blog_content = response.choices[0].message.content
    with open(cache_path, "w") as f:
        f.write(blog_content)
    return blog_content

def generate_image(prompt):
    os.makedirs(".cache/images", exist_ok=True)
    cache_key = hashlib.sha256(prompt.encode()).hexdigest()
    image_path = f".cache/images/{cache_key}.png"
    
    if os.path.exists(image_path):
        print(f"💾 Using cached image for prompt")
        return image_path

    try:
        response = openai.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
    except Exception as e:
        print(f"❌ Failed to generate image with DALL·E: {e}")
        raise

    try:
        image_url = response.data[0].url
    except (AttributeError, IndexError) as e:
        print(f"❌ Unexpected response structure from DALL·E: {e}")
        raise

    try:
        img_resp = requests.get(image_url, timeout=10)
        img_resp.raise_for_status()
        image_data = img_resp.content
    except Exception as e:
        print(f"❌ Failed to download generated image: {e}")
        raise
    with open(image_path, "wb") as f:
        f.write(image_data)
    return image_path

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
        media_json = r.json()
        media_id = media_json.get("id")
        media_url = media_json.get("source_url")

        # Update alt text
        requests.post(f"{WP_URL}/wp-json/wp/v2/media/{media_id}",
                      auth=(WP_USER, WP_PASS),
                      json={"alt_text": alt_text})
        return media_id, media_url

def upload_post(title, slug, content, meta_title, meta_desc, keyphrase, media_id, publish_date=None, publish_date_gmt=None, status="draft"):
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
    if publish_date:
        payload["date"] = publish_date
    if publish_date_gmt:
        payload["date_gmt"] = publish_date_gmt
    r = requests.post(f"{WP_URL}/wp-json/wp/v2/posts", auth=(WP_USER, WP_PASS), json=payload)
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

def parse_generated_text(raw_text):
    parsed = {}
    current = None
    lines = raw_text.splitlines()
    body = []
    for line in lines:
        if line.startswith("TITLE:"):
            parsed['title'] = line.replace("TITLE:", "").strip()
        elif line.startswith("META_TITLE:"):
            parsed['meta_title'] = line.replace("META_TITLE:", "").strip()
        elif line.startswith("META_DESC:"):
            parsed['meta_desc'] = line.replace("META_DESC:", "").strip()
        elif line.startswith("SLUG:"):
            parsed['slug'] = line.replace("SLUG:", "").strip()
        elif line.startswith("KEYPHRASE:"):
            parsed['keyphrase'] = line.replace("KEYPHRASE:", "").strip()
        elif line.startswith("SYNONYMS:"):
            parsed['synonyms'] = line.replace("SYNONYMS:", "").strip()
        elif line.startswith("IMAGE_PROMPT:"):
            parsed['image_prompt'] = line.replace("IMAGE_PROMPT:", "").strip()
        elif line.startswith("IMAGE_ALT:"):
            parsed['alt_text'] = line.replace("IMAGE_ALT:", "").strip()
        elif line.startswith("BODY:"):
            current = 'body'
        elif current == 'body':
            body.append(line)
    parsed['body'] = "\n".join(body)
    return parsed

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, help='Path to CSV file with columns: idea,keyphrase')
    parser.add_argument('--days', type=int, default=0, help='Range of past days for randomized publication date')
    parser.add_argument('--page', action='store_true', help='Create pages instead of posts')
    parser.add_argument('--scan-broken', action='store_true', help='Scan internal post links for 404s and write to broken_pages.txt')
    args = parser.parse_args()
    if args.scan_broken:
        scan_broken_links()
        return

    tasks = []
    if args.csv:
        with open(args.csv, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                tasks.append((row['idea'], row['keyphrase']))
    else:
        parser.add_argument('--idea', required=True)
        parser.add_argument('--keyphrase', required=True)
        tasks.append((args.idea, args.keyphrase))

    published_links = []

    for idx, (idea, keyphrase) in enumerate(tasks, start=1):
        print(f"🧠 Generating blog post {idx}/{len(tasks)} for idea: {idea!r}")
        raw = generate_blog_components(idea, keyphrase)
        parsed = parse_generated_text(raw)

        # Remove duplicated plain-title if present
        title_plain = parsed['title']
        if parsed['body'].lstrip().startswith(title_plain):
            parsed['body'] = parsed['body'].lstrip()[len(title_plain):].lstrip()

        print(f"🎨 Generating contextual image for post {idx}...")
        image_path = generate_image(parsed['image_prompt'])

        print(f"📤 Uploading image {idx}...")
        media_id, media_url = upload_image_to_wp(image_path, parsed['alt_text'])

        # Determine randomized publish date if requested
        pub_date = pub_date_gmt = None
        if args.days > 0:
            # random offset within args.days and within a day
            delta_days = random.uniform(0, args.days)
            delta_secs = random.uniform(0, 86400)
            dt = datetime.datetime.now() - datetime.timedelta(days=delta_days, seconds=delta_secs)
            pub_date = dt.strftime("%Y-%m-%dT%H:%M:%S")
            pub_date_gmt = dt.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
            status = "publish"
        else:
            status = "draft"

        # Upload as page or post based on flag
        print(f"📝 Uploading {'page' if args.page else 'post'} {idx} to WordPress...")
        if args.page:
            result = upload_page(
                parsed['title'], parsed['slug'], parsed['body'],
                media_id=media_id,
                status=status
            )
        else:
            result = upload_post(
                parsed['title'], parsed['slug'], parsed['body'],
                parsed['meta_title'], parsed['meta_desc'],
                parsed['keyphrase'], media_id,
                publish_date=pub_date,
                publish_date_gmt=pub_date_gmt,
                status=status
            )
            # Patch SEO meta only for posts
            update_seo_meta(result.get('id'), parsed['meta_title'], parsed['meta_desc'], parsed['keyphrase'])
        post_link = result.get('link')
        published_links.append(post_link)
        print(f"✅ Published {idx}: {post_link}")

    print("📄 Summary of published post links:")
    for link in published_links:
        print(link)

if __name__ == '__main__':
    main()