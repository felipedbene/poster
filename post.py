import os
import argparse
import openai
import httpx
from openai import OpenAI
import requests
import hashlib
import json
import csv
import datetime
import random
import re
from dotenv import load_dotenv
from PIL import Image
import redis
import logging
import time
import sys

# Load .env
load_dotenv()

# Healthcheck API key (optional)
HC_APIKEY = os.getenv("HC_APIKEY")

# Config
WP_USER = os.getenv("WORDPRESS_USERNAME")
WP_PASS = os.getenv("WORDPRESS_APP_PASSWORD")
WP_URL = os.getenv("WORDPRESS_URL")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

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

openai.api_key = OPENAI_KEY

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


def generate_blog_components(trend):
    os.makedirs(".cache/posts", exist_ok=True)
    cache_key = hashlib.sha256(trend.encode()).hexdigest()
    cache_path = f".cache/posts/{cache_key}.json"
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cached = f.read()
        if "IMAGE_PROMPT:" not in cached:
            print("⚠️ Cached version outdated — regenerating...")
        else:
            print(f"💾 Cached blog retrieved for: {trend}")
            return cached

    prompt = f"""
    You are a thoughtful, well-informed, and engaging writer crafting professional-grade blog content for a general but intelligent audience. Based on the following inspiration:
    Trending Headline: "{trend}"
    Today is {datetime.datetime.now().strftime('%Y-%m-%d')}.
    You are physically located in the United States ( Chicago, IL ) and the blog is for a US-based audience.

    Please generate a polished and insightful blog post (at least 1000 words), written in a clear, professional, and slightly conversational tone that:
      - Seamlessly incorporates the keyphrase **(choose a suitable SEO keyphrase)** at least 2–3 times for SEO purposes.
      - Includes at least one useful and relevant outbound link (<a href="https://">).
      - Balances clarity and depth, making complex topics accessible without oversimplifying.
      - Reflects careful structure and flow, using transitions to connect ideas fluidly.
      - Is suitable for readers with diverse backgrounds — not just technical.
      - Fun

    Additionally generate:
      - A compelling SEO meta title
      - A concise meta description (under 160 characters)
      - A URL-friendly slug
      - An image generation prompt for DALL·E
      - Alt text for the generated image
      - The main idea (title) and the SEO keyphrase used

    ---
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
    for attempt in range(3):
        try:
            print(f"⏳ Calling OpenAI for: {trend} (attempt {attempt + 1})")
            response = openai.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6
            )
            blog_content = response.choices[0].message.content
            with open(cache_path, "w") as f:
                f.write(blog_content)
            return blog_content
        except Exception as e:
            wait = 2 ** attempt
            print(f"⚠️ OpenAI call failed for '{trend}' (attempt {attempt + 1}) — retrying in {wait}s: {e}")
            import time
            time.sleep(wait)

    print(f"❌ Failed to generate blog for: {trend} after 3 attempts")
    return ""

def generate_image(prompt):
    os.makedirs(".cache/images", exist_ok=True)
    cache_key = hashlib.sha256(prompt.encode()).hexdigest()
    png_path = f".cache/images/{cache_key}.png"
    webp_path = f".cache/images/{cache_key}.webp"
    if os.path.exists(webp_path):
        print("🖼️ Cached WebP image used")
        return webp_path
    image_path = png_path

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
    # Convert to WebP to reduce size
    webp_path = os.path.splitext(image_path)[0] + ".webp"
    try:
        img = Image.open(image_path)
        img.save(webp_path, format="WEBP", quality=80)
        return webp_path
    except Exception as e:
        print(f"❌ Failed to convert image to WebP: {e}")
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
    # --- Trim SEO fields to safe limits
    if 'meta_title' in parsed and parsed['meta_title']:
        parsed['meta_title'] = parsed['meta_title'][:60].strip()
    if 'meta_desc' in parsed and parsed['meta_desc']:
        parsed['meta_desc'] = parsed['meta_desc'][:155].strip()
    if 'slug' in parsed and parsed['slug']:
        # Limit slug to 6 hyphenated words
        parsed['slug'] = '-'.join(parsed['slug'].split('-')[:6])
    print("🧠 Parsed SEO metadata →", {
        "meta_title": parsed.get("meta_title"),
        "meta_desc": parsed.get("meta_desc"),
        "keyphrase": parsed.get("keyphrase")
    })
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
            # Avoid linking to self if slug present in body (approximate)
            if f"https://blog.debene.dev/{post_slug}" in parsed_body:
                continue
            links_to_add.append(f'<p>See also: <a href="https://blog.debene.dev/{post_slug}">{post_title}</a></p>')
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
            if not posts:
                break
            for p in posts:
                results.append({
                    "title": p["title"]["rendered"],
                    "slug": p["slug"],
                    "id": p["id"]
                })
            page += 1
        except requests.exceptions.RequestException as e:
            logging.warning(f"⚠️ Failed to fetch page {page}: {e}")
            break
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gnews', action='store_true', help='Fetch trending topics from GNews')
    parser.add_argument('--interval', type=int, default=900, help='Interval in seconds between fetches (default 900s)')
    parser.add_argument('--days', type=int, default=0, help='Range of past days for randomized publication date')
    parser.add_argument('--page', action='store_true', help='Create pages instead of posts')
    parser.add_argument('--scan-broken', action='store_true', help='Scan internal post links for 404s and write to broken_pages.txt')
    args, unknown = parser.parse_known_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler("poster.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    if args.scan_broken:
        scan_broken_links()
        return

    if args.gnews:
        logging.info("📥 Fetching new trending topics from GNews...")
        all_trends = fetch_trending_topics(count=10)
        logging.info(f"🌐 Retrieved {len(all_trends)} headlines.")
        for trend in all_trends:
            logging.info(f"🧵 New trend: {trend}")
            # Redis deduplication
            r = redis.Redis(
                host=REDIS_HOST,
                port=6379,
                password=REDIS_PASSWORD,
                decode_responses=True
            )
            trend_key = f"trend:{hashlib.sha1(trend.encode()).hexdigest()}"
            if r.exists(trend_key):
                logging.info(f"⏩ Skipping already-processed trend: {trend}")
                continue
            r.setex(trend_key, 86400, "seen")  # 24h TTL
            for attempt in range(3):
                try:
                    blog_raw = generate_blog_components(trend)
                    parsed = parse_generated_text(blog_raw)
                    # Determine if post already exists by slug
                    existing_posts = fetch_all_posts_metadata()
                    existing = next((p for p in existing_posts if p["slug"] == parsed["slug"]), None)
                    post_id = existing.get("id") if existing else None
                    idea = parsed['title']
                    keyphrase = parsed['keyphrase']

                    all_posts = existing_posts
                    parsed['body'] = enrich_with_internal_links(parsed['body'], all_posts)

                    title_plain = parsed['title']
                    if parsed['body'].lstrip().startswith(title_plain):
                        parsed['body'] = parsed['body'].lstrip()[len(title_plain):].lstrip()

                    image_path = generate_image(parsed['image_prompt'])
                    media_id, media_url = upload_image_to_wp(image_path, parsed['alt_text'])

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

                    parsed['keyphrase'] = parsed.get('keyphrase') or keyphrase
                    logging.info(f"📝 Posting blog for: {parsed['title']} (from: {trend})")
                    result = upload_post(
                        parsed['title'], parsed['slug'], parsed['body'],
                        parsed['meta_title'], parsed['meta_desc'],
                        parsed['keyphrase'], media_id,
                        publish_date=pub_date,
                        publish_date_gmt=pub_date_gmt,
                        status=status,
                        post_id=post_id
                    )
                    logging.info(f"✅ Upsert complete → {result.get('link')}")
                    update_seo_meta(result.get('id'), parsed['meta_title'], parsed['meta_desc'], parsed['keyphrase'])
                    logging.info(f"📢 Published: {result.get('link')}")
                    break
                except Exception as e:
                    wait = 2 ** attempt
                    logging.warning(f"⚠️ OpenAI call failed (attempt {attempt+1}) for trend: {trend} → retrying in {wait}s\nReason: {e}")
                    time.sleep(wait)
        return
    else:
        parser.add_argument('--idea', required=True)
        parser.add_argument('--keyphrase', required=True)
        args2 = parser.parse_args(unknown, namespace=args)
        tasks = []
        tasks.append((args2.idea, args2.keyphrase))

        published_links = []
        for idx, (idea, keyphrase) in enumerate(tasks, start=1):
            print(f"🧠 Generating blog post {idx}/{len(tasks)} for idea: {idea!r}")
            cached = load_cached_post(idea, keyphrase)
            if cached:
                print(f"💾 Using pre-existing cached content for: {idea}")
                raw = cached
            else:
                raw = generate_blog_components(idea, keyphrase)
            parsed = parse_generated_text(raw)

            all_posts = fetch_all_posts_metadata()
            parsed['body'] = enrich_with_internal_links(parsed['body'], all_posts)

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

            # Fallback for missing keyphrase
            parsed['keyphrase'] = parsed.get('keyphrase') or keyphrase
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

        # Healthcheck ping if enabled
        if HC_APIKEY:
            try:
                requests.get(f"https://hc-ping.com/{HC_APIKEY}", timeout=5)
                print("✅ Healthcheck ping sent.")
            except Exception as e:
                print(f"⚠️ Failed to send healthcheck ping: {e}")

        print("📄 Summary of published post links:")
        for link in published_links:
            print(link)

# TODO: Implement internal link enrichment — scan past posts and inject semantic internal links into parsed['body'] based on relevance

if __name__ == '__main__':
    main()