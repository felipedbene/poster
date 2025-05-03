import argparse
import requests
import os
from dotenv import load_dotenv
load_dotenv()
from bs4 import BeautifulSoup

WP_URL = os.getenv("WP_URL", "https://debene.dev")
WP_USER = os.getenv("WP_USER")
WP_PASS = os.getenv("WP_PASS")

auth = (WP_USER, WP_PASS)

def get_all(endpoint):
    results = []
    page = 1
    while True:
        r = requests.get(f"{WP_URL}/wp-json/wp/v2/{endpoint}?per_page=100&page={page}", auth=auth)
        if r.status_code != 200:
            break
        data = r.json()
        if not data:
            break
        results.extend(data)
        page += 1
    return results

def extract_used_media_ids(posts):
    used_ids = set()
    for post in posts:
        # Featured media
        if post.get("featured_media"):
            used_ids.add(post["featured_media"])
        # Inline media
        soup = BeautifulSoup(post.get("content", {}).get("rendered", ""), "html.parser")
        for img in soup.find_all("img"):
            src = img.get("src", "")
            if "/uploads/" in src:
                id_guess = src.split("/")[-1].split("-")[0]
                used_ids.add(id_guess)
    return used_ids

def main():
    parser = argparse.ArgumentParser(description="Find and optionally delete unused WordPress media.")
    parser.add_argument("--dry-run", action="store_true", help="Only list unused media, do not delete them.")
    args = parser.parse_args()

    print("🔍 Fetching media...")
    media = get_all("media")
    print(f"📸 Total media found: {len(media)}")

    posts = get_all("posts")
    pages = get_all("pages")
    all_content = posts + pages
    used_ids = extract_used_media_ids(all_content)

    unused = [m for m in media if str(m["id"]) not in used_ids]
    print(f"🧹 Unused media: {len(unused)}")

    for item in unused:
        print(f"- {item['id']}: {item['source_url']}")
        if not args.dry_run:
            delete_url = f"{WP_URL}/wp-json/wp/v2/media/{item['id']}?force=true"
            resp = requests.delete(delete_url, auth=auth)
            if resp.status_code == 200:
                print(f"✅ Deleted media ID {item['id']}")
            else:
                print(f"❌ Failed to delete media ID {item['id']} ({resp.status_code})")

if __name__ == "__main__":
    main()