import os
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

load_dotenv()

WP_USER = os.getenv("WORDPRESS_USERNAME")
WP_PASS = os.getenv("WORDPRESS_APP_PASSWORD")
WP_URL = os.getenv("WORDPRESS_URL", "").rstrip("/") + "/wp-json/wp/v2/posts"

if not WP_USER or not WP_PASS or not WP_URL:
    raise SystemExit("❌ Missing env vars: make sure WORDPRESS_USERNAME, WORDPRESS_APP_PASSWORD, and WORDPRESS_URL are set")

auth = HTTPBasicAuth(WP_USER, WP_PASS)

def fetch_post_ids():
    ids = []
    page = 1
    while True:
        res = requests.get(WP_URL, params={"per_page": 100, "page": page}, auth=auth)
        if res.status_code != 200:
            print("❌ Error fetching posts:", res.text)
            break
        posts = res.json()
        if not posts:
            break
        ids += [post["id"] for post in posts]
        page += 1
    return ids

def delete_post(post_id):
    url = f"{WP_URL}/{post_id}?force=true"
    res = requests.delete(url, auth=auth)
    if res.status_code == 200:
        print(f"🗑️ Deleted post {post_id}")
    else:
        print(f"⚠️ Failed to delete post {post_id} → {res.status_code}: {res.text}")

if __name__ == "__main__":
    post_ids = fetch_post_ids()
    print(f"🔍 Found {len(post_ids)} posts")
    for pid in post_ids:
        delete_post(pid)