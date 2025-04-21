# SEO Fixer

Automate end-to-end SEO improvements for your WordPress **posts and pages** using Python, OpenAI, and the WP REST API.

## Overview

**SEO Fixer** bulk-fetches your site content, using OpenAI to generate:
- Optimized HTML `<title>` (≤ 60 chars)  
- Engaging meta descriptions (≤ 155 chars)  
- Focus keywords for Yoast SEO  
- Full HTML rewrites (preserving code blocks & images)  
- Bespoke featured images (default 512×512 WebP via OpenAI)

Then it patches each post or page in place and prints its public URL when done.

## Assumptions

- **WordPress** site with REST API enabled  
- An **Application Password** for your WP user  
- **Yoast SEO** plugin installed and active  
- **OpenAI** account with API key and `images-generations` permissions  
- Python 3.9+ environment  

## Getting Started

1. **Clone the repo**  
   ```bash
   git clone https://github.com/yourusername/seo-fixer.git
   cd seo-fixer
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Create a `.env`** at project root:
   ```ini
   WORDPRESS_URL=https://your-blog.com
   WORDPRESS_USERNAME=your_wp_user
   WORDPRESS_APP_PASSWORD=your_app_password
   OPENAI_API_KEY=sk-...
   ```

4. **(Optional) Register for a free OpenAI account** at https://platform.openai.com/ and add your API key.

## Usage

```bash
# Generate a single draft post
python seo_fixer.py --idea "Your topic here" --keyphrase "your focus keyword"

# Generate a single draft page
python seo_fixer.py --idea "About Us" --keyphrase "company history" --page

# Bulk from CSV: CSV columns are `idea,keyphrase`
python seo_fixer.py --csv ideas.csv

# Publish with randomized dates within the past N days
python seo_fixer.py --idea "Topic" --keyphrase "Keyword" --days 7

# Scan for broken internal links
python seo_fixer.py --scan-broken

# Process both posts and pages
python seo_fixer.py --idea "Your topic here" --keyphrase "your focus keyword" --page
```

- After each upload, the script prints:
  ```
  ✅ Published 1: https://your-blog.com/2025/04/20/your-slug/
  ```

- Note: The script uses a cache directory to store temporary files.

## Features

- **Bulk-fetch** all posts & pages (pagination supported)  
- **AI-driven metadata** update via Yoast fields  
- **HTML rewrite** preserving structure & code  
- **Image generation** and automatic media upload  
- **Broken-link scanner** outputs `broken_pages.txt`  

## Development & Testing

- Run in **dry-run** mode by inspecting console output before publishing.  
- Confirm `.env` is loaded correctly by printing environment variables.

## Troubleshooting

- **WP-CLI** caches: clear any cache plugin (e.g., `wp cache flush`).  
- **Cloudflare**: purge via Dashboard or API:
  ```bash
  curl -X POST "https://api.cloudflare.com/client/v4/zones/$CF_ZONE_ID/purge_cache" \
    -H "Authorization: Bearer $CF_API_TOKEN" \
    -H "Content-Type: application/json" \
    --data '{"purge_everything":true}'
  ```
- **Permissions**: ensure your Application Password user has `edit_posts` and `upload_files`.
- **PHP Upload Limits**: adjust your `php.ini` settings for `upload_max_filesize` and `post_max_size` if you encounter issues with file uploads.
- **Image Size Adjustments**: configure image dimensions in the script if you require a different size than the default 512×512.

## License

MIT © Your Name