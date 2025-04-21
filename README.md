# SEO Fixer

A Python script that automates SEO improvements across your WordPress posts **and** pages using OpenAI and the WP REST API.

## Features

- **Bulk-fetch** all posts & pages (pagination supported).
- **AI‑driven metadata**:  
  - Generates optimized HTML `<title>` (≤ 60 chars)  
  - Crafts compelling meta descriptions (≤ 155 chars)  
  - Suggests focus keywords via Yoast SEO fields
- **Content rewrite**:  
  - Rewrites body HTML (no Markdown)  
  - Preserves code blocks, images & structure  
  - Infuses style influences from Euclid, Lispector, Schopenhauer, Nietzsche, Mario Bortoloto, Beatniks, Linus Torvalds, open‑source ethos, cloud computing & mathematics
- **Featured image generation**:  
  - Auto‑generates a bespoke 1024×1024 PNG via OpenAI  
  - Uploads to WP media library if none exists
- **One‑shot updates**:  
  - Patches `title`, `content`, `yoast_wpseo_metadesc`, `yoast_wpseo_focuskw`, and `featured_media`
  - Prints each item’s public URL after update

## Requirements

- Python 3.9+
- `pip install -r requirements.txt`  
  (includes `openai`, `requests`, `python-dotenv`)

## Setup

1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/seo-fixer.git
   cd seo-fixer