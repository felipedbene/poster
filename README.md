# Trend Poster

Automatically generate, publish, and update SEO-optimized blog posts to WordPress based on trending headlines — powered by OpenAI and the WP REST API.

## Overview

**Trend Poster** continuously fetches trending headlines (via GNews), uses OpenAI to generate:
- Titles, meta descriptions, and keyphrases
- Full HTML blog posts (1000+ words)
- Featured image prompts (and auto-generated DALL·E images)
- SEO metadata integration (Yoast-compatible)
- Seamless updates of existing posts (via slug detection)

## Assumptions

- A **WordPress** site with REST API enabled
- An **Application Password** for your WP user
- **Yoast SEO** plugin (for meta patching)
- A **GNews API key**
- An **OpenAI** account with `chat-completions` and `images-generations` access
- Python 3.9+

## Getting Started

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/poster.git
   cd poster
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create a `.env` file** with WordPress, OpenAI, GNews, and Redis configuration:
   ```ini
   WORDPRESS_URL=https://your-blog.com
   WORDPRESS_USERNAME=your_wp_user
   WORDPRESS_APP_PASSWORD=your_app_password
   OPENAI_API_KEY=sk-...
   GNEWS_API_KEY=your_gnews_key
   REDIS_HOST=redis-master.wp.svc.cluster.local # This is namespace dependant
   REDIS_PASSWORD=your_redis_password  # Only if Redis auth is enabled
   ```
   Ensure your environment (e.g., `poster-env`) contains these variables for Redis integration.

## Usage

```bash
# Manually trigger a one-time CronJob execution:
kubectl create job --from=cronjob/trend-poster trend-poster-now -n wp
kubectl logs -n wp job/trend-poster-now -f
```

**Note:** Redis is used to deduplicate headlines for 24 hours to avoid reposting the same content.

### Example logs:
```text
📡 Starting trend fetch and post
🧵 New trend: Robot see, robot do...
✅ Upsert complete → https://your-blog.com/2025/04/22/ai-learns-from-videos/
📢 Published: https://your-blog.com/2025/04/22/ai-learns-from-videos/
🎉 Job completed successfully
```

## Features

- 🔁 Reuses cached posts and images for speed
- 📤 Automatically publishes or updates matching slugs
- 🧠 Logs to both `poster.log` and stdout (Kubernetes friendly)
- 🌐 Fully async-compatible design

## Development & Debugging

To test OpenAI manually:
```bash
python test_openai.py
```

To clear caches:
```bash
rm -rf .cache/
```

## License

MIT © Felipe De Bene