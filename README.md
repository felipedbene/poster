# Trend Poster

Automatically generate, publish, and update SEO-optimized blog posts to WordPress based on trending headlines — powered by `mlx.llm` and the WP REST API.

## Overview

**Trend Poster** continuously fetches trending headlines using NewsAPI and uses the `mlx.llm` model to generate:
- Titles, meta descriptions, and keyphrases
- Full HTML blog posts (1000+ words)
- Featured image prompts (and auto-generated images)
- SEO metadata integration (Yoast-compatible)
- Seamless updates of existing posts (via slug detection)

### Enhanced Image Generation

The system now generates both text and images directly on Apple Silicon hardware:
- **Apple Silicon Support**: Uses the Neural Processing Unit (NPU) via MLX Core for fast image generation
- **Mistral LLM**: Blog posts are produced in a single call to the local Mistral model running on the NPU

### Code Flow

```mermaid
flowchart TD
    A[Start] --> B{Topic Source}
    B -->|--idea| C[Generate blog components]
    B -->|Trending headlines| D[Fetch trends]
    D --> C
    C --> E[Generate post with Mistral]
    E --> F{Generate images?}
    F --> G[MLX Core]
    G --> J[Upload images]
    J --> K[Publish to WordPress]
    K --> L[Done]
```

## Assumptions

- A **WordPress** site with REST API enabled
- An **Application Password** for your WP user
- **Yoast SEO** plugin (for meta patching)
- A **NewsAPI key**
- Python 3.11+
- For Apple NPU support: Apple Silicon Mac with MLX Core, CoreML dependencies, and the `mlx.llm` package installed

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

3. **Create a `.env` file** with WordPress and NewsAPI configuration:
   ```ini
   WORDPRESS_URL=https://your-blog.com
   WORDPRESS_USERNAME=your_wp_user
WORDPRESS_APP_PASSWORD=your_app_password
NEWSAPI_KEY=your_newsapi_key
MLX_LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2
   ```
    Ensure your environment (e.g., `poster-env`) contains these variables loaded.

## Usage

### Kubernetes Job
```bash
# Manually trigger a one-time CronJob execution
kubectl create job --from=cronjob/trend-poster trend-poster-now -n wp
kubectl logs -n wp job/trend-poster-now -f
```


### Example logs:
```text
📡 Starting trend fetch and post
🧵 New trend: Robot see, robot do...
✅ Upsert complete → https://your-blog.com/2025/04/22/ai-learns-from-videos/
📢 Published: https://your-blog.com/2025/04/22/ai-learns-from-videos/
🎉 Job completed successfully
```


## Running Locally
Generate a blog post with your own topic:

```bash
python post.py --idea "Amazing new tech" --keyphrase "latest tech trends"
```

## Features

- 🔁 Reuses cached posts and images for speed
- 📤 Automatically publishes or updates matching slugs
- 🧠 Logs to both `poster.log` and stdout (Kubernetes friendly)
- 🌐 Fully async-compatible design
- 🍎 Optimized image generation on Apple Silicon using the Neural Processing Unit (NPU)
- ✍️ Text generation powered by the `mlx.llm` API running locally on the Apple NPU

## Development & Debugging

To clear caches:
```bash
rm -rf .cache/
```

## License

MIT © Felipe De Bene

