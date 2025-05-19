
## [Unreleased] - 2025-05-16

### Changed
- Replaced Ollama-based text generation with the `mlx.llm` API
- Removed references to AUTOMATIC1111 and other non-Apple AI services
- Updated README and requirements to reflect the simplified workflow

### Fixed
- Prevented crashes when newer and older versions of `mlx_lm` use
  different argument names for temperature during text generation

## [Unreleased] - 2025-05-15

### Added
- Enhanced image generation with Apple Neural Processing Unit (NPU) support via MLX Core
- Added platform detection to automatically use optimized image generation on Apple Silicon
- Created new `apple_utils.py` module with Apple-specific functionality
- Added unit tests for the enhanced image generation functionality
- Added `test_image_generation.py` script for manual testing of image generation
- Updated requirements.txt with MLX Core and CoreML dependencies

### Changed
- Modified `generate_image()` to rely solely on the Apple NPU
- Simplified blog generation to a single Mistral call on the NPU
- Updated README.md to reflect the streamlined flow

### Improved
- Optimized image generation performance on Apple Silicon devices
- Enhanced code organization with platform-specific utilities
- Added comprehensive test coverage for the new functionality

## [Unreleased] - 2025-04-28

### Added
- Introduced `--category` CLI option (default: "general") for the `--news` command to dynamically specify News API categories.
- Renamed the `--gnews` option to `--news` and updated related help texts and log messages to reflect the change.
- Added environment variable support for `NEWSAPI_KEY` to authenticate with NewsAPI.org.

### Changed
- Completely removed GNews.io integration and replaced with NewsAPI.org as the sole news source.
- Refactored `fetch_trending_topics` to:
  - Use `requests` for NewsAPI.org calls (`/top-headlines` and `/everything` endpoints).
  - Properly assemble query parameters with `requests.get` and `urlencode` style for safety.
  - Enrich each topic entry with `description` (if available) alongside the headline.
- Updated default endpoint from `/newsdata.io/api/1/news` to `/latest` at Newsdata.io before final removal.
- Switched all API consumption logic to exclusively call NewsAPI.org v2 endpoints.

### Fixed
- Addressed HTTP 422 errors ("Unprocessable Entity") by migrating off Newsdata.io filters and consolidating on NewsAPI.org.
- Handled empty or missing article results gracefully by returning an empty list with a warning message.
- Removed legacy `GNEWS_API_KEY` and related environment key checks.

### Improved
- Enhanced prompt in `generate_blog_outline` to produce a **parody article** tone, injecting humor and sarcasm.
- Ensured no `.strip()` calls on `NoneType` by safely defaulting missing fields to empty strings.