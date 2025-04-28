FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install curl only (minimal dependencies), clean cache immediately after
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install ollama
RUN curl https://ollama.ai/install.sh | bash

EXPOSE 11434

ENTRYPOINT ["ollama", "serve"]