#!/bin/bash

uv pip freeze > requirements.txt
docker buildx create --use || true
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t fdebene/poster:latest \
  --push .