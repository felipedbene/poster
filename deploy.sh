#!/bin/bash

set -e

echo "🧼 Syncing Python dependencies..."
uv pip sync requirements.txt

echo "🐳 Building multi-arch image for Docker Hub..."
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t fdebene/poster:latest \
  --push .

echo "🚀 Rolling out new version in Kubernetes..."
kubectl rollout restart deployment trend-poster -n wp

echo "⏳ Waiting 10 seconds for pod to start..."
sleep 10

echo "📜 Tailing logs from trend-poster..."
kubectl logs -n wp -l app=trend-poster -f