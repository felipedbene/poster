#!/bin/bash

set -e

echo "🧼 Syncing Python dependencies..."
uv pip sync requirements.txt

echo "🐳 Building multi-arch image for Docker Hub..."
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t fdebene/poster:latest \
  --push .

echo "🚀 Triggering manual CronJob run..."
kubectl create job --from=cronjob/trend-poster trend-poster-now -n wp
sleep 10
kubectl logs job/trend-poster-now -n wp -f

echo "⏳ Waiting 10 seconds for pod to start..."
sleep 20

echo "📜 Tailing logs from trend-poster..."
kubectl logs -n wp -l app=trend-poster -f