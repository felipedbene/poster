#!/bin/bash

set -e

echo "🧼 Syncing Python dependencies..."
uv pip sync requirements.txt

echo "🐳 Building multi-arch image for Docker Hub..."
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t fdebene/poster:latest \
  --push .

echo "🚀 Recreating CronJob manually..."
kubectl delete job trend-poster-now -n wp --ignore-not-found
kubectl create job --from=cronjob/trend-poster trend-poster-now -n wp

sleep 20
kubectl logs job/trend-poster-now -n wp -f


echo "📜 Tailing logs from trend-poster..."
kubectl logs -n wp -l app=trend-poster -f