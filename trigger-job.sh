#!/bin/bash

set -euo pipefail

echo "🗑️  Deleting previous job (if exists)..."
kubectl delete job trend-poster-now -n wp --ignore-not-found

echo "🚀 Creating new job from CronJob..."
kubectl create job --from=cronjob/trend-poster trend-poster-now -n wp

echo "⏳ Waiting for pod to initialize..."
sleep 20


echo "📺 Tailing job logs..."
kubectl logs job/trend-poster-now -n wp -f
