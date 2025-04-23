#!/bin/bash

set -euo pipefail

echo "🧽 Flushing Redis cache..."
kubectl exec -i redis-master-0 -n wp -- redis-cli -a 'SQQEYnCjYr' EVAL "local keys = redis.call('keys', 'trend:*'); for i=1,#keys,5000 do redis.call('del', unpack(keys, i, math.min(i+4999, #keys))) end return #keys" 0