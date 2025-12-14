# RunPod vLLM Cluster Setup Guide

3x A100 80GB PCIe running Qwen3-8B with Nginx load balancer.

## Overview

- 3 independent vLLM instances (one per GPU)
- Nginx load balancer with `least_conn` strategy
- Single endpoint at port 8000
- vLLM backend ports: 9001, 9002, 9003 (high ports to avoid conflicts)
- Total capacity: ~24 concurrent requests, ~2,400 tok/s throughput

---

## Step 1: Install Dependencies

```bash
# Update system
apt update && apt install -y nginx tmux

# Install vLLM (if not already installed)
pip install vllm openai --break-system-packages
```

---

## Step 2: Create vLLM Launch Scripts

Create a directory for scripts:

```bash
mkdir -p /workspace/vllm_cluster
```

### Create launcher script

```bash
cat > /workspace/vllm_cluster/start_all.sh << 'EOF'
#!/bin/bash

# Kill any existing vLLM processes
pkill -f "vllm.entrypoints" || true
sleep 2

# Configuration - edit these as needed
MODEL_PATH="/workspace/Qwen3/weight/qwen3-8b-finetuned-hard-merged"
MODEL_NAME="qwen3-finetuned-hard"

# Activate venv if exists
[ -f /workspace/Qwen3/venv/bin/activate ] && source /workspace/Qwen3/venv/bin/activate

# Start vLLM instance on GPU 0 (port 9001)
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --max-model-len 32768 \
    --max-num-seqs 8 \
    --port 9001 \
    --disable-log-requests \
    > /workspace/vllm_cluster/gpu0.log 2>&1 &

# Start vLLM instance on GPU 1 (port 9002)
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --max-model-len 32768 \
    --max-num-seqs 8 \
    --port 9002 \
    --disable-log-requests \
    > /workspace/vllm_cluster/gpu1.log 2>&1 &

# Start vLLM instance on GPU 2 (port 9003)
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --max-model-len 32768 \
    --max-num-seqs 8 \
    --port 9003 \
    --disable-log-requests \
    > /workspace/vllm_cluster/gpu2.log 2>&1 &

echo "Started 3 vLLM instances on ports 9001, 9002, 9003"
echo "Model: $MODEL_NAME"
echo "Logs at /workspace/vllm_cluster/gpu*.log"
EOF

chmod +x /workspace/vllm_cluster/start_all.sh
```

### Create stop script

```bash
cat > /workspace/vllm_cluster/stop_all.sh << 'EOF'
#!/bin/bash
pkill -f "vllm.entrypoints" || true
echo "Stopped all vLLM instances"
EOF

chmod +x /workspace/vllm_cluster/stop_all.sh
```

### Create health check script

```bash
cat > /workspace/vllm_cluster/health_check.sh << 'EOF'
#!/bin/bash
echo "Checking vLLM instances..."
for port in 9001 9002 9003; do
    if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
        echo "Port $port: ✓ healthy"
    else
        echo "Port $port: ✗ not responding"
    fi
done

echo ""
echo "Checking load balancer..."
if curl -s "http://localhost:8000/health" > /dev/null 2>&1; then
    echo "Port 8000 (LB): ✓ healthy"
else
    echo "Port 8000 (LB): ✗ not responding"
fi
EOF

chmod +x /workspace/vllm_cluster/health_check.sh
```

---

## Step 3: Configure Nginx Load Balancer

On RunPod, nginx is already running with other services. We need to **add** our vLLM config to the existing config, not replace it.

### Backup original config

```bash
cp /etc/nginx/nginx.conf /etc/nginx/nginx.conf.backup
```

### Add vLLM load balancer to existing nginx config

Edit `/etc/nginx/nginx.conf` and add the following inside the `http { }` block, near the top:

```nginx
    # vLLM cluster load balancer
    upstream vllm_cluster {
        least_conn;
        server 127.0.0.1:9001;
        server 127.0.0.1:9002;
        server 127.0.0.1:9003;
    }

    # Log format showing which backend handled each request
    log_format vllm_lb '$time_local | $remote_addr | $upstream_addr | $request_time s | $status | "$request"';

    server {
        listen 8000;

        access_log /var/log/nginx/vllm_lb.log vllm_lb;
        error_log /var/log/nginx/vllm_error.log;

        location /health {
            proxy_pass http://vllm_cluster;
            proxy_http_version 1.1;
            proxy_connect_timeout 5s;
            proxy_read_timeout 10s;
        }

        location / {
            proxy_pass http://vllm_cluster;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header Connection "";

            # Long timeouts for LLM inference
            proxy_connect_timeout 3600s;
            proxy_read_timeout 3600s;
            proxy_send_timeout 3600s;

            # Disable buffering for streaming
            proxy_buffering off;
            proxy_request_buffering off;
        }
    }
```

### Test and reload Nginx

```bash
nginx -t && nginx -s reload
```

---

## Step 4: Start the Cluster

### Start vLLM instances

```bash
/workspace/vllm_cluster/start_all.sh
```

### Wait for models to load (~2-3 minutes)

```bash
# Watch logs until you see "Application startup complete"
tail -f /workspace/vllm_cluster/gpu0.log
```

### Verify all instances are healthy

```bash
/workspace/vllm_cluster/health_check.sh
```

Expected output:
```
Checking vLLM instances...
Port 9001: ✓ healthy
Port 9002: ✓ healthy
Port 9003: ✓ healthy

Checking load balancer...
Port 8000 (LB): ✓ healthy
```

---

## Step 5: Test the Setup

### Quick test via curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-finetuned-hard",
    "messages": [{"role": "user", "content": "Hello, who are you?"}],
    "max_tokens": 100
  }'
```

### Test with Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="qwen3-finetuned-hard",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)

print(response.choices[0].message.content)
```

### Test load balancing (concurrent requests)

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

async def send_request(i):
    response = await client.chat.completions.create(
        model="qwen3-finetuned-hard",
        messages=[{"role": "user", "content": f"Count from 1 to 10. Request {i}"}],
        max_tokens=100
    )
    return f"Request {i}: {len(response.choices[0].message.content)} chars"

async def main():
    tasks = [send_request(i) for i in range(12)]
    results = await asyncio.gather(*tasks)
    for r in results:
        print(r)

asyncio.run(main())
```

---

## Step 6: Monitoring

### Watch GPU utilization

```bash
watch -n 1 nvidia-smi
```

### Check load balancer logs (see which backend handles each request)

```bash
tail -f /var/log/nginx/vllm_lb.log
```

Example output:
```
14/Dec/2025:20:55:16 +0000 | 127.0.0.1 | 127.0.0.1:9003 | 0.009 s | 200 | "GET /v1/models HTTP/1.1"
14/Dec/2025:20:55:16 +0000 | 127.0.0.1 | 127.0.0.1:9002 | 0.010 s | 200 | "GET /v1/models HTTP/1.1"
14/Dec/2025:20:55:16 +0000 | 127.0.0.1 | 127.0.0.1:9001 | 0.012 s | 200 | "GET /v1/models HTTP/1.1"
```

### Check individual vLLM logs

```bash
tail -f /workspace/vllm_cluster/gpu0.log
tail -f /workspace/vllm_cluster/gpu1.log
tail -f /workspace/vllm_cluster/gpu2.log
```

---

## Quick Reference

| Action | Command |
|--------|---------|
| Start cluster | `/workspace/vllm_cluster/start_all.sh` |
| Stop cluster | `/workspace/vllm_cluster/stop_all.sh` |
| Health check | `/workspace/vllm_cluster/health_check.sh` |
| Restart Nginx | `nginx -s reload` |
| View GPU usage | `nvidia-smi` |
| View LB logs | `tail -f /var/log/nginx/vllm_lb.log` |
| API endpoint | `http://localhost:8000/v1` |

---

## Port Mapping

| Component | Port | Notes |
|-----------|------|-------|
| Nginx Load Balancer | 8000 | Public API endpoint |
| vLLM GPU 0 | 9001 | Internal backend |
| vLLM GPU 1 | 9002 | Internal backend |
| vLLM GPU 2 | 9003 | Internal backend |

> **Note:** We use high ports (9001-9003) for vLLM backends to avoid conflicts with RunPod's default services which use ports like 8001 (vscode), 8080 (code-server), etc.

---

## Troubleshooting

### vLLM not starting
```bash
# Check logs
cat /workspace/vllm_cluster/gpu0.log

# Common fixes:
pip install -U vllm --break-system-packages
```

### OOM errors
Reduce `--max-num-seqs` from 8 to 4 in the launch script.

### Nginx 502 Bad Gateway
vLLM instances not ready yet. Wait for model loading or check health.

### Port already in use
```bash
# Check what's using the port
lsof -i :9001

# Kill vLLM and restart
/workspace/vllm_cluster/stop_all.sh
sleep 5
/workspace/vllm_cluster/start_all.sh
```

### Check if ports are free before starting
```bash
for port in 9001 9002 9003; do
    if lsof -i :$port > /dev/null 2>&1; then
        echo "Port $port is in use!"
    else
        echo "Port $port is free"
    fi
done
```
