# Qwen3-8B Finetuned Model Setup Instructions

This document explains how to set up and serve a finetuned Qwen3-8B model with LoRA weights on RunPod.

## Overview

We have finetuned LoRA adapters stored as compressed archives. The process involves:
1. Extracting LoRA weights from the archive
2. Merging LoRA weights with the base Qwen3-8B model
3. Serving the merged model via vLLM
4. Accessing the API via RunPod's TCP port

---

## Directory Structure

```
/workspace/Qwen3/
├── weight/
│   ├── easy-model-checkpoint.tar.gz    # LoRA weights archive (easy version)
│   ├── hard-model-checkpoint.tar       # LoRA weights archive (hard version)
│   └── qwen3-8b-finetuned-easy-merged/ # Merged model output (created after merge)
├── merge_lora.py                       # Script to merge LoRA with base model
├── inference_lora.py                   # Script for direct LoRA inference
├── venv/                               # Python virtual environment with dependencies
└── SETUP_INSTRUCTIONS.md               # This file
```

---

## Step 1: Activate Virtual Environment

```bash
source /workspace/Qwen3/venv/bin/activate
```

The venv contains all required dependencies: `transformers`, `peft`, `torch`, `vllm`, etc.

---

## Step 2: Extract LoRA Weights

The archive is named `.tar.gz` but is actually a plain tar file.

```bash
cd /workspace/Qwen3/weight
tar -xf easy-model-checkpoint.tar.gz
```

This extracts:
- `adapter_config.json` - LoRA configuration (rank=32, alpha=32, target="all-linear")
- `adapter_model.safetensors` - LoRA weights (~369MB)
- `checkpoint_complete` - Marker file

---

## Step 3: Merge LoRA with Base Model

Use `merge_lora.py` to merge the LoRA adapter into the base Qwen3-8B model:

```bash
cd /workspace/Qwen3
python merge_lora.py \
  --base-model Qwen/Qwen3-8B \
  --adapter-path /workspace/Qwen3/weight \
  --output-path /workspace/Qwen3/weight/qwen3-8b-finetuned-easy-merged
```

This will:
1. Download the base Qwen3-8B model from HuggingFace (if not cached)
2. Load the LoRA adapter
3. Merge weights into the base model
4. Save the merged model (~16GB) to the output path

**Time**: ~2-3 minutes

---

## Step 4: Serve with vLLM

Start the vLLM OpenAI-compatible API server:

```bash
source /workspace/Qwen3/venv/bin/activate
cd /workspace/Qwen3

nohup python -m vllm.entrypoints.openai.api_server \
  --model /workspace/Qwen3/weight/qwen3-8b-finetuned-easy-merged \
  --served-model-name qwen3-finetuned-easy \
  --port 8000 \
  --host 0.0.0.0 \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --api-key "sk-qwen3-262d69fda52130a25880846f2596aabaf637294be779fb2b248378ecdb153153" \
  --max-num-seqs 3 > /tmp/vllm_server.log 2>&1 &
```

**Parameters explained**:
- `--served-model-name`: Name to use in API requests
- `--port 8000`: Internal port (RunPod maps this to external TCP port)
- `--host 0.0.0.0`: Listen on all interfaces
- `--max-model-len 32768`: Maximum context length
- `--dtype bfloat16`: Use bfloat16 precision
- `--api-key`: API key for authentication
- `--max-num-seqs 3`: Maximum parallel requests (prevents OOM)

**Time**: ~1-2 minutes to load model and start serving

---

## Step 5: Verify Server is Running

### Check process:
```bash
ps aux | grep vllm | grep -v grep
```

### Check logs:
```bash
tail -f /tmp/vllm_server.log
```

### Test locally:
```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer sk-qwen3-262d69fda52130a25880846f2596aabaf637294be779fb2b248378ecdb153153"
```

---

## Step 6: Access from External (AWS/Remote)

### Find RunPod TCP Port Mapping

In RunPod dashboard, go to your pod's "Connect" section and find:
```
Direct TCP ports:
<PUBLIC_IP>:<EXTERNAL_PORT> → :8000
```

Example: `62.169.158.26:46161` maps to internal port 8000

### Make API Requests

Replace `<PUBLIC_IP>:<EXTERNAL_PORT>` with your actual values.

**List models:**
```bash
curl http://<PUBLIC_IP>:<EXTERNAL_PORT>/v1/models \
  -H "Authorization: Bearer sk-qwen3-262d69fda52130a25880846f2596aabaf637294be779fb2b248378ecdb153153"
```

**Chat completion:**
```bash
curl http://<PUBLIC_IP>:<EXTERNAL_PORT>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-qwen3-262d69fda52130a25880846f2596aabaf637294be779fb2b248378ecdb153153" \
  -d '{
    "model": "qwen3-finetuned-easy",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 512
  }'
```

**Python (OpenAI SDK):**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://<PUBLIC_IP>:<EXTERNAL_PORT>/v1",
    api_key="sk-qwen3-262d69fda52130a25880846f2596aabaf637294be779fb2b248378ecdb153153"
)

response = client.chat.completions.create(
    model="qwen3-finetuned-easy",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=512
)
print(response.choices[0].message.content)
```

---

## Server Management

### Stop server:
```bash
pkill -f "vllm.entrypoints.openai.api_server"
```

### Restart server:
Run the nohup command from Step 4 again.

### View real-time logs:
```bash
tail -f /tmp/vllm_server.log
```

---

## Why TCP Instead of HTTPS Proxy?

RunPod's HTTPS proxy (`https://<pod_id>-8000.proxy.runpod.net`) goes through Cloudflare, which has a 100-second timeout limit. For large context inference requests that take longer, this causes 524 timeout errors.

Using direct TCP connection (`http://<PUBLIC_IP>:<EXTERNAL_PORT>`) bypasses Cloudflare entirely, allowing unlimited request duration.

---

## Quick Start (All Steps Combined)

```bash
# 1. Activate environment
source /workspace/Qwen3/venv/bin/activate

# 2. Extract LoRA weights
cd /workspace/Qwen3/weight
tar -xf easy-model-checkpoint.tar.gz

# 3. Merge LoRA with base model
cd /workspace/Qwen3
python merge_lora.py \
  --base-model Qwen/Qwen3-8B \
  --adapter-path /workspace/Qwen3/weight \
  --output-path /workspace/Qwen3/weight/qwen3-8b-finetuned-easy-merged

# 4. Start vLLM server
nohup python -m vllm.entrypoints.openai.api_server \
  --model /workspace/Qwen3/weight/qwen3-8b-finetuned-easy-merged \
  --served-model-name qwen3-finetuned-easy \
  --port 8000 \
  --host 0.0.0.0 \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --api-key "sk-qwen3-262d69fda52130a25880846f2596aabaf637294be779fb2b248378ecdb153153" \
  --max-num-seqs 3 > /tmp/vllm_server.log 2>&1 &

# 5. Wait for server to start and verify
sleep 60
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer sk-qwen3-262d69fda52130a25880846f2596aabaf637294be779fb2b248378ecdb153153"
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Activate venv: `source /workspace/Qwen3/venv/bin/activate` |
| `gzip: stdin: not in gzip format` | Use `tar -xf` instead of `tar -xzf` (file is plain tar despite .tar.gz name) |
| Server not responding | Check logs: `tail -50 /tmp/vllm_server.log` |
| Connection refused from external | Check RunPod TCP port mapping in dashboard |
| 524 timeout errors | Use TCP direct connection instead of HTTPS proxy |
| OOM errors | Reduce `--max-num-seqs` or `--max-model-len` |

---

## Files Reference

| File | Purpose |
|------|---------|
| `merge_lora.py` | Merges LoRA adapter into base model for vLLM serving |
| `inference_lora.py` | Direct inference with LoRA (without merging, slower) |
| `how-to-run.md` | Original run commands reference |
| `venv/` | Python environment with all dependencies |

---

## Current Configuration (as of last setup)

- **RunPod ID**: wvrew2whawh4cf
- **Internal Port**: 8000
- **External TCP**: 62.169.158.26:46161 (check dashboard for current mapping)
- **Model Name**: qwen3-finetuned-easy
- **API Key**: sk-qwen3-262d69fda52130a25880846f2596aabaf637294be779fb2b248378ecdb153153
- **Max Parallel Requests**: 3
