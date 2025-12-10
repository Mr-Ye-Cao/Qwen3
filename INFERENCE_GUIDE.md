# Qwen3-8B Finetuned Model Inference Guide

This guide documents how to run inference on the finetuned Qwen3-8B models and host them as an API using vLLM.

## Overview

The finetuned models are LoRA adapters trained on top of `Qwen/Qwen3-8B`. Two checkpoints are available:

| Checkpoint | Path | Description |
|------------|------|-------------|
| Easy | `weight/weights_qwen_agentic_distill_16easy_20251209_032801` | Finetuned on easier examples |
| Hard | `weight/weights_qwen_agentic_distill_hard_20251209_200527` | Finetuned on harder examples |

**LoRA Configuration:**
- Rank (r): 32
- Alpha: 32
- Target modules: all-linear
- Task type: CAUSAL_LM

---

## Setup

### 1. Create Conda Environment

```bash
conda create -n qwen3 python=3.11 -y
conda activate qwen3
```

### 2. Install Dependencies

```bash
# Install PyTorch with CUDA 12.8 (for RTX 5090 / Blackwell GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install transformers, peft, and accelerate
pip install transformers peft accelerate

# Install vLLM for serving
pip install vllm
```

---

## Method 1: Direct Inference with Transformers + PEFT

Use `inference_lora.py` for quick testing and interactive chat.

### Single Prompt Inference

```bash
python inference_lora.py \
  --base-model Qwen/Qwen3-8B \
  --adapter-path weight/weights_qwen_agentic_distill_16easy_20251209_032801 \
  --prompt "What is 25 * 17? Show your reasoning." \
  --max-new-tokens 512
```

### Interactive Chat Mode

```bash
python inference_lora.py \
  --base-model Qwen/Qwen3-8B \
  --adapter-path weight/weights_qwen_agentic_distill_16easy_20251209_032801
```

Type your messages and press Enter. Type `quit` or `exit` to end.

---

## Method 2: Hosting with vLLM (Recommended for Production)

vLLM provides high-performance inference with an OpenAI-compatible API.

### Step 1: Merge LoRA Adapter into Base Model

vLLM's LoRA support doesn't work with adapters trained using `target_modules: "all-linear"`. The solution is to merge the adapter into the base model first.

```bash
python merge_lora.py \
  --base-model Qwen/Qwen3-8B \
  --adapter-path weight/weights_qwen_agentic_distill_16easy_20251209_032801 \
  --output-path weight/qwen3-8b-finetuned-merged
```

This creates a standalone model at `weight/qwen3-8b-finetuned-merged/` that can be served directly.

### Step 2: Start vLLM Server

```bash
python -m vllm.entrypoints.openai.api_server \
  --model weight/qwen3-8b-finetuned-merged \
  --served-model-name qwen3-finetuned \
  --port 8000 \
  --host 0.0.0.0 \
  --max-model-len 32768 \
  --dtype bfloat16
```

**Parameters:**
- `--model`: Path to the merged model
- `--served-model-name`: Name to use in API requests
- `--max-model-len`: Maximum context length (Qwen3-8B supports up to 32,768 natively)
- `--dtype bfloat16`: Use bfloat16 for memory efficiency

### Step 3: Use the API

#### Health Check
```bash
curl http://localhost:8000/health
```

#### List Models
```bash
curl http://localhost:8000/v1/models
```

#### Chat Completions
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-finetuned",
    "messages": [
      {"role": "user", "content": "What is 15 * 23? Show your reasoning."}
    ],
    "max_tokens": 1024,
    "temperature": 0.7
  }'
```

#### Using Python (OpenAI SDK)
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # vLLM doesn't require a real key
)

response = client.chat.completions.create(
    model="qwen3-finetuned",
    messages=[
        {"role": "user", "content": "Explain quantum entanglement simply."}
    ],
    max_tokens=512
)

print(response.choices[0].message.content)
```

---

## Memory Requirements

| Context Length | Model Memory | KV Cache | Total VRAM | Concurrent Requests |
|----------------|--------------|----------|------------|---------------------|
| 8,192 | ~15.3 GB | ~11.4 GB | ~27 GB | ~10x |
| 32,768 | ~15.3 GB | ~11.5 GB | ~27 GB | ~2.5x |

RTX 5090 (32GB VRAM) can comfortably run the model at 32K context.

---

## Model Behavior

The finetuned models use `<think>` tags for chain-of-thought reasoning:

```
<think>
Okay, so I need to figure out what 15 multiplied by 23 is...
</think>

The answer is 345.
```

This reasoning behavior was learned during finetuning for agentic/reasoning tasks.

---

## Troubleshooting

### Issue: vLLM LoRA loading fails with "unexpected modules" error

**Error:**
```
ValueError: expected target modules in {'v_proj', 'q_proj', ...} but received ['model.unembed_tokens']
```

**Solution:** Merge the LoRA adapter first using `merge_lora.py` (see Step 1 above).

### Issue: Out of GPU memory

**Solution:** Reduce `--max-model-len` or use `--gpu-memory-utilization 0.85`:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model weight/qwen3-8b-finetuned-merged \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  ...
```

### Issue: Previous vLLM process blocking GPU memory

```bash
# Find and kill vLLM processes
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -I {} kill -9 {}
```

---

## File Structure

```
Qwen3/
├── weight/
│   ├── weights_qwen_agentic_distill_16easy_20251209_032801/  # LoRA adapter (easy)
│   │   ├── adapter_config.json
│   │   └── adapter_model.safetensors
│   ├── weights_qwen_agentic_distill_hard_20251209_200527/    # LoRA adapter (hard)
│   │   ├── adapter_config.json
│   │   └── adapter_model.safetensors
│   └── qwen3-8b-finetuned-merged/                            # Merged model for vLLM
│       ├── config.json
│       ├── model-*.safetensors
│       └── tokenizer files...
├── inference_lora.py      # Direct inference script
├── merge_lora.py          # LoRA merger script
└── INFERENCE_GUIDE.md     # This file
```

---

## Quick Reference

```bash
# Activate environment
conda activate qwen3

# Run interactive inference
python inference_lora.py --base-model Qwen/Qwen3-8B \
  --adapter-path weight/weights_qwen_agentic_distill_16easy_20251209_032801

# Start vLLM server (32K context)
python -m vllm.entrypoints.openai.api_server \
  --model weight/qwen3-8b-finetuned-merged \
  --served-model-name qwen3-finetuned \
  --port 8000 --host 0.0.0.0 --max-model-len 32768 --dtype bfloat16

# Test API
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-finetuned","messages":[{"role":"user","content":"Hello!"}],"max_tokens":256}'
```
