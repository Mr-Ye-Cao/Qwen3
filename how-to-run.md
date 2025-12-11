# 16 easy finetuned version
conda activate qwen3
python -m vllm.entrypoints.openai.api_server \
  --model /home/ye/ml-experiments/Qwen3/weight/qwen3-8b-finetuned-easy-merged \
  --served-model-name qwen3-finetuned-easy \
  --port 8000 --host 0.0.0.0 --max-model-len 32768 --dtype bfloat16

# base model
conda activate qwen3
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --served-model-name qwen3-8b \
  --port 8000 \
  --host 0.0.0.0 \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --api-key "sk-qwen3-262d69fda52130a25880846f2596aabaf637294be779fb2b248378ecdb153153"

# hard finetuned version
python -m vllm.entrypoints.openai.api_server \
  --model weight/qwen3-8b-finetuned-hard-merged \
  --served-model-name qwen3-finetuned-hard \
  --port 8000 \
  --host 0.0.0.0 \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --api-key "sk-qwen3-262d69fda52130a25880846f2596aabaf637294be779fb2b248378ecdb153153"