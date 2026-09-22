#!/bin/bash
# vLLM serving 地址与模型路径，按需覆盖:
#   ENDPOINT_URL=http://<host>:<port>/v1 MODEL_PATH=/path/to/ckpt bash scripts/wholeeval.sh
ENDPOINT_URL="${ENDPOINT_URL:-http://127.0.0.1:8998/v1}"
MODEL_PATH="${MODEL_PATH:-/tmp/gpt-oss-20b/}"
OUTPUT_DIR="${OUTPUT_DIR:-output/gpt-oss-20b-fix}"

for task in recall rag rerank cite longqa summ icl; do
    uv run python eval.py --config configs/${task}.yaml --model_name_or_path "$MODEL_PATH" --endpoint_url "$ENDPOINT_URL" --use_vllm_serving  --thinking --output_dir "$OUTPUT_DIR" &
    uv run python eval.py --config configs/${task}_short.yaml --model_name_or_path "$MODEL_PATH" --endpoint_url "$ENDPOINT_URL" --use_vllm_serving  --thinking --output_dir "$OUTPUT_DIR" &
   wait
done
