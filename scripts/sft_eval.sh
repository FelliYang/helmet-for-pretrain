#!/bin/bash
set -x
model_name=$1
seq_len_filter=$2
tasks=(recall icl)
# vLLM serving 地址，按需覆盖: ENDPOINT_URL=http://<host>:<port>/v1 bash scripts/sft_eval.sh ...
ip="${ENDPOINT_URL:-http://127.0.0.1:8998/v1}"

for task in "${tasks[@]}"; do
    bname=$(basename "$model_name")
    uv run python eval.py --config configs/${task}_short.yaml --model_name_or_path $model_name --endpoint_url "$ip" --use_vllm_serving --output_dir "output/$bname" --seq_len_filter $seq_len_filter --use_chat_template True --thinking &
done
wait
