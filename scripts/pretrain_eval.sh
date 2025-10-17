#!/bin/bash
set -x
model_name=$1
seq_len_filter=$2
tasks=(recall icl)

for task in "${tasks[@]}"; do
    bname=$(basename "$model_name")
    uv run python eval.py --config configs/${task}_short.yaml --model_name_or_path $model_name --endpoint_url "http://10.52.96.23:8998/v1" --use_vllm_serving --output_dir "output/$bname" --seq_len_filter $seq_len_filter &
done
wait
