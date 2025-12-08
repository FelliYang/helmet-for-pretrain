#!/bin/bash
set -x
model_name=$1
seq_len_filter=$2
tasks=(recall icl)
ip="http://10.52.96.23:8000/v1"
ip="http://10.54.109.139:8998/v1" # RANK0
# ip="http://10.54.109.139:8999/v1" # RANK2
# ip="http://10.54.109.144:8000/v1" # RANK2 - card 0

export proxy_addr="http://cmcproxy:WvUBhef4bQ@10.251.112.50:8128"
export http_proxy="$proxy_addr"
export https_proxy="$proxy_addr"

for task in "${tasks[@]}"; do
    bname=$(basename "$model_name")
    uv run python eval.py --config configs/${task}_short.yaml --model_name_or_path $model_name --endpoint_url "$ip" --use_vllm_serving --output_dir "output/$bname" --seq_len_filter $seq_len_filter &
done
wait

unset http_proxy
unset https_proxy
