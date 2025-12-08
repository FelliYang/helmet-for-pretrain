#!/bin/bash
set -x
model_name=$1
seq_len_filter=$2
# tasks=(recall icl)
ip="http://10.54.109.139:8000/v1"
# ip="http://10.54.109.139:8998/v1" # RANK0
# ip="http://10.54.109.139:8999/v1" # RANK2
# ip="http://10.54.109.144:8000/v1" # RANK2 - card 0

# for task in "${tasks[@]}"; do
bname=$(basename "$model_name")
# balanced 版本
uv run python eval.py --config configs/recall_chinese_poem_balanced.yaml --model_name_or_path $model_name --endpoint_url "$ip" --use_vllm_serving --output_dir "output/$bname" &

# old 版本
uv run python eval.py --config configs/recall_chinese_poem.yaml --model_name_or_path $model_name --endpoint_url "$ip" --use_vllm_serving --output_dir "output/$bname" &
# done
wait
