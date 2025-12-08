#!/bin/bash
export proxy_addr="http://cmcproxy:WvUBhef4bQ@10.251.112.50:8128"
export http_proxy="$proxy_addr"
export https_proxy="$proxy_addr"


bash scripts/pretrain_eval_128k.sh $1 131072
wait
bash scripts/pretrain_eval.sh $1 65536

unset http_proxy
unset htts_proxy
