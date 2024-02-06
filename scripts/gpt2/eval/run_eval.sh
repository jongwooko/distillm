#!/bin/bash

MASTER_PORT=2040
DEVICE=${1}
ckpt=${2}

for seed in 10 20 30 40 50
do
    CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/gpt2/eval/eval_main_dolly.sh ./ ${MASTER_PORT} 1 ${ckpt} --seed $seed  --eval-batch-size 16
    CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/gpt2/eval/eval_main_self_inst.sh ./ ${MASTER_PORT} 1 ${ckpt} --seed $seed  --eval-batch-size 16
    CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/gpt2/eval/eval_main_vicuna.sh ./ ${MASTER_PORT} 1 ${ckpt} --seed $seed  --eval-batch-size 16
    CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/gpt2/eval/eval_main_sinst.sh ./ ${MASTER_PORT} 1 ${ckpt} --seed $seed  --eval-batch-size 16
    CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/gpt2/eval/eval_main_uinst.sh ./ ${MASTER_PORT} 1 ${ckpt} --seed $seed  --eval-batch-size 16
done