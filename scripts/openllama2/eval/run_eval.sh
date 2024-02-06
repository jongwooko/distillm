#!/bin/bash

MASTER_PORT=2040
DEVICE=${1}
ckpt=${2}

# dolly eval
for seed in 10 20 30 40 50
do
    CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/openllama2/eval/eval_main_dolly_lora.sh ./ ${MASTER_PORT} 1 openllama2-3B ${ckpt} --seed $seed  --eval-batch-size 4
    CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/openllama2/eval/eval_main_self_inst_lora.sh ./ ${MASTER_PORT} 1 openllama2-3B ${ckpt} --seed $seed  --eval-batch-size 4
    CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/openllama2/eval/eval_main_vicuna_lora.sh ./ ${MASTER_PORT} 1 openllama2-3B ${ckpt} --seed $seed  --eval-batch-size 4
    CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/openllama2/eval/eval_main_sinst_lora.sh ./ ${MASTER_PORT} 1 openllama2-3B ${ckpt} --seed $seed  --eval-batch-size 4
    CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/openllama2/eval/eval_main_uinst_lora.sh ./ ${MASTER_PORT} 1 openllama2-3B ${ckpt} --seed $seed  --eval-batch-size 4
done