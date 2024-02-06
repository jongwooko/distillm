BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/results/openllama2/gen/openllama2-7B/t1.0-l512 \
    --processed-data-dir ${BASE_PATH}/processed_data/dolly/pseudo \
    --model-path ${BASE_PATH}/checkpoints/openllama2-7B \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num -1 \
    --model-type openllama2

cp ${BASE_PATH}/processed_data/dolly/full/openllama2/valid_0.bin ${BASE_PATH}/processed_data/dolly/pseudo/openllama2/
cp ${BASE_PATH}/processed_data/dolly/full/openllama2/valid_0.idx ${BASE_PATH}/processed_data/dolly/pseudo/openllama2/
cp ${BASE_PATH}/processed_data/dolly/full/openllama2/valid.jsonl ${BASE_PATH}/processed_data/dolly/pseudo/openllama2/
