CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8000 \
    --model pre_train_model/qwen2-7b-chat \
    --served-model-name Qwen2_7B \
    --enable-lora \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --max-model-len 2048 \
    --dtype=half

