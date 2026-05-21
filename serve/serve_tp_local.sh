CUDA_VISIBLE_DEVICES=0,1 lmdeploy serve api_server \
        Qwen/Qwen2.5-7B-Instruct \
        --tp 2 \
        --server-port 23333 \
        --backend pytorch \
        --log-level INFO
