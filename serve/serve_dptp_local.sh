# proxy
lmdeploy serve proxy --server-name --server-port 8000 --routing-strategy 'min_expected_latency' --serving-strategy Hybrid --log-level INFO



# node 0
LMDEPLOY_DP_MASTER_ADDR=0.0.0.0 \
LMDEPLOY_DP_MASTER_PORT=29555 \
lmdeploy serve api_server \
    Qwen/Qwen2.5-7B-Instruct \
    --backend pytorch \
    --tp 2 \
    --dp 2 \
    --proxy-url http://0.0.0.0:8000 \
    --nnodes 1 \
    --node-rank 0 \
    --log-level INFO
