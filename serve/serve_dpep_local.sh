# proxy
lmdeploy serve proxy --server-port 8000 --routing-strategy 'min_expected_latency' --serving-strategy Hybrid --log-level INFO



# node 0
LMDEPLOY_DP_MASTER_ADDR=0.0.0.0 \
LMDEPLOY_DP_MASTER_PORT=29555 \
lmdeploy serve api_server \
    Qwen/Qwen3-235B-A22B-FP8 \
    --backend pytorch \
    --tp 1 \
    --dp 8 \
    --ep 8 \
    --proxy-url http://0.0.0.0:8000 \
    --nnodes 1 \
    --node-rank 0 \
    --log-level INFO
