# proxy
lmdeploy serve proxy --server-name 172.16.4.52 --server-port 8000 --routing-strategy 'min_expected_latency' --serving-strategy Hybrid --log-level INFO



# node 0
LMDEPLOY_DP_MASTER_ADDR=172.16.4.52 \
LMDEPLOY_DP_MASTER_PORT=29555 \
lmdeploy serve api_server \
    deepseek-ai/DeepSeek-V3 \
    --backend pytorch \
    --tp 1 \
    --dp 32 \
    --ep 32 \
    --proxy-url http://172.16.4.52:8000 \
    --nnodes 4 \
    --node-rank 0 \
    --log-level INFO 2>&1 | tee benchmark/api_serve_node0.log



# node 1
LMDEPLOY_DP_MASTER_ADDR=172.16.4.52 \
LMDEPLOY_DP_MASTER_PORT=29555 \
lmdeploy serve api_server \
    deepseek-ai/DeepSeek-V3 \
    --backend pytorch \
    --tp 1 \
    --dp 32 \
    --ep 32 \
    --proxy-url http://172.16.4.52:8000 \
    --nnodes 4 \
    --node-rank 1 \
    --log-level INFO 2>&1 | tee benchmark/api_serve_node1.log



# node 2
LMDEPLOY_DP_MASTER_ADDR=172.16.4.52 \
LMDEPLOY_DP_MASTER_PORT=29555 \
lmdeploy serve api_server \
    deepseek-ai/DeepSeek-V3 \
    --backend pytorch \
    --tp 1 \
    --dp 32 \
    --ep 32 \
    --proxy-url http://172.16.4.52:8000 \
    --nnodes 4 \
    --node-rank 2 \
    --log-level INFO  2>&1 | tee benchmark/api_serve_node2.log



# node 3
LMDEPLOY_DP_MASTER_ADDR=172.16.4.52 \
LMDEPLOY_DP_MASTER_PORT=29555 \
lmdeploy serve api_server \
    deepseek-ai/DeepSeek-V3 \
    --backend pytorch \
    --tp 1 \
    --dp 32 \
    --ep 32 \
    --proxy-url http://172.16.4.52:8000 \
    --nnodes 4 \
    --node-rank 3 \
    --log-level INFO 2>&1 | tee benchmark/api_serve_node3.log
