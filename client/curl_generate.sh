curl http://localhost:23333/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The capital of France is",
    "model": "Qwen/Qwen3-8B",
    "max_tokens": 50,
    "temperature": 0,
    "repetition_penalty": 1.0,
    "stop": ["\n"]
  }'
