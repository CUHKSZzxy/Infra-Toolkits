import datetime
import os

import torch.distributed as dist
from lmdeploy import PytorchEngineConfig, pipeline
from lmdeploy.messages import GenerationConfig
from lmdeploy.serve.openai.api_server import handle_torchrun


def main(rank: int):
    model_path = 'Qwen/Qwen3-30B-A3B-FP8'

    log_level = 'WARNING'
    prompts = [
        'fast fox jump over the lazy dog.',
        'fast fox jump over the lazy dog.',
    ]
    prompts = prompts[rank:rank + 1]

    backend_config = PytorchEngineConfig(
        tp=1,
        dp=2,
        ep=2,
        dp_rank=rank,
        eager_mode=True,
    )
    gen_config = GenerationConfig(
        temperature=1.0,
        top_k=1,
        do_sample=True,
        max_new_tokens=64,
    )

    os.environ['LMDEPLOY_DP_MASTER_ADDR'] = 'localhost'
    os.environ['LMDEPLOY_DP_MASTER_PORT'] = str(8000)
    with pipeline(model_path, backend_config=backend_config, log_level=log_level) as pipe:
        outputs = pipe(prompts, gen_config=gen_config)
        print(outputs)

        dist.barrier()


if __name__ == '__main__':
    handle_torchrun()
    rank = int(os.environ['LOCAL_RANK'])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    dist.init_process_group(backend='nccl', timeout=datetime.timedelta(90))
    try:
        main(rank)
    finally:
        dist.destroy_process_group()

# torchrun --nproc-per-node=2 test_pipeline_ep.py
