"""run.py:"""
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# os.environ['NCCL_DEBUG'] = 'TRACE'
# os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
# os.environ['NCCL_DEBUG_TIMESTAMP_LEVELS'] = 'ALL'


def run(rank, size):
    """Distributed function to perform an all-reduce and synchronize."""
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    # Rank 0: [1,1,1,1,1], Rank 1: [2,2,2,2,2]
    tensor = torch.ones(5, device=device) * (rank + 1)
    print(f'Rank {rank}: Initial tensor = {tensor.tolist()}')

    # all-reduce sum
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(device)
    print(f'Rank {rank}: After all-reduce tensor = {tensor.tolist()}')

    # [3,3,3,3,3] for 2 processes
    expected = torch.ones(5, device=device) * 3
    assert torch.allclose(tensor, expected)


def init_process(rank, size, fn, backend='nccl'):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'xxx.xxx.x.xxx'  # change this
    os.environ['MASTER_PORT'] = '29503'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == '__main__':
    world_size = 2  # change this if > 2 nodes
    processes = []
    if 'google.colab' in sys.modules:
        print('Running in Google Colab')
        mp.get_context('spawn')
    else:
        mp.set_start_method('spawn')
    # for rank in range(world_size):
    #     p = mp.Process(target=init_process, args=(rank, world_size, run))
    #     p.start()
    #     processes.append(p)
    rank = 1  # change this for rank 0 / 1
    p = mp.Process(target=init_process, args=(rank, world_size, run))
    p.start()
    processes.append(p)

    for p in processes:
        p.join()

    print('>> Test finished')
