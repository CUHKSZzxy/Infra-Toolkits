#!/usr/bin/env python3
import argparse
import copy
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List

from lmdeploy import GenerationConfig, PytorchEngineConfig, TurbomindEngineConfig, pipeline
from pipeline_config import MESSAGE_BUILDERS, MODEL_PATHS, TEST_CASES


class RayPrefixFilter:

    def __init__(self, stream):
        self.stream = stream
        self.buffer = ''
        self.pattern = re.compile(r'\([^)]*(?:pid=\d+|ip=[^)]+)[^)]*\)\s*')

    def write(self, data):
        self.buffer += data
        lines = self.buffer.split('\n')
        self.buffer = lines[-1]
        for line in lines[:-1]:
            self.stream.write(self.pattern.sub('', line) + '\n')

    def flush(self):
        if self.buffer:
            self.stream.write(self.pattern.sub('', self.buffer))
            self.buffer = ''
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


sys.stdout = RayPrefixFilter(sys.stdout)
sys.stderr = RayPrefixFilter(sys.stderr)


@dataclass
class InferenceConfig:
    temperature: float = 0.0
    max_new_tokens: int = 128
    log_level: str = 'INFO'
    eager_mode: bool = False
    return_routed_experts: bool = False
    max_batch_size: int = 10


class LMDeployRunner:

    def __init__(self, backend='pt', model_name='qwen3-vl-4b', tp=1, cuda_devices='6,7', config=None):
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices
        os.environ['LMDEPLOY_SKIP_WARMUP'] = '1'
        os.environ['RAY_DEDUP_LOGS'] = '0'

        self.config = config or InferenceConfig()
        self.model_path = MODEL_PATHS.get(model_name, model_name)

        if backend == 'pt':
            backend_config = self._build_pytorch_config(tp)
        elif backend == 'tm':
            backend_config = TurbomindEngineConfig(tp=tp)
        else:
            raise ValueError(f'Unsupported backend: {backend}')

        self.pipe = pipeline(self.model_path,
                             backend_config=backend_config,
                             log_level=self.config.log_level,
                             trust_remote_code=True)
        print(f'\n{"="*50}')
        print(f'Model: {model_name}  TP={tp}  temp={self.config.temperature}  '
              f'max_tokens={self.config.max_new_tokens}')
        print(f'{"="*50}')

    def _build_pytorch_config(self, tp: int):
        kwargs = dict(tp=tp, max_batch_size=self.config.max_batch_size)
        if self.config.eager_mode:
            kwargs['eager_mode'] = True
        if self.config.return_routed_experts:
            kwargs['enable_return_routed_experts'] = True
        return PytorchEngineConfig(**kwargs)

    def run(self, messages: List[Dict], **run_kwargs):
        gen_kwargs = dict(temperature=self.config.temperature, max_new_tokens=self.config.max_new_tokens)
        if self.config.return_routed_experts:
            gen_kwargs['return_routed_experts'] = True
        gen_config = GenerationConfig(**gen_kwargs)
        return self.pipe(messages, gen_config=gen_config, **run_kwargs)


def merge_cli_run_kwargs(case_run_kwargs: dict, args) -> dict:
    run_kwargs = copy.deepcopy(case_run_kwargs)

    if args.thinking != 'case':
        chat_kwargs = run_kwargs.setdefault('chat_template_kwargs', {})
        if args.thinking == 'unset':
            chat_kwargs.pop('enable_thinking', None)
            if not chat_kwargs:
                run_kwargs.pop('chat_template_kwargs', None)
        else:
            chat_kwargs['enable_thinking'] = args.thinking == 'on'

    video_overrides = {}
    if args.video_fps is not None:
        video_overrides['fps'] = args.video_fps
    if args.video_frames is not None:
        video_overrides['num_frames'] = args.video_frames
    if video_overrides:
        run_kwargs.setdefault('media_io_kwargs', {}).setdefault('video', {}).update(video_overrides)

    pixel_overrides = {}
    if args.min_pixels is not None:
        pixel_overrides['min_pixels'] = args.min_pixels
    if args.max_pixels is not None:
        pixel_overrides['max_pixels'] = args.max_pixels
    if pixel_overrides:
        mm_kwargs = run_kwargs.setdefault('mm_processor_kwargs', {})
        mm_kwargs.setdefault('image', {}).update(pixel_overrides)
        mm_kwargs.setdefault('video', {}).update(pixel_overrides)

    return run_kwargs


def run_test(runner: LMDeployRunner, test_id: int, args):
    test_case = TEST_CASES[test_id]
    print(f"\n{'='*50}\nTEST {test_id}: {test_case.name}\n{'='*50}")
    messages = MESSAGE_BUILDERS[test_case.modality](**test_case.kwargs)
    run_kwargs = merge_cli_run_kwargs(test_case.run_kwargs, args)
    print(f'\n{runner.run(messages, **run_kwargs)}')
    print(f'\n{"="*50}\nTest End {test_id}: {test_case.name}\n{"="*50}')


def parse_args():
    parser = argparse.ArgumentParser(description='LMDeploy inference scratch runner')
    parser.add_argument('tests',
                        nargs='*',
                        default=['0'],
                        help=f'Test IDs or "all". Available: {list(TEST_CASES.keys())}')
    parser.add_argument('--backend', default='pt', choices=['pt', 'tm'])
    parser.add_argument('--model',
                        default='qwen3-vl-4b',
                        help=f'Model alias or path. Aliases: {list(MODEL_PATHS.keys())}')
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--cuda', default='6,7')
    parser.add_argument('--temp', type=float, default=0.0)
    parser.add_argument('--tokens', type=int, default=50)
    parser.add_argument('--log', default='INFO')
    parser.add_argument('--max-batch-size', type=int, default=10)
    parser.add_argument('--eager', default=False, action='store_true')
    parser.add_argument('--r3', default=False, action='store_true')
    parser.add_argument('--thinking',
                        default='case',
                        choices=['case', 'on', 'off', 'unset'],
                        help='case: keep test default; unset: omit enable_thinking.')
    parser.add_argument('--video-fps', type=float, default=None)
    parser.add_argument('--video-frames', type=int, default=None)
    parser.add_argument('--min-pixels', type=int, default=None)
    parser.add_argument('--max-pixels', type=int, default=None)
    return parser.parse_args()


def selected_test_ids(args) -> list[int]:
    if 'all' in args.tests:
        return list(TEST_CASES.keys())
    return sorted({int(t) for t in args.tests if t.isdigit() and int(t) in TEST_CASES})


def main():
    args = parse_args()
    test_ids = selected_test_ids(args)
    if not test_ids:
        print(f'No valid tests. Available: {list(TEST_CASES.keys())}')
        return

    config = InferenceConfig(
        temperature=args.temp,
        max_new_tokens=args.tokens,
        log_level=args.log,
        eager_mode=args.eager,
        return_routed_experts=args.r3,
        max_batch_size=args.max_batch_size,
    )
    runner = LMDeployRunner(backend=args.backend,
                            model_name=args.model,
                            tp=args.tp,
                            cuda_devices=args.cuda,
                            config=config)
    for test_id in test_ids:
        run_test(runner, test_id, args)


if __name__ == '__main__':
    main()
"""
python pipeline.py --model qwen3-omni-30b --cuda 6 --tp 1 2
python pipeline.py --model qwen3-vl-4b --cuda 7 --tp 1 2
python pipeline.py --model qwen3-30b --cuda 7 --tp 1 0
python pipeline.py --model qwen35-4b --cuda 7 --tp 1 2
python pipeline.py --model glm-4.1v-9b --cuda 7 --tp 1 1 --thinking on
python pipeline.py --model interns1-pro --cuda 7 --tp 1 8 --thinking unset
python pipeline.py --model internvl3-8b-hf --cuda 7 --tp 1 1
python pipeline.py --model internvl3-1b --cuda 7 --tp 1 1
python pipeline.py --model interns2 --cuda 5 --tp 1 8 --video-fps 2 --video-frames 10
"""
