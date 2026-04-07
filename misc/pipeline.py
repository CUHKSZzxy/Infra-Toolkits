#!/usr/bin/env python3
import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List

from lmdeploy import GenerationConfig, PytorchEngineConfig, TurbomindEngineConfig, pipeline

# ── Output filter ──────────────────────────────────────────────────────────────


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

# ── Config & runner ────────────────────────────────────────────────────────────


@dataclass
class InferenceConfig:
    temperature: float = 0.0
    max_new_tokens: int = 128
    log_level: str = 'INFO'
    eager_mode: bool = False
    return_routed_experts: bool = False


class LMDeployRunner:
    MODELS = {
        'qwen2.5-vl-7b': '/real_model_path',
        'qwen3-8b': '/real_model_path',
        'qwen3-8b-fp8': '/real_model_path',
        'qwen3-30b': '/real_model_path',
        'qwen3-vl-4b': '/real_model_path',
        'qwen3-vl-30b': '/real_model_path',
        'qwen3-omni-30b': '/real_model_path',
        'qwen35-4b': '/real_model_path',
        'qwen35-35b': '/real_model_path',
        'glm-4.1v-9b': '/real_model_path',
        'interns1-mini': '/real_model_path',
        'internvl3-8b': '/real_model_path',
        'internvl35-8b': '/real_model_path',
    }

    def __init__(self, backend='pt', model_name='qwen3-vl-4b', model_path=None, tp=1, cuda_devices='6,7', config=None):
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices
        os.environ['LMDEPLOY_SKIP_WARMUP'] = '1'
        os.environ['RAY_DEDUP_LOGS'] = '0'

        self.config = config or InferenceConfig()
        self.model_path = model_path or self.MODELS.get(model_name, model_name)

        if backend == 'pt':
            pt_kwargs = dict(tp=tp)
            if self.config.eager_mode:
                pt_kwargs['eager_mode'] = True
            if self.config.return_routed_experts:
                pt_kwargs['enable_return_routed_experts'] = True
            backend_config = PytorchEngineConfig(**pt_kwargs)
        elif backend == 'tm':
            backend_config = TurbomindEngineConfig(tp=tp)

        self.pipe = pipeline(self.model_path, backend_config=backend_config, log_level=self.config.log_level)
        print(f'\n{"="*50}')
        print(f'Model: {model_name}  TP={tp}  temp={self.config.temperature}  max_tokens={self.config.max_new_tokens}')
        print(f'{"="*50}')

    def run(self, messages: List[Dict], **run_kwargs):
        gen_kwargs = dict(temperature=self.config.temperature, max_new_tokens=self.config.max_new_tokens)
        if self.config.return_routed_experts:
            gen_kwargs['return_routed_experts'] = True
        gen_config = GenerationConfig(**gen_kwargs)
        return self.pipe(messages, gen_config=gen_config, **run_kwargs)


# ── Message builders ───────────────────────────────────────────────────────────


def _user_msg(content: list) -> List[Dict]:
    final_message = [{'role': 'user', 'content': content}]
    print(f'Input message:\n{final_message}\n')
    return final_message


def _media(media_type: str, urls, prompt: str) -> List[Dict]:
    """Unified builder for single/multi image, video, or audio messages."""
    key = f'{media_type}_url'
    url_list = urls if isinstance(urls, list) else [urls]
    items = [{'type': key, key: {'url': u}} for u in url_list]
    return _user_msg(items + [{'type': 'text', 'text': prompt}])


MESSAGE_BUILDERS = {
    'text':
    lambda prompt: _user_msg([{
        'type': 'text',
        'text': prompt
    }]),
    'image':
    lambda url, prompt='Describe this image': _media('image', url, prompt),
    'multi_image':
    lambda urls, prompt='Describe these images': _media('image', urls, prompt),
    'video':
    lambda url, prompt='Describe this video': _media('video', url, prompt),
    'multi_video':
    lambda urls, prompt='Describe these videos': _media('video', urls, prompt),
    'audio':
    lambda url, prompt='Describe this audio': _media('audio', url, prompt),
    'multi_audio':
    lambda urls, prompt='Describe these audios': _media('audio', urls, prompt),
    'time_series':
    lambda url, sampling_rate, prompt=None: _user_msg([
        {
            'type':
            'text',
            'text':
            prompt or ('Please determine whether an Earthquake event has occurred. '
                       'If so, specify P-wave and S-wave starting indices.')
        },
        {
            'type': 'time_series',
            'time_series_url': {
                'url': url,
                'sampling_rate': sampling_rate
            }
        },
    ]),
}

# ── Test cases ─────────────────────────────────────────────────────────────────


@dataclass
class TestCase:
    name: str
    modality: str  # key into MESSAGE_BUILDERS
    kwargs: dict  # passed to the builder
    run_kwargs: dict = field(default_factory=dict)
    # run_kwargs = {
    #     'media_io_kwargs': {
    #         'video': {
    #             'fps': 2,
    #             'num_frames': 10,
    #         },
    #         'mm_processor_kwargs': {
    #             'min_pixels': 4 * 32 * 32,
    #             'max_pixels': 256 * 32 * 32,
    #         }
    #     }
    # }


TEST_CASES: Dict[int, TestCase] = {
    0:
    TestCase('Text', 'text', {'prompt': 'Who are you?'}),
    1:
    TestCase(
        'Single Image', 'image', {
            'url': 'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg',
            'prompt': 'Describe this image',
        }),
    2:
    TestCase('Single Video', 'video', {
        'url': 'file:///nvme1/zhouxinyu/lmdeploy_fp8/clip_3_removed.mp4',
        'prompt': 'Describe this video',
    }),
    3:
    TestCase('Single Audio', 'audio', {
        'url': 'file:///nvme1/zhouxinyu/lmdeploy_vl/cough.wav',
        'prompt': 'Describe this audio',
    }),
    4:
    TestCase(
        'Multi Image', 'multi_image', {
            'urls': [
                'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg',
                'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg',
            ],
            'prompt':
            'Compare these two images. What are the similarities and differences?',
        }),
    5:
    TestCase(
        'Multi Video', 'multi_video', {
            'urls': [
                'file:///nvme1/zhouxinyu/lmdeploy_fp8/space_woaudio.mp4',
                'file:///nvme1/zhouxinyu/lmdeploy_fp8/space_woaudio.mp4',
            ],
            'prompt':
            'Compare these two videos. What are the similarities and differences?',
        }),
    6:
    TestCase(
        'Multi Audio', 'multi_audio', {
            'urls': [
                'file:///nvme1/zhouxinyu/lmdeploy_vl/cough.wav',
                'file:///nvme1/zhouxinyu/lmdeploy_vl/cough.wav',
            ],
            'prompt': 'Compare these two audios. What are the similarities and differences?',
        }),
    7:
    TestCase('Time Series', 'time_series', {
        'url': 'https://raw.githubusercontent.com/CUHKSZzxy/Online-Data/main/0092638_seism.npy',
        'sampling_rate': 100,
    }),
}

# ── Entry point ────────────────────────────────────────────────────────────────


def run_test(runner: LMDeployRunner, test_id: int):
    tc = TEST_CASES[test_id]
    print(f"\n{'='*50}\nTEST {test_id}: {tc.name}\n{'='*50}")
    messages = MESSAGE_BUILDERS[tc.modality](**tc.kwargs)
    print(runner.run(messages, **tc.run_kwargs))


def main():
    parser = argparse.ArgumentParser(description='LMDeploy Inference')
    parser.add_argument('tests',
                        nargs='*',
                        default=['0'],
                        help=f'Test IDs or "all". Available: {list(TEST_CASES.keys())}')
    parser.add_argument('--backend', default='pt', choices=['pt', 'tm'])
    parser.add_argument('--model',
                        default='qwen3-vl-4b',
                        help=f'Model alias or path. Aliases: {list(LMDeployRunner.MODELS.keys())}')
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--cuda', default='6,7')
    parser.add_argument('--temp', type=float, default=0.0)
    parser.add_argument('--max-tokens', type=int, default=50)
    parser.add_argument('--log-level', default='INFO')
    parser.add_argument('--eager-mode', default=False, action='store_true')
    parser.add_argument('--routed-experts', default=False, action='store_true')
    args = parser.parse_args()

    test_ids = (list(TEST_CASES.keys())
                if 'all' in args.tests else sorted({int(t)
                                                    for t in args.tests if t.isdigit() and int(t) in TEST_CASES}))
    if not test_ids:
        print(f'No valid tests. Available: {list(TEST_CASES.keys())}')
        return

    config = InferenceConfig(
        temperature=args.temp,
        max_new_tokens=args.max_tokens,
        log_level=args.log_level,
        eager_mode=args.eager_mode,
        return_routed_experts=args.routed_experts,
    )
    runner = LMDeployRunner(backend=args.backend,
                            model_name=args.model,
                            tp=args.tp,
                            cuda_devices=args.cuda,
                            config=config)
    for tid in test_ids:
        run_test(runner, tid)


if __name__ == '__main__':
    main()
"""
python 0_pipe.py 2 --model qwen3-omni-30b --cuda 6 --tp 1
python 0_pipe.py 2 --model qwen3-vl-4b --cuda 7 --tp 1
python 0_pipe.py 0 --model qwen3-30b --cuda 7 --tp 1
"""
