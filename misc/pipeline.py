#!/usr/bin/env python3
import argparse
import os
import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional

from lmdeploy import GenerationConfig, PytorchEngineConfig, pipeline


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


class InputType(Enum):
    TEXT = 0
    IMAGE = 1
    VIDEO = 2
    MULTI_IMAGE = 3
    MULTI_VIDEO = 4
    TIME_SERIES = 5


@dataclass
class InferenceConfig:
    temperature: float = 0.0
    max_new_tokens: int = 128
    log_level: str = 'INFO'


class LMDeployRunner:
    MODELS = {
        'qwen3-vl-4b': '/real_model_path',
        'qwen3-vl-30b': '/real_model_path',
        'qwen3-30b': '/real_model_path',
        'qwen3-8b': '/real_model_path',
        'qwen2.5-vl-7b': '/real_model_path',
        'glm-4.1v-9b': '/real_model_path',
    }

    def __init__(self, model_name='qwen3-vl-4b', model_path=None, tp=1, cuda_devices='6,7', config=None):
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices
        os.environ['LMDEPLOY_SKIP_WARMUP'] = '1'
        os.environ['RAY_DEDUP_LOGS'] = '0'

        self.config = config or InferenceConfig()
        self.model_path = model_path or self.MODELS.get(model_name, model_name)

        backend_config = PytorchEngineConfig(tp=tp)
        self.pipe = pipeline(self.model_path, backend_config=backend_config, log_level=self.config.log_level)
        print(f'\n{"="*50}')
        print(f'Model loaded: {model_name} (TP={tp})')
        print(f'Gen Config: temperature={self.config.temperature}, max_new_tokens={self.config.max_new_tokens}')
        print(f'{"="*50}\n')

    def run(self, messages: List[Dict]) -> str:
        gen_config = GenerationConfig(temperature=self.config.temperature, max_new_tokens=self.config.max_new_tokens)
        response = self.pipe(messages, gen_config=gen_config)
        # return response.text if hasattr(response, 'text') else str(response)
        return response


class MessageBuilder:

    @staticmethod
    def text(prompt: str) -> List[Dict]:
        return [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}]

    @staticmethod
    def image(url: str, prompt: str = 'Describe this image') -> List[Dict]:
        return [{
            'role': 'user',
            'content': [{
                'type': 'image_url',
                'image_url': {
                    'url': url
                }
            }, {
                'type': 'text',
                'text': prompt
            }]
        }]

    @staticmethod
    def multi_image(urls: List[str], prompt: str = 'Describe these images') -> List[Dict]:
        content = [{'type': 'image_url', 'image_url': {'url': url}} for url in urls]
        content.append({'type': 'text', 'text': prompt})
        return [{'role': 'user', 'content': content}]

    @staticmethod
    def video(url: str, prompt: str = 'Describe this video') -> List[Dict]:
        return [{
            'role': 'user',
            'content': [{
                'type': 'video_url',
                'video_url': {
                    'url': url
                }
            }, {
                'type': 'text',
                'text': prompt
            }]
        }]

    @staticmethod
    def multi_video(urls: List[str], prompt: str = 'Describe these videos') -> List[Dict]:
        content = [{'type': 'video_url', 'video_url': {'url': url}} for url in urls]
        content.append({'type': 'text', 'text': prompt})
        return [{'role': 'user', 'content': content}]

    @staticmethod
    def time_series(url: str, sampling_rate: int, prompt: Optional[str] = None) -> List[Dict]:
        default = ('Please determine whether an Earthquake event has occurred. '
                   'If so, specify P-wave and S-wave starting indices.')
        return [{
            'role':
            'user',
            'content': [{
                'type': 'text',
                'text': prompt or default
            }, {
                'type': 'time_series',
                'time_series_url': {
                    'url': url,
                    'sampling_rate': sampling_rate
                }
            }]
        }]


BUILDERS: Dict[InputType, Callable] = {
    InputType.TEXT: MessageBuilder.text,
    InputType.IMAGE: MessageBuilder.image,
    InputType.VIDEO: MessageBuilder.video,
    InputType.MULTI_IMAGE: MessageBuilder.multi_image,
    InputType.MULTI_VIDEO: MessageBuilder.multi_video,
    InputType.TIME_SERIES: MessageBuilder.time_series,
}

TEST_CASES = {
    0: ('Text', InputType.TEXT, {
        'prompt': 'Who are you?'
    }),
    1: ('Single Image', InputType.IMAGE, {
        'url': 'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg',
        'prompt': 'Describe this image'
    }),
    2: (
        'Single Video',
        InputType.VIDEO,
        {
            'url': 'file:///nvme1/zhouxinyu/lmdeploy_fp8/space_woaudio.mp4',
            # "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4",
            # "url": "https://raw.githubusercontent.com/CUHKSZzxy/Online-Data/main/clip_3_removed.mp4",
            'prompt': 'Describe this video'
        }),
    3: ('Multi Image', InputType.MULTI_IMAGE, {
        'urls': [
            'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg',
            'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg'
        ],
        'prompt':
        'Compare these two images. What are the similarities and differences?'
    }),
    4: ('Multi Video', InputType.MULTI_VIDEO, {
        'urls': [
            'file:///nvme1/zhouxinyu/lmdeploy_fp8/space_woaudio.mp4',
            'file:///nvme1/zhouxinyu/lmdeploy_fp8/space_woaudio.mp4'
        ],
        'prompt':
        'Compare these two videos. What are the similarities and differences?'
    }),
    5: ('Time Series', InputType.TIME_SERIES, {
        'url': 'https://raw.githubusercontent.com/CUHKSZzxy/Online-Data/main/0092638_seism.npy',
        'sampling_rate': 100
    }),
}


def run_test(runner: LMDeployRunner, test_id: int):
    name, input_type, kwargs = TEST_CASES[test_id]
    print(f"\n{'='*50}")
    print(f'TEST {test_id}: {name}')
    print('=' * 50)

    builder = BUILDERS[input_type]
    messages = builder(**kwargs)
    print(runner.run(messages))


def main():
    parser = argparse.ArgumentParser(description='LMDeploy Inference')
    parser.add_argument('tests', nargs='*', default=['all'], help='Test IDs (0-5) or "all"')
    parser.add_argument('--model', default='qwen3-vl-4b', help='Model name or path')
    parser.add_argument('--tp', type=int, default=1, help='Tensor parallelism')
    parser.add_argument('--cuda', default='6,7', help='CUDA devices')
    parser.add_argument('--temp', type=float, default=0.0, help='temperature')
    parser.add_argument('--max-tokens', type=int, default=50, help='Max new tokens')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    args = parser.parse_args()

    # NOTE: 0: Text, 1: Single Image, 2: Single Video, 3: Multi Image,  4: Multi Video, 5: Time Series
    if 'all' in args.tests:
        test_ids = list(TEST_CASES.keys())
    else:
        test_ids = sorted(set(int(t) for t in args.tests if t.isdigit() and int(t) in TEST_CASES))

    if not test_ids:
        print(f'No valid tests. Available: {list(TEST_CASES.keys())}')
        return

    config = InferenceConfig(temperature=args.temp, max_new_tokens=args.max_tokens, log_level=args.log_level)
    runner = LMDeployRunner(model_name=args.model, tp=args.tp, cuda_devices=args.cuda, config=config)

    for tid in test_ids:
        run_test(runner, tid)


if __name__ == '__main__':
    main()
