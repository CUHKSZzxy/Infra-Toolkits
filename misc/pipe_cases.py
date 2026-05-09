from dataclasses import dataclass, field
from typing import Dict, List

IMAGE_URL = 'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg'
VIDEO_URL = 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4'
AUDIO_URL = 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav'
TIME_SERIES_URL = 'https://raw.githubusercontent.com/CUHKSZzxy/Online-Data/main/0092638_seism.npy'

MODEL_PATHS = {
    'qwen2.5-vl-7b': 'Qwen/Qwen2.5-VL-7B-Instruct',
    'qwen3-8b': 'Qwen/Qwen3-8B',
    'qwen3-8b-fp8': 'Qwen/Qwen3-8B-FP8',
    'qwen3-30b': 'Qwen/Qwen3-30B-A3B',
    'qwen3-vl-4b': 'Qwen/Qwen3-VL-4B-Instruct',
    'qwen3-vl-30b': 'Qwen/Qwen3-VL-30B-A3B-Instruct',
    'qwen3-omni-30b': 'Qwen/Qwen3-Omni-30B-A3B-Instruct',
    'qwen35-4b': 'Qwen/Qwen3.5-4B',
    'qwen35-35b': 'Qwen/Qwen3.5-35B-A3B',
    'glm-4.1v-9b': 'zai-org/GLM-4.1V-9B-Thinking',
    'interns1-mini': 'internlm/Intern-S1-mini',
    'internvl3-1b': 'OpenGVLab/InternVL3-1B',
    'internvl3-8b-hf': 'OpenGVLab/InternVL3-8B-hf',
    'internvl35-8b': 'OpenGVLab/InternVL3_5-8B',
    'interns1-pro': 'internlm/Intern-S1-Pro',
    'interns2': 'internlm/Intern-S2-Preview',
}


def default_run_kwargs() -> dict:
    return {
        'chat_template_kwargs': {
            'enable_thinking': True,
        },
    }


@dataclass
class TestCase:
    name: str
    modality: str
    kwargs: dict
    run_kwargs: dict = field(default_factory=default_run_kwargs)


def _user_msg(content: list) -> List[Dict]:
    final_message = [{'role': 'user', 'content': content}]
    print(f'Input message:\n{final_message}\n')
    return final_message


def _media(media_type: str, urls, prompt: str) -> List[Dict]:
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
    'mixed_image_video':
    lambda image_url, video_url, prompt='Describe this image and video': _user_msg([
        {
            'type': 'image_url',
            'image_url': {
                'url': image_url
            }
        },
        {
            'type': 'video_url',
            'video_url': {
                'url': video_url
            }
        },
        {
            'type': 'text',
            'text': prompt
        },
    ]),
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
            'type': 'time_series_url',
            'time_series_url': {
                'url': url,
                'sampling_rate': sampling_rate
            }
        },
    ]),
}

TEST_CASES: Dict[int, TestCase] = {
    0:
    TestCase('Text', 'text', {'prompt': 'Who are you?'}),
    1:
    TestCase('Single Image', 'image', {
        'url': IMAGE_URL,
        'prompt': 'Describe this image.',
    }),
    2:
    TestCase('Single Video', 'video', {
        'url': VIDEO_URL,
        'prompt': 'Describe this video.',
    }),
    3:
    TestCase('Single Audio', 'audio', {
        'url': AUDIO_URL,
        'prompt': 'Describe this audio.',
    }),
    4:
    TestCase('Multi Image', 'multi_image', {
        'urls': [IMAGE_URL, IMAGE_URL],
        'prompt': 'Compare these two images. What are the similarities and differences?',
    }),
    5:
    TestCase('Multi Video', 'multi_video', {
        'urls': [VIDEO_URL, VIDEO_URL],
        'prompt': 'Compare these two videos. What are the similarities and differences?',
    }),
    6:
    TestCase('Multi Audio', 'multi_audio', {
        'urls': [AUDIO_URL, AUDIO_URL],
        'prompt': 'Compare these two audios. What are the similarities and differences?',
    }),
    7:
    TestCase('Mixed Image+Video', 'mixed_image_video', {
        'image_url': IMAGE_URL,
        'video_url': VIDEO_URL,
        'prompt': 'Describe both the image and the video.',
    }),
    8:
    TestCase('Time Series', 'time_series', {
        'url': TIME_SERIES_URL,
        'sampling_rate': 100,
    }),
}
