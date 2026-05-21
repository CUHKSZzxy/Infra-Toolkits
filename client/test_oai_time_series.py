from lmdeploy.vl.time_series_utils import encode_time_series_base64
from openai import OpenAI

openai_api_key = 'EMPTY'
openai_api_base = 'http://0.0.0.0:8000/v1'
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
model_name = client.models.list().data[0].id
prompt = ('Please determine whether an Earthquake event has occurred in the provided time-series data. '
          'If so, please specify the starting time point indices of the P-wave and S-wave in the event.')


def send_base64(file_path: str, sampling_rate: int = 100):
    """Base64-encoded time-series data."""

    # encode_time_series_base64 accepts local file paths and http urls,
    # encoding time-series data (.npy, .csv, .wav, .mp3, .flac, etc.) into base64 strings.
    base64_ts = encode_time_series_base64(file_path)

    messages = [{
        'role':
        'user',
        'content': [
            {
                'type': 'text',
                'text': prompt
            },
            {
                'type': 'time_series_url',
                'time_series_url': {
                    'url': f'data:time_series/npy;base64,{base64_ts}',
                    'sampling_rate': sampling_rate
                },
            },
        ],
    }]

    return client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
        max_tokens=200,
    )


def send_http_url(url: str, sampling_rate: int = 100):
    """Http(s) url pointing to the time-series data."""
    messages = [{
        'role':
        'user',
        'content': [
            {
                'type': 'text',
                'text': prompt
            },
            {
                'type': 'time_series_url',
                'time_series_url': {
                    'url': url,
                    'sampling_rate': sampling_rate
                },
            },
        ],
    }]

    return client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
        max_tokens=200,
    )


def send_file_url(file_path: str, sampling_rate: int = 100):
    """File url pointing to the time-series data."""
    messages = [{
        'role':
        'user',
        'content': [
            {
                'type': 'text',
                'text': prompt
            },
            {
                'type': 'time_series_url',
                'time_series_url': {
                    'url': f'file://{file_path}',
                    'sampling_rate': sampling_rate
                },
            },
        ],
    }]

    return client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
        max_tokens=200,
    )


response = send_base64('./data/0068636_seism.npy')
# response = send_http_url("https://raw.githubusercontent.com/CUHKSZzxy/Online-Data/main/0068636_seism.npy")
# response = send_file_url("./data/0068636_seism.npy")

print(response.choices[0].message)
