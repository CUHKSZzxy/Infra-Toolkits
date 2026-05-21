#!/usr/bin/env python3
import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional

import requests

# Configuration
BASE_URL = 'http://xxx.xxx.xxx.xxx:xxxx'
MODEL = 'xxx'  # e.g., model identifier or path
IMAGE_PATH = 'xxx'  # path to input image
PROMPT = 'Describe the content of this image in detail.'
NUM_REQUESTS = 8  # number of concurrent requests


def get_image_mime_type(image_path: str) -> str:
    """Determine MIME type based on file extension."""
    ext = os.path.splitext(image_path)[1].lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp'
    }
    return mime_types.get(ext, 'image/jpeg')


def send_single_request(request_id: int, image_b64: str, prompt: str, mime_type: str) -> Optional[Dict]:
    """Send a single API request."""
    payload = {
        'model':
        MODEL,
        'messages': [{
            'role':
            'user',
            'content': [{
                'type': 'text',
                'text': prompt
            }, {
                'type': 'image_url',
                'image_url': {
                    'url': f'data:{mime_type};base64,{image_b64}'
                }
            }]
        }],
        'temperature':
        0.7,
        'max_tokens':
        1000
    }

    try:
        start_time = time.time()
        response = requests.post(f'{BASE_URL}/v1/chat/completions',
                                 headers={'Content-Type': 'application/json'},
                                 json=payload,
                                 timeout=300)
        response.raise_for_status()
        elapsed_time = time.time() - start_time

        result = response.json()
        result['_request_id'] = request_id
        result['_elapsed_time'] = elapsed_time
        return result

    except requests.exceptions.Timeout:
        print(f'❌ Request #{request_id}: Timeout')
        return None
    except requests.exceptions.RequestException as e:
        print(f'❌ Request #{request_id}: Error - {e}')
        if hasattr(e, 'response') and e.response:
            try:
                print(f'   Response: {e.response.text[:200]}')
            except Exception:
                print('   Response: <unable to read response text>')
                return None
        return None
    except Exception as e:
        print(f'❌ Request #{request_id}: Unknown error - {e}')
        return None


def test_image_caption():
    print('🖼️  Reading and encoding image file...')

    if not os.path.exists(IMAGE_PATH):
        print(f'❌ Error: Image file not found: {IMAGE_PATH}')
        return

    file_size_mb = os.path.getsize(IMAGE_PATH) / (1024 * 1024)
    print(f'Image size: {file_size_mb:.2f} MB')

    mime_type = get_image_mime_type(IMAGE_PATH)
    print(f'Image type: {mime_type}')

    print('Encoding image to base64...')
    with open(IMAGE_PATH, 'rb') as f:
        image_b64 = base64.b64encode(f.read()).decode('utf-8')

    print(f'Base64 size: {len(image_b64) / (1024 * 1024):.2f} MB')
    print(f'\n🚀 Sending {NUM_REQUESTS} concurrent requests...\n')

    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=NUM_REQUESTS) as executor:
        future_to_id = {
            executor.submit(send_single_request, i + 1, image_b64, PROMPT, mime_type): i + 1
            for i in range(NUM_REQUESTS)
        }

        completed = 0
        for future in as_completed(future_to_id):
            request_id = future_to_id[future]
            completed += 1
            try:
                result = future.result()
                if result:
                    results.append(result)
                    print(f'✅ Request #{request_id} completed ({completed}/{NUM_REQUESTS})')
                else:
                    print(f'⚠️  Request #{request_id} failed ({completed}/{NUM_REQUESTS})')
            except Exception as e:
                print(f'❌ Request #{request_id} exception: {e}')

    total_time = time.time() - start_time

    print('\n' + '=' * 80)
    print('📊 Summary')
    print('=' * 80)
    print(f'Total requests: {NUM_REQUESTS}')
    print(f'Success: {len(results)}')
    print(f'Failed: {NUM_REQUESTS - len(results)}')
    print(f'Total time: {total_time:.2f} seconds')
    if results:
        avg_time = sum(r.get('_elapsed_time', 0) for r in results) / len(results)
        print(f'Average per-request time: {avg_time:.2f} seconds')
    print('=' * 80)

    for i, result in enumerate(results, 1):
        request_id = result.get('_request_id', i)
        elapsed = result.get('_elapsed_time', 0)

        print(f"\n{'='*80}")
        print(f'📝 Request #{request_id} (Time: {elapsed:.2f}s)')
        print('=' * 80)

        if 'choices' in result and result['choices']:
            content = result['choices'][0]['message']['content']
            print('✅ Model response:')
            print('-' * 80)
            print(content)
            print('-' * 80)

            if 'usage' in result:
                usage = result['usage']
                print('📊 Token usage:')
                print(f"  - Prompt: {usage.get('prompt_tokens', 'N/A')}")
                print(f"  - Completion: {usage.get('completion_tokens', 'N/A')}")
                print(f"  - Total: {usage.get('total_tokens', 'N/A')}")

        else:
            print('⚠️  Unexpected response format:')
            print(json.dumps(result, indent=2, ensure_ascii=False))

    output_file = 'test_image_caption_results.json'
    output_data = {
        'config': {
            'base_url': BASE_URL,
            'model': MODEL,
            'image_path': IMAGE_PATH,
            'prompt': PROMPT,
            'num_requests': NUM_REQUESTS,
            'total_time': total_time
        },
        'results': results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f'\n💾 Results saved to: {output_file}')


if __name__ == '__main__':
    test_image_caption()
