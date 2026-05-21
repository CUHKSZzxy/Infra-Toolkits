import os

from lmdeploy import PytorchEngineConfig, pipeline

# from lmdeploy.vl import load_image

os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'

# configurations
tp = 1
backend_config = PytorchEngineConfig(tp=tp, )

# init pipeline
model_path = 'Qwen/Qwen2.5-VL-72B-Instruct'
pipe = pipeline(model_path, backend_config=backend_config, log_level='INFO')

# inference
url_a = 'https://raw.githubusercontent.com/QwenLM/Qwen-VL/master/assets/mm_tutorial/Beijing_Small.jpeg'
url_b = 'https://raw.githubusercontent.com/QwenLM/Qwen-VL/master/assets/mm_tutorial/Chongqing_Small.jpeg'

messages = [
    dict(role='user',
         content=[
             dict(type='text', text='Describe the two images in detail.'),
             dict(type='image_url', image_url=dict(url=url_a)),
             dict(type='image_url', image_url=dict(url=url_b))
         ])
]

response = pipe(messages)
messages.append(dict(role='assistant', content=response.text))
messages.append(dict(role='user', content='What are the similarities and differences between these two images.'))
response = pipe(messages)
print(response)
