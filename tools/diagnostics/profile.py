import os

from lmdeploy import GenerationConfig, PytorchEngineConfig, pipeline

os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'

# enable profile cpu
os.environ['LMDEPLOY_PROFILE_CPU'] = '1'
# enable profile cuda
os.environ['LMDEPLOY_PROFILE_CUDA'] = '1'
# profile would start after 3 seconds
os.environ['LMDEPLOY_PROFILE_DELAY'] = '3'
# profile 10 seconds
os.environ['LMDEPLOY_PROFILE_DURATION'] = '10'
# prefix path to save profile files
os.environ['LMDEPLOY_PROFILE_OUT_PREFIX'] = '/path/to/save/profile_'

# configurations
tp = 1
backend_config = PytorchEngineConfig(tp=tp, )
gen_config = GenerationConfig(max_new_tokens=10, )

# init pipeline
model_path = 'Qwen/Qwen2.5-7B-Instruct'
pipe = pipeline(model_path=model_path, backend_config=backend_config, log_level='INFO')

# inference
prompt = 'what is AGI?'
response = pipe([prompt], gen_config=gen_config)  # Batch input
print(response)
