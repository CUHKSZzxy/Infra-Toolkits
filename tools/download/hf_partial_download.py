from huggingface_hub import snapshot_download

repo_id = 'internlm/Intern-S1-mini'

download_path = snapshot_download(
    repo_id=repo_id,
    # Ignore common weight file extensions
    ignore_patterns=['*.safetensors', '*.bin', '*.pth', '*.pt'],
    # Optional: Specify a local folder
    local_dir='/nvme1/zhouxinyu/Intern-S1-mini',
    # Some models require login, set to True if you have access tokens
    token=None)

print(f'\nFiles downloaded to: {download_path}')
