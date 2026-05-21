# Useful AI Infra Toolkits

Personal scripts for LMDeploy serving, OpenAI-compatible smoke tests, pipeline
experiments, downloads, Docker, and runtime diagnostics.

Run commands from the repository root unless a script says otherwise. Some
scripts intentionally keep personal model IDs, ports, IPs, CUDA IDs, or
placeholder constants; treat those as editable local templates.

## Folder Structure

```text
Infra-Toolkits/
├── client/                   # OpenAI-compatible curl/Python request scripts
├── data/                     # Small local sample assets used by examples
├── debug/                    # NCCL, Ray log, tensor, NaN/Inf, and profile helpers
├── docker/                   # Docker build/run snippets
├── download/                 # Hugging Face, VS Code, and offline CLI helpers
├── model/                    # Model-file manipulation helpers
├── notes/                    # Investigation notes and captured outputs
├── pipeline/                 # LMDeploy pipeline runners and reusable cases
├── serve/                    # LMDeploy API server/proxy launch snippets
├── .pre-commit-config.yaml
├── .pylintrc
└── README.md
```

## Tool Index

| Need                                                       | Tool                                                                                               | Example command                                                                                    |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Start a local TP LMDeploy API server                       | `serve/serve_tp_local.sh`                                                                          | `bash serve/serve_tp_local.sh`                                                                     |
| Start local DP+TP or DP+EP serving                         | `serve/serve_dptp_local.sh`, `serve/serve_dpep_local.sh`                                           | `bash serve/serve_dpep_local.sh`                                                                   |
| Keep distributed DP+EP launch snippets                     | `serve/serve_dpep_dist.sh`                                                                         | Edit node-specific block, then run on each node                                                    |
| Send OpenAI-compatible chat/generate curl requests         | `client/curl_chat.sh`, `client/curl_generate.sh`                                                   | `bash client/curl_chat.sh`                                                                         |
| Test OpenAI-compatible tool calls                          | `client/test_oai_tool_call.py`, `client/curl_tool_calls.sh`                                        | `python client/test_oai_tool_call.py`                                                              |
| Test OpenAI-compatible image/video/time-series requests    | `client/test_oai_image.py`, `client/test_oai_video.py`, `client/test_oai_time_series.py`           | `python client/test_oai_image.py`                                                                  |
| Stress image/video caption APIs with local base64 payloads | `client/test_oai_image_caption.py`, `client/test_oai_video_caption.py`                             | Edit constants, then `python client/test_oai_image_caption.py`                                     |
| Run reusable LMDeploy pipeline cases                       | `pipeline/pipeline.py`, `pipeline/pipeline_config.py`                                              | `python pipeline/pipeline.py --model qwen3-vl-4b --cuda 7 --tp 1 2`                                |
| Run simple TP/EP pipeline examples                         | `pipeline/test_pipeline_tp.py`, `pipeline/test_pipeline_ep.py`                                     | `python pipeline/test_pipeline_tp.py`                                                              |
| Compare or save/load tensors                               | `debug/comp_tensor_utils.py`                                                                       | Import `compare_tensors`, `save_tensor`, or `load_tensor`                                          |
| Detect NaN/Inf in tensors                                  | `debug/detect_inf_nan.py`                                                                          | Import `contains_inf_or_nan`                                                                       |
| Filter Ray log prefixes                                    | `debug/ray_log_filter.py`                                                                          | Import before noisy Ray/LMDeploy logs                                                              |
| Run a small NCCL all-reduce check                          | `debug/test_internode_nccl.py`                                                                     | Edit rank/master values, then `python debug/test_internode_nccl.py`                                |
| Run LMDeploy with profiler env vars                        | `debug/profile.py`                                                                                 | Edit output path, then `python debug/profile.py`                                                   |
| Download Hugging Face models or metadata                   | `download/hf_download.sh`, `download/hf_partial_download.py`                                       | `bash download/hf_download.sh`                                                                     |
| Prepare offline VS Code server files                       | `download/offline_download_vscode.sh`                                                              | `bash download/offline_download_vscode.sh`                                                         |
| Prepare or install an offline Codex CLI bundle             | `download/prepare_codex_offline_bundle.sh`, `download/install_codex_offline_bundle.sh`             | `bash download/install_codex_offline_bundle.sh --bundle codex-cli-offline-bundle-*.tar.gz`         |
| Prepare or install an offline Claude Code CLI bundle       | `download/prepare_claude_code_offline_bundle.sh`, `download/install_claude_code_offline_bundle.sh` | `bash download/install_claude_code_offline_bundle.sh --bundle claude-code-offline-bundle-*.tar.gz` |
| Build or start Docker containers                           | `docker/build_docker_image.sh`, `docker/start_docker.sh`                                           | Edit image/volume values, then run with `bash`                                                     |
| Create a model directory with one copied file and symlinks | `model/hack_model_file.sh`                                                                         | Set env vars, then `bash model/hack_model_file.sh`                                                 |

## Pre-commit

```bash
pip install -U pre-commit
pre-commit install
pre-commit run --all-files
```
