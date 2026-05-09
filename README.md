# Useful AI Infra Toolkits

Personal scripts for LMDeploy serving, OpenAI-compatible smoke tests, pipeline
experiments, downloads, Docker, and runtime diagnostics.

Run commands from the repository root unless a script says otherwise. Some
scripts intentionally keep personal model IDs, ports, IPs, CUDA IDs, or
placeholder constants; treat those as editable local templates.

## Folder Structure

```text
Infra-Toolkits/
├── data/                     # Small local sample assets used by examples
├── examples/
│   └── notes/                # Investigation notes and captured outputs
├── tools/
│   ├── client/               # OpenAI-compatible curl/Python request scripts
│   ├── diagnostics/          # NCCL, Ray log, tensor, NaN/Inf, and profile helpers
│   ├── docker/               # Docker build/run snippets
│   ├── download/             # Hugging Face and offline VS Code helpers
│   ├── model/                # Model-file manipulation helpers
│   ├── pipeline/             # LMDeploy pipeline runners and reusable cases
│   └── serve/                # LMDeploy API server/proxy launch snippets
├── .pre-commit-config.yaml
├── .pylintrc
└── README.md
```

## Tool Index

| Need                                                       | Tool                                                                                                       | Example command                                                                 |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Start a local TP LMDeploy API server                       | `tools/serve/serve_tp_local.sh`                                                                            | `bash tools/serve/serve_tp_local.sh`                                            |
| Start local DP+TP or DP+EP serving                         | `tools/serve/serve_dptp_local.sh`, `tools/serve/serve_dpep_local.sh`                                       | `bash tools/serve/serve_dpep_local.sh`                                          |
| Keep distributed DP+EP launch snippets                     | `tools/serve/serve_dpep_dist.sh`                                                                           | Edit node-specific block, then run on each node                                 |
| Send OpenAI-compatible chat/generate curl requests         | `tools/client/curl_chat.sh`, `tools/client/curl_generate.sh`                                               | `bash tools/client/curl_chat.sh`                                                |
| Test OpenAI-compatible tool calls                          | `tools/client/test_oai_tool_call.py`, `tools/client/curl_tool_calls.sh`                                    | `python tools/client/test_oai_tool_call.py`                                     |
| Test OpenAI-compatible image/video/time-series requests    | `tools/client/test_oai_image.py`, `tools/client/test_oai_video.py`, `tools/client/test_oai_time_series.py` | `python tools/client/test_oai_image.py`                                         |
| Stress image/video caption APIs with local base64 payloads | `tools/client/test_oai_image_caption.py`, `tools/client/test_oai_video_caption.py`                         | Edit constants, then `python tools/client/test_oai_image_caption.py`            |
| Run reusable LMDeploy pipeline cases                       | `tools/pipeline/pipeline.py`, `tools/pipeline/pipe_cases.py`                                               | `python tools/pipeline/pipeline.py --model qwen3-vl-4b --cuda 7 --tp 1 2`       |
| Run simple TP/EP pipeline examples                         | `tools/pipeline/test_pipeline_tp.py`, `tools/pipeline/test_pipeline_ep.py`                                 | `python tools/pipeline/test_pipeline_tp.py`                                     |
| Compare or save/load tensors                               | `tools/diagnostics/comp_tensor_utils.py`                                                                   | Import `compare_tensors`, `save_tensor`, or `load_tensor`                       |
| Detect NaN/Inf in tensors                                  | `tools/diagnostics/detect_inf_nan.py`                                                                      | Import `contains_inf_or_nan`                                                    |
| Filter Ray log prefixes                                    | `tools/diagnostics/ray_log_filter.py`                                                                      | Import before noisy Ray/LMDeploy logs                                           |
| Run a small NCCL all-reduce check                          | `tools/diagnostics/test_internode_nccl.py`                                                                 | Edit rank/master values, then `python tools/diagnostics/test_internode_nccl.py` |
| Run LMDeploy with profiler env vars                        | `tools/diagnostics/profile.py`                                                                             | Edit output path, then `python tools/diagnostics/profile.py`                    |
| Download Hugging Face models or metadata                   | `tools/download/hf_download.sh`, `tools/download/hf_partial_download.py`                                   | `bash tools/download/hf_download.sh`                                            |
| Prepare offline VS Code server files                       | `tools/download/offline_download_vscode.sh`                                                                | `bash tools/download/offline_download_vscode.sh`                                |
| Build or start Docker containers                           | `tools/docker/build_docker_image.sh`, `tools/docker/start_docker.sh`                                       | Edit image/volume values, then run with `bash`                                  |
| Create a model directory with one copied file and symlinks | `tools/model/hack_model_file.sh`                                                                           | Set env vars, then `bash tools/model/hack_model_file.sh`                        |

## Pre-commit

```bash
pip install -U pre-commit
pre-commit install
pre-commit run --all-files
```
