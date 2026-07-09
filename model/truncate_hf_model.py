# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from safetensors.torch import safe_open, save_file
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

DEFAULT_KEEP_NUM_LAYERS = 4
_LAYER_NAME_RE = re.compile(r'^model\.layers\.(\d+)(?:\.|$)')
_LAYER_PATTERN_STRING_KEYS = {'index_topk_pattern'}
_SKIP_COPY_DIRS = {'.cache', '.git'}
_WEIGHT_INDEX_NAMES = {SAFE_WEIGHTS_INDEX_NAME, 'pytorch_model.bin.index.json'}
_WEIGHT_SUFFIXES = ('.bin', '.safetensors', '.pt', '.pth', '.ckpt')
_CONFIG_NAME = 'config.json'


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Create a small HF checkpoint by keeping the first N decoder layers.')
    parser.add_argument('--input', required=True, help='local HF model directory')
    parser.add_argument('--output', required=True, help='directory to save the truncated model')
    parser.add_argument('--keep-num-layers',
                        type=int,
                        default=DEFAULT_KEEP_NUM_LAYERS,
                        help='number of model.layers.* blocks to keep')
    parser.add_argument('--overwrite', action='store_true', help='overwrite output directory if it already has files')
    return parser.parse_args()


def truncate_hf_model(input_path: str,
                      output_path: str,
                      keep_num_layers: int = DEFAULT_KEEP_NUM_LAYERS,
                      overwrite: bool = False):
    """Create a truncated Hugging Face checkpoint.

    This is intended for debug/accuracy smoke tests where a very large MoE
    checkpoint cannot be loaded in full. It keeps the first N ``model.layers``
    blocks and all non-layer tensors such as embeddings, final norm and lm_head.
    """
    if keep_num_layers <= 0:
        raise ValueError('keep_num_layers must be greater than 0.')

    input_dir = Path(input_path)
    if not input_dir.is_dir():
        raise FileNotFoundError(f'{input_dir} is not a local model directory.')

    output_dir = Path(output_path)
    _prepare_output_dir(input_dir=input_dir, output_dir=output_dir, overwrite=overwrite)

    config = _load_json(input_dir / _CONFIG_NAME)
    original_num_layers = _truncate_config(config, keep_num_layers=keep_num_layers)
    _dump_json(output_dir / _CONFIG_NAME, config)

    _copy_non_weight_files(input_dir=input_dir, output_dir=output_dir)

    if (input_dir / SAFE_WEIGHTS_INDEX_NAME).is_file():
        _truncate_sharded_safetensors(
            input_dir=input_dir,
            output_dir=output_dir,
            keep_num_layers=keep_num_layers,
        )
    elif (input_dir / SAFE_WEIGHTS_NAME).is_file():
        _truncate_single_safetensors(
            input_dir=input_dir,
            output_dir=output_dir,
            keep_num_layers=keep_num_layers,
        )
    else:
        raise RuntimeError('Only safetensors checkpoints are supported by truncate_hf_model.')

    print(f'Truncated {input_dir} from {original_num_layers} to {keep_num_layers} layers at {output_dir}.')


def _prepare_output_dir(input_dir: Path, output_dir: Path, overwrite: bool):
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir == input_dir:
        raise ValueError('output directory must be different from input directory.')
    if output_dir.is_relative_to(input_dir):
        raise ValueError('output directory must not be inside input directory.')
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f'{output_dir} is not empty. Pass --overwrite to replace it.')
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _truncate_config(config: dict[str, Any], keep_num_layers: int) -> int:
    original_num_layers = config.get('num_hidden_layers')
    if original_num_layers is None:
        raise KeyError('config.json does not contain num_hidden_layers.')
    if keep_num_layers > original_num_layers:
        raise ValueError(f'keep_num_layers={keep_num_layers} exceeds num_hidden_layers={original_num_layers}.')

    config['num_hidden_layers'] = keep_num_layers
    for key, value in list(config.items()):
        if isinstance(value, list) and len(value) == original_num_layers:
            config[key] = value[:keep_num_layers]
        elif key in _LAYER_PATTERN_STRING_KEYS and isinstance(value, str) and len(value) == original_num_layers:
            config[key] = value[:keep_num_layers]
    _filter_config_layer_refs(config, keep_num_layers=keep_num_layers)

    return original_num_layers


def _filter_config_layer_refs(value: Any, keep_num_layers: int):
    if isinstance(value, dict):
        for key, item in list(value.items()):
            value[key] = _filter_config_layer_refs(item, keep_num_layers)
    elif isinstance(value, list):
        value = [
            _filter_config_layer_refs(item, keep_num_layers) for item in value
            if not isinstance(item, str) or _keep_tensor_name(item, keep_num_layers)
        ]
    return value


def _truncate_sharded_safetensors(input_dir: Path, output_dir: Path, keep_num_layers: int):
    index = _load_json(input_dir / SAFE_WEIGHTS_INDEX_NAME)
    weight_map = index['weight_map']
    new_weight_map = {name: shard for name, shard in weight_map.items() if _keep_tensor_name(name, keep_num_layers)}
    index['weight_map'] = new_weight_map

    names_by_shard = defaultdict(list)
    for name, shard in new_weight_map.items():
        names_by_shard[shard].append(name)

    total_size = 0
    seen_names = set()
    for shard_name in sorted(set(weight_map.values())):
        input_file = input_dir / shard_name
        output_file = output_dir / shard_name
        keep_names = set(names_by_shard.get(shard_name, ()))
        if not keep_names:
            continue

        state_dict, metadata = _read_safetensors_subset(input_file, lambda name: name in keep_names)
        seen_names.update(state_dict)
        total_size += _state_dict_nbytes(state_dict)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        save_file(state_dict, output_file, metadata=metadata)
        print(f'Save {len(state_dict)} tensors to {output_file}')

    missing_names = set(new_weight_map) - seen_names
    if missing_names:
        raise RuntimeError(f'Missing tensors in source checkpoint: {sorted(missing_names)}')

    if isinstance(index.get('metadata'), dict) and 'total_size' in index['metadata']:
        index['metadata']['total_size'] = total_size
    _dump_json(output_dir / SAFE_WEIGHTS_INDEX_NAME, index)


def _truncate_single_safetensors(input_dir: Path, output_dir: Path, keep_num_layers: int):
    input_file = input_dir / SAFE_WEIGHTS_NAME
    output_file = output_dir / SAFE_WEIGHTS_NAME
    state_dict, metadata = _read_safetensors_subset(input_file, lambda name: _keep_tensor_name(name, keep_num_layers))
    save_file(state_dict, output_file, metadata=metadata)
    print(f'Save {len(state_dict)} tensors to {output_file}')


def _read_safetensors_subset(input_file: Path, keep) -> tuple[dict[str, Any], dict[str, str] | None]:
    state_dict = {}
    with safe_open(input_file, framework='pt', device='cpu') as f:
        metadata = f.metadata()
        for name in f.keys():
            if keep(name):
                state_dict[name] = f.get_tensor(name)
    return state_dict, metadata


def _keep_tensor_name(tensor_name: str, keep_num_layers: int) -> bool:
    match = _LAYER_NAME_RE.match(tensor_name)
    if match is None:
        return True
    return int(match.group(1)) < keep_num_layers


def _state_dict_nbytes(state_dict: dict[str, Any]) -> int:
    total_size = 0
    for tensor in state_dict.values():
        total_size += tensor.numel() * tensor.element_size()
    return total_size


def _copy_non_weight_files(input_dir: Path, output_dir: Path):
    for source in input_dir.iterdir():
        if source.name == _CONFIG_NAME or _is_weight_file(source):
            continue
        if source.is_dir() and source.name in _SKIP_COPY_DIRS:
            continue

        target = output_dir / source.name
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)


def _is_weight_file(path: Path) -> bool:
    if path.name in _WEIGHT_INDEX_NAMES:
        return True
    return path.suffix in _WEIGHT_SUFFIXES


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _dump_json(path: Path, data: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
        f.write('\n')


def main():
    args = parse_args()
    truncate_hf_model(
        input_path=args.input,
        output_path=args.output,
        keep_num_layers=args.keep_num_layers,
        overwrite=args.overwrite,
    )


if __name__ == '__main__':
    main()
