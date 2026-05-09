import os

import torch


def compare_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor, name: str = 'Tensors'):
    """Compares two PyTorch tensors and prints the statistics of their absolute
    difference.

    Args:
        tensor1 (torch.Tensor): The first tensor for comparison.
        tensor2 (torch.Tensor): The second tensor for comparison.
        name (str): An optional name for the tensors being compared, for logging purposes.
    """
    if tensor1.shape != tensor2.shape:
        print(f"Error: Shape mismatch for '{name}'. {tensor1.shape} vs {tensor2.shape}")
        return

    # Ensure both tensors are on the same device and type
    if tensor1.device != tensor2.device:
        tensor1 = tensor1.cpu()
        tensor2 = tensor2.cpu()
    assert tensor1.device == tensor2.device, f"Device mismatch for '{name}': {tensor1.device} vs {tensor2.device}"
    assert tensor1.dtype == tensor2.dtype, f"Data type mismatch for '{name}': {tensor1.dtype} vs {tensor2.dtype}"

    # Calculate the absolute difference
    abs_diff = torch.abs(tensor1.float() - tensor2.float())

    # Calculate statistics
    mean_diff = abs_diff.mean().item()
    max_diff = abs_diff.max().item()
    min_diff = abs_diff.min().item()

    print(f'--- Comparison for: {name} ---')
    print(f'  - Mean Absolute Diff: {mean_diff:.8f}')
    print(f'  - Max Absolute Diff:  {max_diff:.8f}')
    print(f'  - Min Absolute Diff:  {min_diff:.8f}')
    print('-' * (25 + len(name)))


def load_tensor(name: str, load_path: str = '/nvme1/zhouxinyu/0_test'):
    """Loads a PyTorch tensor from a specified directory.

    Args:
        name (str): The base name for the tensor file (e.g., 'q_embed').
        load_path (str): The directory path where the tensor is saved.

    Returns:
        torch.Tensor: The loaded tensor.
    """
    filename = f'gt_{name}.pt'
    full_filepath = os.path.join(load_path, filename)

    if not os.path.exists(full_filepath):
        raise FileNotFoundError(f"Tensor file '{full_filepath}' does not exist.")

    tensor = torch.load(full_filepath)
    print(f"Loaded tensor '{name}' from: {full_filepath}")
    return tensor


def save_tensor(tensor: torch.Tensor, name: str, save_path: str = '/nvme1/zhouxinyu/0_test'):
    """Saves a PyTorch tensor to a specified directory with a 'gt_' prefix.

    The filename will be 'gt_<name>.pt'. The function ensures the
    save directory exists before saving.

    Args:
        tensor (torch.Tensor): The tensor to be saved.
        name (str): The base name for the tensor file (e.g., 'q_embed').
        save_path (str): The directory path where the tensor will be saved.
    """
    # Ensure the save directory exists, create it if it doesn't.
    os.makedirs(save_path, exist_ok=True)

    # Construct the full file path with the 'gt_' prefix and .pt extension.
    filename = f'gt_{name}.pt'
    full_filepath = os.path.join(save_path, filename)

    # Save the tensor.
    torch.save(tensor, full_filepath)

    print(f"Saved ground truth tensor '{name}' to: {full_filepath}")
