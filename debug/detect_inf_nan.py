def contains_inf_or_nan(tensor):
    """Returns True if the input tensor contains any INF or NaN values, False
    otherwise.

    Args:
        tensor (torch.Tensor): Input tensor to check.

    Returns:
        bool: True if tensor contains INF or NaN, False otherwise.
    """
    import torch
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    if has_nan:
        print('=> detect nan')
    if has_inf:
        print('=> detect inf')

    return has_nan or has_inf
