import numpy as np
import torch


def np_to_tensor(np_adj_matrix, device=None) -> torch.Tensor:
    """Convert an array-like adjacency matrix to a float32 PyTorch tensor.

    NumPy arrays, numeric Python sequences, and PyTorch tensors are accepted.
    A single ``[node, node]`` matrix gains a batch dimension; an existing
    ``[batch, node, node]`` array keeps its shape.

    Args:
        np_adj_matrix: A 2D or 3D array-like adjacency matrix.
        device: Destination device (e.g. ``"cpu"``, ``"mps"``, or ``"cuda:0"``).
            Defaults to CUDA if available, otherwise CPU.

    Returns:
        A float32 tensor with shape ``[batch, node, node]`` on ``device``.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if isinstance(np_adj_matrix, torch.Tensor):
        tensor = np_adj_matrix
    else:
        # NumPy handles sequences and other array-like objects, including
        # arrays with negative strides that PyTorch cannot consume directly.
        array = np.ascontiguousarray(np_adj_matrix, dtype=np.float32)
        tensor = torch.from_numpy(array)

    if tensor.ndim not in (2, 3):
        raise ValueError("Expected a 2D or 3D adjacency matrix")

    tensor = tensor.to(device=device, dtype=torch.float32)
    return tensor.unsqueeze(0) if tensor.ndim == 2 else tensor
