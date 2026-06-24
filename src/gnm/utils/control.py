r"""Control networks for network evaluation and comparison."""

import random
import warnings
import torch
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
from .checks import binary_checks
import numpy as np
import networkx as nx
from tqdm import tqdm
from typing import Optional


# Track the last seed set via set_random_seed so we can warn if a
# conflicting explicit seed is passed to any downstream function.
_last_seed_from_set: Optional[int] = None


def _check_seed_consistency(seed: int) -> None:
    """If a seed was set via set_random_seed and *seed* disagrees, warn."""
    if _last_seed_from_set is not None and seed != _last_seed_from_set:
        warnings.warn(
            f"Seed {seed} differs from the seed previously set via "
            f"set_random_seed ({_last_seed_from_set}). "
            f"This may produce unexpected non-deterministic behaviour.",
            stacklevel=3,
        )


@jaxtyped(typechecker=typechecked)
def get_control(
    matrices: Float[torch.Tensor, "*batch num_nodes num_nodes"],
    seed: Optional[int] = None,
) -> Float[torch.Tensor, "*batch num_nodes num_nodes"]:
    """Generate control networks by randomly permuting connections while preserving network properties.

    This function creates randomized versions of the input networks while maintaining:
    - The same number of connections (for binary networks) or weight distribution (for weighted networks)
    - Symmetry (undirected graph structure)
    - No self-connections (zeros on diagonal)

    Args:
        matrices:
            Input adjacency or weight matrices with shape [*batch, num_nodes, num_nodes]

    Returns:
        Permuted control networks with the same shape as input matrices, preserving key properties

    Examples:
        >>> import torch
        >>> from gnm.utils import get_control
        >>> from gnm.defaults import get_binary_network
        >>> # Get a real network
        >>> real_network = get_binary_network()
        >>> # Generate a control with preserved properties
        >>> control_network = get_control(real_network)
        >>> # Check that control has same number of connections
        >>> real_network.sum() == control_network.sum()
        tensor(True)

    Notes:
        - For binary networks, this is equivalent to randomly rewiring all connections
        - For weighted networks, connection weights are preserved but redistributed
    """
    if seed is not None:
        _check_seed_consistency(seed)
        torch.manual_seed(seed)

    *batch_shape, num_nodes, _ = matrices.shape
    batch_size = torch.prod(torch.tensor(batch_shape)).item()

    # Flatten batch dimensions for processing
    matrices_2d = matrices.view(batch_size, num_nodes, num_nodes)

    # Get upper triangular indices (excluding diagonal)
    indices = torch.triu_indices(num_nodes, num_nodes, offset=1, device=matrices.device)
    num_edges = indices.shape[1]

    # Extract all upper triangular values for all matrices at once
    upper_values = matrices_2d[:, indices[0], indices[1]]  # [batch_size, num_edges]

    # Generate different permutations for each matrix in the batch
    rand_vals = torch.rand(batch_size, num_edges, device=matrices.device)
    perm_indices = torch.argsort(rand_vals, dim=1)  # [batch_size, num_edges]

    # Apply permutations using advanced indexing
    batch_indices = torch.arange(batch_size, device=matrices.device).unsqueeze(
        1
    )  # [batch_size, 1]
    permuted_values = upper_values[
        batch_indices, perm_indices
    ]  # [batch_size, num_edges]

    # Create control networks and set permuted values
    control_networks = torch.zeros_like(matrices_2d)
    control_networks[:, indices[0], indices[1]] = permuted_values

    # Ensure symmetry by adding transpose
    control_networks = control_networks + control_networks.transpose(-2, -1)

    # Reshape back to original batch shape
    return control_networks.view(*batch_shape, num_nodes, num_nodes)


@jaxtyped(typechecker=typechecked)
def generate_random_binary_networks(
    num_nodes: int,
    density: float,
    num_networks: int = 1,
    seed: Optional[int] = None,
) -> Float[torch.Tensor, "num_networks num_nodes num_nodes"]:
    r"""Create a batch of random graphs with a specified number of nodes and density.

    Args:
        num_nodes:
            Number of nodes in the graph.
        density:
            Density of the graph, i.e., fraction of possible edges present in the graph. Must be between 0 and 1.
        num_networks:
            Number of graphs to create.
        seed:
            Random seed.

    Returns:
        Tensor:
            Adjacency matrices of shape (num_networks, num_nodes, num_nodes)
    """
    if seed is not None:
        _check_seed_consistency(seed)
        torch.manual_seed(seed)

    assert 0 <= density <= 1, "Density must be between 0 and 1"

    # Calculate number of possible edges in upper triangle (excluding diagonal)
    num_possible_edges = num_nodes * (num_nodes - 1) // 2

    # Calculate exact number of edges for specified density
    num_edges = int(density * num_possible_edges)

    # Get upper triangular indices
    triu_indices = torch.triu_indices(num_nodes, num_nodes, offset=1)

    # Initialize graphs
    graphs = torch.zeros(num_networks, num_nodes, num_nodes, dtype=torch.float)

    if num_edges > 0:  # Only proceed if we need to place edges
        # Generate different permutations for each network
        rand_vals = torch.rand(num_networks, num_possible_edges)
        perm_indices = torch.argsort(
            rand_vals, dim=1
        )  # [num_networks, num_possible_edges]

        # Select first num_edges positions for each network
        selected_indices = perm_indices[:, :num_edges]  # [num_networks, num_edges]

        # Convert selected edge indices to row/col coordinates
        selected_rows = triu_indices[0, selected_indices]  # [num_networks, num_edges]
        selected_cols = triu_indices[1, selected_indices]  # [num_networks, num_edges]

        # Set selected edges to 1 using advanced indexing
        batch_indices = torch.arange(num_networks).unsqueeze(1)  # [num_networks, 1]
        graphs[batch_indices, selected_rows, selected_cols] = 1.0

    # Make symmetric
    graphs = graphs + graphs.transpose(1, 2)

    binary_checks(graphs)

    return graphs


def set_random_seed(seed: int) -> None:
    r"""Set the random seed for Python, NumPy, and PyTorch (CPU and CUDA).

    Use this at the top of scripts or experiments to ensure deterministic
    behaviour across random number generators.

    Args:
        seed:
            Integer seed to set for all RNGs.

    Examples:
        >>> from gnm.utils import set_random_seed
        >>> set_random_seed(42)
        >>> # All subsequent random operations are now deterministic.

    Note:
        - This does **not** enable ``torch.backends.cudnn.deterministic``.
          For full determinism, set that flag separately.
        - In Jupyter notebooks, ``np.random.seed`` only affects the current
          cell's global state and does not propagate across cells.
    """
    global _last_seed_from_set
    _last_seed_from_set = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
