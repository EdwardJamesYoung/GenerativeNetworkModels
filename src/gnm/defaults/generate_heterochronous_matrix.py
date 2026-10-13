import torch
import math
from typing import Sequence, Union

# Function to generate heterochronous matrices
def generate_heterochronous_matrix(
    coord: torch.Tensor,
    reference_coord: Union[Sequence[float], torch.Tensor] = (0, 0, 0),
    sigma: float = 1.0,
    num_nodes: int = 100,
    mseed: int = 0,
    cumulative: bool = False,
    local: bool = True,
    tau: float = 1.0,
) -> torch.Tensor:
    """
    Generate heterochronous matrices based on a dynamic starting node.

    Parameters
    ----------
    coord : torch.Tensor, shape (N, 3)
        Node coordinates.
    reference_coord : 3-vector, default (0,0,0)
        Starting point for distance computation.
    sigma : float
        Standard deviation for the Gaussian profile.
    num_nodes : int
        Number of time steps / nodes (T + mseed).
    mseed : int
        Number of seed nodes excluded from Gaussian schedule.
    cumulative : bool
        If True, apply cumulative maximum over time.
    local : bool
        If True, use outer-product rule; otherwise use global max rule.
    tau : float, default 1.0
        Temporal rate of the unfolding wave:
        - tau < 1: the wave reaches the farthest node early, then stays fully unfolded.
        - tau = 1: linear sweep from the reference point to the farthest node over T steps.
        - tau > 1: truncated; the wave does not reach the farthest node within T steps.

    Returns
    -------
    torch.Tensor, shape (T, N, N)
        Stack of heterochronous adjacency matrices (float32).
    """
    # Ensure torch tensors & float dtype
    coord = coord.float()
    ref   = torch.as_tensor(reference_coord, dtype=coord.dtype,
                            device=coord.device)

    # Compute Euclidean distances from reference node to all nodes
    distances   = torch.linalg.norm(coord - ref, dim=1)            # (N,)
    max_distance= distances.max()

    # Means for Gaussian at each time step (tau = 1 -> linspace(0, max_distance, T))
    if tau <= 0:
        raise ValueError("tau must be > 0")
    T        = num_nodes - mseed
    if T > 1:
        t_idx    = torch.arange(T, dtype=coord.dtype, device=coord.device)
        progress = (t_idx / ((T - 1) * tau)).clamp(0.0, 1.0)     # (T,)
    else:
        progress = torch.zeros(T, dtype=coord.dtype, device=coord.device)
    means    = progress * max_distance                            # (T,)

    # Allocate probability matrix (N, T)
    heterochronous_matrix = torch.empty(coord.size(0), T,
                                        dtype=coord.dtype,
                                        device=coord.device)

    # Gaussian probability function
    coeff = 1.0 / (math.sqrt(2*math.pi) * sigma)
    for t in range(T):
        mu = means[t]
        heterochronous_matrix[:, t] = coeff * torch.exp(
            -((distances - mu) ** 2) / (2 * sigma**2)
        )

    # Apply cumulative maximum if requested
    if cumulative:
        heterochronous_matrix = torch.cummax(
            heterochronous_matrix, dim=1).values                  # (N, T)

    # Rescale the whole (N, T) matrix once with a global min/max, so the
    # relative amplitude across time steps (the unfolding wave) is preserved
    global_min = heterochronous_matrix.min()
    global_max = heterochronous_matrix.max()
    denom = torch.clamp(global_max - global_min, min=1e-12)      # avoid divide-by-zero
    heterochronous_matrix = (heterochronous_matrix - global_min) / denom

    # Convert to matrix form based on `local`
    heterochronous_matrices = []
    for t in range(T):
        H_rescaled = heterochronous_matrix[:, t]                  # (N,)
        if local:
            Hmat = torch.outer(H_rescaled, H_rescaled)            # (N, N)
        else:
            N = H_rescaled.size(0)
            rows = H_rescaled.unsqueeze(1).expand(N, N)
            cols = H_rescaled.unsqueeze(0).expand(N, N)
            Hmat = torch.maximum(rows, cols)                      # (N, N)
        heterochronous_matrices.append(Hmat)

    heterochronous_matrices_tensor = torch.stack(
        heterochronous_matrices, dim=-1).float()                  # (N, N, T)

    heterochronous_matrices_tensor = heterochronous_matrices_tensor.permute(2, 0, 1).contiguous()  # (T, N, N)


    return heterochronous_matrices_tensor