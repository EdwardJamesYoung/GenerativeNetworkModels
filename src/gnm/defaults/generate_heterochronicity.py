import numpy as np
import torch

def generate_heterochronicity(
    coordinates,
    origin_point=[0,0,0],
    sigma=1.0,
    num_connections=100,
    mseed=0,
    cumulative=True,
    local=False
):
    """
    Generate heterochronous matrices based on a dynamic starting node.

    Parameters:
    - coord (array): Coordinates of nodes.
    - starting_node_index (int): Index of the node to use as the starting point.
    - sigma (float): Standard deviation for the Gaussian.
    - num_connections (int): Number of time steps/edges.
    - mseed (int): Number of seed nodes to exclude from computation.
    - cumulative (bool): Whether to apply cumulative maximum.
    - local (bool): Whether to generate connections locally or globally.

    Returns:
    - torch.tensor: Heterochronous matrices tensor.
    """

    # Compute euclidean distances from coordinates
    distances = np.sqrt(np.sum((coordinates - origin_point) ** 2, axis=1))
    max_distance = np.max(distances)
    
    # initialise the mean of gaussian heterochronous gradient and heterocrhonicity matrix
    means = np.linspace(0, max_distance, num_connections - mseed)
    heterochronous_matrix = np.zeros((len(distances), num_connections - mseed))
    
    # define the gaussian
    P = lambda d, mu: (1.0 / (np.sqrt(2*np.pi)*sigma))*np.exp(-((d-mu)**2)/(2*sigma**2))
    
    # fill matrix across time
    for t in range(num_connections - mseed):
        mu = means[t]
        heterochronous_matrix[:, t] = P(distances, mu)
    
    # Cumulative max if requested
    if cumulative:
        heterochronous_matrix = np.maximum.accumulate(heterochronous_matrix, axis=1)
    
    # Compute global min & max across all time slices
    global_min = heterochronous_matrix.min()
    global_max = heterochronous_matrix.max()
    
    # Avoid divide-by-zero if global_min == global_max
    eps = 1e-12  # small number
    denom = max(global_max - global_min, eps)
    
    # Rescale the entire 2D matrix [N x T] once
    heterochronous_matrix_rescaled = (heterochronous_matrix - global_min) / denom
    
    # Build adjacency matrices slice by slice
    heterochronous_matrices = []
    for t in range(num_connections - mseed):
        Ht = heterochronous_matrix_rescaled[:, t]
        if local:
            # Local = outer product
            Hmat = np.outer(Ht, Ht)
        else:
            # Global = max of node i and j
            N = len(Ht)
            rows, cols = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
            Hmat = np.maximum(Ht[rows], Ht[cols])
        heterochronous_matrices.append(Hmat)
    
    heterochronous_matrices_tensor = torch.tensor(
    np.stack(heterochronous_matrices, axis=0), dtype=torch.float32)

    return heterochronous_matrices_tensor
