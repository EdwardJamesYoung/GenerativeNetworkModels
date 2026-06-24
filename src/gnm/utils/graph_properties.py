r"""Graph theory metrics for analyzing network properties.

This module provides various metrics from graph theory for characterising network
structures in both binary and weighted networks. These metrics include node strengths,
clustering coefficients, communicability, and betweenness centrality.
"""

from jaxtyping import Float, jaxtyped
from typeguard import typechecked
from typing import Optional, Union
import torch
import networkx as nx
import numpy as np
from warnings import warn
from tqdm import tqdm

from .checks import binary_checks, weighted_checks
from .control import get_control, _check_seed_consistency


@jaxtyped(typechecker=typechecked)
def node_strengths(
    adjacency_matrix: Float[torch.Tensor, "*batch num_nodes num_nodes"],
) -> Float[torch.Tensor, "*batch num_nodes"]:
    r"""Compute the node strengths (or nodal degree) for each node in the network.

    For binary networks, this is equivalent to the node degree (number of connections).
    For weighted networks, this represents the sum of all edge weights connected to each node.

    Args:
        adjacency_matrix:
            Adjacency matrix (binary or weighted) with shape [*batch, num_nodes, num_nodes]

    Returns:
        Vector of node strengths for each node in the network with shape [*batch, num_nodes]

    Examples:
        >>> import torch
        >>> from gnm.utils import node_strengths
        >>> # Create a sample binary network
        >>> adj_matrix = torch.zeros(1, 4, 4)
        >>> adj_matrix[0, 0, 1] = 1
        >>> adj_matrix[0, 1, 0] = 1
        >>> adj_matrix[0, 1, 2] = 1
        >>> adj_matrix[0, 2, 1] = 1
        >>> strength = node_strengths(adj_matrix)
        >>> strength
        tensor([[1., 2., 1., 0.]])

    See Also:
        - [`evaluation.DegreeKS`][gnm.evaluation.DegreeKS]: Binary evaluation criterion which compares the distribution of node degrees between two binary networks.
        - [`evaluation.WeightedNodeStrengthKS`][gnm.evaluation.WeightedNodeStrengthKS]: Weighted evaluation criterion which compares the distribution of node strengths between two weighted networks.
        - [`evaluation.DegreeCorrelation`][gnm.evaluation.DegreeCorrelation]: Binary evaluation criterion which compares the correlations between the node degrees between two binary networks.
    """
    return adjacency_matrix.sum(dim=-1)


@jaxtyped(typechecker=typechecked)
def binary_clustering_coefficients(
    adjacency_matrix: Float[torch.Tensor, "*batch num_nodes num_nodes"],
) -> Float[torch.Tensor, "*batch num_nodes"]:
    r"""Compute the clustering coefficients for each node in a binary network.

    The clustering coefficient measures the degree to which nodes in a graph tend to cluster together.
    For a node i, it quantifies how close its neighbors are to being a complete subgraph (clique).

    The clustering coefficient for a node $i$ is computed as
    $$
        c_i = \\frac{ 2t_i }{ s_i (s_i - 1) },
    $$
    where $t_i$ is the number of (unordered) triangles around node $i$, and $s_i$ is the degree of node $i$.
    The number of triangles can be computed via
    $$
        t_i = \\frac{1}{2} (A^3)_{ii},
    $$
    *i.e.*, the number of paths of length 3 that start and end at node $i$, divided by 2 to account for the fact that $i \to j \to k \to i$ and $i \to k \to j \to i$ are the same triangle.

    Args:
        adjacency_matrix:
            Binary adjacency matrix with shape [*batch, num_nodes, num_nodes]

    Returns:
        The clustering coefficients for each node with shape [*batch, num_nodes]

    Examples:
        >>> import torch
        >>> from gnm.utils import binary_clustering_coefficients
        >>> # Create a binary network with a triangle
        >>> adj_matrix = torch.zeros(1, 4, 4)
        >>> adj_matrix[0, 0, 1] = 1
        >>> adj_matrix[0, 1, 0] = 1
        >>> adj_matrix[0, 1, 2] = 1
        >>> adj_matrix[0, 2, 1] = 1
        >>> adj_matrix[0, 0, 2] = 1
        >>> adj_matrix[0, 2, 0] = 1
        >>> clustering = binary_clustering_coefficients(adj_matrix)
        >>> clustering
        tensor([[1., 1., 1., 0.]])

    See Also:
        - [`utils.weighted_clustering_coefficients`][gnm.utils.weighted_clustering_coefficients]: For calculating clustering coefficient in weighted networks.
        - [`evaluation.ClusteringKS`][gnm.evaluation.ClusteringKS]: Binary evaluation criterion which compares the distribution of clustering coefficients between two binary networks.
        - [`evaluation.ClusteringCorrelation`][gnm.evaluation.ClusteringCorrelation]: Binary evaluation criterion which compares the correlations between the clustering coefficients between two binary networks.
    """
    binary_checks(adjacency_matrix)

    degrees = adjacency_matrix.sum(dim=-1)
    number_of_pairs = degrees * (degrees - 1)

    number_of_triangles = torch.diagonal(
        torch.matmul(
            torch.matmul(adjacency_matrix, adjacency_matrix), adjacency_matrix
        ),
        dim1=-2,
        dim2=-1,
    )

    clustering = torch.zeros_like(number_of_triangles)
    mask = number_of_pairs > 0

    clustering[mask] = number_of_triangles[mask] / number_of_pairs[mask]
    return clustering


@jaxtyped(typechecker=typechecked)
def weighted_clustering_coefficients(
    weight_matrices: Float[torch.Tensor, "*batch num_nodes num_nodes"],
) -> Float[torch.Tensor, "*batch num_nodes"]:
    r"""Compute weighted clustering coefficients based on Onnela et al. (2005) definition.

    This implementation uses the geometric mean of triangle weights. For each node $i$,
    the clustering coefficient is:

    $$
    c_i = \frac{1}{s_i (s_i - 1)} \sum_{jk} (\hat{w}_{ij} \times \hat{w}_{jk} \times \hat{w}_{ki})^{1/3},
    $$

    where $s_i$ is the degree of node $i$, and $\hat{w}_{ij}$ is the weight of the edge between nodes $i$ and $j$,
    *after* normalising by dividing by the maximum weight in the network.

    Args:
        weight_matrices:
            Batch of weighted adjacency matrices with shape [*batch, num_nodes, num_nodes].
            Weights should be non-negative.

    Returns:
        Clustering coefficients for each node in each network with shape [*batch, num_nodes]

    Examples:
        >>> import torch
        >>> from gnm.utils import weighted_clustering_coefficients
        >>> # Create a weighted network with a triangle
        >>> weight_matrix = torch.zeros(1, 4, 4)
        >>> weight_matrix[0, 0, 1] = 0.5
        >>> weight_matrix[0, 1, 0] = 0.5
        >>> weight_matrix[0, 1, 2] = 0.8
        >>> weight_matrix[0, 2, 1] = 0.8
        >>> weight_matrix[0, 0, 2] = 0.6
        >>> weight_matrix[0, 2, 0] = 0.6
        >>> clustering = weighted_clustering_coefficients(weight_matrix)
        >>> clustering.shape
        torch.Size([1, 4])

    See Also:
        - [`utils.binary_clustering_coefficients`][gnm.utils.binary_clustering_coefficients]: For calculating clustering in binary networks.
        - [`evaluation.WeightedClusteringKS`][gnm.evaluation.WeightedClusteringKS]: Weighted evaluation criterion which compares the distribution of (weighted) clustering coefficients between two weighted networks.
    """
    weighted_checks(weight_matrices)

    # Normalise the weights by dividing by the maximum weight in the network
    max_weight = weight_matrices.amax(dim=(-2, -1), keepdim=True)  # [*batch, 1, 1]
    normalised_w = weight_matrices / max_weight

    # Take the cube root
    cube_root_w = torch.pow(normalised_w, 1 / 3)

    # For each node u, compute the geometric mean of triangle weights:
    # (w_uv * w_vw * w_wu) ^ (1/3)
    triangles = torch.diagonal(
        torch.matmul(torch.matmul(cube_root_w, cube_root_w), cube_root_w),
        dim1=-2,
        dim2=-1,
    )  # [*batch, num_nodes]

    # Get node degree (number of connections)
    degree = torch.sum(weight_matrices > 0, dim=-1)  # [*batch, num_nodes]

    # Compute denominator s_i * (s_i - 1)
    denom = degree * (degree - 1)  # [*batch, num_nodes]

    # Handle division by zero - set clustering to 0 where s_i <= 1
    clustering = torch.zeros_like(triangles)
    mask = denom > 0
    clustering[mask] = triangles[mask] / denom[mask]

    return clustering


@jaxtyped(typechecker=typechecked)
def communicability(
    weight_matrix: Float[torch.Tensor, "*batch num_nodes num_nodes"],
) -> Float[torch.Tensor, "*batch num_nodes num_nodes"]:
    r"""Compute the communicability matrix for a network.

    Communicability measures the ease of information flow between nodes, taking into
    account all possible paths between them.

    To compute the communicability matrix, we go through the following steps:

    1. Compute the diagonal node strength matrix, $S_{ii} = \sum_j W_{ij}$ (plus a small constant to prevent division by zero).
    2. Compute the normalised weight matrix, $S^{-1/2} W S^{-1/2}$.
    3. Compute the communicability matrix by taking the matrix exponential, $\exp( S^{-1/2} W S^{-1/2} )$.

    Args:
        weight_matrix:
            Weighted adjacency matrix with shape [*batch, num_nodes, num_nodes]

    Returns:
        Communicability matrix with shape [*batch, num_nodes, num_nodes]

    Examples:
        >>> import torch
        >>> from gnm.utils import communicability
        >>> # Create a simple weighted network
        >>> weight_matrix = torch.zeros(1, 3, 3)
        >>> weight_matrix[0, 0, 1] = 0.5
        >>> weight_matrix[0, 1, 0] = 0.5
        >>> weight_matrix[0, 1, 2] = 0.8
        >>> weight_matrix[0, 2, 1] = 0.8
        >>> comm_matrix = communicability(weight_matrix)
        >>> comm_matrix.shape
        torch.Size([1, 3, 3])

    See Also:
        - [`weight_criteria.Communicability`][gnm.weight_criteria.Communicability]: weight optimisation criterion which minimises total communicability.
        - [`weight_criteria.NormalisedCommunicability`][gnm.weight_criteria.NormalisedCommunicability]: weight optimisation criterion which minimises total communicability, divided by the maximum communicability.
        - [`weight_criteria.DistanceWeightedCommunicability`][gnm.weight_criteria.DistanceWeightedCommunicability]: weight optimisation criterion which minimises total communicability, weighted by the distance between nodes.
        - [`weight_criteria.NormalisedDistanceWeightedCommunicability`][gnm.weight_criteria.NormalisedDistanceWeightedCommunicability]: weight optimisation criterion which minimises total communicability, weighted by the distance between nodes and divided by the maximum distance-weighted communicability.
    """
    # Compute the node strengths, with a small constant addition to prevent division by zero.
    node_strengths = (
        0.5 * (weight_matrix.sum(dim=-1) + weight_matrix.sum(dim=-2)) + 1e-6
    )

    # Create diagonal matrix for each batch element
    batch_shape = weight_matrix.shape[:-2]
    num_nodes = weight_matrix.shape[-1]
    inv_sqrt_node_strengths = torch.zeros(
        *batch_shape, num_nodes, num_nodes, device=weight_matrix.device
    )

    # Set diagonal values for each batch element
    diag_indices = torch.arange(num_nodes)
    inv_sqrt_node_strengths[..., diag_indices, diag_indices] = 1.0 / torch.sqrt(
        node_strengths
    )

    # Compute the normalised weight matrix
    normalised_weight_matrix = torch.matmul(
        torch.matmul(inv_sqrt_node_strengths, weight_matrix), inv_sqrt_node_strengths
    )

    # Compute the communicability matrix
    communicability_matrix = torch.matrix_exp(normalised_weight_matrix)

    return communicability_matrix


@jaxtyped(typechecker=typechecked)
def shortest_paths(
    matrices: Float[torch.Tensor, "*batch num_nodes num_nodes"],
    invert_weights: bool = False,
) -> tuple[
    Float[torch.Tensor, "*batch num_nodes num_nodes"],
    Float[torch.Tensor, "*batch num_nodes num_nodes"],
]:
    r"""Compute shortest path distances and path counts using Floyd-Warshall algorithm.
    Accepts both binary and weighted networks.

    Args:
        matrices:
            Batch of either binary or weighted adjacency matrices with shape [*batch, num_nodes, num_nodes].
        invert_weights:
            If True, treats weights as inverse distances (so higher weights mean shorter distances).
            If False, weights are treated as distances (higher weights mean longer distances).
            Defaults to False.

    Returns:
        distances:
            Shortest path distances for each pair of nodes in each network with shape [*batch, num_nodes, num_nodes].
            Non-connected nodes will have a distance of infinity.
        path_counts:
            Number of shortest paths for each pair of nodes in each network with shape [*batch, num_nodes, num_nodes].
            Diagonal entries will be 1 (self-paths), and non-connected nodes will have a count of 0.

    """
    num_nodes = matrices.shape[-1]
    device = matrices.device

    # Initialize distance matrix
    distances = matrices.clone().float()

    # Invert weights if they represent strengths/affinities
    if invert_weights:
        # Only invert non-zero entries
        mask = distances > 0
        distances[mask] = 1.0 / distances[mask]

    # Set zero entries to infinity (no connection)
    distances[distances == 0] = torch.inf

    # Set diagonal to 0 (no self-distance)
    diag_idx = torch.arange(num_nodes, device=device)
    distances[..., diag_idx, diag_idx] = 0

    # Initialize path count matrix
    path_counts = torch.zeros_like(distances)
    path_counts[matrices > 0] = 1  # Use original matrix for path counting
    path_counts[..., diag_idx, diag_idx] = 1

    # Floyd-Warshall with path counting
    for k in range(num_nodes):
        dist_ik = distances[..., :, k].unsqueeze(-1)
        dist_kj = distances[..., k, :].unsqueeze(-2)
        new_dist = dist_ik + dist_kj

        count_ik = path_counts[..., :, k].unsqueeze(-1)
        count_kj = path_counts[..., k, :].unsqueeze(-2)
        new_count = count_ik * count_kj

        is_shorter = new_dist < distances
        is_equal = torch.isclose(new_dist, distances)

        distances[is_shorter] = new_dist[is_shorter]
        path_counts[is_shorter] = new_count[is_shorter]
        path_counts[is_equal] += new_count[is_equal]

    return distances, path_counts


@jaxtyped(typechecker=typechecked)
def binary_characteristic_path_length(
    adjacency_matrices: Float[torch.Tensor, "*batch num_nodes num_nodes"],
) -> Float[torch.Tensor, "*batch"]:
    r"""Compute the characteristic path length for each binary network.

    In a binary network, a path is a sequence of node in which each node is connected to the next one by an edge,
    and no node is repeated. A path between two nodes is a path which starts at one node and ends at the other.
    Shortest paths are those paths between two nodes which, among all paths between those nodes, have the minimum number of edges.

    The characteristic path length of a network is the average shortest path length, where the average is taken over all pairs of nodes in the network.

    Args:
        adjacency_matrices:
            Binary adjacency matrix with shape [*batch, num_nodes, num_nodes]

    Returns:
        Characteristic path length for each network with shape [*batch]

    Examples:
        >>> import torch
        >>> from gnm.utils import binary_characteristic_path_length
        >>> # Create a sample binary network in which all nodes are connected
        >>> connectome = torch.tensor([[[0, 1, 1, 1],
        ...                             [1, 0, 1, 1],
        ...                             [1, 1, 0, 1],
        ...                             [1, 1, 1, 0]]], dtype=torch.float32)
        >>> path_length = binary_characteristic_path_length(connectome)
        >>> print(path_length)
        tensor([1.0000])

    """
    binary_checks(adjacency_matrices)

    distances, _ = shortest_paths(adjacency_matrices, invert_weights=False)

    batch_shape = adjacency_matrices.shape[:-2]
    n_nodes = adjacency_matrices.shape[-1]

    # Mask diagonal (self-distances)
    mask = ~torch.eye(n_nodes, dtype=bool, device=adjacency_matrices.device)

    shortest_paths_flat = distances[..., mask].reshape(
        *batch_shape, n_nodes, n_nodes - 1
    )

    # Mean over all node pairs
    path_length = shortest_paths_flat.mean(dim=(-1, -2))

    return path_length


@jaxtyped(typechecker=typechecked)
def weighted_characteristic_path_length(
    adjacency_matrices: Float[torch.Tensor, "*batch num_nodes num_nodes"],
    invert_weights: bool = False,
) -> Float[torch.Tensor, "*batch"]:
    r"""Compute the characteristic path length for each weighted network.

    In a weighted network, a path is a sequence of nodes in which each node is connected to the next one by an edge,
    and no node is repeated. A path between two nodes is a path which starts at one node and ends at the other.
    Shortest paths are those paths between two nodes which, among all paths between those nodes, have the minimum total weight.

    The characteristic path length of a network is the average shortest path length, where the average is taken over all pairs of nodes in the network.

    Args:
        adjacency_matrices:
            Weighted adjacency matrix with shape [*batch, num_nodes, num_nodes]
        invert_weights:
            If True, treats weights as inverse distances (so higher weights mean shorter distances).
            If False, weights are treated as distances (higher weights mean longer distances).
            Defaults to False.

    Returns:
        Characteristic path length for each network with shape [*batch]

    Examples:
        >>> import torch
        >>> from gnm.utils import weighted_characteristic_path_length
        >>> # Create a sample weighted network
        >>> connectome = torch.tensor([[[0, 0.5, 0.8, 0.3],
        ...                             [0.5, 0, 0.4, 0.6],
        ...                             [0.8, 0.4, 0, 0.2],
        ...                             [0.3, 0.6, 0.2, 0]]], dtype=torch.float32)
        >>> # Treating weights as distances
        >>> path_length = weighted_characteristic_path_length(connectome, invert_weights=False)
        >>> # Treating weights as connection strengths
        >>> path_length = weighted_characteristic_path_length(connectome, invert_weights=True)

    """
    weighted_checks(adjacency_matrices)

    distances, _ = shortest_paths(adjacency_matrices, invert_weights=invert_weights)

    batch_shape = adjacency_matrices.shape[:-2]
    n_nodes = adjacency_matrices.shape[-1]

    # Mask diagonal (self-distances)
    mask = ~torch.eye(n_nodes, dtype=bool, device=adjacency_matrices.device)

    shortest_paths_flat = distances[..., mask].reshape(
        *batch_shape, n_nodes, n_nodes - 1
    )

    # Mean over all node pairs
    path_length = shortest_paths_flat.mean(dim=(-1, -2))

    return path_length


def binary_betweenness_centrality_nx(
    matrices: Float[torch.Tensor, "num_matrices num_nodes num_nodes"],
) -> Float[torch.Tensor, "num_matrices num_nodes"]:
    r"""Compute betweenness centrality for each node in binary networks.

    Betweenness centrality quantifies the number of times a node acts as a bridge along
    the shortest path between two other nodes. It identifies nodes that control information
    flow in a network.

    This function uses NetworkX for calculation and is intended for binary networks.

    Args:
        matrices:
            Batch of binary adjacency matrices with shape [num_matrices, num_nodes, num_nodes]

    Returns:
        Array of betweenness centralities for each node in each network with shape [num_matrices, num_nodes]

    Examples:
        >>> import torch
        >>> from gnm.utils import binary_betweenness_centrality
        >>> from gnm import defaults
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> binary_connectome = defaults.get_binary_network(device=DEVICE)
        >>> betweenness = binary_betweenness_centrality(binary_connectome)
        >>> betweenness.shape
        torch.Size([1, 4])

    Notes:
        This function converts PyTorch tensors to NumPy arrays for NetworkX processing,
        then converts the results back to PyTorch tensors. For large networks or batches,
        this may be computationally expensive.

    See Also:
        - [`evaluation.BetweennessKS`][gnm.evaluation.BetweennessKS]: Binary evaluation criterion which compares the distribution of betweenness centralities between two binary networks.
    """

    warn(
        """
        This implementation of betweeness centrality is depriciated.
        "Use binary_betweenness_centrality instead: 
        https://generative-network-models-toolbox.readthedocs.io/en/latest/api-reference/utils/#:~:text=gnm.utils.binary_betweenness_centrality
        """
    )

    graphs = [nx.from_numpy_array(matrix.cpu().numpy()) for matrix in matrices]
    betweenness_values = [
        np.array(list(nx.betweenness_centrality(g).values())) for g in graphs
    ]
    return torch.tensor(np.array(betweenness_values), dtype=matrices.dtype)


@jaxtyped(typechecker=typechecked)
def binary_betweenness_centrality(
    connectome: Float[torch.Tensor, "*batch num_nodes num_nodes"],
    normalised: bool = True,
) -> Float[torch.Tensor, "*batch num_nodes"]:
    r"""Compute betweenness centrality for each node in binary networks.

    Betweenness centrality quantifies the number of times a node acts as a bridge along
    the shortest path between two other nodes. It identifies nodes that control information
    flow in a network.

    Args:
        connectome:
            Batch of binary adjacency matrices with shape [*batch, num_nodes, num_nodes]
        normalised:
            If True (default), the betweenness values are normalised by dividing
            by the number of possible pairs of nodes, which is `(n-1)(n-2)/2` for
            undirected graphs, where n is the number of nodes.

    Returns:
        Array of betweenness centralities for each node in each network with shape [*batch, num_nodes]

    Examples:
        >>> import torch
        >>> from gnm.utils import binary_betweenness_centrality
        >>> from gnm import defaults
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> binary_connectome = defaults.get_binary_network(device=device)
        >>> betweenness = binary_betweenness_centrality(binary_connectome)
        >>> betweenness.shape
        torch.Size([1, 4])

    Notes:
        This function uses the Floyd-Warshall algorithm to compute shortest paths and path counts.
        For large networks or batches, this may be computationally expensive.

    See Also:
        - [`evaluation.BetweennessKS`][gnm.evaluation.BetweennessKS]: Binary evaluation criterion which compares the distribution of betweenness centralities between two binary networks.
    """
    binary_checks(connectome)

    device = connectome.device
    *batch_shape, num_nodes, _ = connectome.shape
    batch_size = torch.prod(torch.tensor(batch_shape)).item()

    # Reshape to a 2D batch for easier processing
    connectome_2d = connectome.view(batch_size, num_nodes, num_nodes)

    # Get shortest paths and path counts
    distances, path_counts = shortest_paths(connectome_2d, invert_weights=False)

    # Betweenness centrality computation
    betweenness = torch.zeros(batch_size, num_nodes, device=device)

    # Avoid division by zero, replace sigma = 0 with 1 (as it won't be used anyway)
    sigma_no_zeros = torch.where(
        path_counts == 0, torch.ones_like(path_counts), path_counts
    )

    for v in range(num_nodes):
        # Get distances and path counts relative to node v
        dist_sv = distances[:, :, v].unsqueeze(2)
        dist_vt = distances[:, v, :].unsqueeze(1)

        sigma_sv = path_counts[:, :, v].unsqueeze(2)
        sigma_vt = path_counts[:, v, :].unsqueeze(1)

        # Condition for v being on a shortest path between s and t
        is_on_path = torch.isclose(distances, dist_sv + dist_vt)

        # Number of shortest paths from s to t passing through v
        sigma_st_v = sigma_sv * sigma_vt

        # Pair dependency: sigma_st(v) / sigma_st
        pair_dependency = sigma_st_v / sigma_no_zeros

        # Only consider dependencies where v is on the shortest path
        dependency_v = torch.where(
            is_on_path, pair_dependency, torch.zeros_like(pair_dependency)
        )

        # Exclude endpoints s and t from the sum (s!=v!=t)
        dependency_v.diagonal(dim1=-2, dim2=-1).fill_(0)  # case s=t
        dependency_v[:, v, :] = 0  # case s=v
        dependency_v[:, :, v] = 0  # case t=v

        # Sum all pair dependencies for node v
        betweenness[:, v] = dependency_v.sum(dim=(-1, -2))

    # Adjust for undirected networks
    betweenness /= 2.0

    if normalised:
        # Normalise by the number of possible pairs
        if num_nodes > 2:
            norm_factor = ((num_nodes - 1) * (num_nodes - 2)) / 2.0
            if norm_factor > 0:
                betweenness /= norm_factor
        else:
            # For n<=2, betweenness is always 0, no normalization needed
            betweenness.fill_(0)

    # Reshape back to original batch shape
    return betweenness.view(*batch_shape, num_nodes)


@jaxtyped(typechecker=typechecked)
def weighted_betweenness_centrality(
    connectome: Float[torch.Tensor, "*batch num_nodes num_nodes"],
    normalised: bool = True,
    invert_weights: bool = False,
) -> Float[torch.Tensor, "*batch num_nodes"]:
    r"""Compute weighted betweenness centrality for each node in a batch of graphs.

    This function calculates the betweenness centrality for graphs with weighted edges,
    adapting Brandes' algorithm for batch processing on PyTorch tensors.

    The algorithm consists of two main stages:
    1.  An all-pairs shortest path calculation using a Floyd-Warshall algorithm
        to find the distance and number of shortest paths between all node pairs.
    2.  An accumulation stage that computes the betweenness centrality for each node
        by summing the dependencies over all source-target pairs.

    Args:
        connectome:
            A batch of weighted adjacency matrices with shape
            [*batch, num_nodes, num_nodes]. Edge weights should be positive.
        normalised:
            If True (default), the betweenness values are normalised by dividing
            by the number of possible pairs of nodes, `(n-1)(n-2)/2`
        invert_weights:
            If True, treats weights as connection strengths (higher = closer)
            and inverts them to get distances (1/weight). If False, treats
            weights as distances directly. Default: False.

    Returns:
        A tensor of betweenness centrality values for each node in each graph,
        with shape [*batch, num_nodes].

    Examples:
        >>> import torch
        >>> # Create a sample weighted graph (batch of 1)
        >>> #      (1) --2-- (2)
        >>> #     / |       /
        >>> #    1  3      1
        >>> #   /   |     /
        >>> # (0) --4-- (3)
        >>> graph = torch.tensor([[[0, 1, 2, 4],
        ...                        [1, 0, 0, 3],
        ...                        [2, 0, 0, 1],
        ...                        [4, 3, 1, 0]]], dtype=torch.float32)
        >>> bc = weighted_betweenness_centrality(graph)
        >>> print(bc)
        tensor([[0.5000, 0.0000, 0.5000, 1.0000]])

    """
    weighted_checks(connectome)

    device = connectome.device
    *batch_shape, num_nodes, _ = connectome.shape
    batch_size = torch.prod(torch.tensor(batch_shape)).item()

    # Reshape to a 2D batch for easier processing
    connectome_2d = connectome.view(batch_size, num_nodes, num_nodes)

    # Get shortest paths and path counts
    distances, path_counts = shortest_paths(
        connectome_2d, invert_weights=invert_weights
    )

    # Betweenness centrality computation
    betweenness = torch.zeros(batch_size, num_nodes, device=device)

    # Avoid division by zero, replace sigma=0 with 1 (as it won't be used anyway)
    sigma_no_zeros = torch.where(
        path_counts == 0, torch.ones_like(path_counts), path_counts
    )

    for v in range(num_nodes):
        # Get distances and path counts relative to node v
        dist_sv = distances[:, :, v].unsqueeze(2)
        dist_vt = distances[:, v, :].unsqueeze(1)

        sigma_sv = path_counts[:, :, v].unsqueeze(2)
        sigma_vt = path_counts[:, v, :].unsqueeze(1)

        # Condition for v being on a shortest path between s and t
        is_on_path = torch.isclose(distances, dist_sv + dist_vt)

        # Number of shortest paths from s to t passing through v
        sigma_st_v = sigma_sv * sigma_vt

        # Pair dependency: sigma_st(v) / sigma_st
        pair_dependency = sigma_st_v / sigma_no_zeros

        # Only consider dependencies where v is on the shortest path
        dependency_v = torch.where(
            is_on_path, pair_dependency, torch.zeros_like(pair_dependency)
        )

        # Exclude endpoints s and t from the sum (s!=v!=t)
        dependency_v.diagonal(dim1=-2, dim2=-1).fill_(0)  # case s=t
        dependency_v[:, v, :] = 0  # case s=v
        dependency_v[:, :, v] = 0  # case t=v

        # Sum all pair dependencies for node v
        betweenness[:, v] = dependency_v.sum(dim=(-1, -2))

    # Adjust for undirected networks
    betweenness /= 2.0

    if normalised:
        # Normalise by the number of possible pairs
        if num_nodes > 2:
            norm_factor = ((num_nodes - 1) * (num_nodes - 2)) / 2.0
            if norm_factor > 0:
                betweenness /= norm_factor
        else:
            # For n<=2, betweenness is always 0, no normalisation needed
            betweenness.fill_(0)

    # Reshape back to original batch shape
    return betweenness.view(*batch_shape, num_nodes)


@jaxtyped(typechecker=typechecked)
def binary_small_worldness(
    adjacency_matrices: Float[torch.Tensor, "*batch num_nodes num_nodes"],
    num_random_networks: int = 100,
    seed: Optional[int] = None,
):
    r"""Compute the (binary) small-worldness for each network in a batch.

    Small-worldness quantifies the degree to which a network exhibits small-world properties,
    which are characterised by high clustering and short path lengths between nodes.

    The small-worldness is computed as:
    $$
    \frac{C/C_{\rm random}}{L/L_{\rm random}},
    $$
    where $C$ is the clustering coefficient of the network, $L$ is the characteristic path length,
    and the subscripts "random" denote the average of that quantity for a random network with the same number of nodes and edges.
    $C_{\rm random}$ and $L_{\rm random}$ are estimated by generating and then averaging over a collection of random networks.

    Args:
        adjacency_matrices:
            Binary adjacency matrix with shape [*batch, num_nodes, num_nodes]
        num_random_networks:
            Number of random networks to generate for calculating average clustering and path length.
            Defaults to 100.
        seed:
            Random seed for reproducibility. If None, no seed is set.
            Defaults to None.

    Returns:
        Small-worldness for each network with shape [*batch]

    Examples:
        >>> import torch
        >>> from gnm.utils import binary_small_worldness
        >>> from gnm import defaults
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> binary_connectome = defaults.get_binary_network(device=DEVICE)
        >>> small_worldness = binary_small_worldness(binary_connectome)
    """
    if seed is not None:
        _check_seed_consistency(seed)
        torch.manual_seed(seed)

    binary_checks(adjacency_matrices)

    # Real network measures
    binary_clustering = binary_clustering_coefficients(adjacency_matrices)
    global_clustering = binary_clustering.mean(dim=-1)  # [*batch]
    path_length = binary_characteristic_path_length(adjacency_matrices)  # [*batch]

    # Generate random networks using get_control
    # Repeat each network num_random_networks times
    expanded_matrices = adjacency_matrices.unsqueeze(0).repeat(
        num_random_networks, *([1] * len(adjacency_matrices.shape))
    )  # [num_random_networks, *batch, num_nodes, num_nodes]

    # Generate random control networks
    random_networks = get_control(
        expanded_matrices
    )  # [num_random_networks, *batch, num_nodes, num_nodes]

    # Calculate measures for random networks
    random_local_clustering = binary_clustering_coefficients(
        random_networks
    )  # [num_random_networks, *batch, num_nodes]
    random_global_clustering = random_local_clustering.mean(
        dim=-1
    )  # [num_random_networks, *batch]
    average_random_global_clustering = random_global_clustering.mean(dim=0)  # [*batch]

    random_path_length = binary_characteristic_path_length(
        random_networks
    )  # [num_random_networks, *batch]
    average_random_path_length = random_path_length.mean(dim=0)  # [*batch]

    # Small-worldness
    small_worldness = (global_clustering / average_random_global_clustering) / (
        path_length / average_random_path_length
    )

    return small_worldness


@jaxtyped(typechecker=typechecked)
def weighted_small_worldness(
    weight_matrices: Float[torch.Tensor, "*batch num_nodes num_nodes"],
    num_random_networks: int = 100,
    invert_weights: bool = False,
    seed: Optional[int] = None,
):
    r"""Compute the (weighted) small-worldness for each network in a batch.

    Small-worldness quantifies the degree to which a network exhibits small-world properties,
    which are characterised by high clustering and short path lengths between nodes.

    The small-worldness is computed as:
    $$
    \frac{C/C_{\rm random}}{L/L_{\rm random}},
    $$
    where $C$ is the clustering coefficient of the network, $L$ is the characteristic path length,
    and the subscripts "random" denote the average of that quantity for a random network with the same number of nodes and edges.
    $C_{\rm random}$ and $L_{\rm random}$ are estimated by generating and then averaging over a collection of random networks.

    Args:
        weight_matrices:
            A batch of adjacency matrices representing the connectomes. The tensor
            should have shape (*batch, num_nodes, num_nodes) and contain edge weights.
        num_random_networks:
            Number of random networks to generate for calculating average clustering and path length.
            Defaults to 100.
        invert_weights:
            If True, treats weights as connection strengths (higher = closer)
            and inverts them to get distances (1/weight). If False, treats
            weights as distances directly.
            Default to False.
        seed:
            Random seed for reproducible results.
            If None, no seed is set.
            Defaults to None.

    Returns:
        Small-worldness for each network with shape [*batch].

    Example:
        >>> connectome = torch.rand(5, 10, 10)  # Batch of 5 connectomes with 10 nodes each
        >>> small_worldness = weighted_small_worldness(connectome)
        >>> print(small_worldness.shape)
        torch.Size([5])
    """
    if seed is not None:
        _check_seed_consistency(seed)
        torch.manual_seed(seed)

    weighted_checks(weight_matrices)

    # Real network measures
    weighted_clustering = weighted_clustering_coefficients(weight_matrices)
    global_clustering = weighted_clustering.mean(dim=-1)  # [*batch]
    path_length = weighted_characteristic_path_length(
        weight_matrices, invert_weights=invert_weights
    )  # [*batch]

    # Generate random networks using get_control
    # Repeat each network num_random_networks times
    expanded_matrices = weight_matrices.unsqueeze(0).repeat(
        num_random_networks, *([1] * len(weight_matrices.shape))
    )  # [num_random_networks, *batch, num_nodes, num_nodes]

    # Generate random control networks
    random_networks = get_control(
        expanded_matrices
    )  # [num_random_networks, *batch, num_nodes, num_nodes]

    # Calculate measures for random networks
    random_local_clustering = weighted_clustering_coefficients(
        random_networks
    )  # [num_random_networks, *batch, num_nodes]
    random_global_clustering = random_local_clustering.mean(
        dim=-1
    )  # [num_random_networks, *batch]
    average_random_global_clustering = random_global_clustering.mean(dim=0)  # [*batch]

    random_path_length = weighted_characteristic_path_length(
        random_networks, invert_weights=invert_weights
    )  # [num_random_networks, *batch]
    average_random_path_length = random_path_length.mean(dim=0)  # [*batch]

    # Small-worldness
    small_worldness = (global_clustering / average_random_global_clustering) / (
        path_length / average_random_path_length
    )

    return small_worldness
