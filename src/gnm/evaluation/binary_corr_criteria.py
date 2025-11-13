r"""Correlation-based criteria for evaluating binary networks.

This module provides evaluation criteria based on spatial correlations of network
properties between synthetic and real binary networks. These criteria quantify
the similarity in the spatial patterns of degrees, clustering coefficients, and
betweenness centrality, which is important for capturing the spatial organization
of brain networks.
"""

import torch
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
from .evaluation_base import CorrelationCriterion, BinaryEvaluationCriterion

from gnm.utils import binary_clustering_coefficients, binary_betweenness_centrality


class DegreeCorrelation(CorrelationCriterion, BinaryEvaluationCriterion):
    r"""Compare spatial patterns of node degrees between binary networks using correlation.

    This criterion measures the similarity between the spatial patterns of node degrees in
    synthetic and real networks using Pearson correlation. Unlike KS statistics which
    compare distributions, correlation criteria assess whether the same nodes have
    similar relative values in both networks.

    Examples:
        >>> import torch
        >>> from gnm.defaults import get_binary_network, get_distance_matrix
        >>> from gnm.evaluation import DegreeCorrelation
        >>> from gnm.utils import get_control
        >>> # Load a default binary network
        >>> real_network = get_binary_network()
        >>> # Create smoothing matrix (here using identity for example)
        >>> num_nodes = real_network.shape[-1]
        >>> smoothing_matrix = torch.eye(num_nodes)
        >>> # Create a random network
        >>> random_network = get_control(real_network)
        >>> # Calculate degree spatial correlation
        >>> criterion = DegreeCorrelation(smoothing_matrix)
        >>> dissimilarity = criterion(random_network, real_network)

    See Also:
        - [`utils.node_strengths`][gnm.utils.node_strengths]: The function used to calculate node degrees
        - [`evaluation.CorrelationCriterion`][gnm.evaluation.CorrelationCriterion]: The base class for correlation criteria, from which this class inherits.
        - [`evaluation.BinaryEvaluationCriterion`][gnm.evaluation.BinaryEvaluationCriterion]: The base class for binary evaluation criteria, from which this class inherits.
    """

    def __init__(self, smoothing_matrix: Float[torch.Tensor, "num_nodes num_nodes"]):
        r"""
        Args:
            smoothing_matrix:
                Matrix used to spatially smooth the degree values, which can
                help account for registration errors or spatial uncertainty in brain networks.
                Shape [num_nodes, num_nodes].
        """
        CorrelationCriterion.__init__(self, smoothing_matrix)
        BinaryEvaluationCriterion.__init__(self)

    @jaxtyped(typechecker=typechecked)
    def _get_graph_statistics(
        self, matrices: Float[torch.Tensor, "num_networks num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_networks num_nodes"]:
        r"""Extract degree values for each node in the networks.

        Computes the sum of connections for each node, which represents the
        node's degree in an undirected binary network.

        Args:
            matrices:
                Batch of adjacency matrices with shape [num_networks, num_nodes, num_nodes]

        Returns:
            Tensor of node degrees with shape [num_networks, num_nodes]
        """
        return matrices.sum(dim=-1)


class BinaryClusteringCorrelation(CorrelationCriterion, BinaryEvaluationCriterion):
    r"""Compare spatial patterns of clustering coefficients using correlation.

    This criterion measures the similarity between the spatial patterns of clustering
    coefficients in synthetic and real networks using Pearson correlation. The clustering
    coefficient measures the degree to which nodes in a graph tend to cluster together.

    Examples:
        >>> import torch
        >>> from gnm.defaults import get_binary_network, get_distance_matrix
        >>> from gnm.evaluation import ClusteringCorrelation
        >>> from gnm.utils import get_control
        >>> # Load a default binary network
        >>> real_network = get_binary_network()
        >>> # Create smoothing matrix (here using identity for example)
        >>> num_nodes = real_network.shape[-1]
        >>> smoothing_matrix = torch.eye(num_nodes)
        >>> # Create a random network
        >>> random_network = get_control(real_network)
        >>> # Calculate clustering coefficient spatial correlation
        >>> criterion = ClusteringCorrelation(smoothing_matrix)
        >>> dissimilarity = criterion(random_network, real_network)

    See Also:
        - [`utils.binary_clustering_coefficients`][gnm.utils.binary_clustering_coefficients]: Function used to calculate clustering coefficients
        - [`evaluation.CorrelationCriterion`][gnm.evaluation.CorrelationCriterion]: Base class for correlation criteria, from which this class inherits.
        - [`evaluation.BinaryEvaluationCriterion`][gnm.evaluation.BinaryEvaluationCriterion]: Base class for binary evaluation criteria, from which this class inherits.
    """

    def __init__(self, smoothing_matrix: Float[torch.Tensor, "num_nodes num_nodes"]):
        r"""
        Args:
            smoothing_matrix:
                Matrix used to spatially smooth the clustering coefficient values, which can
                help account for registration errors or spatial uncertainty in brain networks.
                Shape [num_nodes, num_nodes].
        """
        CorrelationCriterion.__init__(self, smoothing_matrix)
        BinaryEvaluationCriterion.__init__(self)

    @jaxtyped(typechecker=typechecked)
    def _get_graph_statistics(
        self, matrices: Float[torch.Tensor, "num_networks num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_networks num_nodes"]:
        r"""Extract clustering coefficient values for each node in the networks.

        Computes the clustering coefficient for each node, representing the
        fraction of possible triangles through that node that exist.

        Args:
            matrices:
                Batch of adjacency matrices with shape [num_networks, num_nodes, num_nodes]

        Returns:
            Tensor of clustering coefficients with shape [num_networks, num_nodes]
        """
        return binary_clustering_coefficients(matrices)


class BinaryBetweennessCorrelation(CorrelationCriterion, BinaryEvaluationCriterion):
    r"""Compare spatial patterns of betweenness centrality using correlation.

    This criterion measures the similarity between the spatial patterns of betweenness
    centrality in synthetic and real networks using Pearson correlation. Betweenness
    centrality quantifies the number of times a node acts as a bridge along the
    shortest path between two other nodes.

    Examples:
        >>> import torch
        >>> from gnm.defaults import get_binary_network, get_distance_matrix
        >>> from gnm.evaluation import BetweennessCorrelation
        >>> from gnm.utils import get_control
        >>> # Load a default binary network
        >>> real_network = get_binary_network()
        >>> # Create smoothing matrix (here using identity for example)
        >>> num_nodes = real_network.shape[-1]
        >>> smoothing_matrix = torch.eye(num_nodes)
        >>> # Create a random network
        >>> random_network = get_control(real_network)
        >>> # Calculate betweenness centrality spatial correlation
        >>> criterion = BetweennessCorrelation(smoothing_matrix)
        >>> dissimilarity = criterion(random_network, real_network)

    See Also:
        - [`utils.binary_betweenness_centrality`][gnm.utils.binary_betweenness_centrality]: Function used to calculate betweenness centrality
        - [`evaluation.CorrelationCriterion`][gnm.evaluation.CorrelationCriterion]: Base class for correlation criteria, from which this class inherits.
        - [`evaluation.BinaryEvaluationCriterion`][gnm.evaluation.BinaryEvaluationCriterion]: Base class for binary evaluation criteria, from which this class inherits.
    """

    def __init__(self, smoothing_matrix: Float[torch.Tensor, "num_nodes num_nodes"]):
        r"""
        Args:
            smoothing_matrix:
                Matrix used to spatially smooth the betweenness centrality values, which can
                help account for registration errors or spatial uncertainty in brain networks.
                Shape [num_nodes, num_nodes].
        """

        CorrelationCriterion.__init__(self, smoothing_matrix)
        BinaryEvaluationCriterion.__init__(self)

    @jaxtyped(typechecker=typechecked)
    def _get_graph_statistics(
        self, matrices: Float[torch.Tensor, "num_networks num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_networks num_nodes"]:
        r"""Extract betweenness centrality values for each node in the networks.

        Computes the betweenness centrality for each node, which measures how often
        a node lies on the shortest path between other nodes.

        Args:
            matrices:
                Batch of adjacency matrices with shape [num_networks, num_nodes, num_nodes]

        Returns:
            Tensor of betweenness centrality values with shape [num_networks, num_nodes]

        Notes:
            This method uses NetworkX for calculation and temporarily converts tensors
            to NumPy arrays, which may affect performance for large networks.
        """
        return binary_betweenness_centrality(matrices)
