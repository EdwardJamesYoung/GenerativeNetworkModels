r"""Correlation-based criteria for evaluating weighted networks.

This module provides evaluation criteria based on spatial correlations of network
properties between synthetic and real weighted networks. These criteria quantify
the similarity in the spatial patterns of node strengths, weighted clustering coefficients,
and weighted betweenness centralitynetworks.
"""

import torch
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
from .evaluation_base import CorrelationCriterion, WeightedEvaluationCriterion
from gnm.utils import (
    weighted_betweenness_centrality,
    weighted_clustering_coefficients,
    node_strengths,
)


class StrengthCorrelation(CorrelationCriterion, WeightedEvaluationCriterion):
    r"""Compare spatial patterns of node strengths between weighted networks using correlation.

    This criterion measures the similarity between the spatial patterns of node strengths in
    synthetic and real weighted networks using Pearson correlation. Unlike KS statistics which
    compare distributions, correlation criteria assess whether the same nodes have
    similar relative values in both networks.

    Examples:
        >>> import torch
        >>> from gnm.defaults import get_weighted_network
        >>> from gnm.evaluation import StrengthCorrelation
        >>> from gnm.utils import get_control
        >>> # Load a default weighted network
        >>> real_network = get_weighted_network()
        >>> # Create smoothing matrix (here using identity for example)
        >>> num_nodes = real_network.shape[-1]
        >>> smoothing_matrix = torch.eye(num_nodes)
        >>> # Create a random network
        >>> random_network = get_control(real_network)
        >>> # Calculate strength spatial correlation
        >>> criterion = StrengthCorrelation(smoothing_matrix)
        >>> dissimilarity = criterion(random_network, real_network)

    See Also:
        - [`evaluation.CorrelationCriterion`][gnm.evaluation.CorrelationCriterion]: The base class for correlation criteria, from which this class inherits.
        - [`evaluation.WeightedEvaluationCriterion`][gnm.evaluation.WeightedEvaluationCriterion]: The base class for weighted evaluation criteria, from which this class inherits.
    """

    def __init__(
        self,
        smoothing_matrix: Float[torch.Tensor, "num_nodes num_nodes"],
        normalise: bool = True,
    ):
        r"""
        Args:
            smoothing_matrix:
                Matrix used to spatially smooth the strength values, which can
                help account for registration errors or spatial uncertainty in brain networks.
                Shape [num_nodes, num_nodes].
        """
        CorrelationCriterion.__init__(self, smoothing_matrix)
        WeightedEvaluationCriterion.__init__(self)
        self.normalise = normalise

    @jaxtyped(typechecker=typechecked)
    def _get_graph_statistics(
        self, matrices: Float[torch.Tensor, "num_networks num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_networks num_nodes"]:
        r"""Extract strength values for each node in the networks.

        Computes the sum of edge weights for each node, which represents the
        node's strength in a weighted network.

        Args:
            matrices:
                Batch of weighted adjacency matrices with shape [num_networks, num_nodes, num_nodes]

        Returns:
            Tensor of node strengths with shape [num_networks, num_nodes]
        """
        if self.normalise:
            return node_strengths(matrices / matrices.amax(dim=(-1, -2), keepdim=True))
        else:
            return node_strengths(matrices)


class WeightedClusteringCorrelation(CorrelationCriterion, WeightedEvaluationCriterion):
    r"""Compare spatial patterns of weighted clustering coefficients using correlation.

    This criterion measures the similarity between the spatial patterns of weighted clustering
    coefficients in synthetic and real weighted networks using Pearson correlation. The weighted
    clustering coefficient measures the degree to which nodes in a weighted graph tend to cluster
    together, taking into account the weights of the edges.

    Examples:
        >>> import torch
        >>> from gnm.defaults import get_weighted_network
        >>> from gnm.evaluation import WeightedClusteringCorrelation
        >>> from gnm.utils import get_control
        >>> # Load a default weighted network
        >>> real_network = get_weighted_network()
        >>> # Create smoothing matrix (here using identity for example)
        >>> num_nodes = real_network.shape[-1]
        >>> smoothing_matrix = torch.eye(num_nodes)
        >>> # Create a random network
        >>> random_network = get_control(real_network)
        >>> # Calculate weighted clustering coefficient spatial correlation
        >>> criterion = WeightedClusteringCorrelation(smoothing_matrix)
        >>> dissimilarity = criterion(random_network, real_network)

    See Also:
        - [`utils.weighted_clustering_coefficients`][gnm.utils.weighted_clustering_coefficients]: Function used to calculate weighted clustering coefficients
        - [`evaluation.CorrelationCriterion`][gnm.evaluation.CorrelationCriterion]: Base class for correlation criteria, from which this class inherits.
        - [`evaluation.WeightedEvaluationCriterion`][gnm.evaluation.WeightedEvaluationCriterion]: Base class for weighted evaluation criteria, from which this class inherits.
    """

    def __init__(
        self,
        smoothing_matrix: Float[torch.Tensor, "num_nodes num_nodes"],
    ):
        r"""
        Args:
            smoothing_matrix:
                Matrix used to spatially smooth the clustering coefficient values, which can
                help account for registration errors or spatial uncertainty in brain networks.
                Shape [num_nodes, num_nodes].
        """
        CorrelationCriterion.__init__(self, smoothing_matrix)
        WeightedEvaluationCriterion.__init__(self)

    @jaxtyped(typechecker=typechecked)
    def _get_graph_statistics(
        self, matrices: Float[torch.Tensor, "num_networks num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_networks num_nodes"]:
        r"""Extract clustering coefficient values for each node in the networks.

        Computes the weighted clustering coefficient for each node, representing the
        fraction of possible triangles through that node that exist.

        Args:
            matrices:
                Batch of adjacency matrices with shape [num_networks, num_nodes, num_nodes]

        Returns:
            Tensor of clustering coefficients with shape [num_networks, num_nodes]
        """
        return weighted_clustering_coefficients(matrices)


class WeightedBetweennessCorrelation(CorrelationCriterion, WeightedEvaluationCriterion):
    r"""Compare spatial patterns of weighted betweenness centrality using correlation.

    This criterion measures the similarity between the spatial patterns of weighted betweenness
    centrality in synthetic and real weighted networks using Pearson correlation. Weighted
    betweenness centrality quantifies the number of times a node acts as a bridge along the
    shortest path between two other nodes, taking into account the weights of the edges.

    Examples:
        >>> import torch
        >>> from gnm.defaults import get_weighted_network
        >>> from gnm.evaluation import WeightedBetweennessCorrelation
        >>> from gnm.utils import get_control
        >>> # Load a default weighted network
        >>> real_network = get_weighted_network()
        >>> # Create smoothing matrix (here using identity for example)
        >>> num_nodes = real_network.shape[-1]
        >>> smoothing_matrix = torch.eye(num_nodes)
        >>> # Create a random network
        >>> random_network = get_control(real_network)
        >>> # Calculate weighted betweenness centrality spatial correlation
        >>> criterion = WeightedBetweennessCorrelation(smoothing_matrix)
        >>> dissimilarity = criterion(random_network, real_network)

    See Also:
        - [`utils.weighted_betweenness_centrality`][gnm.utils.weighted_betweenness_centrality]: Function used to calculate weighted betweenness centrality
        - [`evaluation.CorrelationCriterion`][gnm.evaluation.CorrelationCriterion]: Base class for correlation criteria, from which this class inherits.
        - [`evaluation.WeightedEvaluationCriterion`][gnm.evaluation.WeightedEvaluationCriterion]: Base class for weighted evaluation criteria, from which this class inherits.
    """

    def __init__(
        self,
        smoothing_matrix: Float[torch.Tensor, "num_nodes num_nodes"],
        normalise: bool = True,
        invert_weights: bool = True,
    ):
        r"""
        Args:
            smoothing_matrix:
                Matrix used to spatially smooth the betweenness centrality values, which can
                help account for registration errors or spatial uncertainty in brain networks.
                Shape [num_nodes, num_nodes].
        """

        CorrelationCriterion.__init__(self, smoothing_matrix)
        WeightedEvaluationCriterion.__init__(self)
        self.normalise = normalise
        self.invert_weights = invert_weights

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
        """
        return weighted_betweenness_centrality(
            matrices,
            normalised=self.normalise,
            invert_weights=self.invert_weights,
        )
