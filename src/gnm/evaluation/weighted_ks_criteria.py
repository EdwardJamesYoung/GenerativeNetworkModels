r"""Kolmogorov-Smirnov criteria for evaluating weighted networks.

This module provides evaluation criteria based on the Kolmogorov-Smirnov (KS) test
to compare distributions of various network properties between synthetic and real
weighted networks. These criteria quantify differences in node strength distributions,
weighted clustering coefficients, and weighted betweenness centrality.
"""

import torch
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
from typing import Optional

from .evaluation_base import KSCriterion, WeightedEvaluationCriterion

from gnm.utils import (
    node_strengths,
    weighted_clustering_coefficients,
    weighted_betweenness_centrality,
)


class WeightedNodeStrengthKS(KSCriterion, WeightedEvaluationCriterion):
    r"""Compare node strength distributions between weighted networks using KS statistic.

    This criterion measures the dissimilarity between the node strength distributions of
    synthetic and real weighted networks using the Kolmogorov-Smirnov test. Node strength
    is the weighted equivalent of node degree - it is the sum of the weights of all edges
    connected to a node.

    Examples:
        >>> import torch
        >>> from gnm.defaults import get_weighted_network
        >>> from gnm.evaluation import WeightedNodeStrengthKS
        >>> from gnm.utils import get_control
        >>> # Load a default weighted network
        >>> real_network = get_weighted_network()
        >>> # Create a random network with the same weight distribution
        >>> random_network = get_control(real_network)
        >>> # Calculate node strength distribution dissimilarity
        >>> criterion = WeightedNodeStrengthKS()
        >>> dissimilarity = criterion(random_network, real_network)

    See Also:
        - [`utils.node_strengths`][gnm.utils.node_strengths]: The function used to calculate node strengths.
        - [`evaluation_base.KSCriterion`][gnm.evaluation.evaluation_base.KSCriterion]: The base class for KS criteria, from which this class inherits.
        - [`evaluation_base.WeightedEvaluationCriterion`][gnm.evaluation.evaluation_base.WeightedEvaluationCriterion]: The base class for weighted criteria, from which this class inherits.
    """

    @jaxtyped(typechecker=typechecked)
    def __init__(self, normalise: Optional[bool] = True):
        r"""
        Args:
            normalise:
                If True, normalise the weights of each network by its maximum weight
                before computing node strengths. This can be useful when comparing
                networks with different weight scales. Defaults to True.
        """
        KSCriterion.__init__(self)
        WeightedEvaluationCriterion.__init__(self)
        self.normalise = normalise

    @jaxtyped(typechecker=typechecked)
    def _get_graph_statistics(
        self, matrices: Float[torch.Tensor, "num_networks num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_networks num_nodes"]:
        r"""Extract node strength values for each node in the networks.

        Computes the sum of connection weights for each node, optionally normalizing
        by the maximum weight in each network.

        Args:
            matrices:
                Batch of weight matrices with shape [num_networks, num_nodes, num_nodes]

        Returns:
            Tensor of node strengths with shape [num_networks, num_nodes]
        """
        if self.normalise:
            return node_strengths(matrices / matrices.amax(dim=(-1, -2), keepdim=True))
        else:
            return node_strengths(matrices)


class WeightedBetweennessKS(KSCriterion, WeightedEvaluationCriterion):
    r"""Compare weighted betweenness centrality distributions using KS statistic.

    This criterion measures the dissimilarity between the weighted betweenness centrality
    distributions of synthetic and real networks using the Kolmogorov-Smirnov test.

    We compute betweenness centrality using Brandes algorithm. Betweenness
    centrality for a node $u$ in a weighted network is:

    $$
    c_B(u) = \sum_{v,w} \\frac{\sigma(v,w|u)}{\sigma(v,w)},
    $$

    where $\sigma(v,w)$ is the number of shortest paths from $v$ to $w$, and
    $\sigma(v,w|u)$ is the number of those that pass through $u$.
    For weighted networks, path lengths are computed using the edge weights as distance.

    Examples:
        >>> import torch
        >>> from gnm.defaults import get_weighted_network
        >>> from gnm.evaluation import WeightedBetweennessKS
        >>> from gnm.utils import get_control
        >>> # Load a default weighted network
        >>> real_network = get_weighted_network()
        >>> # Create a random network with the same weight distribution
        >>> random_network = get_control(real_network)
        >>> # Calculate weighted betweenness centrality distribution dissimilarity
        >>> criterion = WeightedBetweennessKS()
        >>> dissimilarity = criterion(random_network, real_network)

    See Also:
        - [`evaluation_base.KSCriterion`][gnm.evaluation.evaluation_base.KSCriterion]: The base class for KS criteria, from which this class inherits.
        - [`evaluation_base.WeightedEvaluationCriterion`][gnm.evaluation.evaluation_base.WeightedEvaluationCriterion]: The base class for weighted criteria, from which this class inherits.
    """

    @jaxtyped(typechecker=typechecked)
    def __init__(
        self,
        normalise: bool = True,
        invert_weights: bool = True,
    ):
        """
        Args:
            normalise: If True, normalise the weights of the network by the maximum weight in the network. Defaults to True.
        """
        KSCriterion.__init__(self)
        WeightedEvaluationCriterion.__init__(self)
        self.normalise = normalise
        self.invert_weights = invert_weights

    @jaxtyped(typechecker=typechecked)
    def _get_graph_statistics(
        self, matrices: Float[torch.Tensor, "num_networks num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_networks num_nodes"]:
        """Compute weighted betweenness centrality for each node in the network.

        Args:
            matrices: Weight matrices of the network

        Returns:
            torch.Tensor: array of weighted betweenness centralities
        """
        return weighted_betweenness_centrality(
            matrices,
            normalised=self.normalise,
            invert_weights=self.invert_weights,
        )


class WeightedClusteringKS(KSCriterion, WeightedEvaluationCriterion):
    r"""Compare weighted clustering coefficient distributions using KS statistic.

    This criterion measures the dissimilarity between the weighted clustering coefficient
    distributions of synthetic and real networks using the Kolmogorov-Smirnov test.

    Implements the Onnela et al. (2005) definition of weighted clustering, which uses
    the geometric mean of triangle weights. For each node $u$, the clustering coefficient is:

    $$
    c(u) = \\frac{1}{k_u(k_u-1)} \sum_{v,w} (\hat{w}_{uv} \\times \hat{w}_{uw} \\times \hat{w}_{vw})^{1/3},
    $$

    where $k_u$ is the node strength of node $u$, and $\hat{w}_{uv}$ is the weight of the edge between nodes $u$ and $v$,
    after normalising by dividing by the maximum weight in the network.

    Examples:
        >>> import torch
        >>> from gnm.defaults import get_weighted_network
        >>> from gnm.evaluation import WeightedClusteringKS
        >>> from gnm.utils import get_control
        >>> # Load a default weighted network
        >>> real_network = get_weighted_network()
        >>> # Create a random network with the same weight distribution
        >>> random_network = get_control(real_network)
        >>> # Calculate weighted clustering coefficient distribution dissimilarity
        >>> criterion = WeightedClusteringKS()
        >>> dissimilarity = criterion(random_network, real_network)

    See Also:
        - [`utils.weighted_clustering_coefficients`][gnm.utils.weighted_clustering_coefficients]: The function used to calculate weighted clustering
        - [`evaluation_base.KSCriterion`][gnm.evaluation.evaluation_base.KSCriterion]: The base class for KS criteria, from which this class inherits.
        - [`evaluation_base.WeightedEvaluationCriterion`][gnm.evaluation.evaluation_base.WeightedEvaluationCriterion]: The base class for weighted criteria, from which this class inherits.
    """

    def __init__(self):
        KSCriterion.__init__(self)
        WeightedEvaluationCriterion.__init__(self)

    @jaxtyped(typechecker=typechecked)
    def _get_graph_statistics(
        self, matrices: Float[torch.Tensor, "num_networks num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_networks num_nodes"]:
        r"""Extract weighted clustering coefficient values for each node.

        Computes the weighted clustering coefficient for each node using the
        Onnela et al. (2005) definition, which uses the geometric mean of
        triangle weights.

        Args:
            matrices:
                Batch of weight matrices with shape [num_networks, num_nodes, num_nodes]

        Returns:
            Tensor of weighted clustering coefficients with shape [num_networks, num_nodes]
        """
        with torch.no_grad():
            return weighted_clustering_coefficients(matrices)
