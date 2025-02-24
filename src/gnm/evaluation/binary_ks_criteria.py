import torch
import networkx as nx
import numpy as np
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
from .evaluation_base import KSCriterion, BinaryEvaluationCriterion

from gnm.utils import (
    binary_clustering_coefficients,
    binary_betweenness_centrality,
    ks_statistic,
)


class DegreeKS(KSCriterion, BinaryEvaluationCriterion):
    """KS statistic comparing degree distributions between networks."""

    def __init__(self):
        KSCriterion.__init__(self)
        BinaryEvaluationCriterion.__init__(self)

    def __str__(self) -> str:
        return "Binary degree KS"

    @jaxtyped(typechecker=typechecked)
    def _get_graph_statistics(
        self, matrices: Float[torch.Tensor, "num_networks num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_networks _"]:
        """Compute degree for each node in the network.

        Args:
            matrix:
                Adjacency matrix of the network

        Returns:
            Vector of node degrees
        """
        return matrices.sum(dim=-1)


class ClusteringKS(KSCriterion, BinaryEvaluationCriterion):
    """KS statistic comparing clustering coefficient distributions between networks."""

    def __init__(self):
        KSCriterion.__init__(self)
        BinaryEvaluationCriterion.__init__(self)

    def __str__(self) -> str:
        return "Binary clustering coefficient KS"

    @jaxtyped(typechecker=typechecked)
    def _get_graph_statistics(
        self, matrices: Float[torch.Tensor, "num_networks num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_networks _"]:
        """Compute clustering coefficient for each node in the network.

        Args:
            matrix:
                Adjacency matrix of the network

        Returns:
            Vector of clustering coefficients
        """
        return binary_clustering_coefficients(matrices)


class BetweennessKS(KSCriterion, BinaryEvaluationCriterion):
    """KS statistic comparing betweenness centrality distributions between networks."""

    def __init__(self):
        KSCriterion.__init__(self)
        BinaryEvaluationCriterion.__init__(self)

    def __str__(self) -> str:
        return "Binary betweenness centrality KS"

    @jaxtyped(typechecker=typechecked)
    def _get_graph_statistics(
        self, matrices: Float[torch.Tensor, "num_networks num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_networks _"]:
        """Compute betweenness centrality for each node in the network.

        Args:
            matrices: Adjacency matrix of the network

        Returns:
            torch.Tensor: Vector of betweenness centralities
        """
        return binary_betweenness_centrality(matrices)


class EdgeLengthKS(BinaryEvaluationCriterion):
    """KS statistic comparing edge length distributions between networks."""

    def __init__(self, distance_matrix: Float[torch.Tensor, "num_nodes num_nodes"]):
        """Initialise the criterion.

        Args:
            distance_matrix:
                Distance matrix of the real networks
        """
        BinaryEvaluationCriterion.__init__(self)
        self.distance_matrix = distance_matrix

    def __str__(self) -> str:
        return "Binary edge length KS"

    @jaxtyped(typechecker=typechecked)
    def _evaluate(
        self,
        synthetic_matrices: Float[
            torch.Tensor, "num_synthetic_networks num_nodes num_nodes"
        ],
        real_matrices: Float[torch.Tensor, "num_real_networks num_nodes num_nodes"],
    ) -> Float[torch.Tensor, "num_synthetic_networks num_real_networks"]:
        """Compute the KS statistic between edge length distributions.

        Args:
            synthetic_matrices:
                Batch of adjacency matrices of the synthetic networks
            real_matrices:
                Adjacency matrices of the real networks

        Returns:
            KS statistics for all pairs of synthetic and real networks
        """
        num_synthetic_networks = synthetic_matrices.shape[0]
        num_real_networks = real_matrices.shape[0]
        ks_distances = torch.zeros(
            num_synthetic_networks, num_real_networks, dtype=synthetic_matrices.dtype
        )
        synthetic_edge_lengths = []
        real_edge_lengths = []
        # Iterate through all pairs of synthetic and real networks
        for i in range(num_synthetic_networks):
            synthetic_edge_lengths.append(
                self._get_edge_lengths(synthetic_matrices[i, :, :])
            )
        for j in range(num_real_networks):
            real_edge_lengths.append(self._get_edge_lengths(real_matrices[j, :, :]))

        for i in range(num_synthetic_networks):
            for j in range(num_real_networks):
                ks_distances[i, j] = ks_statistic(
                    synthetic_edge_lengths[i], real_edge_lengths[j]
                )

        return ks_distances

    @jaxtyped(typechecker=typechecked)
    def _get_edge_lengths(
        self, adjacency_matrix: Float[torch.Tensor, "num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "1 num_non_zero_edges"]:
        """Compute edge lengths for each network.

        Args:
            matrices: Adjacency matrix of the network

        Returns:
            1D tensor of edge lengths
        """
        adj = torch.triu(adjacency_matrix, diagonal=1)
        return self.distance_matrix[adj.bool()].flatten().unsqueeze(0)
