r"""Base classes for network evaluation criteria.

This module defines the abstract base classes and common functionality for
building evaluation criteria to compare synthetic and real networks.
"""

import torch
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
from abc import ABC, abstractmethod

from gnm.utils import ks_statistic

from gnm.utils import binary_checks, weighted_checks

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EvaluationCriterion(ABC):
    r"""Abstract base class for network evaluation criteria.

    This class provides a framework for defining various criteria to evaluate the similarity
    between a synthetic (generated) network and a real (target) network. Each criterion
    computes a dissimilarity measure between the two networks based on specific network
    properties.

    Subclasses must implement:

    1. `_pre_call` to perform validation checks on input matrices.
    2. `_evaluate` to compute the actual dissimilarity measure.

    See Also:
        - [`evaluation.BinaryEvaluationCriterion`][gnm.evaluation.BinaryEvaluationCriterion]: Base class for binary network evaluations.
        - [`evaluation.WeightedEvaluationCriterion`][gnm.evaluation.WeightedEvaluationCriterion]: Base class for weighted network evaluations.
        - [`fitting.perform_evalutions`][gnm.fitting.perform_evaluations]: Function to evaluate networks using an evaluation criterion.
        - [`fitting.optimise_evaluation`][gnm.fitting.optimise_evaluation]: Function to optimise model parameters using an evaluation.
    """

    def __init__(self, device: torch.device = None):
        r"""
        Args:
            device:
                PyTorch device to use for computations. If None, uses CUDA if available,
                otherwise CPU.
        """
        self.device = DEVICE if device is None else device

    def __str__(self) -> str:
        r"""Return a string representation of the criterion."""
        return self.__class__.__name__

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self,
        synthetic_matrices: Float[
            torch.Tensor, "num_synthetic_networks num_nodes num_nodes"
        ],
        real_matrices: Float[torch.Tensor, "num_real_networks num_nodes num_nodes"],
    ) -> Float[torch.Tensor, "num_synthetic_networks num_real_networks"]:
        r"""Compute the dissimilarity between synthetic and real networks.

        This method validates the input matrices and then calls the _evaluate method
        to compute the actual dissimilarity measure.

        Args:
            synthetic_matrices:
                Batch of adjacency/weight matrices of the synthetic networks with shape
                [num_synthetic_networks, num_nodes, num_nodes]
            real_matrices:
                Adjacency/weight matrices of the real networks with shape
                [num_real_networks, num_nodes, num_nodes]

        Returns:
            Tensor of dissimilarity values with shape [num_synthetic_networks, num_real_networks],
            where higher values indicate greater dissimilarity
        """
        self._pre_call(synthetic_matrices)
        self._pre_call(real_matrices)
        return self._evaluate(synthetic_matrices, real_matrices).to(DEVICE)

    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def _pre_call(
        self, matrices: Float[torch.Tensor, "num_network num_nodes num_nodes"]
    ):
        r"""Perform validation checks on input matrices.

        This abstract method should be implemented by subclasses to ensure that input
        matrices meet the required criteria (*e.g.*, binary values, symmetry).

        Args:
            matrices:
                Adjacency/weight matrices with shape [num_networks, num_nodes, num_nodes]
        """
        pass

    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def _evaluate(
        self,
        synthetic_matrices: Float[
            torch.Tensor, "num_synthetic_networks num_nodes num_nodes"
        ],
        real_matrices: Float[torch.Tensor, "num_real_networks num_nodes num_nodes"],
    ) -> Float[torch.Tensor, "num_synthetic_networks num_real_networks"]:
        r"""Compute the actual dissimilarity measure between networks.

        This abstract method should be implemented by subclasses to define how
        to calculate the dissimilarity between synthetic and real networks.

        Args:
            synthetic_matrices:
                Batch of adjacency/weight matrices of the synthetic networks with shape
                [num_synthetic_networks, num_nodes, num_nodes]
            real_matrices:
                Adjacency/weight matrices of the real networks with shape
                [num_real_networks, num_nodes, num_nodes]

        Returns:
            Tensor of dissimilarity values with shape [num_synthetic_networks, num_real_networks]
        """
        pass


class BinaryEvaluationCriterion(EvaluationCriterion, ABC):
    r"""Base class for evaluation criteria specialized for binary networks.

    This class extends EvaluationCriterion to specifically handle binary (unweighted)
    networks. It implements validation checks to ensure input matrices contain only
    binary values (0 or 1), are symmetric, and contain no self-connections.

    Subclasses must implement the `_evaluate` method to compute the actual dissimilarity measure.

    See Also:
        - [`evaluation.EvaluationCriterion`][gnm.evaluation.EvaluationCriterion]: Parent abstract base class.
        - [`evaluation.WeightedEvaluationCriterion`][gnm.evaluation.WeightedEvaluationCriterion]: Base class for weighted network evaluations.
    """

    def __init__(self):
        r"""The initialisation method sets the accepts attribute to 'binary' to indicate that this criterion
        works with binary networks.
        """
        self.accepts = "binary"

    @jaxtyped(typechecker=typechecked)
    def _pre_call(
        self, matrices: Float[torch.Tensor, "num_networks num_nodes num_nodes"]
    ):
        r"""Perform validation checks on binary matrices.

        Validates that the input matrices contain only binary values (0 or 1),
        are symmetric, and have no self-connections.

        Args:
            matrices:
                Binary adjacency matrices with shape [num_networks, num_nodes, num_nodes]
        """
        binary_checks(matrices)


class WeightedEvaluationCriterion(EvaluationCriterion, ABC):
    r"""Base class for evaluation criteria specialized for weighted networks.

    This class extends EvaluationCriterion to specifically handle weighted networks.
    It implements validation checks to ensure input matrices contain only non-negative
    values which are symmetric and contain no self-connections.

    Subclasses must implement the `_evaluate` to compute the actual dissimilarity measure.

    See Also:
        - [`evaluation.EvaluationCriterion`][gnm.evaluation.EvaluationCriterion]: Parent abstract base class.
        - [`evaluation.BinaryEvaluationCriterion`][gnm.evaluation.BinaryEvaluationCriterion]: Base class for binary network evaluations.
    """

    def __init__(self):
        r"""The initialisation method sets the accepts attribute to 'weighted' to indicate that this criterion
        works with weighted networks.
        """
        self.accepts = "weighted"

    @jaxtyped(typechecker=typechecked)
    def _pre_call(
        self, matrices: Float[torch.Tensor, "num_networks num_nodes num_nodes"]
    ):
        r"""Perform validation checks on weighted matrices.

        Validates that the input matrices contain only non-negative values,
        are symmetric, and have no self-connections.

        Args:
            matrices:
                Weighted adjacency matrices with shape [num_networks, num_nodes, num_nodes]
        """
        weighted_checks(matrices)


class KSCriterion(EvaluationCriterion, ABC):
    r"""Base class for Kolmogorov-Smirnov (KS) distance based network evaluation.

    This class implements network comparison using the KS test statistic between
    distributions of network properties (*e.g.*, degree distribution, clustering
    coefficients). The KS statistic measures the maximum difference between two
    cumulative distribution functions, providing a measure of how different two
    distributions are.

    Subclasses must implement the `_get_graph_statistics` method to define the
    specific network property to use in the KS test.

    See Also:
        - [`evaluation.DegreeKS`][gnm.evaluation.DegreeKS]: KS test on degree distributions
        - [`evaluation.ClusteringKS`][gnm.evaluation.ClusteringKS]: KS test on clustering coefficient distributions
        - [`evaluation.BetweennessKS`][gnm.evaluation.BetweennessKS]: KS test on betweenness centrality distributions
        - [`utils.ks_statistic`][gnm.utils.ks_statistic]: Function to compute KS statistics, used internally to this class.
    """

    @jaxtyped(typechecker=typechecked)
    def _evaluate(
        self,
        synthetic_matrices: Float[
            torch.Tensor, "num_synthetic_networks num_nodes num_nodes"
        ],
        real_matrices: Float[torch.Tensor, "num_real_networks num_nodes num_nodes"],
    ) -> Float[torch.Tensor, "num_synthetic_networks num_real_networks"]:
        r"""Compute KS statistics between network property distributions.

        Extracts network properties using _get_graph_statistics and then computes
        the Kolmogorov-Smirnov statistic between all pairs of synthetic and real
        property distributions.

        Args:
            synthetic_matrices:
                Batch of adjacency/weight matrices of the synthetic networks with shape
                [num_synthetic_networks, num_nodes, num_nodes]
            real_matrices:
                Adjacency/weight matrices of the real networks with shape
                [num_real_networks, num_nodes, num_nodes]

        Returns:
            KS statistics for all pairs with shape [num_synthetic_networks, num_real_networks]
        """
        # Compute network property values for each network
        synthetic_statistics = self._get_graph_statistics(synthetic_matrices)
        real_statistics = self._get_graph_statistics(real_matrices)

        # Compute KS statistics between all pairs of distributions
        return ks_statistic(synthetic_statistics, real_statistics)

    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def _get_graph_statistics(
        self, matrices: Float[torch.Tensor, "num_networks num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_networks _"]:
        r"""Compute network properties for KS comparison.

        This abstract method should be implemented by subclasses to define which
        network property to extract for comparison (*e.g.*, degrees, clustering coefficients).

        Args:
            matrices:
                Adjacency/weight matrices with shape [num_networks, num_nodes, num_nodes]

        Returns:
            Network property values with shape [num_networks, num_values]
        """
        pass


class CorrelationCriterion(EvaluationCriterion, ABC):
    r"""Base class for correlation-based network evaluation criteria.

    This class implements network comparison using correlation coefficients between
    spatial patterns of network properties (*e.g.*, node degree, clustering coefficients).
    Higher correlation indicates greater similarity in the spatial organisation of
    network properties.

    Subclasses must implement the `_get_graph_statistics` method to define the specific
    network property to use in the correlation calculation.

    Args:
        smoothing_matrix:
            Matrix used to spatially smooth the network property values, which can
            help account for registration errors or spatial uncertainty in brain networks.

    See Also:
        - [`evaluation.DegreeCorrelation`][gnm.evaluation.DegreeCorrelation]: Correlation of node degree patterns
        - [`evaluation.ClusteringCorrelation`][gnm.evaluation.ClusteringCorrelation]: Correlation of clustering coefficient patterns
        - [`evaluation.BetweennessCorrelation`][gnm.evaluation.BetweennessCorrelation]: Correlation of betweenness centrality patterns
    """

    def __init__(self, smoothing_matrix: Float[torch.Tensor, "num_nodes num_nodes"]):
        r"""
        Args:
            smoothing_matrix:
                Matrix used to spatially smooth the network property values with shape
                [num_nodes, num_nodes]. This can help account for registration errors
                or spatial uncertainty in brain networks.
        """
        # Verify that the rows of the smoothing matrix sum to 1

        if not torch.allclose(
            torch.einsum("ij->i", smoothing_matrix),
            torch.tensor(1.0, device=smoothing_matrix.device),
        ):
            raise ValueError("Rows of the smoothing matrix must sum to 1.")

        self.smoothing_matrix = smoothing_matrix

    @jaxtyped(typechecker=typechecked)
    def _evaluate(
        self,
        synthetic_matrices: Float[
            torch.Tensor, "num_synthetic_networks num_nodes num_nodes"
        ],
        real_matrices: Float[torch.Tensor, "num_real_networks num_nodes num_nodes"],
    ) -> Float[torch.Tensor, "num_synthetic_networks num_real_networks"]:
        r"""Compute correlation coefficients between network property spatial patterns.

        Extracts network properties using _get_graph_statistics, applies spatial smoothing,
        and then computes Pearson correlation coefficients between all pairs of synthetic
        and real property spatial patterns.

        Args:
            synthetic_matrices:
                Batch of adjacency/weight matrices of the synthetic networks with shape
                [num_synthetic_networks, num_nodes, num_nodes]
            real_matrices:
                Adjacency/weight matrices of the real networks with shape
                [num_real_networks, num_nodes, num_nodes]

        Returns:
            Correlation coefficients for all pairs with shape [num_synthetic_networks, num_real_networks]
        """
        # Compute network property values for each network
        synthetic_statistics = self._get_graph_statistics(synthetic_matrices)
        real_statistics = self._get_graph_statistics(real_matrices)

        smoothed_synthetic_statistics = torch.einsum(
            "nj,ij->ni", synthetic_statistics, self.smoothing_matrix
        )  # Shape [num_synthetic_networks num_nodes]
        smoothed_real_statistics = torch.einsum(
            "nj,ij->ni", real_statistics, self.smoothing_matrix
        )  # Shape [num_real_networks num_nodes]

        # Compute correlation coefficients between all pairs of distributions
        # Center the data
        real_centered = smoothed_real_statistics - smoothed_real_statistics.mean(
            dim=1, keepdim=True
        )
        synth_centered = (
            smoothed_synthetic_statistics
            - smoothed_synthetic_statistics.mean(dim=1, keepdim=True)
        )

        # Compute standard deviations
        real_std = torch.sqrt((real_centered**2).sum(dim=1, keepdim=True) + 1e-12)
        synth_std = torch.sqrt((synth_centered**2).sum(dim=1, keepdim=True) + 1e-12)

        # Normalize the data
        real_normalised = real_centered / (real_std + 1e-12)
        synth_normalised = synth_centered / (synth_std + 1e-12)

        # Compute correlation matrix using matrix multiplication
        corr_matrix = torch.mm(synth_normalised, real_normalised.t())

        # Divide by number of nodes to get Pearson R
        corr_matrix = corr_matrix / smoothed_real_statistics.shape[1]

        return corr_matrix

    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def _get_graph_statistics(
        self, matrices: Float[torch.Tensor, "num_networks num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_networks num_nodes"]:
        r"""Compute network properties for correlation comparison.

        This abstract method should be implemented by subclasses to define which
        network property to extract for comparison (*e.g.*, degrees, clustering coefficients).

        Args:
            matrices:
                Adjacency/weight matrices with shape [num_networks, num_nodes, num_nodes]

        Returns:
            Network property values with shape [num_networks, num_nodes]
        """
        pass


class CompositeCriterion(EvaluationCriterion, ABC):
    r"""Base class for composite evaluation criteria combining multiple metrics.

    This class allows combining multiple evaluation criteria into a single composite
    criterion. Subclasses define how to combine the individual criteria results
    (*e.g.*, maximum, mean, weighted sum).

    Args:
        criteria:
            List of EvaluationCriterion objects to combine

    Notes:
        All criteria in the list must accept the same type of network (binary or weighted).

    See Also:
        - [`evaluation.MaxCriteria`][gnm.evaluation.MaxCriteria]: Takes the maximum value across all criteria
        - [`evaluation.MeanCriteria`][gnm.evaluation.MeanCriteria]: Takes the mean value across all criteria
        - [`evaluation.WeightedSumCriteria`][gnm.evaluation.WeightedSumCriteria]: Takes a weighted sum of all criteria
    """

    def __init__(self, criteria: list[EvaluationCriterion]):
        r"""
        Args:
            criteria:
                List of EvaluationCriterion objects to combine. All criteria must
                accept the same type of network (binary or weighted).

        Raises:
            AssertionError: If no criteria are provided or if they accept different network type
        """
        assert len(criteria) > 0, "Must provide at least one criterion"
        self.criteria = criteria
        self.accepts = self.criteria[0].accepts
        assert all(
            criterion.accepts == self.accepts for criterion in self.criteria
        ), "All criteria must accept the same type of network"

    @jaxtyped(typechecker=typechecked)
    def _pre_call(
        self, matrices: Float[torch.Tensor, "num_networks num_nodes num_nodes"]
    ):
        r"""Perform validation checks using the first criterion in the list.

        Delegates validation to the first criterion in the list, assuming all
        criteria accept the same type of network.

        Args:
            matrices:
                Adjacency/weight matrices with shape [num_networks, num_nodes, num_nodes]
        """
        self.criteria[0]._pre_call(matrices)
