r"""Network evaluation criteria for comparing generated networks to real networks.

This subpackage provides classes for quantifying the similarity or dissimilarity
between synthetic (generated) networks and real (target) networks. These evaluation
criteria can be used to:

1. Assess the quality of generated networks.
2. Guide optimization processes to find optimal model parameters.
3. Compare different generative models parameters.

The module includes:

- Base classes for defining novel evaluation criteria.
- Criteria based on the Kolmogorov-Smirnov test comparing network property distributions.
- Correlation-based criteria comparing spatial patterns of network properties.
- Composite criteria combining multiple evaluation metrics.

Both binary (unweighted) and weighted network evaluations are supported.
"""

from .evaluation_base import (
    EvaluationCriterion,
    BinaryEvaluationCriterion,
    WeightedEvaluationCriterion,
    KSCriterion,
    CorrelationCriterion,
    CompositeCriterion,
)
from .composite_criteria import MaxCriteria, MeanCriteria, WeightedSumCriteria
from .binary_ks_criteria import (
    BetweennessKS,
    ClusteringKS,
    DegreeKS,
    EdgeLengthKS,
)
from .weighted_ks_criteria import (
    WeightedNodeStrengthKS,
    WeightedBetweennessKS,
    WeightedClusteringKS,
)

from .binary_corr_criteria import (
    DegreeCorrelation,
    BinaryBetweennessCorrelation,
    BinaryClusteringCorrelation,
)

from .weighted_corr_criteria import (
    StrengthCorrelation,
    WeightedBetweennessCorrelation,
    WeightedClusteringCorrelation,
)

__all__ = [
    "EvaluationCriterion",
    "BinaryEvaluationCriterion",
    "WeightedEvaluationCriterion",
    "KSCriterion",
    "CorrelationCriterion",
    "CompositeCriterion",
    "MaxCriteria",
    "MeanCriteria",
    "WeightedSumCriteria",
    "BetweennessKS",
    "ClusteringKS",
    "DegreeKS",
    "EdgeLengthKS",
    "WeightedNodeStrengthKS",
    "WeightedBetweennessKS",
    "WeightedClusteringKS",
    "DegreeCorrelation",
    "BinaryClusteringCorrelation",
    "BinaryBetweennessCorrelation",
    "StrengthCorrelation",
    "WeightedBetweennessCorrelation",
    "WeightedClusteringCorrelation",
]
