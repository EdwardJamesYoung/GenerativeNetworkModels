r"""Core implementation of generative network models.

This script implements the core classes for generative network models, including both binary
and weighted networks. It provides a unified framework for growing networks using generative
processes that combine physical distance constraints, topological similarity, and developmental
timing to create realistic brain-like networks.

The main classes are:
- BinaryGenerativeParameters: Controls how binary networks grow
- WeightedGenerativeParameters: Controls how weights evolve in weighted networks
- GenerativeNetworkModel: The main class implementing the generative network model
"""

from jaxtyping import Float, jaxtyped, Int
from typing import Optional, Tuple, Union, Any
from typeguard import typechecked
import gc

from .weight_criteria import OptimisationCriterion
from .generative_rules import GenerativeRule

import torch
import torch.optim as optim

from tqdm import tqdm
from dataclasses import dataclass

from gnm.utils import binary_checks, weighted_checks

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class BinaryGenerativeParameters:
    r"""Parameters controlling the binary generative network model's evolution.

    This dataclass encapsulates the parameters that determine how a binary generative
    network model grows and forms connections. The parameters control three main aspects
    of network generation:

    1. The influence of physical distance, $\eta$
    2. The influence of topological similarity, $\gamma$
    3. The influence of developmental factors, $\lambda$

    Each influence can be modeled using either a power law or exponential relationship,
    as specified by the relationship type parameters. The total wiring probability is
    proportional to the product of a distance factor $d_{ij}$, a preferential wiring
    factor $k_{ij}$, and a developmental factor $h_{ij}$:
    $$
        P_{ij} \propto d_{ij} \\times k_{ij} \\times h_{ij}
    $$

    Attributes:
        eta (float):
            Parameter ($\eta$) controlling the influence of Euclidean distances $D_{ij}$
            on wiring probability. More negative values indicate lower wiring probabilities
            between nodes that are futher away.

            - For power law: $d_{ij} = D_{ij}^{\eta}$
            - For exponential: $d_{ij} = \exp(\eta D_{ij})$

        gamma (float):
            Parameter ($\gamma$) controlling the influence of the preferential wiring rule $K_{ij}$
            on wiring probability. Larger values indicate stronger preference creating
            connections between nodes that have high preferential value.

            - For power law: $k_{ij} = K_{ij}^{\gamma}$
            - For exponential: $k_{ij} = \exp(\gamma K_{ij})$

        lambdah (float):
            Parameter ($\lambda$) controlling the influence of heterochronicity $H_{ij}$ on wiring
            probability. Larger values indicate stronger temporal dependence in development.

            - For power law: $h_{ij} = H_{ij}^{\lambda}$
            - For exponential: $h_{ij} = \exp(\lambda H_{ij})$

        distance_relationship_type (str):
            The relationship between distance $D_{ij}$ and distance factor $d_{ij}$.
            Must be one of ['powerlaw', 'exponential'].

        preference_relationship_type (str):
            The relationship between the generative rule output $K_{ij}$ and preferential wiring factor $k_{ij}$.
            Must be one of ['powerlaw', 'exponential'].

        heterochronicity_relationship_type (str):
            The relationship between heterochronicity $H_{ij}$ and developmental factor $h_{ij}$.
            Must be one of ['powerlaw', 'exponential'].

        generative_rule (GenerativeRule):
            The generative rule that transforms the adjacency matrix to a matching index matrix.
            This computes the preferential wiring rule $K_{ij}$ from the adjacency matrix $A_{ij}$.

        num_iterations (int):
            The number of iterations to train the model for.

        prob_offset (float, optional):
            Small constant added to unnormalized probabilities to prevent division by zero.
            Defaults to 1e-6.

        binary_updates_per_iteration (int, optional):
            The number of binary update steps to do per model iteration. Defaults to 1.


    Examples:
        >>> from gnm.generative_rules import MatchingIndex
        >>> from gnm import BinaryGenerativeParameters
        >>> binary_parameters = BinaryGenerativeParameters(
        ...     eta=1.0,
        ...     gamma=0.5,
        ...     lambdah=2.0,
        ...     distance_relationship_type='powerlaw',
        ...     preferential_relationship_type='exponential',
        ...     heterochronicity_relationship_type='powerlaw',
        ...     generative_rule=MatchingIndex(divisor='mean'),
        ...     num_iterations=400
        ... )

    See Also:
        - [`generative_rules.GenerativeRule`][gnm.generative_rules.GenerativeRule]: A base class for generative rules that transform an adjacency matrix $A_{ij}$ into a preferential wiring matrix $K_{ij}$
    """

    eta: float
    gamma: float
    lambdah: float
    distance_relationship_type: str
    preferential_relationship_type: str
    heterochronicity_relationship_type: str
    generative_rule: GenerativeRule
    num_iterations: int
    prob_offset: float = 1e-6
    binary_updates_per_iteration: int = 1

    def __post_init__(self):
        # Perform checks on the distance and matching index relationship type.
        if self.distance_relationship_type not in ["powerlaw", "exponential"]:
            raise NotImplementedError(
                f"Distance relationship type '{self.distance_relationship_type}' is not supported for the binary generative network model."
            )
        if self.preferential_relationship_type not in ["powerlaw", "exponential"]:
            raise NotImplementedError(
                f"Matching relationship type '{self.preferential_relationship_type}' is not supported for the binary generative network model."
            )
        if self.heterochronicity_relationship_type not in ["powerlaw", "exponential"]:
            raise NotImplementedError(
                f"Matching relationship type '{self.heterochronicity_relationship_type}' is not supported for the binary generative network model."
            )

    def __dict__(self) -> dict[str, Any]:
        return {
            "eta": self.eta,
            "gamma": self.gamma,
            "lambdah": self.lambdah,
            "distance_relationship_type": self.distance_relationship_type,
            "preferential_relationship_type": self.preferential_relationship_type,
            "heterochronicity_relationship_type": self.heterochronicity_relationship_type,
            "generative_rule": str(self.generative_rule),
            "prob_offset": self.prob_offset,
            "binary_updates_per_iteration": self.binary_updates_per_iteration,
            "num_iterations": self.num_iterations,
        }


@dataclass
class WeightedGenerativeParameters:
    r"""Parameters controlling the weighted generative network model's evolution.

    This dataclass encapsulates the parameters that determine how weights evolve in a
    weighted generative network model. While the binary parameters control network
    topology, these parameters control the optimisation of connection weights through
    gradient descent. The optimisation process minimises (or maximises) an objective
    function.

    At each step, the weights are updated according to:
    $$
    W_{ij} \gets W_{ij} - \\alpha \\frac{\partial L}{\partial W_{ij}},
    $$
    where $L$ is the optimisation criterion and $\alpha$ is the learning rate.
    Note that only those weights present in the binary network adjacency matrix $A_{ij}$
    are updated.
    Additionally, symmetry is enforced so that we always have $W_{ij} = W_{ji}$.

    Attributes:
        alpha (float):
            Learning rate ($\alpha$) for gradient descent optimisation of weights.
            Controls how much weights change in response to gradients:
            larger values mean bigger steps but potential instability,
            smaller values mean more stable but slower optimisation.

        optimisation_criterion (OptimisationCriterion):
            The objective function $L(W)$ to optimise. This determines what
            properties the final weight configuration will exhibit.
            See OptimisationCriterion class for available options like
            distance-weighted communicability or weighted distance.

        weight_lower_bound (float, optional):
            Minimum allowed value for any weight ($W_{\\rm lower}$). All weights
            will be clipped to stay above this value. Must be non-negative.
            Defaults to 0.0.

        weight_upper_bound (float, optional):
            Maximum allowed value for any weight ($W_{\\rm upper}$). All weights
            will be clipped to stay below this value. Must be greater
            than weight_lower_bound. Defaults to infinity.

        maximise_criterion (bool, optional):
            Whether to maximise rather than minimise the optimisation criterion.
            When True, gradients are flipped to ascend rather than descend.
            Defaults to False.

        weighted_updates_per_iteration (int, optional):
            The number of weight update steps to do per model iteration. Defaults to 1.


    Examples:
        >>> from gnm.weight_criteria import Communicability
        >>> from gnm import WeightedGenerativeParameters
        >>> communicability_optimisation_criterion = Communicability(normalisation=False, omega=1.0)
        >>> weighted_parameters = WeightedGenerativeParameters(
        ...     alpha=0.01,
        ...     optimisation_criterion=communicability_optimisation_criterion,
        ...     weight_lower_bound=0.0,
        ...     weight_upper_bound=1.0,
        ...     maximise_criterion=False,
        ...     weight_updates_per_iteration=100
        ... )

    See Also:
        - [`weight_criteria.OptimisationCriterion`][gnm.weight_criteria.OptimisationCriterion]: Base class for optimisation criteria for weighted networks
    """

    alpha: float
    optimisation_criterion: OptimisationCriterion
    weight_lower_bound: float = 0.0
    weight_upper_bound: float = float("inf")
    maximise_criterion: bool = False
    weight_updates_per_iteration: int = 1

    def __dict__(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "optimisation_criterion": str(self.optimisation_criterion),
            "weight_lower_bound": self.weight_lower_bound,
            "weight_upper_bound": self.weight_upper_bound,
            "maximise_criterion": self.maximise_criterion,
            "weight_updates_per_iteration": self.weight_updates_per_iteration,
        }


class GenerativeNetworkModel:
    r"""A class implementing both binary and weighted Generative Network Models (GNM).

    This class provides a unified framework for growing networks using both binary and weighted
    generative processes. The model works in two phases:

    **Binary Growth Phase:**
    The network's topology is determined by iteratively adding edges to an adjacency matrix
    $A_{ij}$ based on three factors (a) Physical distance between nodes, (b) Topological similarity
    (through the generative rule), (c) Developmental timing (heterochronicity).

    **Weight Optimisation Phase (Optional):**
    If weighted parameters are provided, the model also optimises edge weights $W_{ij}$
    through gradient descent on a loss, $L(W)$.

    Attributes:
        num_simulations (int):
            Number of simulations to run in parallel.
        seed_adjacency_matrix (torch.Tensor):
            Initial binary adjacency matrix (num_nodes, num_nodes).
        adjacency_matrix (torch.Tensor):
            Current state of the network's adjacency matrix.
        distance_matrix (torch.Tensor):
            Matrix of (Euclidean) distances between nodes.
        num_nodes (int):
            Number of nodes in the network.
        binary_parameters (BinaryGenerativeParameters):
            Parameters controlling binary network growth.
        distance_factor (torch.Tensor):
            Precomputed distance influence on edge formation.
        seed_weight_matrix (torch.Tensor, optional):
            Initial weight matrix if using weighted GNM.
        weight_matrix (torch.Tensor, optional):
            Current state of the weight matrix.
        weighted_parameters (WeightedGenerativeParameters, optional):
            Parameters controlling weight optimisation.
        optimiser (torch.optim.Optimizer, optional):
            Optimiser for weight updates.

    Examples:
        >>> from gnm import BinaryGenerativeParameters, WeightedGenerativeParameters, GenerativeNetworkModel
        >>> from gnm.defaults import get_distance_matrix
        >>> from gnm.generative_rules import Neighbours
        >>> from gnm.weight_criteria import WeightedDistance
        >>> binary_parameters = BinaryGenerativeParameters(
        ...     eta=1.0,
        ...     gamma=-0.5,
        ...     lambdah=1.0,
        ...     distance_relationship_type='exponential',
        ...     preferential_relationship_type='powerlaw',
        ...     heterochronicity_relationship_type='powerlaw',
        ...     generative_rule=Neighbours(),
        ...     num_iterations=250,
        ...     binary_updates_per_iteration=1,
        ... )
        >>> weighted_parameters = WeightedGenerativeParameters(
        ...     alpha=0.003,
        ...     optimisation_criterion=WeightedDistance(),
        ...     weighted_updates_per_iteration=200,
        ... )
        ... distance_matrix = get_distance_matrix()
        >>> model = GenerativeNetworkModel(
        ...     binary_parameters=binary_parameters,
        ...     num_simulations=100, # Run 100 networks in parallel
        ...     distance_matrix=distance_matrix,
        ...     weighted_parameters=weighted_parameters,
        ... )
        >>> model.run_model()

    See Also:
        - BinaryGenerativeParameters: Parameters controlling binary network growth
        - WeightedGenerativeParameters: Parameters controlling weight optimisation
    """

    @jaxtyped(typechecker=typechecked)
    def __init__(
        self,
        binary_parameters: BinaryGenerativeParameters,
        num_simulations: Optional[int] = None,
        num_nodes: Optional[int] = None,
        seed_adjacency_matrix: Optional[
            Union[
                Float[torch.Tensor, "num_simulations num_nodes num_nodes"],
                Float[torch.Tensor, "num_nodes num_nodes"],
            ]
        ] = None,
        distance_matrix: Optional[
            Union[
                Float[torch.Tensor, "num_simulations num_nodes num_nodes"],
                Float[torch.Tensor, "num_nodes num_nodes"],
            ]
        ] = None,
        weighted_parameters: Optional[WeightedGenerativeParameters] = None,
        seed_weight_matrix: Optional[
            Union[
                Float[torch.Tensor, "num_simulations num_nodes num_nodes"],
                Float[torch.Tensor, "num_nodes num_nodes"],
            ]
        ] = None,
        device: Optional[torch.device] = None,
        verbose: Optional[bool] = False,
    ):
        r"""The initialisation process for the Generative Network Model:

        1. Validates input matrices (symmetry, binary values, etc.).
        2. Stores the binary parameters and optionally the weighted parameters.
        3. Precomputes a distance factor matrix based on distance_relationship_type.
        4. If weighted parameters are provided, prepares the weight matrix and optimiser.

        Args:
            binary_parameters:
                Parameters controlling network growth.
            num_simulations:
                Number of simulations to run in parallel
            seed_adjacency_matrix:
                Initial network structure(s). Must be a binary symmetric matrix.
            distance_matrix:
                Physical distances between nodes. Must be symmetric and non-negative. If not provided,
                all distances are set to 1.
            weighted_parameters:
                Parameters controlling weight optimisation. If None, only binary growth is performed.
            seed_weight_matrix:
                Initial weight matrix for weighted networks(s). If None but weighted parameters
                are provided, a matrix matching the adjacency support is used.
            device:
                The device (either CPU or CUDA-based GPU) responsible for running model. Leave blank for auto.
            verbose:
                Explicitly output warnings and model fit progress. True = show outputs, False = silence outputs.
                False by default.

        Raises:
            ValueError: If input matrices don't meet requirements (binary, symmetric, etc.) or
                        if weight matrix doesn't match adjacency support.
        """

        self.verbose = verbose
        self.binary_parameters = binary_parameters

        if device is not None:
            self.device = device
        else:
            self.device = DEVICE

        # ---- Set the number of nodes in the network ----

        if num_nodes is not None:
            self.num_nodes = num_nodes
        elif distance_matrix is not None:
            self.vprint("Number of nodes unspecified. Extracting from distance matrix.")
            self.num_nodes = distance_matrix.shape[-1]
        elif seed_adjacency_matrix is not None:
            self.vprint(
                "Number of nodes unspecified. Extracting from seed adjacency matrix."
            )
            self.num_nodes = seed_adjacency_matrix.shape[-1]
        elif seed_weight_matrix is not None:
            self.vprint(
                "Number of nodes unspecified. Extracting from seed weight matrix."
            )
            self.num_nodes = seed_weight_matrix.shape[-1]
        else:
            raise ValueError(
                "Number of nodes unspecified. Please pass in either a distance matrix, seed adjacency matrix, seed weight matrix, or number of nodes."
            )

        self.vprint(f"Number of nodes set to {self.num_nodes}")

        # ---- Set the number of simulations to run ----

        self.num_simulations = None

        if num_simulations is not None:
            self.num_simulations = num_simulations

        if distance_matrix is not None and self.num_simulations is None:
            if len(distance_matrix.shape) == 3:
                self.vprint(
                    "Number of simulations unspecified. Extracting from distance matrix."
                )
                self.num_simulations = distance_matrix.shape[0]

        if seed_adjacency_matrix is not None and self.num_simulations is None:
            if len(seed_adjacency_matrix.shape) == 3:
                self.vprint(
                    "Number of simulations unspecified. Extracting from seed adjacency matrix."
                )
                self.num_simulations = seed_adjacency_matrix.shape[0]

        if seed_weight_matrix is not None and self.num_simulations is None:
            if len(seed_weight_matrix.shape) == 3:
                self.vprint(
                    "Number of simulations unspecified. Extracting from seed weight matrix."
                )
                self.num_simulations = seed_weight_matrix.shape[0]

        if self.num_simulations is None:
            self.vprint("Number of simulations unspecified. Defaulting to 1.")
            self.num_simulations = 1

        self.vprint(f"Number of simulations set to {self.num_simulations}")

        # ---- Perform reshaping and checks on the seed adjacency matrix ----

        if seed_adjacency_matrix is None:
            self.vprint("Seed adjacency matrix unspecified. Assuming empty network.")
            self.seed_adjacency_matrix = torch.zeros(
                (self.num_simulations, self.num_nodes, self.num_nodes),
                dtype=torch.float32,
                device=self.device,
            )
        elif len(seed_adjacency_matrix.shape) == 2:
            assert seed_adjacency_matrix.shape == (
                self.num_nodes,
                self.num_nodes,
            ), f"Seed adjacency matrix is incorrect shape. Expected ({self.num_nodes}, {self.num_nodes}), got {seed_adjacency_matrix.shape}"
            self.seed_adjacency_matrix = (
                seed_adjacency_matrix.unsqueeze(0)
                .expand(self.num_simulations, -1, -1)
                .to(self.device)
            )
        else:
            assert seed_adjacency_matrix.shape == (
                self.num_simulations,
                self.num_nodes,
                self.num_nodes,
            ), f"Seed adjacency matrix is incorrect shape. Expected ({self.num_simulations}, {self.num_nodes}, {self.num_nodes}), got {seed_adjacency_matrix.shape}"
            self.seed_adjacency_matrix = seed_adjacency_matrix.to(self.device)

        binary_checks(self.seed_adjacency_matrix)

        self.adjacency_matrix = self.seed_adjacency_matrix.clone()

        # ---- Perform reshaping and checks on the distance matrix ----

        if distance_matrix is None:
            self.vprint("Distance matrix unspecified. Assuming uniform distances.")
            self.distance_matrix = torch.ones(
                (self.num_simulations, self.num_nodes, self.num_nodes),
                dtype=torch.float32,
                device=self.device,
            )
            # Remove the diagonals
            self.distance_matrix[
                :, torch.arange(self.num_nodes), torch.arange(self.num_nodes)
            ] = 0.0
        elif len(distance_matrix.shape) == 2:
            assert distance_matrix.shape == (
                self.num_nodes,
                self.num_nodes,
            ), f"Distance matrix is incorrect shape. Expected ({self.num_nodes}, {self.num_nodes}), got {distance_matrix.shape}"

            self.distance_matrix = (
                distance_matrix.unsqueeze(0)
                .expand(self.num_simulations, -1, -1)
                .to(self.device)
            )

        else:
            assert distance_matrix.shape == (
                self.num_simulations,
                self.num_nodes,
                self.num_nodes,
            ), f"Distance matrix is incorrect shape. Expected ({self.num_simulations}, {self.num_nodes}, {self.num_nodes}), got {distance_matrix.shape}"

            self.distance_matrix = distance_matrix.to(self.device)

        weighted_checks(self.distance_matrix)

        # ---- Precompute distance factor for binary growth ----

        if self.binary_parameters.distance_relationship_type == "powerlaw":
            self.distance_factor = torch.pow(
                self.distance_matrix, self.binary_parameters.eta
            )
        elif self.binary_parameters.distance_relationship_type == "exponential":
            self.distance_factor = torch.exp(
                self.binary_parameters.eta * self.distance_matrix
            )
        else:
            raise ValueError(
                f"Unsupported distance relationship: {self.binary_parameters.distance_relationship_type}"
            )

        # ---- Weighted network initialisation ----

        if weighted_parameters is not None:
            self.weighted_initialisation(weighted_parameters, seed_weight_matrix)
        else:
            self.weighted_parameters = None
            self.seed_weight_matrix = None
            self.weight_matrix = None
            self.optimiser = None

        # ---- Numerical stability tracking ----
        self._clamp_count = 0
        self._clamp_warned = False

    def vprint(self, msg):
        if self.verbose:
            print(msg)

    @jaxtyped(typechecker=typechecked)
    def weighted_initialisation(
        self,
        weighted_parameters: WeightedGenerativeParameters,
        seed_weight_matrix: Optional[
            Union[
                Float[
                    torch.Tensor,
                    "{self.num_simulations} {self.num_nodes} {self.num_nodes}",
                ],
                Float[torch.Tensor, "{self.num_nodes} {self.num_nodes}"],
            ]
        ] = None,
        device: Optional[torch.device] = None,
    ):
        r"""Initialise the weight matrix and optimiser for the weighted GNM.
        If weighted parameters are not passed in during initialisation, this method
        must be called before any weighted updates can be performed.

        Args:
            weighted_parameters:
                Parameters controlling weight optimisation.
            seed_weight_matrix:
                A seed weight matrix to initialise $W_{ij}$.
                If this is not provided, then the weight matrix is initialised to the
                current adjacency matrix, $W_{ij} \gets A_{ij}$.
                If provided, the matrix must be symmetric, non-negative, and have support
                only where the adjacency matrix is non-zero.
                Defaults to None.

        Raises:
            ValueError: If the seed_weight_matrix is not symmetric, non-negative, or has
                        support where the adjacency matrix is zero.

        See Also:
            - WeightedGenerativeParameters: Parameters controlling weight optimisation
            - weighted_update: Method that uses these parameters
            - __init__: Initialisation method that calls this function if weighted_parameters are provided.
        """
        if device is not None:
            self.device = device
        elif self.device is None:
            self.device = DEVICE

        self.weighted_parameters = weighted_parameters

        # If user didn't provide seed_weight_matrix, initialise from adjacency.
        if seed_weight_matrix is None:
            self.vprint(
                "No seed weight matrix provided. Initialising from seed adjacency matrix."
            )
            seed_weight_matrix = self.adjacency_matrix.clone()
        elif len(seed_weight_matrix.shape) == 2:
            assert seed_weight_matrix.shape == (
                self.num_nodes,
                self.num_nodes,
            ), f"Seed weight matrix is incorrect shape. Expected ({self.num_nodes}, {self.num_nodes}), got {seed_weight_matrix.shape}"

            seed_weight_matrix = seed_weight_matrix.unsqueeze(0).expand(
                self.num_simulations, -1, -1
            )
        else:
            assert seed_weight_matrix.shape == (
                self.num_simulations,
                self.num_nodes,
                self.num_nodes,
            ), f"Seed weight matrix is incorrect shape. Expected ({self.num_simulations}, {self.num_nodes}, {self.num_nodes}), got {seed_weight_matrix.shape}"

        weighted_checks(seed_weight_matrix)

        # Check whether the seed weight matrix has support where the binary adjacency matrix is zero.
        if torch.any(seed_weight_matrix * (1 - self.adjacency_matrix) > 0):
            self.vprint(
                "Warning: Seed weight matrix has support where the adjacency matrix is zero. Setting these weights to zero."
            )
            seed_weight_matrix = seed_weight_matrix * self.adjacency_matrix

        self.seed_weight_matrix = seed_weight_matrix

        # Create a copy for the actual weight matrix that will be optimised.
        self.weight_matrix = (
            self.seed_weight_matrix.clone().to(self.device).requires_grad_(True)
        )

        # Initialise optimiser.
        self.optimiser = optim.SGD(
            [self.weight_matrix],
            lr=self.weighted_parameters.alpha,
            maximize=self.weighted_parameters.maximise_criterion,
        )

    @jaxtyped(typechecker=typechecked)
    def binary_update(
        self,
        heterochronous_matrix: Optional[
            Union[
                Float[
                    torch.Tensor,
                    "{self.num_simulations} {self.num_nodes} {self.num_nodes}",
                ],
                Float[torch.Tensor, "{self.num_nodes} {self.num_nodes}"],
            ]
        ] = None,
    ) -> Tuple[
        Int[
            torch.Tensor, "{self.num_simulations} 2"
        ],  # added edges for each simulation
        Float[
            torch.Tensor, "{self.num_simulations} {self.num_nodes} {self.num_nodes}"
        ],  # updated adjacency matrices
    ]:
        r"""
        Performs one update step of the adjacency matrix for the binary GNM.
        To perform an update, the model calculates the unnormalised wiring probabilities for each edge
        not currently present within the adjacency matrix (i.e., all notes with $A_{ij} = 0$).
        The wiring probability $(i,j)$ based on a distance factor $d_{ij}$, a preferential wiring
        factor $k_{ij}$, and a developmental factor $h_{ij}$.
        The unnormalised probability is proportional to the product of these factors:
        $$
        P_{ij} = d_{ij} \\times k_{ij} \\times h_{ij}
        $$
        These probabilities are then postprocessed by:

        1. Set the probability for all existing connections to be zero, $P_{ij} \gets P_{ij} \times (1 - A_{ij})$
        2. Set the probability of self-connections to be zero, $P_{ii} \gets 0$
        3. Add on a small offset to prevent division by zero, $P_{ij} \gets P_{ij} + \epsilon$
        4. Normalise the probabilities to sum to one, $P_{ij} \gets P_{ij} / \sum_{kl} P_{kl}$

        An edge $(a,b)$ is then sampled from the normalised probabilities.
        This edge is added to the adjacency matrix, $A_{ab} \gets 1, A_{ba} \gets 1$.
        If the model is weighted, the edge is also added to the weight matrix, $W_{ab} \gets 1, W_{ba} \gets 1$.

        Args:
            heterochronous_matrix:
                The heterochronous development matrix $H_{ij}$ for this time step. Can be provided
                for each simulation in the batch or as a single matrix to be used across all simulations.
                Defaults to None.

        Returns:
            added_edges: The edges that were added to each adjacency matrix in the batch
            adjacency_matrices: (A copy of) the updated adjacency matrices after the binary update

        See Also:
            - BinaryGenerativeParameters: Parameters controlling binary network growth
            - GenerativeRule: Base class for generative rules that transform an adjacency matrix $A_{ij}$ into a preferential wiring matrix $K_{ij}$
        """
        # Handle heterochronous matrix
        if heterochronous_matrix is None:
            heterochronous_matrix = torch.ones(
                (self.num_simulations, self.num_nodes, self.num_nodes),
                dtype=self.seed_adjacency_matrix.dtype,
            )

        # If heterochronous matrix isn't batched, expand it
        if len(heterochronous_matrix.shape) == 2:
            heterochronous_matrix = heterochronous_matrix.unsqueeze(0).expand(
                self.num_simulations, -1, -1
            )
        elif heterochronous_matrix.shape[0] != self.num_simulations:
            raise ValueError(
                f"Heterochronous matrix batch size ({heterochronous_matrix.shape[0]}) "
                f"does not match number of simulations ({self.num_simulations})"
            )

        heterochronous_matrix = heterochronous_matrix.to(self.device)

        # Implement generative rule - already handles batch dimensions
        affinity_matrix = self.binary_parameters.generative_rule(self.adjacency_matrix)

        # Add prob_offset to prevent zero to the power of negative number
        affinity_matrix[affinity_matrix == 0] += self.binary_parameters.prob_offset

        # Check that the affinity matrix is everywhere strictly positive:
        assert torch.all(
            affinity_matrix > 0
        ), f"Affinity matrix must be strictly positive. Probability offset term is {self.binary_parameters.prob_offset}."

        # Calculate factors - broadcasting handles batch dimensions
        if self.binary_parameters.preferential_relationship_type == "powerlaw":
            matching_factor = affinity_matrix.pow(self.binary_parameters.gamma)
        elif self.binary_parameters.preferential_relationship_type == "exponential":
            matching_factor = torch.exp(self.binary_parameters.gamma * affinity_matrix)
        else:
            raise ValueError(
                f"Preferential relationship type must be either 'powerlaw' or 'exponential'. Got {self.binary_parameters.preferential_relationship_type}"
            )

        if self.binary_parameters.heterochronicity_relationship_type == "powerlaw":
            heterochronous_factor = heterochronous_matrix.pow(
                self.binary_parameters.lambdah
            )
        elif self.binary_parameters.heterochronicity_relationship_type == "exponential":
            heterochronous_factor = torch.exp(
                self.binary_parameters.lambdah * heterochronous_matrix
            )
        else:
            raise ValueError(
                f"Heterochronicity relationship type must be either 'powerlaw' or 'exponential'. Got {self.binary_parameters.heterochronicity_relationship_type}"
            )

        # Calculate unnormalised wiring probabilities for each edge

        unnormalised_wiring_probabilities = (
            heterochronous_factor * self.distance_factor * matching_factor
        )

        # Add prob_offset to prevent division by zero
        unnormalised_wiring_probabilities += self.binary_parameters.prob_offset

        # Set probability for existing connections to zero
        unnormalised_wiring_probabilities = unnormalised_wiring_probabilities * (
            1 - self.adjacency_matrix
        )

        # Set diagonal to zero to prevent self-connections
        diagonal_indices = torch.arange(
            self.num_nodes, device=self.adjacency_matrix.device
        )
        unnormalised_wiring_probabilities[..., diagonal_indices, diagonal_indices] = 0.0

        # Clamp inf/nan values to prevent torch.multinomial errors.
        # inf can arise when extreme parameter values (e.g. eta, gamma ≈ ±10)
        # produce overflow in the powerlaw computations above.
        inf_mask = torch.isinf(unnormalised_wiring_probabilities)
        nan_mask = torch.isnan(unnormalised_wiring_probabilities)
        if inf_mask.any() or nan_mask.any():
            unnormalised_wiring_probabilities = torch.nan_to_num(
                unnormalised_wiring_probabilities, nan=0.0, posinf=1e30, neginf=0.0
            )
            self._clamp_count += 1
            if not self._clamp_warned:
                self._clamp_warned = True
                import warnings

                warnings.warn(
                    f"Numerical overflow detected in wiring probabilities "
                    f"(eta={self.binary_parameters.eta}, gamma={self.binary_parameters.gamma}, "
                    f"lambdah={self.binary_parameters.lambdah}, "
                    f"relationship_types={self.binary_parameters.distance_relationship_type}/"
                    f"{self.binary_parameters.preferential_relationship_type}/"
                    f"{self.binary_parameters.heterochronicity_relationship_type}). "
                    f"Values clamped to safe range. Further occurrences will be counted silently."
                )

        # Normalize the wiring probabilities for each simulation
        wiring_probability = (
            unnormalised_wiring_probabilities
            / unnormalised_wiring_probabilities.sum(dim=(-2, -1), keepdim=True)
        )

        # Sample edges for each simulation in the batch
        flattened_probs = wiring_probability.view(self.num_simulations, -1)
        edge_indices = torch.multinomial(flattened_probs, num_samples=1).squeeze(-1)

        # Convert to node pairs
        first_nodes = edge_indices // self.num_nodes
        second_nodes = edge_indices % self.num_nodes
        added_edges = torch.stack([first_nodes, second_nodes], dim=-1)

        # Add edges to adjacency matrices
        batch_indices = torch.arange(
            self.num_simulations, device=self.adjacency_matrix.device
        )
        self.adjacency_matrix[batch_indices, first_nodes, second_nodes] = 1.0
        self.adjacency_matrix[batch_indices, second_nodes, first_nodes] = 1.0

        # Add edges to weight matrices if they exist
        if self.weight_matrix is not None:
            self.weight_matrix.data[batch_indices, first_nodes, second_nodes] = 1.0
            self.weight_matrix.data[batch_indices, second_nodes, first_nodes] = 1.0

        # Return the added edges and copies of updated adjacency matrices
        return added_edges, self.adjacency_matrix.clone()

    @jaxtyped(typechecker=typechecked)
    def weighted_update(
        self,
    ) -> Float[
        torch.Tensor, "{self.num_simulations} {self.num_nodes} {self.num_nodes}"
    ]:
        r"""
        Performs one update step of the weight matrix $W_{ij}$ for the weighted GNM. The weights are updated
        using gradient descent on the specified optimisation criterion, with the learning rate $\\alpha$:
        $$
        W_{ij} \gets W_{ij} - \\alpha \\frac{\partial L}{\partial W_{ij}}
        $$
        Following the update step, the following postprocessing steps are performed:

        1. Symmetry: The weight matrix is made symmetric by averaging with its transpose, $W \gets (1/2)(W + W^T)$.
        2. Clipping: The weights are clipped to the specified bounds $W_{\rm lower} \leq W_{ij} \leq W_{\rm upper}$.
        3. Consistency with binary adjacency: All weights where the adjacency matrix is zero are set to zero, so that if $A_{ij} = 0$ then $W_{ij} = 0$.

        Raises:
            AttributeError: If the model does not have a weight matrix, optimisation criterion, or optimiser.

        Returns:
            weight_matrix: (A detached copy of) the updated weight matrix, $W_{ij}$
        """
        # Check that the model has a weight matrix, optimisation criterion, and optimiser
        if self.weighted_parameters is None:
            raise AttributeError(
                "Model does not have weighted update parameters. Cannot perform weighted updates. Please call the weighted_initialisation method first."
            )
        if self.weight_matrix is None:
            raise AttributeError(
                "Model does not have a weight matrix. Cannot perform weighted updates. Please call the weighted_initialisation method first."
            )
        if self.optimiser is None:
            raise AttributeError(
                "Model does not have an optimiser. Cannot perform weighted updates. Please call the weighted_initialisation method first."
            )

        # Perform the optimisation step on the weights.
        # Compute the loss - criterion outputs shape (num_simulations,)
        loss = self.weighted_parameters.optimisation_criterion(self.weight_matrix)
        # Sum over the batch to get a scalar loss
        total_loss = torch.sum(loss)

        # Compute the gradients
        self.optimiser.zero_grad()
        total_loss.backward()

        # Update the weights
        self.optimiser.step()

        # Ensure the weight matrix is symmetric
        self.weight_matrix.data = 0.5 * (
            self.weight_matrix.data + self.weight_matrix.data.transpose(-2, -1)
        )

        # Clip the weights to the specified bounds
        self.weight_matrix.data = torch.clamp(
            self.weight_matrix.data,
            self.weighted_parameters.weight_lower_bound,
            self.weighted_parameters.weight_upper_bound,
        )

        # Zero out all weights where the adjacency matrix is zero
        self.weight_matrix.data = self.weight_matrix.data * self.adjacency_matrix

        # Return the updated weight matrix
        return self.weight_matrix.detach().clone().cpu()

    @jaxtyped(typechecker=typechecked)
    def run_model(
        self,
        heterochronous_matrix: Optional[
            Union[
                Float[
                    torch.Tensor,
                    "num_binary_updates {self.num_simulations} {self.num_nodes} {self.num_nodes}",
                ],
                Float[
                    torch.Tensor, "num_binary_updates {self.num_nodes} {self.num_nodes}"
                ],
            ]
        ] = None,
    ) -> Tuple[
        Optional[
            Int[torch.Tensor, "num_binary_updates {self.num_simulations} 2"]
        ],  # added edges for each update
        Optional[
            Float[
                torch.Tensor,
                "num_binary_updates {self.num_simulations} {self.num_nodes} {self.num_nodes}",
            ]
        ],  # Adjacency snapshots
        Optional[
            Float[
                torch.Tensor,
                "num_weight_updates {self.num_simulations} {self.num_nodes} {self.num_nodes}",
            ]  # Weight snapshots
        ],
    ]:
        r"""Trains the network for a specified number of iterations.
        At each iteration, a number of binary updates and weighted updates are performed.

        Args:
            heterochronous_matrix:
                The heterochronous development probability matrix, $H_{ij}(t)$, for each binary update step $t$.
                Can be provided for each simulation in the batch or as a single matrix sequence to be used
                across all simulations. Defaults to None.

        Returns:
            added_edges: The edges $(a,b)$ that were added to the adjacency matrices $A_{ij}$ at each iteration.
            adjacency_snapshots: The adjacency matrices $A_{ij}$ at each binary update step.
            weight_snapshots: The weight matrices $W_{ij}$ at each iteration of the weighted updates.
        """
        num_iterations = self.binary_parameters.num_iterations
        added_edges_list = []
        binary_updates_per_iteration = (
            self.binary_parameters.binary_updates_per_iteration
        )
        total_binary_updates = num_iterations * binary_updates_per_iteration

        # Initialize snapshots with steps and batch dimensions at start
        adjacency_snapshots = (
            torch.zeros(
                (
                    total_binary_updates,
                    self.num_simulations,
                    self.num_nodes,
                    self.num_nodes,
                ),
                device=self.device,
                dtype=self.adjacency_matrix.dtype,
            )
            if total_binary_updates > 0
            else None
        )

        if self.weighted_parameters is not None:
            weight_updates_per_iteration = (
                self.weighted_parameters.weight_updates_per_iteration
            )
            total_weighted_updates = num_iterations * weight_updates_per_iteration
            weight_snapshots = torch.zeros(
                (
                    total_weighted_updates,
                    self.num_simulations,
                    self.num_nodes,
                    self.num_nodes,
                ),
                device=self.device,
                dtype=self.adjacency_matrix.dtype,
            )
        else:
            weight_updates_per_iteration = 0
            weight_snapshots = None

        # Handle heterochronous matrix
        if heterochronous_matrix is None:
            heterochronous_matrix = torch.ones(
                (
                    total_binary_updates,
                    self.num_nodes,
                    self.num_nodes,
                ),
                dtype=self.adjacency_matrix.dtype,
                device=self.device,
            )

        # Expand heterochronous matrix if it doesn't have batch dimension
        if len(heterochronous_matrix.shape) == 3:
            heterochronous_matrix = heterochronous_matrix.unsqueeze(1).expand(
                -1, self.num_simulations, -1, -1
            )
        elif heterochronous_matrix.shape[1] != self.num_simulations:
            raise ValueError(
                f"Heterochronous matrix batch size ({heterochronous_matrix.shape[1]}) "
                f"does not match number of simulations ({self.num_simulations})"
            )

        for ii in tqdm(range(num_iterations), disable=not self.verbose):
            for jj in range(binary_updates_per_iteration):
                update_idx = ii * binary_updates_per_iteration + jj
                added_edges, adjacency_matrix = self.binary_update(
                    heterochronous_matrix[update_idx]
                )
                adjacency_snapshots[update_idx] = adjacency_matrix
                added_edges_list.append(added_edges)

            for jj in range(weight_updates_per_iteration):
                update_idx = ii * weight_updates_per_iteration + jj
                weight_matrix = self.weighted_update()
                weight_snapshots[update_idx] = weight_matrix

        # stack the added edges
        added_edges = (
            torch.stack(added_edges_list, dim=0) if len(added_edges_list) > 0 else None
        )

        return added_edges, adjacency_snapshots, weight_snapshots

    def to_device(self, device: Union[torch.device, str]):
        r"""Move the model to a new device.

        Args:
            device: The device to move the model to.
        """
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.adjacency_matrix = self.adjacency_matrix.to(device)
        self.distance_matrix = self.distance_matrix.to(device)
        self.distance_factor = self.distance_factor.to(device)
        if self.seed_weight_matrix is not None:
            self.seed_weight_matrix = self.seed_weight_matrix.to(device)
        if self.weight_matrix is not None:
            self.weight_matrix = self.weight_matrix.to(device)

        gc.collect()
        torch.cuda.empty_cache()
