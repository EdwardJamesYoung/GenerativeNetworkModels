r"""Data structures for parameter exploration and experimental results storage.

This module defines dataclasses that represent parameter spaces, experiment configurations,
and results from generative network model simulations. These structures organize parameter
sweeps and store the outputs for subsequent analysis.
"""

import torch
from typing import List, Iterator, Optional, Any, Dict, Union
from itertools import product
from jaxtyping import Float, Int, jaxtyped
from typeguard import typechecked
from dataclasses import dataclass, field
import gc

# import wandb

from gnm import (
    BinaryGenerativeParameters,
    WeightedGenerativeParameters,
    GenerativeNetworkModel,
)
from gnm.generative_rules import GenerativeRule
from gnm.weight_criteria import OptimisationCriterion


@dataclass
class BinarySweepParameters:
    r"""Parameter space definition for binary generative network models.

    This class defines a multidimensional parameter space to explore for binary network
    generation. It contains lists of parameter values that will be combined to create
    different configurations of binary generative models.

    When iterated, this class yields all possible combinations of parameters as
    BinaryGenerativeParameters instances, creating a comprehensive parameter sweep.

    Attributes:
        eta (torch.Tensor):
            Parameter values ($\eta$) controlling the influence of Euclidean distances $D_{ij}$
            on wiring probability. More negative values indicate lower wiring probabilities
            between nodes that are further away.

        gamma (torch.Tensor):
            Parameter values ($\gamma$) controlling the influence of preferential attachment.
            Higher values increase the influence of the preferential attachment factor.

        distance_relationship_type (List[str]):
            Types of distance-dependent relationships to use (*e.g.*, "powerlaw", "exponential").
            Defines the functional form of distance dependence in the model.

        preferential_relationship_type (List[str]):
            Types of preferential attachment relationships to use (*e.g.*, "powerlaw", "exponential").
            Defines the functional form of degree dependence in the model.

        generative_rule (List[GenerativeRule]):
            Generative rules to use for network creation. These define the rule by which the
            preferential attachment factor is computed.

        num_iterations (List[int]):
            Numbers of iterations to run the generative process for each parameter set.
            Controls the number of new connections added to the network.

        lambdah (torch.Tensor):
            Parameter values ($\lambda$) controlling the influence of heterochronicity or
            temporal distance between nodes. Affects the probability of connections between
            nodes with different developmental timing. Defaults to [torch.Tensor([0])]. Set to 
            torch.Tensor([0]) to omit heterochronicity from the model.

        heterochronicity_relationship_type (List[str]):
            Types of heterochronicity relationships to use (*e.g.*, "powerlaw", "exponential").
            Defines the functional form of temporal dependence in the model. Defaults to ['powerlaw'].

        prob_offset (List[float]):
            Small probability offsets added to avoid numerical issues with zero probabilities.
            Defaults to [1e-6].

        binary_updates_per_iteration (List[int]):
            Number of connection updates to perform in each iteration of the binary network
            generation process. **Don't touch this unless you know what you're doing**.
            Defaults to [1].

    Examples:
        >>> import torch
        >>> from gnm.generative_rules import MatchingIndex
        >>> from gnm.fitting import BinarySweepParameters
        >>> # Define parameter ranges to explore
        >>> eta_values = torch.tensor([-3.0, -2.0, -1.0])
        >>> gamma_values = torch.tensor([0.1, 0.2, 0.3])
        >>> lambda_values = torch.tensor([0.0])
        >>> sweep_params = BinarySweepParameters(
        ...     eta=eta_values,
        ...     gamma=gamma_values,
        ...     lambdah=lambda_values,
        ...     distance_relationship_type=["powerlaw"],
        ...     preferential_relationship_type=["powerlaw"],
        ...     heterochronicity_relationship_type=["powerlaw"],
        ...     generative_rule=[MatchingIndex()],
        ...     num_iterations=[100],
        ... )
        >>> # Count total parameter combinations
        >>> len(list(sweep_params))
        9

    See Also:
        - [`model.BinaryGenerativeParameters`][gnm.model.BinaryGenerativeParameters]: Parameters for a single binary generative model configuration
        - [`fitting.SweepConfig`][gnm.fitting.SweepConfig]: Configuration for a complete parameter sweep
    """

    eta: Float[torch.Tensor, "eta_samples"]
    gamma: Float[torch.Tensor, "gamma_samples"]
    distance_relationship_type: List[str]
    preferential_relationship_type: List[str]
    generative_rule: List[GenerativeRule]
    num_iterations: List[int]
    lambdah: Float[torch.Tensor, "lambda_samples"] = field(default_factory=lambda: torch.Tensor([0.0]))
    heterochronicity_relationship_type: List[str] = field(default_factory=lambda: ["powerlaw"])
    prob_offset: List[float] = field(default_factory=lambda: [1e-6])
    binary_updates_per_iteration: List[int] = field(default_factory=lambda: [1])

    def __iter__(self) -> Iterator[BinaryGenerativeParameters]:
        r"""Creates an iterator over all combinations of parameters.
        Each combination is used to create a BinaryGenerativeParameters instance.

        Returns:
            Iterator yielding BinaryGenerativeParameters instances, one for each
            combination of parameters.
        """
        # Get all parameter names and their corresponding lists
        param_names = [
            "eta",
            "gamma",
            "lambdah",
            "distance_relationship_type",
            "preferential_relationship_type",
            "heterochronicity_relationship_type",
            "generative_rule",
            "num_iterations",
            "prob_offset",
            "binary_updates_per_iteration",
        ]
        param_lists = [getattr(self, name) for name in param_names]

        for values in product(*param_lists):
            params = dict(zip(param_names, values))
            yield BinaryGenerativeParameters(**params)


@dataclass
class WeightedSweepParameters:
    r"""Parameter space definition for weighted generative network models.

    This class defines a multidimensional parameter space to explore for the weight
    optimization phase of weighted network generation. It contains lists of parameter
    values that will be combined to create different configurations of weighted
    generative models.

    When iterated, this class yields all possible combinations of parameters as
    WeightedGenerativeParameters instances, creating a comprehensive parameter sweep.

    Attributes:
        alpha (torch.Tensor):
            Parameter values ($\alpha$) controlling the size of the gradient step during
            weight optimization. Higher values lead to larger updates to edge weights.

        optimisation_criterion (List[OptimisationCriterion]):
            Optimisation criteria to use for weight optimization. These define the objective
            function on which gradients are taken during the weight optimization process.

        weight_lower_bound (List[float]):
            Lower bounds for edge weights. Defaults to [0.0].

        weight_upper_bound (List[float]):
            Upper bounds for edge weights. Defaults to [inf].

        maximise_criterion (List[bool]):
            Whether to maximise the optimisation criterion. Defaults to [False].

        weight_updates_per_iteration (List[int]):
            Number of weight updates to perform in each iteration of the weight optimisation
            process. Defaults to [1].

    Examples:
        >>> import torch
        >>> from gnm.weight_criteria import Communicability
        >>> from gnm.fitting import WeightedSweepParameters
        >>> # Define parameter ranges to explore
        >>> sweep_params = WeightedSweepParameters(
        ...     alpha=torch.tensor([0.01, 0.05, 0.1]),
        ...     optimisation_criterion=[Communicability(omega=1.0)],
        ...     maximise_criterion=[True, False],
        ...     weight_updates_per_iteration=[1, 5, 10],
        ... )
        >>> # Count total parameter combinations
        >>> len(list(sweep_params))
        18

    See Also:
        - [`model.WeightedGenerativeParameters`][gnm.model.WeightedGenerativeParameters]: Parameters for a single weighted generative model configuration
        - [`fitting.SweepConfig`][gnm.fitting.SweepConfig]: Configuration for a complete parameter sweep
    """

    alpha: Float[torch.Tensor, "alpha_samples"]
    optimisation_criterion: List[OptimisationCriterion]
    weight_lower_bound: List[float] = field(default_factory=lambda: [0.0])
    weight_upper_bound: List[float] = field(default_factory=lambda: [float("inf")])
    maximise_criterion: List[bool] = field(default_factory=lambda: [False])
    weight_updates_per_iteration: List[int] = field(default_factory=lambda: [1])

    def __iter__(self) -> Iterator[WeightedGenerativeParameters]:
        r"""Creates an iterator over all combinations of parameters.
        Each combination is used to create a WeightedGenerativeParameters instance.

        Returns:
            Iterator yielding WeightedGenerativeParameters instances, one for each
            combination of parameters.
        """
        # Get all parameter names and their corresponding lists
        param_names = [
            "alpha",
            "optimisation_criterion",
            "weight_lower_bound",
            "weight_upper_bound",
            "maximise_criterion",
            "weight_updates_per_iteration",
        ]
        param_lists = [getattr(self, name) for name in param_names]

        for values in product(*param_lists):
            params = dict(zip(param_names, values))
            yield WeightedGenerativeParameters(**params)


@dataclass
class RunConfig:
    r"""Configuration for a single generative network model run.

    This class encapsulates all the parameters and inputs needed for a single run of
    the generative network model. It contains both binary and (optionally) weighted
    parameters, as well as input matrices like distance matrices and seed networks.

    Attributes:
        binary_parameters (BinaryGenerativeParameters):
            Parameters for binary network generation. These define the rules and
            relationships used to generate binary networks.

        num_simulations (int):
            Number of simulations to run in parallel. Each simulation generates a
            separate network using the same parameters.

        seed_adjacency_matrix (Optional[Float[torch.Tensor, "num_simulations num_nodes num_nodes"]]):
            Seed adjacency matrix for the binary network generation process. If provided,
            this matrix is used as the starting point for network generation. If unspecified,
            the network is generated from scratch.

        distance_matrix (Optional[Float[torch.Tensor, "num_nodes num_nodes"]]):
            Distance matrix for the network. This matrix defines the spatial relationships
            between nodes and is used in the generative process. If unspecified, constant distances
            are used.

        weighted_parameters (Optional[WeightedGenerativeParameters]):
            Parameters for weight optimization. If provided, the model will perform a
            weight optimization phase after generating the binary network. If unspecified,
            the model will only generate binary networks.

        seed_weight_matrix (Optional[Float[torch.Tensor, "num_simulations num_nodes num_nodes"]]):
            Seed weight matrix for the weight optimization process. If provided, this matrix
            is used as the starting point for weight optimization. If unspecified, the weights
            are optimised from scratch.

        heterochronous_matrix (Optional[Float[torch.Tensor, "num_binary_updates num_nodes num_nodes"]]):
            The heterochronous development matrix for each binary update step. Can be provided
            for each simulation in the batch or as a single matrix to be used across all simulations.
            Defaults to None, which means that there is no heterochronicity.

    Examples:
        >>> from gnm import BinaryGenerativeParameters
        >>> from gnm.generative_rules import ClusteringMin
        >>> from gnm.fitting import RunConfig
        >>> from gnm.defaults import get_distance_matrix
        >>> # Create binary parameters
        >>> binary_params = BinaryGenerativeParameters(
        ...     eta=-2.0,
        ...     gamma=0.3,
        ...     lambdah=0.0,
        ...     distance_relationship_type="powerlaw",
        ...     preferential_relationship_type="powerlaw",
        ...     heterochronicity_relationship_type="powerlaw",
        ...     generative_rule=ClusteringMin(),
        ...     num_iterations=200,
        ... )
        >>> # Create run configuration
        >>> config = RunConfig(
        ...     binary_parameters=binary_params,
        ...     num_simulations=100,
        ...     distance_matrix=get_distance_matrix(),
        ... )

    See Also:
        - [`model.BinaryGenerativeParameters`][gnm.model.BinaryGenerativeParameters]: Parameters for binary network generation
        - [`model.WeightedGenerativeParameters`][gnm.model.WeightedGenerativeParameters]: Parameters for weight optimization
        - [`fitting.SweepConfig`][gnm.fitting.SweepConfig]: Configuration for a parameter sweep containing multiple run configurations
        - [`fitting.perform_run`][gnm.fitting.perform_run]: Function that executes a run using this configuration
    """

    binary_parameters: BinaryGenerativeParameters
    num_simulations: Optional[int] = None
    seed_adjacency_matrix: Optional[
        Union[
            Float[torch.Tensor, "num_simulations num_nodes num_nodes"],
            Float[torch.Tensor, "num_nodes num_nodes"],
        ]
    ] = None
    distance_matrix: Optional[Float[torch.Tensor, "num_nodes num_nodes"]] = None
    weighted_parameters: Optional[WeightedGenerativeParameters] = None
    seed_weight_matrix: Optional[
        Union[
            Float[torch.Tensor, "num_simulations num_nodes num_nodes"],
            Float[torch.Tensor, "num_nodes num_nodes"],
        ]
    ] = None
    heterochronous_matrix: Optional[
        Union[
            Float[torch.Tensor, "num_simulations num_nodes num_nodes"],
            Float[torch.Tensor, "num_nodes num_nodes"],
        ]
    ] = None

    def to_device(self, device: Union[torch.device, str]):
        r"""Moves all tensors in the run configuration to a specified device.

        Args:
            device: The device to move all tensors to.
        """
        if self.seed_adjacency_matrix is not None:
            self.seed_adjacency_matrix = self.seed_adjacency_matrix.to(device)
        if self.distance_matrix is not None:
            self.distance_matrix = self.distance_matrix.to(device)
        if self.seed_weight_matrix is not None:
            self.seed_weight_matrix = self.seed_weight_matrix.to(device)
        if self.heterochronous_matrix is not None:
            self.heterochronous_matrix = self.heterochronous_matrix.to(device)


@dataclass
class SweepConfig:
    r"""Configuration for a comprehensive parameter sweep.

    This class defines a complete parameter sweep by combining binary parameter spaces,
    weighted parameter spaces, and various input matrices. When iterated, it yields
    RunConfig instances for each unique parameter combination in the sweep.

    The sweep includes all combinations of binary parameters, weighted parameters
    (if provided), and input matrices, creating a thorough exploration of the
    parameter space.

    Attributes:
        binary_sweep_parameters (BinarySweepParameters):
            Parameters for binary network generation. These define the rules and
            relationships used to generate binary networks.

        num_simulations (Optional[int]):
            Number of simulations to run in parallel. Each simulation generates a
            separate network using the same parameters.

        seed_adjacency_matrix (Optional[List[Float[torch.Tensor, "num_simulations num_nodes num_nodes"]]]):
            Seed adjacency matrices for the binary network generation process. If provided,
            these matrices are used as the starting points for network generation. If unspecified,
            the networks are generated from scratch.

        distance_matrices (Optional[List[Float[torch.Tensor, "num_nodes num_nodes"]]]):
            Distance matrices for the network. These matrices define the spatial relationships
            between nodes and are used in the generative process. If unspecified, constant distances
            are used.

        weighted_sweep_parameters (Optional[WeightedSweepParameters]):
            Parameters for weight optimisation. If provided, the model will perform a
            weight optimisation phase after generating the binary network. If unspecified,
            the model will only generate binary networks.

        seed_weight_matrix (Optional[List[Float[torch.Tensor, "num_simulations num_nodes num_nodes"]]]):
            Seed weight matrices for the weight optimisation process. If provided, these matrices
            are used as the starting points for weight optimisation. If unspecified, the weights
            are optimised from scratch.

        heterochronous_matrix (Optional[List[Float[torch.Tensor, "num_binary_updates num_simulations num_nodes num_nodes"]]]):
            The heterochronous development matrices for each binary update step. Can be provided
            for each simulation in the batch or as a single matrix to be used across all simulations.
            Defaults to None, which means that there is no heterochronicity.

    Examples:
        >>> import torch
        >>> from gnm.generative_rules import MatchingIndex
        >>> from gnm.weight_criteria import NormalisedCommunicability, WeightedDistance
        >>> from gnm.fitting import BinarySweepParameters, WeightedSweepParameters, SweepConfig
        >>> from gnm.defaults import get_distance_matrix
        >>> # Define binary parameter space
        >>> binary_sweep = BinarySweepParameters(
        ...     eta=torch.tensor([-3.0, -2.0, -1.0]),
        ...     gamma=torch.tensor([0.2]),
        ...     lambdah=torch.tensor([0.0]),
        ...     distance_relationship_type=["powerlaw"],
        ...     preferential_relationship_type=["powerlaw"],
        ...     heterochronicity_relationship_type=["powerlaw"],
        ...     generative_rule=[MatchingIndex()],
        ...     num_iterations=[100],
        ... )
        >>> # Define weighted parameter space
        >>> weighted_sweep = WeightedSweepParameters(
        ...     alpha=torch.tensor([0.01, 0.02, 0.03, 0.04, 0.05]),
        ...     optimisation_criterion=[NormalisedCommunicability(), WeightedDistance()],
        ... )
        >>> # Create sweep configuration
        >>> sweep_config = SweepConfig(
        ...     binary_sweep_parameters=binary_sweep,
        ...     weighted_sweep_parameters=weighted_sweep,
        ...     num_simulations=10,
        ...     distance_matrices=[get_distance_matrix()],
        ... )
        >>> # Count total run configurations
        >>> len(list(sweep_config))
        30

    See Also:
        - [`fitting.BinarySweepParameters`][gnm.fitting.BinarySweepParameters]: Parameter space for binary models
        - [`fitting.WeightedSweepParameters`][gnm.fitting.WeightedSweepParameters]: Parameter space for weighted models
        - [`fitting.RunConfig`][gnm.fitting.RunConfig]: Configuration for a single run
        - [`fitting.perform_sweep`][gnm.fitting.perform_sweep]: Function that executes a parameter sweep using this configuration
    """

    binary_sweep_parameters: BinarySweepParameters
    num_simulations: Optional[int] = None
    seed_adjacency_matrix: Optional[
        List[
            Union[
                Float[torch.Tensor, "num_simulations num_nodes num_nodes"],
                Float[torch.Tensor, "num_nodes num_nodes"],
            ]
        ]
    ] = None
    distance_matrix: Optional[List[Float[torch.Tensor, "num_nodes num_nodes"]]] = None
    weighted_sweep_parameters: Optional[WeightedSweepParameters] = None
    seed_weight_matrix: Optional[
        List[
            Union[
                Float[torch.Tensor, "num_simulations num_nodes num_nodes"],
                Float[torch.Tensor, "num_nodes num_nodes"],
            ]
        ]
    ] = None
    heterochronous_matrix: Optional[
        List[
            Union[
                Float[
                    torch.Tensor,
                    "num_binary_updates num_simulations num_nodes num_nodes",
                ],
                Float[torch.Tensor, "num_binary_updates num_nodes num_nodes"],
            ]
        ]
    ] = None

    def __iter__(self) -> Iterator[RunConfig]:
        r"""Creates an iterator over all combinations of run parameters.
        Yields pairs of parameter objects representing every possible combination
        of parameters

        Returns:
            Iterator yielding dictionaries of parameters
        """
        # Create lists for optional parameters, using empty list if None
        seed_adj_list = (
            self.seed_adjacency_matrix
            if self.seed_adjacency_matrix is not None
            else [None]
        )
        distance_list = (
            self.distance_matrix if self.distance_matrix is not None else [None]
        )
        heterochronous_list = (
            self.heterochronous_matrix
            if self.heterochronous_matrix is not None
            else [None]
        )
        seed_weight_list = (
            self.seed_weight_matrix if self.seed_weight_matrix is not None else [None]
        )

        # Get weighted parameters iterator if it exists, otherwise use single None
        weighted_sweep_parameters = (
            iter(self.weighted_sweep_parameters)
            if self.weighted_sweep_parameters is not None
            else [None]
        )

        # Create product of all parameter combinations
        for params in product(
            iter(self.binary_sweep_parameters),
            seed_adj_list,
            distance_list,
            weighted_sweep_parameters,
            seed_weight_list,
            heterochronous_list,
        ):
            # Unpack the values
            (
                binary_params,
                seed_adj,
                distance_matrix,
                weighted_parameters,
                seed_weights,
                heterochronous_matrix,
            ) = params

            run_config = RunConfig(
                binary_parameters=binary_params,
                num_simulations=self.num_simulations,
                seed_adjacency_matrix=seed_adj,
                distance_matrix=distance_matrix,
                weighted_parameters=weighted_parameters,
                seed_weight_matrix=seed_weights,
                heterochronous_matrix=heterochronous_matrix,
            )

            yield run_config


@dataclass
class EvaluationResults:
    r"""Storage for network evaluation results.

    This class stores the results of evaluating generated networks against real networks
    using various evaluation criteria. It contains separate dictionaries for binary and
    weighted evaluation results, where each entry maps a criterion name to a tensor of
    evaluation scores.

    Each evaluation tensor has shape [num_real_networks, num_simulations], containing
    the evaluation score for each combination of real network and simulated network.

    Attributes:
        binary_evaluations (Dict[str, Float[torch.Tensor, "num_real_binary_networks num_simulations"]]):
            Dictionary of binary evaluation results. Each entry maps a criterion name to
            a tensor of evaluation scores for binary networks.

        weighted_evaluations (Dict[str, Float[torch.Tensor, "num_real_weighted_networks num_simulations"]]):
            Dictionary of weighted evaluation results. Each entry maps a criterion name to
            a tensor of evaluation scores for weighted networks.

    Examples:
        >>> import torch
        >>> from gnm.fitting import EvaluationResults
        >>> from gnm.evaluation import DegreeKS
        >>> from gnm.defaults import get_binary_network
        >>> from gnm.utils import get_control
        >>> # Create evaluation results
        >>> real_matrices = get_binary_network()
        >>> control_matrices = get_control(real_matrices)
        >>> degree_ks_eval = DegreeKS()
        >>> binary_evaluations = {str(degree_ks_eval): degree_ks_eval(real_matrices, control_matrices)}
        >>> results = EvaluationResults(
        ...     binary_evaluations=binary_evaluations,
        ...     weighted_evaluations={},
        ... )
        >>> # Get evaluation scores for a specific criterion
        >>> results.binary_evaluations[str(degree_ks_eval)]


    See Also:
        - [`evaluation.BinaryEvaluationCriterion`][gnm.evaluation.BinaryEvaluationCriterion]: Criteria for evaluating binary networks
        - [`evaluation.WeightedEvaluationCriterion`][gnm.evaluation.WeightedEvaluationCriterion]: Criteria for evaluating weighted networks
        - [`fitting.perform_evaluations`][gnm.fitting.perform_evaluations]: Function that generates evaluation results
    """

    binary_evaluations: Dict[
        str, Float[torch.Tensor, "num_real_binary_networks num_simulations"]
    ]
    weighted_evaluations: Dict[
        str, Float[torch.Tensor, "num_real_weighted_networks num_simulations"]
    ]

    def to_device(self, device: Union[torch.device, str]):
        r"""Moves the evalution results to a specified device.

        Args:
            device: The device to move all tensors to.
        """
        for key, value in self.binary_evaluations.items():
            self.binary_evaluations[key] = value.to(device)
        for key, value in self.weighted_evaluations.items():
            self.weighted_evaluations[key] = value.to(device)


@dataclass
class RunHistory:
    r"""Record of network evolution during a model run.

    This class stores the history of how networks evolved during the generative process.
    It records which edges were added at each step and maintains snapshots of the adjacency
    and weight matrices at regular intervals.

    This history can be used to visualize network growth, analyse the order in which
    connections formed, and track how weights evolved over time.

    Attributes:
        added_edges (Int[torch.Tensor, "num_binary_updates num_simulations 2]):
            Tensor containing the edges added at each binary update step. Each row
            corresponds to a single update, with columns [source, target] indicating
            the nodes that were connected in that step.

        adjacency_snapshots (Float[torch.Tensor, "num_binary_updates num_simulations num_nodes num_nodes"]):
            Tensor containing snapshots of the adjacency matrix at each binary update step.

        weight_snapshots (Optional[Float[torch.Tensor, "num_weight_updates num_simulations num_nodes num_nodes"]]):
            Tensor containing snapshots of the weight matrix at each weight update step.
            If the model did not perform weight optimisation, this tensor is None.

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
        >>> added_edges, adjacency_snapshots, weight_snapshots = model.run_model()
        >>> history = RunHistory(
        ...     added_edges=added_edges,
        ...     adjacency_snapshots=adjacency_snapshots,
        ...     weight_snapshots=weight_snapshots,
        ... )


    See Also:
        - [`model.GenerativeNetworkModel`][gnm.model.GenerativeNetworkModel]: The model that generates this history
        - [`fitting.perform_run`][gnm.fitting.perform_run]: Function that produces run histories
    """

    added_edges: Optional[Int[torch.Tensor, "num_binary_updates num_simulations 2"]]
    adjacency_snapshots: Optional[
        Float[torch.Tensor, "num_binary_updates num_simulations num_nodes num_nodes"]
    ]
    weight_snapshots: Optional[
        Float[torch.Tensor, "num_weight_updates num_simulations num_nodes num_nodes"]
    ]

    def to_device(self, device: Union[torch.device, str]):
        r"""Moves the run history to a specified device.

        Args:
            device: The device to move all tensors to.
        """
        if self.added_edges is not None:
            self.added_edges = self.added_edges.to(device)
        if self.adjacency_snapshots is not None:
            self.adjacency_snapshots = self.adjacency_snapshots.to(device)
        if self.weight_snapshots is not None:
            self.weight_snapshots = self.weight_snapshots.to(device)


@dataclass
class Experiment:
    r"""Complete record of a generative network model experiment.

    This class encapsulates the entire experiment, including the configuration used,
    the results of evaluations, the model instance, and the history of network evolution.
    It provides a comprehensive record that can be saved, loaded, and analysed.

    The `to_device` method allows moving all tensors in the experiment to a specified
    device, which is useful for efficient computation or visualization.

    Attributes:
        run_config (RunConfig):
            Configuration for the experiment, including parameters and input matrices.

        evaluation_results (EvaluationResults):
            Results of evaluating the generated networks against real networks.

        model (Optional[GenerativeNetworkModel]):
            Instance of the generative network model used in the experiment. If the model
            was not saved, this field is None.

        run_history (Optional[RunHistory]):
            History of network evolution during the model run. If the history was not saved,
            this field is None.

    Examples:
        >>> from gnm.fitting import RunConfig, EvaluationResults, Experiment
        >>> from gnm import GenerativeNetworkModel, BinaryGenerativeParameters
        >>> from gnm.generative_rules import MatchingIndex
        >>> # Create minimal example (without actual data)
        >>> config = RunConfig(
        ...     binary_parameters=BinaryGenerativeParameters(
        ...         eta=-2.0,
        ...         gamma=0.3,
        ...         lambdah=0.0,
        ...         distance_relationship_type="powerlaw",
        ...         preferential_relationship_type="powerlaw",
        ...         heterochronicity_relationship_type="powerlaw",
        ...         generative_rule=MatchingIndex(),
        ...         num_iterations=100,
        ...     )
        ... )
        >>> results = EvaluationResults(
        ...     binary_evaluations={},
        ...     weighted_evaluations={},
        ... )
        >>> experiment = Experiment(
        ...     run_config=config,
        ...     evaluation_results=results,
        ... )

    See Also:
        - [`fitting.RunConfig`][gnm.fitting.RunConfig]: Configuration for the experiment
        - [`fitting.EvaluationResults`][gnm.fitting.EvaluationResults]: Results of network evaluations
        - [`fitting.RunHistory`][gnm.fitting.RunHistory]: History of network evolution
        - [`fitting.perform_run`][gnm.fitting.perform_run]: Function that produces experiments
    """

    run_config: RunConfig
    evaluation_results: EvaluationResults
    model: Optional[GenerativeNetworkModel] = None
    run_history: Optional[RunHistory] = None

    def to_device(self, device: Union[torch.device, str]):
        r"""Move all tensors in the experiment, including the model, to a specified device.

        Args:
            device: The device to move all tensors to.
        """
        self.evaluation_results.to_device(device)
        self.run_config.to_device(device)
        if self.model is not None:
            self.model.to_device(device)
        if self.run_history is not None:
            self.run_history.to_device(device)

        gc.collect()
        torch.cuda.empty_cache()
