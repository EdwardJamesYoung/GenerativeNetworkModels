r"""Default data and resources for generative network models.

This subpackage provides access to pre-packaged datasets that can be used
for experimenting with generative network models without requiring external data.
These defaults include:

- Distance matrices: Physical distances between brain regions
- Coordinates: 3D spatial positions of brain regions
- Binary networks: Example structural connectivity networks (presence/absence of connections)
- Weighted networks: Example weighted connectivity networks

The module provides simple functions to list available datasets and load them
with appropriate tensor formats for immediate use in network modeling.

Functions:
    display_available_defaults: Show all available default datasets
    get_distance_matrix: Load a default distance matrix
    get_coordinates: Load default 3D coordinates
    get_binary_network: Load a default binary network
    get_weighted_network: Load a default weighted network
"""

import torch
import json
import os
import pickle
from jaxtyping import Float, jaxtyped
from typing import Optional
from typeguard import typechecked

from gnm.utils import binary_checks, weighted_checks
from gnm.fitting.experiment_dataclasses import Experiment

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BASE_PATH = os.path.dirname(__file__)


def display_available_defaults():
    r"""Print all available default datasets that can be loaded.

    This function prints a formatted list of all available default datasets organized
    by category:

    - Distance matrices
    - Coordinates
    - Binary networks
    - Weighted networks

    Each category displays the names of available files that can be loaded with the
    corresponding get_* functions.

    Examples:
        >>> from gnm.defaults import display_available_defaults
        >>> display_available_defaults()
        === Distance matrices ===
        AAL_DISTANCES
        === Coordinates ===
        AAL_COORDINATES
        === Binary networks ===
        CALM_BINARY_CONSENSUS
        === Weighted networks ===
        CALM_WEIGHTED_CONSENSUS

    See Also:
        - [`defaults.get_distance_matrix`][gnm.defaults.get_distance_matrix]: Load a specific distance matrix
        - [`defaults.get_coordinates`][gnm.defaults.get_coordinates]: Load specific coordinate data
        - [`defaults.get_binary_network`][gnm.defaults.get_binary_network]: Load a specific binary network
        - [`defaults.get_weighted_network`][gnm.defaults.get_weighted_network]: Load a specific weighted network
    """

    print("=== Distance matrices ===")
    distance_matrices_path = os.path.join(BASE_PATH, "distance_matrices")
    for file in os.listdir(distance_matrices_path):
        print(file.split(".")[0])
    print("=== Coordinates ===")
    coordinates_path = os.path.join(BASE_PATH, "coordinates")
    for file in os.listdir(coordinates_path):
        print(file.split(".")[0])
    print("=== Binary networks ===")
    binary_consensus_networks_path = os.path.join(BASE_PATH, "binary_networks")
    for file in os.listdir(binary_consensus_networks_path):
        print(file.split(".")[0])
    print("=== Weighted networks ===")
    weighted_consensus_networks_path = os.path.join(BASE_PATH, "weighted_networks")
    for file in os.listdir(weighted_consensus_networks_path):
        print(file.split(".")[0])
    print("=== Experiment examples ===")
    experiment_examples_path = os.path.join(BASE_PATH, "experiment_dataclass_examples")
    for file in os.listdir(experiment_examples_path):
        print(file.split(".")[0])


@jaxtyped(typechecker=typechecked)
def get_distance_matrix(
    name: Optional[str] = None, device: Optional[torch.device] = None
) -> Float[torch.Tensor, "num_nodes num_nodes"]:
    r"""Load a default distance matrix.

    Provides access to pre-packaged distance matrices that represent physical distances
    between brain regions in standard atlas parcellations.

    Available distance matrices:

    - AAL_DISTANCES: Distance matrix for the Automated Anatomical Labeling atlas

    Args:
        name: Name of the distance matrix to load. If unspecified,
            the AAL_DISTANCES distance matrix is loaded.
        device: Device to load the distance matrix on. If unspecified,
            automatically uses CUDA if available, otherwise CPU.

    Returns:
        A Pytorch tensor containing the requested distance matrix with shape [num_nodes, num_nodes].

    Examples:
        >>> from gnm.defaults import get_distance_matrix
        >>> # Load default distance matrix
        >>> dist_matrix = get_distance_matrix()
        >>> # Load a specific distance matrix and place on CPU
        >>> import torch
        >>> dist_matrix = get_distance_matrix(name="AAL_DISTANCES", device=torch.device("cpu"))
        >>> dist_matrix.shape
        torch.Size([90, 90])

    See Also:
        - [`defaults.get_coordinates`][gnm.defaults.get_coordinates]: For loading spatial coordinates of nodes
        - [`defaults.get_binary_network`][gnm.defaults.get_binary_network]: For loading binary connectivity networks
    """
    if device is None:
        device = DEVICE

    if name is None:
        name = "AAL_DISTANCES"

    distance_matrix = torch.load(
        os.path.join(BASE_PATH, f"distance_matrices/{name.split('.')[0].upper()}.pt"),
        map_location=device,
        weights_only=True,
    )

    weighted_checks(distance_matrix.unsqueeze(0))

    return distance_matrix


@jaxtyped(typechecker=typechecked)
def get_coordinates(
    name: Optional[str] = None, device: Optional[torch.device] = None
) -> Float[torch.Tensor, "num_nodes 3"]:
    r"""Load a default set of 3D coordinates.

    Provides access to pre-packaged coordinate sets that represent the spatial positions
    of brain regions in standard atlas parcellations.

    Available coordinate sets:

    - AAL_COORDINATES: 3D coordinates for the Automated Anatomical Labeling atlas

    Args:
        name: Name of the coordinates to load. If unspecified,
            the AAL_COORDINATES coordinates are loaded.
        device: Device to load the coordinates on. If unspecified,
            automatically uses CUDA if available, otherwise CPU.

    Returns:
        A tensor containing the requested coordinates with shape [num_nodes, 3].

    Examples:
        >>> from gnm.defaults import get_coordinates
        >>> # Load default coordinates
        >>> coords = get_coordinates()
        >>> # Load a specific coordinate set
        >>> coords = get_coordinates(name="AAL_COORDINATES")
        >>> coords.shape
        torch.Size([90, 3])

    See Also:
        - [`defaults.get_distance_matrix`][gnm.defaults.get_distance_matrix]: For loading distance matrices between nodes
    """
    if device is None:
        device = DEVICE

    if name is None:
        name = "AAL_COORDINATES"

    return torch.load(
        os.path.join(BASE_PATH, f"coordinates/{name.split('.')[0].upper()}.pt"),
        map_location=device,
        weights_only=True,
    )


@jaxtyped(typechecker=typechecked)
def get_binary_network(
    name: Optional[str] = None, device: Optional[torch.device] = None
) -> Float[torch.Tensor, "dataset_size num_nodes num_nodes"]:
    r"""Load a default binary network.

    Provides access to pre-packaged binary networks that represent structural
    connectivity with edges indicated as either present (1) or absent (0).

    Available binary networks:

    - CALM_BINARY_CONSENSUS: Binary consensus network from the CALM dataset

    Args:
        name: Name of the binary network to load. If unspecified,
            the CALM_BINARY_CONSENSUS binary network is loaded.
        device: Device to load the binary network on. If unspecified,
            automatically uses CUDA if available, otherwise CPU.

    Returns:
        A tensor containing the requested binary network with shape
        [dataset_size, num_nodes, num_nodes].

    Examples:
        >>> from gnm.defaults import get_binary_network
        >>> # Load default binary network
        >>> bin_net = get_binary_network()
        >>> # Load a specific binary network
        >>> bin_net = get_binary_network(name="CALM_BINARY_CONSENSUS")
        >>> bin_net.shape
        torch.Size([1, 90, 90])

    See Also:
        - [`defaults.get_weighted_network`][gnm.defaults.get_weighted_network]: For loading weighted connectivity networks
        - [`utils.binary_checks`][gnm.utils.binary_checks]: For validating binary networks
    """
    if device is None:
        device = DEVICE

    if name is None:
        name = "CALM_BINARY_CONSENSUS"

    binary_networks = torch.load(
        os.path.join(BASE_PATH, f"binary_networks/{name.split('.')[0].upper()}.pt"),
        map_location=device,
        weights_only=True,
    )

    binary_checks(binary_networks)

    return binary_networks


@jaxtyped(typechecker=typechecked)
def get_weighted_network(
    name: Optional[str] = None, device: Optional[torch.device] = None
) -> Float[torch.Tensor, "dataset_size num_nodes num_nodes"]:
    r"""Load a default weighted network.

    Provides access to pre-packaged weighted networks that represent structural
    connectivity with connections represented by continuous weights.

    Available weighted networks:

    - CALM_WEIGHTED_CONSENSUS: Weighted consensus network from the CALM dataset

    Args:
        name: Name of the weighted network to load. If unspecified,
            the CALM_WEIGHTED_CONSENSUS weighted network is loaded.
        device: Device to load the weighted network on. If unspecified,
            automatically uses CUDA if available, otherwise CPU.

    Returns:
        A tensor containing the requested weighted network with shape
        [dataset_size, num_nodes, num_nodes].

    Examples:
        >>> from gnm.defaults import get_weighted_network
        >>> # Load default weighted network
        >>> wt_net = get_weighted_network()
        >>> # Load a specific weighted network
        >>> wt_net = get_weighted_network(name="CALM_WEIGHTED_CONSENSUS")
        >>> wt_net.shape
        torch.Size([1, 90, 90])

    See Also:
        - [`defaults.get_binary_network`][gnm.defaults.get_binary_network]: For loading binary connectivity networks
        - [`utils.weighted_checks`][gnm.utils.weighted_checks]: For validating weighted networks
    """
    if device is None:
        device = DEVICE

    if name is None:
        name = "CALM_WEIGHTED_CONSENSUS"

    weighted_networks = torch.load(
        os.path.join(BASE_PATH, f"weighted_networks/{name.split('.')[0].upper()}.pt"),
        map_location=device,
        weights_only=True,
    )

    weighted_checks(weighted_networks)

    return weighted_networks


def get_smoothing_matrix(
    distance_matrix: Float[torch.Tensor, "num_nodes num_nodes"],
    scale: float = 0.1,
) -> Float[torch.Tensor, "num_nodes num_nodes"]:
    r"""Create a Gaussian smoothing matrix from a distance matrix.

    This function generates a smoothing matrix based on the provided distance matrix
    using a Gaussian kernel with a specified sigma value. The resulting matrix can be
    used to spatially smooth network properties.

    Args:
        distance_matrix: Distance matrix with shape [num_nodes, num_nodes].
        sigma: Standard deviation for the Gaussian kernel. Default is 15.0.

    Returns:
        A tensor representing the smoothing matrix with shape [num_nodes, num_nodes].

    Examples:
        >>> import torch
        >>> from gnm.defaults import get_distance_matrix, get_smoothing_matrix
        >>> dist_matrix = get_distance_matrix()
        >>> smoothing_matrix = get_smoothing_matrix(dist_matrix)
        >>> smoothing_matrix.shape
        torch.Size([90, 90])
    """
    median_distance = distance_matrix[distance_matrix > 0].median().item()

    sigma = scale * median_distance

    smoothing_matrix = torch.exp(-(distance_matrix**2) / (2 * sigma**2))

    # Ensure that each row sums to 1
    smoothing_matrix = smoothing_matrix / smoothing_matrix.sum(axis=1, keepdims=True)

    return smoothing_matrix


def get_experiment_index_file() -> dict[str]:
    r"""Load the default experiment index file.

    This function loads the default experiment index file, which contains metadata
    about pre-defined experiment configurations for generative network models.

    Returns:
        A dictionary representing the contents of the experiment index file.
    
    Examples:
        >>> from gnm.defaults import get_experiment_index_file
        >>> index_file = get_experiment_index_file()
        >>> index_file.keys()
    """

    PATH = os.path.join(
        BASE_PATH, "experiment_dataclass_examples", "gnm_index.json"
    )

    with open(PATH, "r") as f:
        index_file = json.load(f)

    return index_file