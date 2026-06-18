r"""Utility functions for working with generative network models.

This subpackage provides various utility functions that support the core generative
network modeling functionality. It includes:

- **Statistical measures**: Functions for statistical comparisons between networks
- **Graph properties**: Various network metrics and measures for analysing graph structure
- **Data validation**: Functions to verify the validity of network data structures
- **Control networks**: Functions for generating control networks with preserved properties
- **Convert Datatypes**: Functions to convert numpy and additional datatypes to GNM-compatible tensor

These utilities handle both binary and weighted networks and are optimised
for use with PyTorch tensors.
"""

from .statistics import ks_statistic
from .graph_properties import (
    node_strengths,
    binary_clustering_coefficients,
    weighted_clustering_coefficients,
    communicability,
    shortest_paths,
    binary_characteristic_path_length,
    weighted_characteristic_path_length,
    binary_betweenness_centrality,
    weighted_betweenness_centrality,
    binary_small_worldness,
    weighted_small_worldness,
)
from .checks import binary_checks, weighted_checks
from .control import get_control, generate_random_binary_networks, set_random_seed
from .convert_datatypes import np_to_tensor

__all__ = [
    "ks_statistic",
    "node_strengths",
    "binary_clustering_coefficients",
    "weighted_clustering_coefficients",
    "communicability",
    "shortest_paths",
    "binary_characteristic_path_length",
    "weighted_characteristic_path_length",
    "binary_betweenness_centrality",
    "weighted_betweenness_centrality",
    "binary_small_worldness",
    "weighted_small_worldness",
    "binary_checks",
    "weighted_checks",
    "get_control",
    "generate_random_binary_networks",
    "set_random_seed",
    "np_to_tensor",
]
