r"""Default data and resources for generative network models.

This subpackage provides access to pre-packaged datasets that can be used
for experimenting with generative network models without requiring external data.
These defaults include:

- **Distance matrices**: Physical distances between brain regions
- **Coordinates**: 3D spatial positions of brain regions
- **Binary networks**: Example binary connectivity networks (presence/absence of connections)
- **Weighted networks**: Example weighted connectivity networks

The module provides simple functions to list available datasets and load them
with appropriate tensor formats for immediate use in network modeling.

Functions:
    display_available_defaults: Show all available default datasets
    get_distance_matrix: Load a default distance matrix
    get_coordinates: Load default 3D coordinates
    get_binary_network: Load a default binary network
    get_weighted_network: Load a default weighted network
"""

from .get_defaults import (
    display_available_defaults,
    get_distance_matrix,
    get_coordinates,
    get_binary_network,
    get_weighted_network,
    get_smoothing_matrix,
)

__all__ = [
    "display_available_defaults",
    "get_distance_matrix",
    "get_coordinates",
    "get_binary_network",
    "get_weighted_network",
    "get_smoothing_matrix",
]
