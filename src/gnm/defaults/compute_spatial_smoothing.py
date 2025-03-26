import numpy as np
from scipy.spatial.distance import squareform, pdist


def compute_spatial_smoothing(coordinates, distance_matrix=None, sigma=None):
    """
    Compute a spatial weight matrix W based on the given distance matrix and sigma.

    Parameters
    ----------
    coordinates : x-y-z coordinates of the network nodes
    distance_matrix : numpy.ndarray
        A 2D square array (N x N) of pairwise distances between points.
    sigma : float, optional
        The spatial scale parameter for the Gaussian kernel. If None, defaults to
        0.5 * np.std(distance_matrix).

    Returns
    -------
    numpy.ndarray
        A 2D array (N x N)
    """
    if distance_matrix is None:
        distance_matrix = squareform(pdist(coordinates))
    else:
        # Ensure distance_matrix is a NumPy array
        distance_matrix = np.asarray(distance_matrix)

    # If sigma is not specified, use half the standard deviation of the distance matrix
    if sigma is None:
        sigma = 0.5 * np.std(coordinates)

    # Compute the spatial weight matrix
    W = np.exp(-distance_matrix**2 / (2 * sigma**2))

    return W
