import numpy as np

def compute_spatial_smoothing(distance_matrix, sigma=None):
    """
    Compute a spatial weight matrix W based on the given distance matrix and sigma.

    Parameters
    ----------
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
    # Ensure distance_matrix is a NumPy array
    distance_matrix = np.asarray(distance_matrix)

    # If sigma is not specified, use half the standard deviation of the distance matrix
    if sigma is None:
        sigma = 0.5 * np.std(distance_matrix)

    # Compute the spatial weight matrix
    W = np.exp(-distance_matrix**2 / (2 * sigma**2))

    return W
