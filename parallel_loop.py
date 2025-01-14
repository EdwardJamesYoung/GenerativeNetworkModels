# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:24:25 2024

@author: fp02
"""

from joblib import Parallel, delayed
from tqdm import tqdm

import importlib
import torch
import matplotlib.pyplot as plt
import numpy as np
# from tqdm.notebook import tqdm
import GNM
importlib.reload(GNM)
from GNM import GenerativeNetworkModel
import sample_brain_coordinates

import scipy.io
# from nilearn import plotting
# import plotly
import networkx as nx
import pandas as pd
from scipy.stats import ks_2samp, pearsonr
from scipy.spatial.distance import squareform, pdist
#import seaborn as sns
from netneurotools.networks import networks_utils
#from nilearn import plotting
from itertools import product

from generate_heterochronous_matrix import generate_heterochronous_matrix

# must be improved to deal with symmetry on x axis

from scipy.spatial import ConvexHull, Delaunay

def sample_brain_coordinates(coordinates, num_samples):
    """
    Generates sample points within the bounding box of brain coordinates and returns
    the unique x, y, z values from the points inside the brain using the convex hull approach.

    Parameters:
    - coordinates (np.ndarray): An (N, 3) array of brain coordinates.
    - num_samples (list or tuple): A list of three integers [n_x, n_y, n_z] specifying
      the number of samples along the x, y, and z axes.

    Returns:
    - x_unique (np.ndarray): Unique x-values from the inside points.
    - y_unique (np.ndarray): Unique y-values from the inside points.
    - z_unique (np.ndarray): Unique z-values from the inside points.
    """

    # Ensure that num_samples has three elements
    if len(num_samples) != 3:
        raise ValueError("num_samples must be a list or tuple with three integers [n_x, n_y, n_z]")

    n_x, n_y, n_z = num_samples

    # Compute the convex hull of the brain coordinates
    hull = ConvexHull(coordinates)

    # Get the min and max values for each axis
    x_min, y_min, z_min = coordinates.min(axis=0)
    x_max, y_max, z_max = coordinates.max(axis=0)

    # Handle cases where n_x, n_y, or n_z is 0 or 1
    def generate_axis_samples(n, min_val, max_val):
        if n <= 1:
            # If n is 0 or 1, return the midpoint
            return np.array([(min_val + max_val) / 2])
        else:
            return np.linspace(min_val, max_val, n)

    # Generate sample points along each axis
    x_samples = generate_axis_samples(n_x, x_min, x_max)
    y_samples = generate_axis_samples(n_y, y_min, y_max)
    z_samples = generate_axis_samples(n_z, z_min, z_max)

    # Create a meshgrid of the sample points
    X, Y, Z = np.meshgrid(x_samples, y_samples, z_samples, indexing='ij')
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()

    # Combine into a single array of sample points
    sample_points = np.vstack((X_flat, Y_flat, Z_flat)).T

    # Function to check if points are inside the convex hull
    def in_hull(points, hull):
        """
        Test if points in `points` are inside the convex hull `hull`.
        """
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull.points[hull.vertices])
        return hull.find_simplex(points) >= 0

    # Check which sample points are inside the convex hull
    inside = in_hull(sample_points, hull)

    # Separate the inside points
    inside_points = sample_points[inside]

    # Extract unique x, y, z values from the inside points
    x_unique = np.unique(inside_points[:, 0])
    y_unique = np.unique(inside_points[:, 1])
    z_unique = np.unique(inside_points[:, 2])

    return x_unique, y_unique, z_unique


# Define a function to encapsulate the model running logic
def run_model(x_value, y_value, z_value, 
              eta, gamma, lambdah, alpha, 
              run, seed_adjacency_matrix, distance_matrix, W, 
              consensus_matrix, coord, num_seed_edges, 
              distance_relationship_type, matching_relationship_type, beta,
              set_cumulative, set_local, sigma,
              return_gnm = False):
    """
    Function to run a single model and return the result.
    """
    reference_coord = [x_value, y_value, z_value]
    heterochronous_matrix = generate_heterochronous_matrix(
        coord, 
        reference_coord=reference_coord, 
        sigma=sigma, 
        num_nodes=int(num_seed_edges), 
        mseed=0, 
        cumulative=set_cumulative, 
        local=set_local
    )

    # Initialize the model with the current set of parameters
    gnm = GenerativeNetworkModel(
        seed_adjacency_matrix=seed_adjacency_matrix,
        distance_matrix=distance_matrix,
        eta=eta,
        gamma=gamma,
        lambdah=lambdah,
        distance_relationship_type=distance_relationship_type,
        matching_relationship_type=matching_relationship_type,
        alpha=alpha,
        optimisation_criterion_kwargs={"omega": 1},
        optimisation_normalisation=True
    )

    # Train the model
    edges, adj, wts = gnm.train_loop(
        num_iterations=int(num_seed_edges),
        binary_updates_per_iteration=1,
        weighted_updates_per_iteration=0,
        heterochronous_matrix = heterochronous_matrix
    )
    print(wts)
    # Take the weighted adjacency matrix (i.e., final synthetic network)
    Abin = gnm.adjacency_matrix.numpy()

    # Convert weighted adjacency matrices to binary
    Atgt = (consensus_matrix > 0).astype(int)

    # Create graphs from binary adjacency matrices
    Greal_bin = nx.from_numpy_array(Atgt)
    G_bin = nx.from_numpy_array(Abin)

    # Compute binary network metrics for the real network
    real_degree_list = np.array([degree for node, degree in Greal_bin.degree()])
    real_clustering_coefficients_list = np.array(list(nx.clustering(Greal_bin).values()))
    real_betweenness_centrality_list = np.array(
        list(nx.betweenness_centrality(Greal_bin, normalized=False).values())
    )

    # Extract distances for connected nodes in the real network
    real_connected_indices = np.triu(Atgt, k=1) > 0
    real_connected_distances = distance_matrix.numpy()[real_connected_indices]

    # Compute binary network metrics for the synthetic network
    degree_list = np.array([degree for node, degree in G_bin.degree()])
    clustering_coefficients_list = np.array(list(nx.clustering(G_bin).values()))
    betweenness_centrality_list = np.array(
        list(nx.betweenness_centrality(G_bin, normalized=False).values())
    )

    # Extract distances for connected nodes in the synthetic network
    connected_indices = np.triu(Abin, k=1) > 0
    connected_distances = distance_matrix.numpy()[connected_indices]

    # --- Compute Topological Energy (KS Statistics) ---
    topology_energy = max(
        [
            ks_2samp(real_degree_list, degree_list).statistic,
            ks_2samp(real_clustering_coefficients_list, clustering_coefficients_list).statistic,
            ks_2samp(real_betweenness_centrality_list, betweenness_centrality_list).statistic,
            ks_2samp(real_connected_distances, connected_distances).statistic,
        ]
    )

    # --- Compute Topographical Energy (Spatially Weighted Correlations) ---
    degree_correlation = pearsonr(W.dot(real_degree_list), W.dot(degree_list))[0]
    clustering_correlation = pearsonr(W.dot(real_clustering_coefficients_list), W.dot(clustering_coefficients_list))[0]
    betweenness_correlation = pearsonr(W.dot(real_betweenness_centrality_list), W.dot(betweenness_centrality_list))[0]

    degree_error = 1 - (degree_correlation + 1) / 2
    clustering_error = 1 - (clustering_correlation + 1) / 2
    betweenness_error = 1 - (betweenness_correlation + 1) / 2

    topography_energy = max([degree_error, clustering_error, betweenness_error])

    # --- Combine Energies ---
    total_energy = beta * topology_energy + (1 - beta) * topography_energy

    results = {
        'x_value': x_value,
        'y_value': y_value,
        'z_value': z_value,
        'eta': eta,
        'gamma': gamma,
        'lambdah': lambdah,
        'alpha': alpha,
        'degree_correlation': degree_correlation,
        'clustering_correlation': clustering_correlation,
        'betweenness_correlation': betweenness_correlation,
        'topology_energy': topology_energy,
        'topography_energy': topography_energy,
        'total_energy': total_energy,
    }

    if return_gnm:
        results['gnm'] = gnm

    return results


def run_parallel_models(
    x_values, y_values, z_values, eta_values, gamma_values, lambdah_values, alpha_values,
    num_runs, seed_adjacency_matrix, distance_matrix, W, consensus_matrix,
    coord, num_seed_edges, distance_relationship_type, matching_relationship_type, beta,
    set_cumulative, set_local, sigma):
    # Generate all combinations of parameters
    param_combinations = list(product(
        x_values, y_values, z_values, eta_values, gamma_values, lambdah_values, alpha_values, range(num_runs)
    ))
    total_models = len(param_combinations)

    # Use joblib's Parallel to run models in parallel with tqdm progress bar
    results = Parallel(n_jobs=10)(
        delayed(run_model)(
            x_value, y_value, z_value, eta, gamma, lambdah, alpha, run,
            seed_adjacency_matrix, distance_matrix, W, consensus_matrix, coord, num_seed_edges,
            distance_relationship_type, matching_relationship_type, beta,
            set_cumulative, set_local, sigma
        )
        for x_value, y_value, z_value, eta, gamma, lambdah, alpha, run in tqdm(
            param_combinations, desc="Running Models", total=total_models
        )
    )

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Save or visualize results
    results_df.to_csv("results_cumulative_global.csv", index=False)
    print("Modeling complete. Results saved.")

if __name__ == "__main__":    
    # Load the provided .mat file to explore its contents
    mat_file_path = r'C:\Users\fp02\Downloads\Consensus_Connectomes.mat'
    # on the cluster, data is in \imaging\Astle\fp02\wGNM\prepare\consensus_wgm_h.mat

    # Load the .mat file
    mat_contents = scipy.io.loadmat(mat_file_path)

    res_parcellation = 0  # zero is low res, two is high res
    consensus_mat = scipy.io.loadmat(
        mat_file_path,
        simplify_cells=True,
        squeeze_me=True,
        chars_as_strings=True,
    )
    connectivity = consensus_mat["LauConsensus"]["Matrices"][res_parcellation][0]

    fc = consensus_mat["LauConsensus"]["Matrices"][res_parcellation][2]
    fiber_lengths = consensus_mat["LauConsensus"]["Matrices"][res_parcellation][1]
    coordinates = consensus_mat["LauConsensus"]["Matrices"][res_parcellation][3]
    labels = consensus_mat["LauConsensus"]["Matrices"][res_parcellation][4][:, 0]
    fc_modules = consensus_mat["LauConsensus"]["Matrices"][res_parcellation][4][:, 2]

    euclidean_distances = squareform(pdist(coordinates))

    consensus_matrix = networks_utils.threshold_network(connectivity, 10)

    #Atgt = (consensus_matrix > 0).astype(int)

    coordinates = coordinates[:, [1, 0, 2]]
    coordinates = coordinates - np.mean(coordinates, axis = 0)
    
    # vary y and z of origin!
    x_values, y_values, z_values = sample_brain_coordinates(coordinates, [0, 7, 7])
    x_values = [0]


    # Import the distance matrix
    distance_matrix = torch.from_numpy(euclidean_distances)

    # Set number of nodes and edges
    num_nodes = len(consensus_matrix)
    num_edges = np.sum(consensus_matrix > 0) / 2

    # Start off with an empty seed
    seed_adjacency_matrix = torch.zeros(num_nodes, num_nodes)

    # Define parameter ranges for eta, gamma, lambdah, and alpha
    eta_values = -np.linspace(0, 5, 10)
    gamma_values = np.linspace(0, 1, 10)
    lambdah_values = np.linspace(0,5,10)
    alpha_values = [0.0]  # np.arange(0,0.2,0.05)
    num_runs = 10
    sigma = np.std(coordinates)

    distance_relationship_type = "powerlaw"
    matching_relationship_type = "powerlaw"

    # Define the energy (beta = 1 for 'topology' or beta = 0 for 'topography')
    beta = 0.5
    set_cumulative = True
    set_local = False

    # Initialize an empty list to store results
    results = []
    added_edges_snapshots = {}
    adjacency_snapshots = {}
    weight_snapshots_dict = {}

    ##################### weights for topography #####################
    distance_matrix_np = distance_matrix.numpy()

    # Define spatial weight matrix W
    sigma_w = 15  # Adjust sigma as needed
    W = np.exp(-distance_matrix_np**2 / (2 * sigma_w**2))


    # Normalize the weights for each node
    W = W / W.sum(axis=1, keepdims=True)

    ##################################################################

    run_parallel_models(
    x_values, y_values, z_values, eta_values, gamma_values, lambdah_values, alpha_values,
    num_runs, seed_adjacency_matrix, distance_matrix, W, consensus_matrix,
    coordinates, num_edges, "powerlaw", "powerlaw", beta,
    set_cumulative, set_local, sigma
)

















