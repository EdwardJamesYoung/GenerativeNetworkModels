# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 11:48:24 2024

@author: fp02
"""

#import bct
import importlib
import torch
import matplotlib.pyplot as plt
import numpy as np
# from tqdm.notebook import tqdm
import sample_brain_coordinates
import GNM
importlib.reload(GNM)
from GNM import GenerativeNetworkModel
import scipy.io
# from nilearn import plotting
# import plotly
import networkx as nx
import pandas as pd
from scipy.stats import ks_2samp, pearsonr
from scipy.spatial.distance import squareform, pdist
import seaborn as sns
#import netneurotools as nt
from nilearn import plotting
from netneurotools.networks import networks_utils


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

thresh_conn = networks_utils.threshold_network(connectivity, 10)

# Plot final adjacency and weight matrices
plt.figure(figsize=(5, 5))
plt.imshow(thresh_conn)
plt.title("Final adjacency matrix")
plt.show()

Greal = nx.from_numpy_array(thresh_conn)
nx.density(Greal)

Atgt = (thresh_conn > 0).astype(int)

degree_list = np.array([degree for node, degree in Greal.degree()])
clustering_coefficients_list = np.array(list(nx.clustering(Greal).values()))
betweenness_centrality_list = np.array(
    list(nx.betweenness_centrality(Greal, normalized=False).values())
)

coordinates = coordinates[:, [1, 0, 2]]
coordinates = coordinates - np.mean(coordinates, axis = 0)

plt.scatter(
        coordinates[:, 1],
        coordinates[:, 2],
        #coordinates[:, 2],
        c=degree_list
    )



coordinates2 = coordinates.copy()
coordinates2[:,2] = coordinates[:, 2]*1.2+18
coordinates2[:,1] = coordinates[:, 1]*-1.05-20
coordinates2[:,0] = coordinates[:, 0]*1.2


# Plot using nilearn.plotting.plot_markers
plotting.plot_markers(
    node_values=degree_list,
    node_coords=coordinates2,
    node_size='auto',  # Automatically scale node sizes
    node_cmap=plt.cm.viridis,  # Colormap for nodes
    alpha=0.7,  # Transparency of markers
    display_mode='ortho',  # Orthogonal views
    annotate=True,  # Add annotations for positions
    colorbar=True,  # Display a colorbar
    title=' '
)
plt.show()

# Calculate the min and max for each axis
x_min, y_min, z_min = coordinates.min(axis=0)
x_max, y_max, z_max = coordinates.max(axis=0)

# vary y and z of origin!
y_values = np.linspace(y_min, y_max, 3)
z_values = np.linspace(z_min, z_max, 3)
x_values = [0] # centre!


consensus_matrix = thresh_conn

# Import the distance matrix
distance_matrix = torch.from_numpy(euclidean_distances)

# Import the coordinates
coord = coordinates

# Set number of nodes and edges
num_nodes = len(consensus_matrix)
num_seed_edges = np.sum(consensus_matrix > 0) / 2

# Start off with an empty seed
seed_adjacency_matrix = torch.zeros(num_nodes, num_nodes)

# Define parameter ranges for eta, gamma, lambdah, and alpha
eta_values = -np.arange(0, 5, 1.0)
gamma_values = np.arange(0, .5, 0.1)
lambdah_values = np.arange(0,5,1.0)
alpha_values = np.arange(0,0.2,0.05)
num_runs = 10

distance_relationship_type = "powerlaw"
matching_relationship_type = "powerlaw"

# Define the energy (beta = 1 for 'topology' or beta = 0 for 'topography')
beta = 0.5

# Initialize an empty list to store results
results = []
added_edges_snapshots = {}
adjacency_snapshots = {}
weight_snapshots_dict = {}


##################### Heterochronicity #####################

# Function to generate heterochronous matrices
def generate_heterochronous_matrix(
    coord,
    reference_coord=[0,0,0],
    sigma=1.0,
    num_nodes=100,
    mseed=0,
    cumulative=False,
    local=True
):
    """
    Generate heterochronous matrices based on a dynamic starting node.
    
    Parameters:
    - coord (array): Coordinates of nodes.
    - starting_node_index (int): Index of the node to use as the starting point.
    - sigma (float): Standard deviation for the Gaussian.
    - num_nodes (int): Number of time steps/nodes.
    - mseed (int): Number of seed nodes to exclude from computation.
    - cumulative (bool): Whether to apply cumulative maximum.
    - local (bool): Whether to generate matrices locally or globally.
    
    Returns:
    - torch.tensor: Heterochronous matrices tensor.
    """
    
    # Compute Euclidean distances from the reference node to all nodes
    distances = np.sqrt(np.sum((coord - reference_coord) ** 2, axis=1))
    max_distance = np.max(distances)  # Maximum distance for setting Gaussian means

    # Calculate means for Gaussian function at each time step
    means = np.linspace(0, max_distance, num_nodes - mseed)
    heterochronous_matrix = np.zeros((len(distances), num_nodes - mseed))
    P = lambda d, mu: (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((d - mu) ** 2) / (2 * sigma ** 2))

    # Calculate probabilities at each time step
    for t in range(num_nodes - mseed):
        mu = means[t]
        heterochronous_matrix[:, t] = P(distances, mu)
        #heterochronous_matrix[:, -1 - t] = P(distances, mu)

    # Apply cumulative maximum if requested
    if cumulative:
        heterochronous_matrix = np.maximum.accumulate(heterochronous_matrix, axis=1)

    # Convert to matrix form based on local parameter
    heterochronous_matrices = []
    for t in range(num_nodes - mseed):
        Ht = heterochronous_matrix[:, t]
        H_rescaled = (Ht - np.min(Ht)) / (np.max(Ht) - np.min(Ht))
        if local:
            Hmat = np.outer(H_rescaled, H_rescaled)
        else:
            N = len(H_rescaled)
            rows, cols = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
            Hmat = np.maximum(H_rescaled[rows], H_rescaled[cols])
        heterochronous_matrices.append(Hmat)

    heterochronous_matrices_tensor = torch.tensor(np.stack(heterochronous_matrices, axis=-1), dtype=torch.float32)
    return heterochronous_matrices_tensor


##################### weights for topography #####################
distance_matrix_np = distance_matrix.numpy()

# Define spatial weight matrix W
sigma = 15  # Adjust sigma as needed
W = np.exp(-distance_matrix_np**2 / (2 * sigma**2))

# Ensure the diagonal is zero (no self-weighting)
#np.fill_diagonal(W, 0)

# Normalize the weights for each node
W = W / W.sum(axis=1, keepdims=True)

##################################################################

kcoord = len(x_values)*len(y_values)*len(z_values) #coord[47,:]

# Total number of models to run
total_models = len(eta_values) * len(gamma_values) * len(lambdah_values) * len(alpha_values) * num_runs * kcoord
current_model = 0

# Loop over all combinations of parameter values
#x = heterochronous_matrix[:,:,1].numpy()

#x2 = np.exp(x*2)
#sns.heatmap(x2)


for x_value in x_values:
    for y_value in y_values:
        for z_value in z_values:
            reference_coord = [x_value, y_value, z_value]
            heterochronous_matrix = generate_heterochronous_matrix(
                coord, 
                reference_coord=reference_coord, 
                sigma=np.std(coord), 
                num_nodes=int(num_seed_edges), 
                mseed=0, 
                cumulative=False, 
                local=False
            )
            for eta in eta_values:
                for gamma in gamma_values:
                    for lambdah in lambdah_values:
                        for alpha in alpha_values:
                            total_energies = []  # List to store energies for each run
                            topology_energies = [] 
                            topography_energies = [] 
                            
                            degree_correlations = []
                            clustering_correlations = []
                            betweenness_correlations = []
            
            
                            for run in range(num_runs):
                                # Update and print progress
                                current_model += 1
                                progress_percentage = (current_model / total_models) * 100
                                print(f"Progress: {progress_percentage:.2f}% ({current_model}/{total_models})")
                
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
                                added_edges_list, adjacency_snapshot, weight_snapshots = gnm.train_loop(
                                    num_iterations=int(num_seed_edges),
                                    binary_updates_per_iteration=1,
                                    weighted_updates_per_iteration=0,
                                    heterochronous_matrix = heterochronous_matrix
                                )
                
                                # Save added edges, adjacency snapshots, and weight snapshots for each parameter combination
                                param_key = (eta, gamma, lambdah, alpha)
                                added_edges_snapshots[param_key] = added_edges_list
                                adjacency_snapshots[param_key] = adjacency_snapshot
                                weight_snapshots_dict[param_key] = weight_snapshots
                
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
                                # Compute KS statistics
                                degree_KS_statistic = ks_2samp(real_degree_list, degree_list).statistic
                                clustering_KS_statistic = ks_2samp(
                                    real_clustering_coefficients_list, clustering_coefficients_list
                                ).statistic
                                betweenness_KS_statistic = ks_2samp(
                                    real_betweenness_centrality_list, betweenness_centrality_list
                                ).statistic
                                edge_length_KS_statistic = ks_2samp(
                                    real_connected_distances, connected_distances
                                ).statistic
                
                                # Compute the maximum KS statistic for topology
                                topology_energy = max(
                                    [
                                        degree_KS_statistic,
                                        clustering_KS_statistic,
                                        betweenness_KS_statistic,
                                        edge_length_KS_statistic,
                                    ]
                                )
                
                                # --- Compute Topographical Energy (Spatially Weighted Correlations) ---
                                # Apply the spatial weighting
                                real_degree_list = W.dot(real_degree_list)
                                degree_list = W.dot(degree_list)
                
                                real_clustering_coefficients_list = W.dot(real_clustering_coefficients_list)
                                clustering_coefficients_list = W.dot(clustering_coefficients_list)
                
                                real_betweenness_centrality_list = W.dot(real_betweenness_centrality_list)
                                betweenness_centrality_list = W.dot(betweenness_centrality_list)
                
                                def compute_correlation(x, y):
                                    if np.std(x) == 0 or np.std(y) == 0:
                                        return 0.0
                                    else:
                                        return pearsonr(x, y)[0]
                
                                degree_correlation = compute_correlation(real_degree_list, degree_list)
                                clustering_correlation = compute_correlation(
                                    real_clustering_coefficients_list, clustering_coefficients_list
                                )
                                betweenness_correlation = compute_correlation(
                                    real_betweenness_centrality_list, betweenness_centrality_list
                                )
                
                                # Compute error as 1 - absolute correlation
                                degree_error = 1 - (degree_correlation + 1)/2
                                clustering_error = 1 - (clustering_correlation + 1)/2
                                betweenness_error = 1 - (betweenness_correlation +1)/2
                
                                # Compute the maximum error statistic for topography
                                topography_energy = max([degree_error, clustering_error, betweenness_error])
                
                                # --- Combine Energies ---
                                total_energy = beta * topology_energy + (1 - beta) * topography_energy
                            
                                # Append the energy to the energies list
                                total_energies.append(total_energy)
                                topology_energies.append(topology_energy)
                                topography_energies.append(topography_energy)
                                
                                degree_correlations.append(degree_correlation)
                                clustering_correlations.append(clustering_correlation)
                                betweenness_correlations.append(betweenness_correlation)
                
                            # Compute the average and standard deviation of energies for this parameter combination
                            avg_energy = np.mean(total_energies)
                            std_energy = np.std(total_energies)
                            
                            avg_lenergy = np.mean(topology_energies)
                            std_lenergy = np.std(topology_energies)
                            
                            avg_genergy = np.mean(topography_energies)
                            std_genergy = np.std(topography_energies)
                            
                            
                            avg_degree_correlations = np.mean(degree_correlations)            
                            avg_clustering_correlations = np.mean(clustering_correlations) 
                            avg_betweenness_correlations = np.mean(betweenness_correlations)
            
                            # Store the results
                            results.append(
                                {
                                    'x_value': x_value,
                                    'y_value': y_value,
                                    'z_value': z_value,
                                    'eta': eta,
                                    'gamma': gamma,
                                    'lambdah': lambdah,
                                    'alpha': alpha,
                                    #'degree_KS_statistic': degree_KS_statistic,
                                    #'clustering_KS_statistic': clustering_KS_statistic,
                                    #'betweenness_KS_statistic': betweenness_KS_statistic,
                                    #'edge_length_KS_statistic': edge_length_KS_statistic,
                                    'degree_correlation': avg_degree_correlations,
                                    'clustering_correlation': avg_clustering_correlations,
                                    'betweenness_correlation': avg_betweenness_correlations,
                                    'topology_energy': avg_lenergy,
                                    'topography_energy': avg_genergy,
                                    'total_energy': avg_energy,
                                    'sd_total_energy': std_energy,
                                    'sd_topology_energy': std_lenergy,
                                    'sd_topography_energy': std_genergy,
                                }
                            )

# Convert results to a DataFrame
results_df = pd.DataFrame(results)
# save
results_df.to_csv(r'../Data/results_xyz_simple_global.csv')

results_df= pd.read_csv(r'../Data/results_xyz_simple_global.csv')

# all models are wrong but some are better than other.
# i.e., what are you trying to capture?
beta = 0.5
results_df["total_energy"] = beta * results_df["topology_energy"] + (1 - beta) * results_df["topography_energy"]

sns.regplot(results_df, x = 'topology_energy', y = 'topography_energy', scatter=True)
plt.show()

which_energy = 'total_energy'
# Dynamically extract the best parameter values for the selected starting node
best_params = results_df.loc[results_df[which_energy].idxmin()]
best_eta = best_params['eta']
best_gamma = best_params['gamma']
best_lambdah = best_params['lambdah']
best_alpha = best_params['alpha']
best_y = best_params['y_value']
best_z = best_params['z_value']


# Plot energy values for each combination of eta and gamma, fixing lambdah and alpha
eta_gamma_fixed_df = results_df[
    (results_df['lambdah'] == best_lambdah) & 
    (results_df['alpha'] == best_alpha) &
    (results_df['y_value'] == best_y) &
    (results_df['z_value'] == best_z)
]
eta_gamma_pivot = eta_gamma_fixed_df.pivot(index='eta', columns='gamma', values=which_energy)

plt.figure(figsize=(10, 8))
sns.heatmap(eta_gamma_pivot, annot=True, cmap='viridis')
plt.title(f'Energy Heatmap for eta vs gamma (lambdah={best_lambdah}, alpha={best_alpha})')
plt.xlabel('gamma')
plt.ylabel('eta')
plt.show()

# Plot energy values for each combination of eta and lambdah, fixing gamma and alpha
eta_lambdah_fixed_df = results_df[
    (results_df['gamma'] == best_gamma) & 
    (results_df['alpha'] == best_alpha) &
    (results_df['y_value'] == best_y) &
    (results_df['z_value'] == best_z)
]
eta_lambdah_pivot = eta_lambdah_fixed_df.pivot(index='eta', columns='lambdah', values=which_energy)

plt.figure(figsize=(10, 8))
sns.heatmap(eta_lambdah_pivot, annot=True, cmap='viridis')
plt.title(f'Energy Heatmap for eta vs lambdah (gamma={best_gamma}, alpha={best_alpha})')
plt.xlabel('lambdah')
plt.ylabel('eta')
plt.show()

# Plot energy values for each combination of gamma and alpha, fixing eta and lambdah
gamma_alpha_fixed_df = results_df[
    (results_df['eta'] == best_eta) & 
    (results_df['lambdah'] == best_lambdah) &
    (results_df['y_value'] == best_y) &
    (results_df['z_value'] == best_z)
]
gamma_alpha_pivot = gamma_alpha_fixed_df.pivot(index='gamma', columns='alpha', values=which_energy)

plt.figure(figsize=(10, 8))
sns.heatmap(gamma_alpha_pivot, annot=True, cmap='viridis')
plt.title(f'Energy Heatmap for gamma vs alpha (eta={best_eta}, lambdah={best_lambdah})')
plt.xlabel('alpha')
plt.ylabel('gamma')
plt.show()

# Plot energy values for each combination of gamma and alpha, fixing eta and lambdah
y_z_fixed_df = results_df[
    (results_df['eta'] == best_eta) & 
    (results_df['lambdah'] == best_lambdah) &
    (results_df['gamma'] == best_gamma) &
    (results_df['alpha'] == best_alpha)
]
y_z_pivot = y_z_fixed_df.pivot(index='z_value', columns='y_value', values=which_energy)

plt.figure(figsize=(10, 8))
sns.heatmap(y_z_pivot, annot=True, cmap='viridis')
plt.xlabel('y')
plt.ylabel('z')
plt.show()

# one is consistently best (thalamus?) but there might be multiple gradients


y_z_coordinates = np.zeros((len(y_z_fixed_df),3))
y_z_coordinates[:,2] = y_z_fixed_df['z_value']*1.2+18
y_z_coordinates[:,1] = y_z_fixed_df['y_value']#*-1.05-20
y_z_coordinates[:,0] = y_z_fixed_df['x_value']*1.2


# Convert to an array aligned with node indices
node_values = y_z_fixed_df['total_energy'].values
node_values = y_z_fixed_df['topology_energy'].values
node_values = y_z_fixed_df['topography_energy'].values

node_coords = y_z_coordinates

# Plot using nilearn.plotting.plot_markers
plotting.plot_markers(
    node_values=node_values,
    node_coords=node_coords,
    node_size='auto',  # Automatically scale node sizes
    node_cmap=plt.cm.viridis,  # Colormap for nodes
    alpha=0.7,  # Transparency of markers
    display_mode='ortho',  # Orthogonal views
    annotate=True,  # Add annotations for positions
    colorbar=True,  # Display a colorbar
    title='Total Energy'
)
plt.show()

# best model has heterochron fitted on topography and homophily on topology?

# Create a figure with 8 subplots arranged in one row and 8 columns
fig, axes = plt.subplots(1, 8, figsize=(32, 4))

# Plot the evolution of the network over time using nilearn's plot_markers in 8 subplots
n_steps = 8
for t in range(n_steps):
    ax = axes[t]
    Ht = heterochronous_matrix[:, :, int(np.round(num_seed_edges/n_steps*t))].numpy()
    node_values = Ht.diagonal()  # Use the diagonal elements to represent nodal values for coloring
    display = plotting.plot_markers(
        node_values,
        node_coords=coord,
        node_cmap='viridis',
        node_size=50,
        display_mode='x',
        axes=ax,
        output_file=None,
        colorbar=(t == n_steps - 1)  # Show colorbar only for the last subplot
    )
    display._colorbar = None # Remove colorbar for all but the last plot

#plt.tight_layout()
plt.show()



degree_correlation = pearsonr(real_degree_list, degree_list)
plt.scatter(real_degree_list, degree_list)

clustering_correlation = pearsonr(real_clustering_coefficients_list, clustering_coefficients_list)
plt.scatter(real_clustering_coefficients_list, clustering_coefficients_list)


betweenness_correlation = pearsonr(real_betweenness_centrality_list, betweenness_centrality_list)
plt.scatter(real_betweenness_centrality_list, betweenness_centrality_list)



plt.scatter(
        coordinates[:, 2],
        coordinates[:, 0],
        coordinates[:, 1],
        c=real_degree_list
    )








# Iterate over multiple values of beta and calculate total energy
beta_values = np.linspace(0, 1, 20)
best_params_list = []

for beta in beta_values:
    results_df["total_energy"] = beta * results_df["topology_energy"] + (1 - beta) * results_df["topography_energy"]
    best_params = results_df.loc[results_df["total_energy"].idxmin()]
    best_params_list.append({
        'beta': beta,
        'eta': best_params['eta'],
        'gamma': best_params['gamma'],
        'lambdah': best_params['lambdah'],
        'alpha': best_params['alpha'],
        'y_value': best_params['y_value'],
        'z_value': best_params['z_value'],
        'total_energy': best_params['total_energy'],
        'topology_energy': best_params['topology_energy'],
        'topography_energy': best_params['topography_energy']
    })

# Convert best parameters to a DataFrame
best_params_df = pd.DataFrame(best_params_list)

# Plot parameter values against beta
parameters_to_plot = ['eta', 'gamma', 'lambdah', 'alpha', 'y_value', 'z_value']

for param in parameters_to_plot:
    plt.figure()
    plt.plot(best_params_df['beta'], best_params_df[param], marker='o')
    plt.title(f"{param} vs Beta")
    plt.xlabel('Beta')
    plt.ylabel(param)
    plt.grid()
    plt.show()


# Plot energies as a function of beta
plt.figure(figsize=(10, 6))
plt.plot(best_params_df['beta'], best_params_df['total_energy'], label='Total Energy', marker='o')
plt.plot(best_params_df['beta'], best_params_df['topology_energy'], label='Topology Energy', marker='s')
plt.plot(best_params_df['beta'], best_params_df['topography_energy'], label='Topography Energy', marker='^')
plt.title('Energies vs Beta')
plt.xlabel('Beta')
plt.ylabel('Energy')
plt.legend()
plt.grid()
plt.show()





def pareto_frontier(data, objectives):
    """
    Identifies the Pareto front for a given set of objectives.
    Args:
        data (pd.DataFrame): DataFrame containing the objective values.
        objectives (list of str): Column names for the objectives to minimize.

    Returns:
        pd.DataFrame: Subset of the original DataFrame representing the Pareto front.
    """
    # Sort the data by the first objective
    sorted_data = data.sort_values(by=objectives[0], ascending=True).reset_index(drop=True)
    
    pareto_front = [sorted_data.iloc[0]]  # Start with the first point
    
    # Iterate through the sorted data
    for i in range(1, len(sorted_data)):
        current = sorted_data.iloc[i]
        last_pareto = pareto_front[-1]
        
        # Add to the Pareto front if it's better in at least one dimension
        if current[objectives[1]] < last_pareto[objectives[1]]:
            pareto_front.append(current)
    
    return pd.DataFrame(pareto_front)

# Simulate example data
pareto_data = results_df.copy()

# Normalize the energies
pareto_data['topology_energy'] = (pareto_data['topology_energy'] - pareto_data['topology_energy'].min()) / \
                                      (pareto_data['topology_energy'].max() - pareto_data['topology_energy'].min())

pareto_data['topography_energy'] = (pareto_data['topography_energy'] - pareto_data['topography_energy'].min()) / \
                                        (pareto_data['topography_energy'].max() - pareto_data['topography_energy'].min())

# Use normalized columns for Pareto front identification
pareto_front = pareto_frontier(pareto_data, ['topology_energy_norm', 'topography_energy_norm'])

# Identify the Pareto front for topology and topography energies
pareto_front = pareto_frontier(pareto_data, ['topology_energy', 'topography_energy'])

# Plot the Pareto front
plt.figure(figsize=(10, 6))
plt.scatter(pareto_data['topology_energy'], pareto_data['topography_energy'], label='All Points', alpha=0.5)
plt.plot(pareto_front['topology_energy'], pareto_front['topography_energy'], color='red', marker='o', label='Pareto Front')
plt.title('Pareto Front for Topology vs. Topography Energies')
plt.xlabel('Topology Energy')
plt.ylabel('Topography Energy')
plt.legend()
plt.grid()
plt.show()





inside_points, outside_points = sample_brain_coordinates(coordinates,[0,10,10])


# Plot the brain coordinates and the sample points
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the original brain coordinates
ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2],
           color='blue', s=20, label='Brain Coordinates')

# Plot the sample points inside the convex hull
ax.scatter(inside_points[:, 0], inside_points[:, 1], inside_points[:, 2],
           color='green', s=50, label='Inside Points')

# Optionally, plot the sample points outside the convex hull
ax.scatter(outside_points[:, 0], outside_points[:, 1], outside_points[:, 2],
           color='red', s=50, label='Outside Points')

# Set labels
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.legend()
plt.title('Sample Points Inside and Outside the Brain Convex Hull')
plt.show()



