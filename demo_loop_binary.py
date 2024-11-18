# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 18:32:05 2024

@author: fp02
"""

import importlib
import torch
import matplotlib.pyplot as plt
import numpy as np
#from tqdm.notebook import tqdm
import GNM
importlib.reload(GNM)
from GNM import GenerativeNetworkModel
import scipy.io
#from nilearn import plotting
#import plotly
import networkx as nx
import pandas as pd
from scipy.stats import ks_2samp
import seaborn as sns

# Load the provided .mat file to explore its contents
mat_file_path = r'../Data/Consensus/consensus_wgm_h.mat'
# on the cluster, data is in \imaging\Astle\fp02\wGNM\prepare\consensus_wgm_h.mat

# Load the .mat file
mat_contents = scipy.io.loadmat(mat_file_path)
# Extract the first element from 'k_atlases' to explore its structure
k_atlas_element = mat_contents['k_atlases'][0]

# Import consensus network for aal (unused in gnm for now)
atlas = k_atlas_element[0]
consensus_matrix = atlas['consensus'][0, 0]

# Import the distance matrix
distance_matrix = torch.tensor(atlas['euclidean'][0, 0])

# Import the coordinates
coord = atlas['coordinates'][0, 0]

# set number of nodes and edges
num_nodes = len(consensus_matrix)
num_seed_edges = np.sum(consensus_matrix > 0) / 2

# start off with an empty seed
seed_adjacency_matrix = torch.zeros(num_nodes, num_nodes)

# Define parameter ranges for eta, gamma, lambdah, and alpha
eta_values = -np.arange(2,5,0.2)
gamma_values = np.arange(0,0.5,0.05)
lambdah_values = [0.0]
alpha_values = [0.0] #np.arange(0,0.2,0.05)

distance_relationship_type = "powerlaw"
matching_relationship_type = "powerlaw"

# Initialize an empty list to store results
results = []
added_edges_snapshots = {}
adjacency_snapshots = {}
weight_snapshots_dict = {}

# Total number of models to run
total_models = len(eta_values) * len(gamma_values) * len(lambdah_values) * len(alpha_values)
current_model = 0

# Loop over all combinations of parameter values
for eta in eta_values:
    for gamma in gamma_values:
        for lambdah in lambdah_values:
            for alpha in alpha_values:
                # Update and print progress
                current_model += 1
                progress_percentage = (current_model / total_models) * 100
                print(f"Progress: {progress_percentage:.2f}% ({current_model}/{total_models})")

                # Initialize the model with the current set of parameters
                gnm = GenerativeNetworkModel(seed_adjacency_matrix=seed_adjacency_matrix,
                                             distance_matrix=distance_matrix,
                                             eta=eta,
                                             gamma=gamma,
                                             lambdah=lambdah,
                                             distance_relationship_type=distance_relationship_type,
                                             matching_relationship_type=matching_relationship_type,
                                             alpha=alpha,
                                             optimisation_criterion_kwargs={"omega": 1},
                                             optimisation_normalisation=True)

                # Train the model
                added_edges_list, adjacency_snapshot, weight_snapshots = gnm.train_loop(num_iterations=int(num_seed_edges),
                                                                                       binary_updates_per_iteration=1,
                                                                                       weighted_updates_per_iteration=0)

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
                real_degree_list = [degree for node, degree in Greal_bin.degree()]
                real_clustering_coefficients_list = list(nx.clustering(Greal_bin).values())
                real_betweenness_centrality_list = list(nx.betweenness_centrality(Greal_bin, normalized=False).values())
                
                # Extract distances for connected nodes in the real network
                real_connected_indices = np.triu(Atgt, k=1) > 0
                real_connected_distances = distance_matrix[real_connected_indices]
                
                # Compute binary network metrics for the synthetic network
                degree_list = [degree for node, degree in G_bin.degree()]
                clustering_coefficients_list = list(nx.clustering(G_bin).values())
                betweenness_centrality_list = list(nx.betweenness_centrality(G_bin, normalized=False).values())
                
                # Extract distances for connected nodes in the synthetic network
                connected_indices = np.triu(Abin, k=1) > 0
                connected_distances = distance_matrix[connected_indices]
                
                # Compute KS statistics
                degree_KS_statistic = ks_2samp(real_degree_list, degree_list).statistic
                clustering_KS_statistic = ks_2samp(real_clustering_coefficients_list, clustering_coefficients_list).statistic
                betweenness_KS_statistic = ks_2samp(real_betweenness_centrality_list, betweenness_centrality_list).statistic
                edge_length_KS_statistic = ks_2samp(real_connected_distances, connected_distances).statistic
                
                # Compute the maximum KS statistic
                max_KS_statistic = max([
                    degree_KS_statistic,
                    clustering_KS_statistic,
                    betweenness_KS_statistic,
                    edge_length_KS_statistic
                ])

                # Store the results
                results.append({
                    'eta': eta,
                    'gamma': gamma,
                    'lambdah': lambdah,
                    'alpha': alpha,
                    'degree_KS_statistic': degree_KS_statistic,
                    'clustering_KS_statistic': clustering_KS_statistic,
                    'betweenness_KS_statistic': betweenness_KS_statistic,
                    'edge_length_KS_statistic': edge_length_KS_statistic,
                    'max_KS_statistic': max_KS_statistic
                })

# Convert results to a DataFrame and save to a CSV file
results_df = pd.DataFrame(results)
#results_df.to_csv('gnm_parameter_sweep_results.csv', index=False)


# Plot heatmaps for energy (max(KS))
parameter_combinations = results_df[['eta', 'gamma', 'lambdah', 'alpha', 'max_KS_statistic']]

# Find the best combination for lambdah and alpha
best_lambdah_alpha = parameter_combinations.loc[parameter_combinations['max_KS_statistic'].idxmin(), ['lambdah', 'alpha']]
best_lambdah = best_lambdah_alpha['lambdah']
best_alpha = best_lambdah_alpha['alpha']

# Plot energy values for each combination of eta and gamma, fixing lambdah and alpha to their best values
eta_gamma_fixed_df = results_df[(results_df['lambdah'] == best_lambdah) & (results_df['alpha'] == best_alpha)]
eta_gamma_pivot = eta_gamma_fixed_df.pivot(index='eta', columns='gamma', values='max_KS_statistic')
plt.figure(figsize=(10, 8))
sns.heatmap(eta_gamma_pivot, annot=True, cmap='viridis')
plt.title(f'Energy (max KS) Heatmap for eta vs gamma (lambdah={best_lambdah}, alpha={best_alpha})')
plt.xlabel('gamma')
plt.ylabel('eta')
plt.show()

# Plot energy values for each combination of eta and lambdah, fixing gamma and alpha to their best values
best_gamma_alpha = parameter_combinations.loc[parameter_combinations['max_KS_statistic'].idxmin(), ['gamma', 'alpha']]
best_gamma = best_gamma_alpha['gamma']
best_alpha = best_gamma_alpha['alpha']
eta_lambdah_fixed_df = results_df[(results_df['gamma'] == best_gamma) & (results_df['alpha'] == best_alpha)]
eta_lambdah_pivot = eta_lambdah_fixed_df.pivot(index='eta', columns='lambdah', values='max_KS_statistic')
plt.figure(figsize=(10, 8))
sns.heatmap(eta_lambdah_pivot, annot=True, cmap='viridis')
plt.title(f'Energy (max KS) Heatmap for eta vs lambdah (gamma={best_gamma}, alpha={best_alpha})')
plt.xlabel('lambdah')
plt.ylabel('eta')
plt.show()

# Plot energy values for each combination of gamma and alpha, fixing eta and lambdah to their best values
best_eta_lambdah = parameter_combinations.loc[parameter_combinations['max_KS_statistic'].idxmin(), ['eta', 'lambdah']]
best_eta = best_eta_lambdah['eta']
best_lambdah = best_eta_lambdah['lambdah']
gamma_alpha_fixed_df = results_df[(results_df['eta'] == best_eta) & (results_df['lambdah'] == best_lambdah)]
gamma_alpha_pivot = gamma_alpha_fixed_df.pivot(index='gamma', columns='alpha', values='max_KS_statistic')
plt.figure(figsize=(10, 8))
sns.heatmap(gamma_alpha_pivot, annot=True, cmap='viridis')
plt.title(f'Energy (max KS) Heatmap for gamma vs alpha (eta={best_eta}, lambdah={best_lambdah})')
plt.xlabel('alpha')
plt.ylabel('gamma')
plt.show()

# Plot final adjacency and weight matrices
plt.figure(figsize=(5, 5))
plt.imshow(gnm.adjacency_matrix.numpy())
plt.title("Final adjacency matrix")
plt.show()

plt.figure(figsize=(5, 5))
plt.imshow(gnm.weight_matrix.detach().numpy())
plt.title("Final weight matrix")
plt.show()


