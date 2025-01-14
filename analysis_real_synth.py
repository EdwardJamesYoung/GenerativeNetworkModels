# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:48:36 2024

@author: fp02
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from parallel_loop import run_model

from joblib import Parallel, delayed
from tqdm import tqdm

import importlib
import torch
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


##############################################################
## put all models together

results_df = pd.read_csv(r'../Data/results_simple_local.csv')
results_df["Type"] = "Simple"
results_df["Connections"] = "Local"

results_df1 = pd.read_csv(r'../Code/results_simple_global2.csv')
results_df1["Type"] = "Simple"
results_df1["Connections"] = "Global"

results_df2 = pd.read_csv(r'../Code/results_cumulative_local.csv')
results_df2["Type"] = "Cumulative"
results_df2["Connections"] = "Local"

results_df3 = pd.read_csv(r'../Code/results_cumulative_global.csv')
results_df3["Type"] = "Cumulative"
results_df3["Connections"] = "Global"

results_df = pd.concat([results_df, results_df1, results_df2, results_df3])

beta = 0.5
results_df["total_energy"] = beta * results_df["topology_energy"] + (1 - beta) * results_df["topography_energy"]


results_sd = results_df.groupby(["z_value", "y_value", "eta", "gamma", 
                                 "lambdah", "Type", 
                                 "Connections"])[results_df.columns[7:-2]].std().reset_index()
results_df = np.round(results_df,2)

# average across runs
results_df = results_df.groupby(["z_value", "y_value", "eta", "gamma", 
                                 "lambdah", "Type", 
                                 "Connections"])[results_df.columns[7:-2]].mean().reset_index()


results_df["sd_topography"] = results_sd["topography_energy"]
results_df["sd_topology"] = results_sd["topology_energy"]
results_df["sd_total"] = results_sd["total_energy"]
results_df["sd_degree"] = results_sd["degree_correlation"]
results_df["sd_clustering"] = results_sd["clustering_correlation"]
results_df["sd_betweenness"] = results_sd["betweenness_correlation"]

##############################################################
## pick the "best" model in terms of topography and topology
# Dynamically extract the best parameter values for the selected starting node
best_topol = results_df.loc[results_df["topology_energy"].idxmin()]

best_topog = results_df.loc[results_df["topography_energy"].idxmin()]

best_combo = results_df.loc[results_df["total_energy"].idxmin()]


#####################################################################
# All models

ax = sns.scatterplot(
    data=results_df,
    x='topology_energy',
    y='topography_energy',
    hue='Type',
    style='Connections',
    markers={'Local': 'o', 'Global': 's'}
)

ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

plt.show()

# best models
# Compute the 5% quantiles for topology and topography energies
low_topology_thresh = results_df['topology_energy'].quantile(0.01)
low_topography_thresh = results_df['topography_energy'].quantile(0.01)

# Filter the DataFrame to include only models in the 5% lowest range on both dimensions
subset_df = results_df[
    (results_df['topology_energy'] <= low_topology_thresh) &
    (results_df['topography_energy'] <= low_topography_thresh)
]

# Plot using the filtered subset
ax = sns.scatterplot(
    data=subset_df,
    x='topology_energy',
    y='topography_energy',
    hue='Type',
    style='Connections',
    markers={'Local': 'o', 'Global': 's'}
)

# Move legend outside the plot
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

subset_df["dummy"] = 1 + np.random.normal(loc=0, scale=0.02, size=len(subset_df))

plt.figure(figsize=(1, 4))

ax = sns.scatterplot(
    data=subset_df,
    x='dummy',
    y='total_energy',
    hue='Type',
    style='Connections',
    markers={'Local': 'o', 'Global': 's'}
)

# Remove the legend
ax.get_legend().remove()

# Remove x-axis ticks and labels
ax.set_xticks([])
ax.set_xticklabels([])
ax.set_xlabel('')
ax.set_xlim(.95,1.05)



sns.countplot(
    data=subset_df,
    x='Type',
    hue='Connections'
)


#####################################################################
results_df.loc[results_df['lambdah'] == 0.0, 'Type'] = "No heterochronicity"
results_df.loc[results_df['lambdah'] == 0.0, 'Connections'] = "No heterochronicity"
best_combo = results_df.loc[results_df["total_energy"].idxmin()]



# 2d plot, eta gamma 
which_energy = "total_energy"

best_eta = best_combo['eta']
best_gamma = best_combo['gamma']
best_lambdah = best_combo['lambdah']
#best_alpha = best_combo['alpha']
best_y = best_combo['y_value']
best_z = best_combo['z_value']
best_type = best_combo['Type']
best_connect = best_combo['Connections']


# Plot energy values for each combination of eta and gamma, fixing lambdah and alpha
eta_gamma_fixed_df = results_df[
    (results_df['lambdah'] == best_lambdah) & 
    #(results_df['alpha'] == best_alpha) &
    (results_df['y_value'] == best_y) &
    (results_df['z_value'] == best_z) &
    (results_df['Type'] == best_type) &
    (results_df['Connections'] == best_connect)
]
eta_gamma_pivot = eta_gamma_fixed_df.pivot(index='eta', columns='gamma', values=which_energy)

plt.figure(figsize=(10, 8))
sns.heatmap(eta_gamma_pivot, annot=True, cmap='viridis')
plt.title(f'Energy Heatmap for eta vs gamma (lambdah={best_lambdah})')
plt.xlabel('gamma')
plt.ylabel('eta')
plt.show()

# Filter the dataframe to include multiple lambdah values, but keep other parameters fixed
subset_df = results_df[
    (results_df['y_value'] == best_y) &
    (results_df['z_value'] == best_z) &
    (results_df['Type'] == best_type) &
    (results_df['Connections'] == best_connect)
]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Extract axes
xs = subset_df['eta'].values
ys = subset_df['gamma'].values
zs = subset_df['lambdah'].values
cs = subset_df[which_energy].values

# Create a 3D scatter plot
sc = ax.scatter(xs, ys, zs, c=cs, cmap='viridis')

# Add colorbar
cbar = fig.colorbar(sc, ax=ax, pad=0.1)
cbar.set_label(which_energy)

# Set labels and title
ax.set_xlabel('eta')
ax.set_ylabel('gamma')
ax.set_zlabel('lambdah')
ax.set_title(f'3D plot of {which_energy} for varying eta, gamma, and lambdah')

plt.tight_layout()
plt.show()






# 1d plot for all params, across type * connections + no HC (i.e., lambda = 0)

best_combos = results_df.loc[results_df.groupby(["Connections", "Type"])["total_energy"].idxmin()]


plt.figure(figsize=(10, 6))

for _, row in best_combos.iterrows():
    # Extract the best parameters for this (Connections, Type) pair
    best_gamma = row['gamma']
    best_lambdah = row['lambdah']
    best_y = row['y_value']
    best_z = row['z_value']
    best_connections = row['Connections']
    best_type = row['Type']

    # Filter the original dataframe, varying only eta
    filtered_df = results_df[
        (results_df['gamma'] == best_gamma) &
        (results_df['lambdah'] == best_lambdah) &
        (results_df['y_value'] == best_y) &
        (results_df['z_value'] == best_z) &
        (results_df['Connections'] == best_connections) &
        (results_df['Type'] == best_type)
    ]

    # Plot total_energy vs eta for this (Connections, Type)
    # Each line corresponds to a different (Connections, Type) pair
    sns.lineplot(data=filtered_df, x='eta', y='total_energy', label=f"{best_type} - {best_connections}")

plt.title('Total Energy vs Eta for each (Connections, Type) best combination')
plt.xlabel('eta')
plt.ylabel('total_energy')
plt.legend()
plt.show()








# Filter subsets based on the specified conditions
subset_cumulative_global = results_df[
    (results_df['Type'] == "Cumulative") & 
    (results_df['Connections'] == "Global")
]

subset_no_hetero = results_df[
    (results_df['Type'] == "No heterochronicity")
]

# Combine subsets into one DataFrame
subset_combined = pd.concat([subset_cumulative_global, subset_no_hetero])

subset_combined["total_energy"] = -subset_combined["total_energy"]+1


# Create a 2D KDE plot for each Type separately and overlay them
plt.figure(figsize=(5, 4))

colors = {"Cumulative": "red", "No heterochronicity": "blue"}
handles = []
labels = ["With", "Whitout"]
d = -1
for t, color in colors.items():
    d += 1
    subset = subset_combined[subset_combined["Type"] == t]
    sns.kdeplot(
        data=subset, 
        x="eta", 
        y="gamma", 
        weights="total_energy",  # Use normalized total_energy for weighted KDE
        fill=True, 
        alpha=0.95,
        levels=5,
        thresh=0.96,
        color=color,
        label=t
    )
    handles.append(plt.Line2D([0], [0], marker='o', color=color, 
                              lw=0, markersize=10, label=labels[d]))

# Add a legend where colors indicate Type
plt.legend(handles=handles, title="Heterochronicity")

# Customize the plot
plt.title('2D Distribution of Total Energy Across Eta and Gamma Split by Type (Overlaid)')
plt.xlabel('Eta')
plt.ylabel('Gamma')
plt.ylim((0, 0.5))
plt.xlim((-4, -1))
plt.show()

















#####################################################################
# run again that model only, saving the networks - does it replicate?

# Load the provided .mat file to explore its contents
mat_file_path = r'C:\Users\fp02\Downloads\Consensus_Connectomes.mat'
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
coordinates = coordinates[:, [1, 0, 2]]
coordinates = coordinates - np.mean(coordinates, axis = 0)
distance_matrix = torch.from_numpy(euclidean_distances)
num_nodes = len(consensus_matrix)
num_edges = np.sum(consensus_matrix > 0) / 2
seed_adjacency_matrix = torch.zeros(num_nodes, num_nodes)
sigma = np.std(coordinates)
distance_relationship_type = "powerlaw"
matching_relationship_type = "powerlaw"
beta = 0.5
set_cumulative = True
set_local = False


distance_matrix_np = distance_matrix.numpy()

# Define spatial weight matrix W
sigma_w = 15  # Adjust sigma as needed
W = np.exp(-distance_matrix_np**2 / (2 * sigma_w**2))


# Normalize the weights for each node
W = W / W.sum(axis=1, keepdims=True)


run = 1
x_value = 0.0
alpha = 0.0
results = []
for i in range(10):
    result = run_model(x_value, best_combo['y_value'], best_combo['z_value'], 
              best_combo['eta'], best_combo['gamma'], best_combo['lambdah'], alpha, 
              run, seed_adjacency_matrix, distance_matrix, W, 
              consensus_matrix, coordinates, num_edges, 
              distance_relationship_type, matching_relationship_type, beta,
              set_cumulative, set_local, sigma, return_gnm = True)
    results.append(result)

results_best = pd.DataFrame(results)
best_model = results_best.loc[results_best["total_energy"].idxmin()]


gnms = results_best['gnm']
results_best = results_best.drop(columns ='gnm')

# average across runs
results_best_avg = results_best.groupby(["z_value", "y_value", "eta", "gamma", 
                                 "lambdah"])[results_best.columns[7:]].mean().reset_index()

# let's take the best model in terms of combined energy!
adj_mat = best_model['gnm'].adjacency_matrix

# Extract the columns of interest
cols_of_interest = ["degree_correlation", "clustering_correlation", "betweenness_correlation"]

# Get the average row as a NumPy array
avg_values = results_best_avg[cols_of_interest].iloc[0].values

# Compute Euclidean distances from each row in results_best to the average values
distances = results_best[cols_of_interest].apply(
    lambda row: np.linalg.norm(row.values - avg_values), axis=1
)

# Identify the closest row
closest_row_index = distances.idxmin()
closest_row = results_best.loc[closest_row_index]



##############
from nilearn import plotting

Greal = nx.from_numpy_array(consensus_matrix)
nx.density(Greal)

Atgt = (consensus_matrix > 0).astype(int)

degree_list = np.array([degree for node, degree in Greal.degree()])
clustering_coefficients_list = np.array(list(nx.clustering(Greal).values()))
betweenness_centrality_list = np.array(
    list(nx.betweenness_centrality(Greal, normalized=False).values())
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
    title=''
)
plt.show()




Gsynth = nx.from_numpy_array(adj_mat.numpy())

syn_degree_list = np.array([degree for node, degree in Gsynth.degree()])
syn_clustering_coefficients_list = np.array(list(nx.clustering(Gsynth).values()))
syn_betweenness_centrality_list = np.array(
    list(nx.betweenness_centrality(Gsynth, normalized=False).values())
)



# Plot using nilearn.plotting.plot_markers
plotting.plot_markers(
    node_values=syn_degree_list,
    node_coords=coordinates2,
    node_size='auto',  # Automatically scale node sizes
    node_cmap=plt.cm.viridis,  # Colormap for nodes
    alpha=0.7,  # Transparency of markers
    display_mode='ortho',  # Orthogonal views
    annotate=True,  # Add annotations for positions
    colorbar=True,  # Display a colorbar
    title=''
)
plt.show()


weight_metrics = {
    "rdegree": W.dot(degree_list),
    "rclustering": W.dot(clustering_coefficients_list),
    "rbetweenness": W.dot(betweenness_centrality_list),
    "sdegree": W.dot(syn_degree_list),
    "sclustering": W.dot(syn_clustering_coefficients_list),
    "sbetweenness": W.dot(syn_betweenness_centrality_list)
}


weight_metrics = pd.DataFrame(weight_metrics)

sns.regplot(data = weight_metrics, x = "rdegree", y = "sdegree", 
            scatter = True)

sns.regplot(data = weight_metrics, x = "rclustering", y = "sclustering", 
            scatter = True)

sns.regplot(data = weight_metrics, x = "rbetweenness", y = "sbetweenness", 
            scatter = True)



### no HC

wo_results = []
best_combo_wo = best_combos.loc[best_combos["Type"]=="No heterochronicity"]
for i in range(1):
    result = run_model(x_value, float(best_combo_wo['y_value']), float(best_combo_wo['z_value']), 
              float(best_combo_wo['eta']), float(best_combo_wo['gamma']), float(best_combo_wo['lambdah']), alpha, 
              run, seed_adjacency_matrix, distance_matrix, W, 
              consensus_matrix, coordinates, num_edges, 
              distance_relationship_type, matching_relationship_type, beta,
              set_cumulative, set_local, sigma, return_gnm = True)
    wo_results.append(result)


results_best_wo = pd.DataFrame(wo_results)
best_model_wo = results_best_wo.loc[results_best_wo["total_energy"].idxmin()]


# let's take the best model in terms of combined energy!
adj_mat_wo = best_model_wo['gnm'].adjacency_matrix



Gsynth = nx.from_numpy_array(adj_mat_wo.numpy())

syn_degree_list = np.array([degree for node, degree in Gsynth.degree()])
syn_clustering_coefficients_list = np.array(list(nx.clustering(Gsynth).values()))
syn_betweenness_centrality_list = np.array(
    list(nx.betweenness_centrality(Gsynth, normalized=False).values())
)

weight_metrics_wo = {
    "rdegree": W.dot(degree_list),
    "rclustering": W.dot(clustering_coefficients_list),
    "rbetweenness": W.dot(betweenness_centrality_list),
    "sdegree": W.dot(syn_degree_list),
    "sclustering": W.dot(syn_clustering_coefficients_list),
    "sbetweenness": W.dot(syn_betweenness_centrality_list)
}


weight_metrics_wo = pd.DataFrame(weight_metrics_wo)

sns.regplot(data = weight_metrics_wo, x = "rdegree", y = "sdegree", 
            scatter = True)

sns.regplot(data = weight_metrics_wo, x = "rclustering", y = "sclustering", 
            scatter = True)

sns.regplot(data = weight_metrics_wo, x = "rbetweenness", y = "sbetweenness", 
            scatter = True)








import numpy as np
from scipy.spatial import ConvexHull, Delaunay

def sample_points_within_brain(coordinates, num_samples):
    """
    Generates sample points within the bounding box of brain coordinates and determines
    which points are inside the brain using the convex hull approach.

    Parameters:
    - coordinates (np.ndarray): An (N, 3) array of brain coordinates.
    - num_samples (list or tuple): A list of three integers [n_x, n_y, n_z] specifying
      the number of samples along the x, y, and z axes.

    Returns:
    - inside_points (np.ndarray): An (M, 3) array of points inside the brain.
    - outside_points (np.ndarray): An (K, 3) array of points outside the brain.
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
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()

    # Combine into a single array of sample points
    sample_points = np.vstack((X, Y, Z)).T

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

    # Separate the inside and outside points
    inside_points = sample_points[inside]
    outside_points = sample_points[~inside]

    return inside_points, outside_points


# Call the function
inside_points, outside_points = sample_points_within_brain(coordinates, [0, 7, 7])

# Print the results
print("Inside Points:")
print(inside_points)

print("\nOutside Points:")
print(outside_points)


x_unique, y_unique, z_unique = sample_brain_coordinates(coordinates, [0, 7, 7])
x_values = [0]

# Create a meshgrid using the unique x, y, z values
X_mesh, Y_mesh, Z_mesh = np.meshgrid(x_unique, y_unique, z_unique, indexing='ij')

# Flatten the meshgrid to get sample points
sample_points_inside = np.vstack((X_mesh.flatten(), Y_mesh.flatten(), Z_mesh.flatten())).T


fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the brain coordinates
ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2],
           color='blue', s=20, label='Brain Coordinates')

# Plot the sample points
ax.scatter(sample_points_inside[:, 0], sample_points_inside[:, 1], sample_points_inside[:, 2],
           color='green', s=50, label='Sample Points')

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.legend()
plt.title('Unique Sample Points within the Brain')
plt.show()

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



# 1d plot for all params, best type & connections vs no HC (i.e., lambda = 0)
# x y z overimposed on brain figure



y_z_fixed_df = results_df[
    (results_df['lambdah'] == best_lambdah) & 
    (results_df['eta'] == best_eta) & 
    (results_df['gamma'] == best_gamma) &
    (results_df['Type'] == best_type) &
    (results_df['Connections'] == best_connect)
]

y_z_coordinates = np.zeros((len(y_z_fixed_df),3))
y_z_coordinates[:,2] = y_z_fixed_df['z_value']*1.2+18
y_z_coordinates[:,1] = y_z_fixed_df['y_value']*-1.05-20
y_z_coordinates[:,0] = 0#y_z_fixed_df['x_value']*1.2


# Convert to an array aligned with node indices
node_values = y_z_fixed_df['total_energy'].values
#node_values = y_z_fixed_df['topology_energy'].values
#node_values = y_z_fixed_df['topography_energy'].values

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





# plot r across values of weight
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import pearsonr

##############################################################################
# After you have run the script and obtained these:
#   best_model      -> best fit (with heterochronicity)
#   best_model_wo   -> best fit (without heterochronicity)
#   consensus_matrix-> real network adjacency
#   distance_matrix -> torch Tensor with pairwise distances
#   coordinates     -> node coordinates
#   ...
##############################################################################

# Here, I extract adjacency matrices from the best-fit models
adj_mat    = best_model['gnm'].adjacency_matrix           # with HC
adj_mat_wo = best_model_wo['gnm'].adjacency_matrix        # without HC

# Here, I convert each adjacency matrix into a NetworkX graph
Greal = nx.from_numpy_array(consensus_matrix)
Gsynth_with = nx.from_numpy_array(adj_mat.numpy())
Gsynth_without = nx.from_numpy_array(adj_mat_wo.numpy())

# Next, I compute node-level metrics for the real network
degree_list = np.array([deg for _, deg in Greal.degree()])
clustering_coefficients_list = np.array(list(nx.clustering(Greal).values()))
betweenness_centrality_list = np.array(
    list(nx.betweenness_centrality(Greal, normalized=False).values())
)

# Here, I do the same for the synthetic network (with heterochronicity)
syn_degree_list_with = np.array([deg for _, deg in Gsynth_with.degree()])
syn_clustering_coefficients_list_with = np.array(list(nx.clustering(Gsynth_with).values()))
syn_betweenness_centrality_list_with = np.array(
    list(nx.betweenness_centrality(Gsynth_with, normalized=False).values())
)

# And for the synthetic network (without heterochronicity)
syn_degree_list_without = np.array([deg for _, deg in Gsynth_without.degree()])
syn_clustering_coefficients_list_without = np.array(list(nx.clustering(Gsynth_without).values()))
syn_betweenness_centrality_list_without = np.array(
    list(nx.betweenness_centrality(Gsynth_without, normalized=False).values())
)

# Here, I convert the torch distance matrix to a NumPy array for weighting
distance_matrix_np = distance_matrix.numpy()

# I choose a range of sigma_w values to explore
sigma_values = [2, 5, 10, 15, 20, 25, 30]

results = []

# Loop over each sigma_w, build the W matrix, and compute correlations
for sigma_w in sigma_values:
    # Here, I build the Gaussian weighting matrix W and normalize row-wise
    W = np.exp(-distance_matrix_np**2 / (2 * sigma_w**2))
    W = W / W.sum(axis=1, keepdims=True)

    # Weighted real metrics
    w_rdegree = W.dot(degree_list)
    w_rclust  = W.dot(clustering_coefficients_list)
    w_rbetw   = W.dot(betweenness_centrality_list)

    # Weighted synthetic metrics (with heterochronicity)
    w_sdegree_with = W.dot(syn_degree_list_with)
    w_sclust_with  = W.dot(syn_clustering_coefficients_list_with)
    w_sbetw_with   = W.dot(syn_betweenness_centrality_list_with)

    # Weighted synthetic metrics (without heterochronicity)
    w_sdegree_wo = W.dot(syn_degree_list_without)
    w_sclust_wo  = W.dot(syn_clustering_coefficients_list_without)
    w_sbetw_wo   = W.dot(syn_betweenness_centrality_list_without)

    # I compute Pearson r for each metric in each model
    corr_with_degree = pearsonr(w_rdegree, w_sdegree_with)[0]
    corr_with_clust  = pearsonr(w_rclust,  w_sclust_with )[0]
    corr_with_betw   = pearsonr(w_rbetw,   w_sbetw_with  )[0]

    corr_wo_degree   = pearsonr(w_rdegree, w_sdegree_wo  )[0]
    corr_wo_clust    = pearsonr(w_rclust,  w_sclust_wo   )[0]
    corr_wo_betw     = pearsonr(w_rbetw,   w_sbetw_wo    )[0]

    # I store these correlation values in a tidy DataFrame format
    results.append({
        "sigma_w": sigma_w, "metric": "degree",
        "model": "with_HC", "correlation": corr_with_degree
    })
    results.append({
        "sigma_w": sigma_w, "metric": "clustering",
        "model": "with_HC", "correlation": corr_with_clust
    })
    results.append({
        "sigma_w": sigma_w, "metric": "betweenness",
        "model": "with_HC", "correlation": corr_with_betw
    })
    results.append({
        "sigma_w": sigma_w, "metric": "degree",
        "model": "without_HC", "correlation": corr_wo_degree
    })
    results.append({
        "sigma_w": sigma_w, "metric": "clustering",
        "model": "without_HC", "correlation": corr_wo_clust
    })
    results.append({
        "sigma_w": sigma_w, "metric": "betweenness",
        "model": "without_HC", "correlation": corr_wo_betw
    })

# Convert to a DataFrame for plotting
corr_df = pd.DataFrame(results)

# Plot correlation vs. sigma_w with multiple lines
plt.figure(figsize=(7, 5))
sns.lineplot(
    data=corr_df,
    x="sigma_w",
    y="correlation",
    hue="model",
    style="metric",
    markers=True
)
plt.title("Correlation vs. sigma_w (Real vs. Synthetic Networks)")
plt.xlabel("sigma_w")
plt.ylabel("Pearson r")
plt.legend(title="Model / Metric", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()


## check what we are getting wrong

# edge there in both - only in sim - only in real:
# look at length on connection and location, distance from HC start point
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

##############################################################################
# After you have run your script, you have:
#   consensus_matrix -> real adjacency matrix (NumPy array)
#   adj_mat          -> best synthetic adjacency (with heterochronicity), torch.Tensor
#   adj_mat_wo       -> best synthetic adjacency (without heterochronicity), torch.Tensor
#   distance_matrix  -> torch Tensor with pairwise distances
#   coordinates      -> (N,3) array of node coordinates
##############################################################################

# Convert adjacency matrices to NumPy arrays if they are torch Tensors
Areal = (consensus_matrix > 0).astype(int)        # Real adjacency (binarize if needed)
Awith = (adj_mat.numpy()    > 0).astype(int)      # Synthetic with HC
Awo   = (adj_mat_wo.numpy() > 0).astype(int)      # Synthetic w/o HC

# Convert distance matrix to NumPy
dist_mat = distance_matrix.numpy()

# We only consider the upper triangle if your adjacency is undirected
# (since i-j and j-i are the same edge). But if adjacency is directed,
# you might skip this step.
triu_idx = np.triu_indices_from(Areal, k=1)

##############################################################################
# 1) Categorize edges for the "with heterochronicity" adjacency
##############################################################################
# Overlap edges: present in both real and synthetic
overlap_mask_with = (Areal[triu_idx] == 1) & (Awith[triu_idx] == 1)

# Missing edges: present in real but absent in synthetic
missing_mask_with = (Areal[triu_idx] == 1) & (Awith[triu_idx] == 0)

# Extra edges: absent in real but present in synthetic
extra_mask_with = (Areal[triu_idx] == 0) & (Awith[triu_idx] == 1)

##############################################################################
# 2) Categorize edges for the "without heterochronicity" adjacency
##############################################################################
overlap_mask_wo = (Areal[triu_idx] == 1) & (Awo[triu_idx] == 1)
missing_mask_wo = (Areal[triu_idx] == 1) & (Awo[triu_idx] == 0)
extra_mask_wo   = (Areal[triu_idx] == 0) & (Awo[triu_idx] == 1)

##############################################################################
# 3) Look at the lengths (distance) of edges in these categories
##############################################################################
distances = dist_mat[triu_idx]  # distances for upper-triangular pairs only

df_with = pd.DataFrame({
    "distance": distances,
    "type": np.select(
        [overlap_mask_with, missing_mask_with, extra_mask_with],
        ["overlap", "missing", "extra"],
        default="none"  # For pairs that are absent in both real and synthetic
    ),
    "model": "with_HC"
})

df_wo = pd.DataFrame({
    "distance": distances,
    "type": np.select(
        [overlap_mask_wo, missing_mask_wo, extra_mask_wo],
        ["overlap", "missing", "extra"],
        default="none"
    ),
    "model": "without_HC"
})

# Combine into one dataframe for easier plotting
df_all = pd.concat([df_with, df_wo], ignore_index=True)

# Filter out the “none” category (which is edges missing from both real and synthetic)
df_all = df_all[df_all["type"] != "none"]

##############################################################################
# 4) Plot the distribution of distances for each category and model
##############################################################################
plt.figure(figsize=(8, 5))
sns.kdeplot(
    data=df_all,
    x="distance",
    hue="type",
    multiple="stack",
    fill=True,
    alpha=0.7
)
plt.title("Distribution of Connection Distances (with_HC example)")
plt.xlabel("Edge Euclidean Distance")
plt.ylabel("Density")
plt.legend(title="Edge Category")
plt.show()

# If we want separate subplots for overlap vs. missing vs. extra, grouped by model
g = sns.FacetGrid(df_all, col="type", hue="model", margin_titles=True)
g.map(sns.histplot, "distance", kde=True, alpha=0.7)
g.add_legend()
plt.show()


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

##############################################################################
# After running your script, you have:
#   consensus_matrix -> Real adjacency (NumPy array)
#   adj_mat          -> best synthetic adjacency (with HC), torch.Tensor
#   adj_mat_wo       -> best synthetic adjacency (without HC), torch.Tensor
#   distance_matrix  -> torch.Tensor with pairwise distances, same shape
##############################################################################

# Convert adjacency matrices to binary (1 if edge present, else 0).
# If your real adjacency is already binary, you can skip the “>0”.
Areal  = (consensus_matrix > 0).astype(int)        
Awith  = (adj_mat.numpy()    > 0).astype(int)      
Awo    = (adj_mat_wo.numpy() > 0).astype(int)

# Convert the distance matrix to a NumPy array
dist_mat = distance_matrix.numpy()

# Upper-triangle indices (assuming undirected adjacency)
triu_idx = np.triu_indices_from(Areal, k=1)

# Extract distances only in the upper triangle
upper_distances = dist_mat[triu_idx]

# For each adjacency, gather the edge lengths
real_distances = upper_distances[Areal[triu_idx] == 1]
with_distances = upper_distances[Awith[triu_idx] == 1]
wo_distances   = upper_distances[Awo[triu_idx]   == 1]

# Combine into a single DataFrame for plotting
df_plot = pd.DataFrame({
    "distance": np.concatenate([real_distances, with_distances, wo_distances]),
    "model": (
        ["Real"] * len(real_distances) 
        + ["With_HC"] * len(with_distances) 
        + ["No_HC"]   * len(wo_distances)
    )
})

##############################################################################
# Plot the number of connections by length for each adjacency
##############################################################################
plt.figure(figsize=(8, 5))
sns.histplot(
    data=df_plot,
    x="distance",
    hue="model",
    stat="count",        # show counts (the number of edges)
    element="step",      # step-like histogram
    multiple="layer",    # overlap the histograms
    alpha=0.5,
    bins=30              # adjust as desired
)
plt.title("Number of Connections by Length")
plt.xlabel("Edge Length")
plt.ylabel("Number of Connections")
#plt.legend(title="Model")
plt.tight_layout()
plt.show()



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from nilearn import plotting

##############################################################################
# After your script, you have:
#   consensus_matrix -> the real adjacency (NumPy array)
#   adj_mat          -> best synthetic adjacency (with heterochronicity), torch.Tensor
#   adj_mat_wo       -> best synthetic adjacency (without heterochronicity), torch.Tensor
#   coordinates      -> (N, 3) array of node coordinates in MNI-like space
##############################################################################

##############################################################################
# 1) Convert adjacency matrices to NumPy arrays (and binarize if you want presence/absence).
##############################################################################
A_real = (consensus_matrix > 0).astype(int)
A_with = (adj_mat.numpy()    > 0).astype(int)  # with heterochronicity
A_wo   = (adj_mat_wo.numpy() > 0).astype(int)  # without heterochronicity

##############################################################################
# 2) Build a color-coded adjacency for "with HC":
#    overlap  ->  0.5
#    missing  -> -0.5
#    extra    ->  1.5
#    none     ->  0.0 (edge absent in both)
##############################################################################
Aplot_with = np.zeros_like(A_real, dtype=float)

overlap_mask  = (A_real == 1) & (A_with == 1)
missing_mask  = (A_real == 1) & (A_with == 0)
extra_mask    = (A_real == 0) & (A_with == 1)

Aplot_with[overlap_mask] = 0.5
Aplot_with[missing_mask] = -0.5
Aplot_with[extra_mask]   = 1.5

# To ensure symmetry if your network is undirected:
Aplot_with = np.triu(Aplot_with) + np.triu(Aplot_with, k=1).T

##############################################################################
# 3) Build a color-coded adjacency for "without HC":
##############################################################################
Aplot_wo = np.zeros_like(A_real, dtype=float)

overlap_mask_wo = (A_real == 1) & (A_wo == 1)
missing_mask_wo = (A_real == 1) & (A_wo == 0)
extra_mask_wo   = (A_real == 0) & (A_wo == 1)

Aplot_wo[overlap_mask_wo] = 0.5
Aplot_wo[missing_mask_wo] = -0.5
Aplot_wo[extra_mask_wo]   = 1.5

Aplot_wo = np.triu(Aplot_wo) + np.triu(Aplot_wo, k=1).T

##############################################################################
# 4) Define a discrete colormap:
#    We want:
#      - -0.5 = "darkred"     (missing edges)
#      -  0.0 = "white"       (none)
#      -  0.5 = "darkgreen"   (overlap)
#      -  1.5 = "darkblue"    (extra)
#
#    We set up “bounds” so that we map data values [-1, -0.1], [0.1, 1.0], etc.
##############################################################################
colors = ["darkred", "white", "darkgreen", "darkblue"]
bounds = [-1.0, -0.1, 0.1, 1.0, 2.0]
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(bounds, cmap.N)

##############################################################################
# 5) Plot both connectomes using nilearn
##############################################################################

# (A) With heterochronicity
plt.figure(figsize=(8, 6))
plotting.plot_connectome(
    adjacency_matrix=Aplot_with,
    node_coords=coordinates2,
    edge_cmap=cmap,
    edge_vmin=-1,
    edge_vmax=2,
    node_color='black',         # Each node can get a random color if you prefer
    node_size=20,
    edge_kwargs={'alpha': 0.25},  
    edge_threshold=None,       # Keep all edges; we can also pass e.g. '0%' or None
    display_mode='ortho',
    figure=plt.gcf(),
    title="With HC vs. Real",
    annotate=True,
    black_bg=False,
    alpha=0.9,
    colorbar=True
)
plt.show()

# (B) Without heterochronicity
plt.figure(figsize=(8, 6))
plotting.plot_connectome(
    adjacency_matrix=Aplot_wo,
    node_coords=coordinates2,
    edge_cmap=cmap,
    edge_vmin=-1,
    edge_vmax=2,
    node_color='black',
    node_size=20,
    edge_kwargs={'alpha': 0.25},  
    edge_threshold=None,
    display_mode='ortho',
    figure=plt.gcf(),
    title="Without HC vs. Real",
    annotate=True,
    black_bg=False,
    alpha=0.9,
    colorbar=True
)
plt.show()



############################

##############################################################################
# We assume from your script we already have:
#   consensus_matrix -> real adjacency (NumPy array)
#   adj_mat          -> best synthetic adjacency (with heterochronicity), torch.Tensor
#   distance_matrix  -> pairwise distances (torch.Tensor)
#   W                -> NxN spatial weight matrix (NumPy array), row-normalized
#   coordinates2     -> (N, 3) node coordinates (NumPy array)
##############################################################################

sigma_w = 15
W = np.exp(-distance_matrix_np**2 / (2 * sigma_w**2))
W = W / W.sum(axis=1, keepdims=True)

# 1) Convert key variables to NumPy arrays and binarize real and synthetic.
#    If your adjacency is weighted, adapt as needed.
A_real = (consensus_matrix > 0).astype(float)
A_syn  = (adj_mat.numpy()    > 0).astype(float)

# 2) Construct fuzzy adjacency: A_fuzzy = W @ A_syn @ W^T
A_fuzzy = W @ A_syn @ W.T


min_val = A_fuzzy.min()
max_val = A_fuzzy.max()

A_fuzzy = (A_fuzzy - min_val) / (max_val - min_val)

sns.heatmap(A_fuzzy)


# 3) Define a continuous probability p_match where:
#    - If real[i,j] = 1, p_match[i,j] = A_fuzzy[i,j].
#      (Larger => better match for a real edge)
#    - If real[i,j] = 0, p_match[i,j] = 1 - A_fuzzy[i,j].
#      (Smaller A_fuzzy => better match for a real non-edge)
p_match = np.zeros_like(A_real, dtype=float)
p_match[A_real == 1] = A_fuzzy[A_real == 1]
#p_match[A_real == 0] = 1 - A_fuzzy[A_real == 0]

sns.heatmap(p_match)

# Because our network is undirected, ensure symmetry if desired:
p_match = np.triu(p_match) + np.triu(p_match, k=1).T

##############################################################################
# 4) Plot p_match with nilearn’s plot_connectome
#    - Now, each edge has a value from ~0 (poor match) to ~1 (good match)
##############################################################################

plt.figure(figsize=(8, 6))

plotting.plot_connectome(
    adjacency_matrix=p_match,
    node_coords=coordinates2,
    edge_vmin=0,
    edge_vmax=1,
    edge_cmap='viridis',          # or another continuous colormap
    node_color='black',           # color of nodes
    node_size=50,
    edge_threshold=None,          # show all edges
    edge_kwargs={'alpha': 0.8},   # reduce edges alpha for clarity
    display_mode='ortho',
    title="",
    annotate=True,
    black_bg=False,
    colorbar=True
)
plt.show()



# plot various network metrics for each seed for R, S and R-S

# long vs short connections?
# plot connection length on x axis, metrics on y axis - with SD!
# 

model = gnms[1]
dir(best_model.gnm)
a = best_model.gnm.adjacency_snapshots









import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
from nilearn import plotting

def animate_connectome_in_time(
    added_edges_list,
    node_coords,
    out_gif_path="connectome_evolution.gif",
    duration=0.5,
    edge_cmap="viridis"
):
    """
    Create a GIF of the network growing edge by edge, showing only the 
    nodes connected so far.

    Parameters
    ----------
    added_edges_list : list of tuples
        Each tuple is (i, j), indicating the i-th and j-th node were 
        connected at this time step. Must be in chronological order.
    node_coords : np.ndarray, shape (num_nodes, 3)
        3D coordinates for each node (e.g., MNI space).
    out_gif_path : str
        File path to save the resulting GIF.
    duration : float
        Number of seconds to display each frame in the GIF.
    edge_cmap : str
        Name of a matplotlib colormap for coloring edges (e.g. 'viridis').

    Returns
    -------
    None
    """

    # Number of total nodes
    num_nodes = node_coords.shape[0]
    # Create a full adjacency array, initially empty
    adjacency = np.zeros((num_nodes, num_nodes), dtype=float)

    # Make a folder to store temporary frames
    os.makedirs("frames", exist_ok=True)
    frame_paths = []

    # Step through the edges in chronological order
    for step_index, (i, j) in enumerate(added_edges_list, start=1):
        # Add the new edge to the adjacency
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0

        # Determine which nodes have appeared so far
        # i.e., any node that has at least one edge
        connected_indices = np.unique(np.where(adjacency > 0)[0])
        
        # Subset the adjacency and the node coordinates
        subA = adjacency[np.ix_(connected_indices, connected_indices)]
        sub_coords = node_coords[connected_indices]

        # --- Plotting ---
        # We use Nilearn’s "plot_connectome" to plot only subA & sub_coords
        display = plotting.plot_connectome(
            adjacency_matrix=subA,
            node_coords=sub_coords,
            edge_threshold=None,     # Show all edges > 0
            edge_cmap=edge_cmap,
            node_color='black',      # Make nodes black
            node_size=50,            # You can adjust as you like
            display_mode='ortho',    # Or "x", "y", "z", etc.
            colorbar=True,
            edge_vmin=0, edge_vmax=1,
            title=f"Step {step_index}: just added edge ({i}, {j})",
        )
        
        # Save the image to a file
        frame_filename = os.path.join("frames", f"frame_{step_index:03d}.png")
        display.savefig(frame_filename, dpi=100)
        display.close()
        frame_paths.append(frame_filename)

    # --- Build a GIF from the saved frames ---
    frames = [imageio.imread(fname) for fname in frame_paths]
    imageio.mimsave(out_gif_path, frames, duration=duration)
    print(f"GIF saved to {out_gif_path}")


animate_connectome_in_time(
    added_edges_list=best_model.gnm.added_edges_list,
    node_coords=coordinates2,       # shape (num_nodes, 3)
    out_gif_path="network_growth.gif",
    duration=0.5
)
