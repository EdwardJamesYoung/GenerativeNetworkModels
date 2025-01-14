# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 18:54:23 2024

@author: fp02
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


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


def rank_pareto_solutions(pareto_df, objectives, objective_weights):
    """
    Implements the R-method to rank Pareto-optimal solutions and select the best solution.
    
    Args:
        pareto_df (pd.DataFrame): DataFrame containing Pareto-optimal solutions and their objective values.
        objectives (list of str): List of objective columns in the DataFrame to consider.
        objective_weights (dict): Dictionary of weights assigned to each objective based on their rank.

    Returns:
        pd.DataFrame: Pareto DataFrame with composite scores and ranks added.
        pd.Series: The best Pareto-optimal solution.
    """
    # Normalize objective values for fair comparison
    for obj in objectives:
        pareto_df[f"{obj}_norm"] = (pareto_df[obj] - pareto_df[obj].min()) / (pareto_df[obj].max() - pareto_df[obj].min())

    # Assign ranks to Pareto solutions for each objective
    for obj in objectives:
        pareto_df[f"{obj}_rank"] = pareto_df[f"{obj}_norm"].rank(method='min', ascending=True)

    # Compute weights for each Pareto solution based on objective ranks
    num_solutions = len(pareto_df)
    pareto_df["composite_score"] = 0
    for obj in objectives:
        obj_weight = objective_weights[obj]
        pareto_df["composite_score"] += obj_weight * (1 / pareto_df[f"{obj}_rank"])

    # Assign composite ranks based on the composite score
    pareto_df["composite_rank"] = pareto_df["composite_score"].rank(method='min', ascending=False)

    # Select the best Pareto solution (highest composite score)
    best_solution = pareto_df.loc[pareto_df["composite_rank"].idxmin()]

    return pareto_df, best_solution



results_df= pd.read_csv(r'../Data/results_simple_local.csv')

results_df = np.round(results_df,2)
results_df2= pd.read_csv(r'../Code/results_cumulative_local.csv')
results_df2 = np.round(results_df2,2)

# average across runs
results_df = results_df.groupby(["z_value", "y_value", "eta", "gamma", "lambdah"]).mean().reset_index()
results_df2 = results_df2.groupby(["z_value", "y_value", "eta", "gamma", "lambdah"]).mean().reset_index()
#results_df2 = results_df.groupby(["z_value", "y_value", "eta", "gamma", "lambdah"]).std().reset_index()

#results_df["sd_topography"] = results_df2["topography_energy"]
#results_df["sd_topology"] = results_df2["topology_energy"]
#results_df["sd_total"] = results_df2["total_energy"]

results_df["Type"] = "Simple"
results_df["Connections"] = "Local"

results_df2["Type"] = "Cumulative"
results_df2["Connections"] = "Local"

#results_df = pd.concat([results_df, results_df2])

# all models are wrong but some are better than other.
# i.e., what are you trying to capture?
beta = 0.5
results_df["total_energy"] = beta * results_df["topology_energy"] + (1 - beta) * results_df["topography_energy"]
results_df2["total_energy"] = beta * results_df2["topology_energy"] + (1 - beta) * results_df2["topography_energy"]

sns.regplot(results_df, x = 'topology_energy', y = 'topography_energy', scatter=False)
plt.show()

# Simulate example data
pareto_data = results_df.copy()

# Normalize the energies
#pareto_data['topology_energy'] = (pareto_data['topology_energy'] - pareto_data['topology_energy'].min()) / \
#                                      (pareto_data['topology_energy'].max() - pareto_data['topology_energy'].min())

#pareto_data['topography_energy'] = (pareto_data['topography_energy'] - pareto_data['topography_energy'].min()) / \
#                                        (pareto_data['topography_energy'].max() - pareto_data['topography_energy'].min())


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

# Determine the number of rows and columns for the subplot grid
num_vars = len(pareto_front.columns)
num_cols = 4  # You can adjust this number based on your preference
num_rows = (num_vars + num_cols - 1) // num_cols  # Ceiling division to get enough rows

# Set up the matplotlib figure and axes
fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*4, num_rows*4))

# Flatten the axes array for easy indexing
axes = axes.flatten()

# Create a violin plot for each numerical variable
for i, col in enumerate(pareto_front.columns):
    sns.violinplot(y=pareto_front[col], ax=axes[i], color='skyblue')
    axes[i].set_title(col)

# Remove any unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

pareto_front2 = pareto_frontier(results_df2, ['topology_energy', 'topography_energy'])

pareto_front = pd.concat([pareto_front, pareto_front2])

pareto_front = pareto_front.drop(columns=["x_value","alpha"])

results_df0 = pd.concat([results_df, results_df2])
results_df0 = results_df.loc[results_df["lambdah"] <0.0001,:]
results_df0["Type"] = "No HC"
pareto_front0 = pareto_frontier(results_df0, ['topology_energy', 'topography_energy'])
pareto_front0 = pareto_front0.drop(columns=["x_value","alpha"])

pareto_front = pd.concat([pareto_front, pareto_front0])


sns.kdeplot(pareto_front, x = "total_energy", hue = "Type")

sns.kdeplot(pareto_front, x = "topology_energy", hue = "Type")

sns.kdeplot(pareto_front, x = "topography_energy", hue = "Connections")




sns.kdeplot(pareto_front, x = "eta", hue = "Connections")
plt.xlim((-5,0))

sns.kdeplot(pareto_front, x = "gamma", hue = "Connections")
plt.xlim((0,1))

sns.kdeplot(pareto_front, x = "lambdah", hue = "Connections")
plt.xlim((0,5))

sns.kdeplot(pareto_front, x = "z_value", hue = "Connections")

sns.kdeplot(pareto_front, x = "y_value", hue = "Connections")







#results_df = results_df.loc[results_df["lambdah"] <0.0001,:]


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
    (results_df['eta'] == best_eta) & 
    (results_df['alpha'] == best_alpha) &
    (results_df['y_value'] == best_y) &
    (results_df['z_value'] == best_z)
]
eta_lambdah_pivot = eta_lambdah_fixed_df.pivot(index='gamma', columns='lambdah', values=which_energy)

plt.figure(figsize=(10, 8))
sns.heatmap(eta_lambdah_pivot, annot=True, cmap='viridis')
plt.title(f'Energy Heatmap for gamma vs lambdah (gamma={best_gamma}, alpha={best_alpha})')
plt.xlabel('lambdah')
plt.ylabel('gamma')
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






###############################################################################


which_energy = 'total_energy'
# Dynamically extract the best parameter values for the selected starting node
best_params = results_df.loc[results_df[which_energy].idxmin()]
best_eta = best_params['eta']
best_gamma = best_params['gamma']
best_lambdah = best_params['lambdah']
best_alpha = best_params['alpha']
best_y = best_params['y_value']
best_z = best_params['z_value']



y_fixed_df = results_df[
    (results_df['y_value'] == best_y) & 
    (results_df['lambdah'] == best_lambdah) &
    (results_df['gamma'] == best_gamma) &
    (results_df['alpha'] == best_alpha) &
    (results_df['z_value'] == best_z)
]

best_params = results_df2.loc[results_df[which_energy].idxmin()]
best_eta = best_params['eta']
best_gamma = best_params['gamma']
best_lambdah = best_params['lambdah']
best_alpha = best_params['alpha']
best_y = best_params['y_value']
best_z = best_params['z_value']



y_fixed_df2 = results_df2[
    (results_df2['y_value'] == best_y) & 
    (results_df2['lambdah'] == best_lambdah) &
    (results_df2['gamma'] == best_gamma) &
    (results_df2['alpha'] == best_alpha) &
    (results_df2['z_value'] == best_z)
]

y_fixed_df = pd.concat([y_fixed_df, y_fixed_df2])


best_params = results_df0.loc[results_df0[which_energy].idxmin()]
best_eta = best_params['eta']
best_gamma = best_params['gamma']
best_lambdah = best_params['lambdah']
best_alpha = best_params['alpha']
best_y = best_params['y_value']
best_z = best_params['z_value']



y_fixed_df2 = results_df0[
    (results_df0['y_value'] == best_y) & 
    (results_df0['lambdah'] == best_lambdah) &
    (results_df0['gamma'] == best_gamma) &
    (results_df0['alpha'] == best_alpha) &
    (results_df0['z_value'] == best_z)
]

y_fixed_df = pd.concat([y_fixed_df, y_fixed_df2])

plt.figure(figsize=(10, 8))
sns.kdeplot(y_fixed_df, x = "eta", y = "total_energy", hue="Connections")

plt.legend()
plt.show()




plt.figure(figsize=(10, 8))
sns.kdeplot(pareto_front, x = "z_value")
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



















######################

















