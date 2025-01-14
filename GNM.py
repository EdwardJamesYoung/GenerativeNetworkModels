import jaxtyping
from jaxtyping import Float, jaxtyped
from typing import Optional, Tuple, Union, List
from typeguard import typechecked

from matching_index import matching_index
from optimisation_criteria import DistanceWeightedCommunicability, WeightedDistance

import torch
import torch.optim as optim


class GenerativeNetworkModel():
    @jaxtyped(typechecker=typechecked)
    def __init__(self,
                 # The first set of arguments are for both the binary and weighted GNM. 
                 seed_adjacency_matrix: Float[torch.Tensor, "num_nodes num_nodes"],
                 distance_matrix: Float[torch.Tensor, "num_nodes num_nodes"],
                 eta: float,
                 gamma: float,
                 lambdah: float,
                 distance_relationship_type: str,
                 matching_relationship_type: str,
                 prob_offset: float = 1e-6,
                 # The following arguments are for the weighted GNM. Leave unspecified for binary GNM.
                 seed_weight_matrix: Optional[Float[torch.Tensor, "num_nodes num_nodes"]] = None,
                 alpha: Optional[float] = None,
                 optimisation_criterion: Optional[str] = None,
                 optimisation_criterion_kwargs: Optional[dict] = None,
                 optimisation_normalisation: Optional[bool] = None,
                 weight_lower_bound: Optional[float] = None,
                 weight_upper_bound: Optional[float] = None,
                 maximise_criterion: Optional[bool] = False
                 ):
        """
        Initilisation method for the generative network model.

        Parameters:
                ------ The following arguments are for both the binary and weighted GNM. ------
            - seed_adjacency_matrix (Pytorch tensor of shape (num_nodes, num_nodes)): The initial adjacency matrix for the graph.
            - distance_matrix (Pytorch tensor of shape (num_nodes, num_nodes)): The distances between each pair of nodes in the graph.
            - eta (float): Parameter controlling the influence of distance on wiring probability.
            - gamma (float): Parameter controlling the influence of matching index on wiring probability.
            - lambdah (float): Parameter controlling the influence of heterochronicity on wiring probability.
            - distance_relationship_type (str): The relationship between distance and wiring probability. Must be one of 'powerlaw' or 'exponential'. 
            - matching_relationship_type (str): The relationship between the matching index and wiring probability. Must be one of 'powerlaw' or 'exponential'.
            - prob_offset (float): Small constant added to unnormalised probabilities to prevent division by zero. Defaults to 1e-6.
                ------ The following arguments are for the weighted GNM. Leave unspecified for binary GNM. ------
            - seed_weight_matrix (Pytorch tensor of shape (num_nodes, num_nodes), optional): The initial weight matrix for the graph. Must be symmetric, non-negative, and only non-zero on those elements that the adjacency matrix is non-zero. Defaults to None.
            - alpha (float, optional): The learning rate to apply to the weights for optimisation. Defaults to None.
            - optimisation_criterion (str, optional): Which function to perform optimisation on in order to normalise the weights. Defaults to None.
            - optimisation_criterion_kwargs (dict, optional): The keyword arguments to pass to the optimisation criterion. Defaults to None.
            - optimisation_normalisation (bool, optional): Whether to normalise the function before applying gradients. Defaults to None.
            - weight_lower_bound (float, optional): The smallest value a weight is allowed to reach. Defaults to None.
            - weight_upper_bound (float, optional): The largest value a weight is allowed to reach. Defaults to None.
            - maximise_criterion (bool, optional): Whether to maximise the optimisation criterion. Defaults to False.
        """
        self.seed_adjacency_matrix = seed_adjacency_matrix
        self.distance_matrix = distance_matrix
        
        # Perform various checks the seed adjacency matrix and the distance matrix.
        # Check that the seed_adjacency_matrix is binary.
        if not torch.all((self.seed_adjacency_matrix == 0) | (self.seed_adjacency_matrix == 1)):
            raise ValueError(f"The seed_adjacency_matrix must be binary (contain only 0s and 1s). Recieved {self.seed_adjacency_matrix}.")
        # Check that the distance matrix is symmetric.
        if not torch.allclose(self.distance_matrix, self.distance_matrix.T):
            raise ValueError(f"The distance matrix must be symmetric. Recieved {self.distance_matrix}.")
        # Check that the distance matrix is non-negative.
        if torch.any(self.distance_matrix < 0):
            raise ValueError(f"The distance matrix must be non-negative. Recieved {self.distance_matrix}.")
        # Check that the diagonals are all zero.
        if torch.any(torch.diag(self.seed_adjacency_matrix) != 0):
            print("The adjacency matrix contains self-connections. Removing them.")
            self.seed_adjacency_matrix.fill_diagonal_(0)

        # Perform checks on the distance and matching index relationship type.
        if distance_relationship_type not in ['powerlaw', 'exponential']:
            raise NotImplementedError(f"Distance relationship type '{distance_relationship_type}' is not supported for the binary GNM.")
        if matching_relationship_type not in ['powerlaw', 'exponential']:
            raise NotImplementedError(f"Matching relationship type '{matching_relationship_type}' is not supported for the binary GNM.")

        # Initialise the remainder of the arguments for the binary GNM.
        # This will be updated as the model runs, while the seed is kept static.
        self.adjacency_matrix = self.seed_adjacency_matrix.clone() 
        # The number of nodes in the graph.
        self.num_nodes = self.seed_adjacency_matrix.shape[0] 
        # The parameter controlling the influence of distance.
        self.eta = eta 
        # The parameter controlling the influence of the matching index.
        self.gamma = gamma 
        # The parameter controlling the influence of heterochronicity.
        self.lambdah = lambdah 
        # The relationship type for distance.
        self.distance_relationship_type = distance_relationship_type 
        # The relationship type for the matching index.
        self.matching_relationship_type = matching_relationship_type 
        # The small constant added to unnormalised probabilities to prevent division by zero.
        self.prob_offset = prob_offset 

        # Initialise the distance cost matrix for the binary GNM.
        if self.distance_relationship_type == 'powerlaw':
            self.distance_factor = distance_matrix.pow(self.eta)
        elif self.distance_relationship_type == 'exponential':
            self.distance_factor = torch.exp(eta * distance_matrix)

        
        # If alpha is passed in as an argument, assume we have a weighted GNM. 
        if alpha is not None:
            # The learning rate to use to update the weights. 
            self.alpha = alpha
            
            if seed_weight_matrix is None:
                # If the seed weight matrix is unspecified, then initialise it at the seed adjacency matrix.
                self.seed_weight_matrix = self.seed_adjacency_matrix.clone()
            else:
                # Perform checks on the passed in seed weight matrix.
                # Check that the seed_weight_matrix is symmetric.
                if not torch.allclose(seed_weight_matrix, seed_weight_matrix.T):
                    raise ValueError(f"The seed_weight_matrix must be symmetric. Recieved {seed_weight_matrix}.")
                # Check that the seed_weight_matrix is non-negative.
                if torch.any(seed_weight_matrix < 0):
                    raise ValueError(f"The seed_weight_matrix must be non-negative. Recieved {seed_weight_matrix}.")
                # Check that the seed weight matrix has support only where the seed adjacency matrix is non-zero.
                if torch.any((self.seed_adjacency_matrix == 0) & (seed_weight_matrix != 0)):
                    raise ValueError(f"The seed_weight_matrix must have support only where the seed_adjacency_matrix is non-zero. Recieved seed adjacency matrix {self.seed_adjacency_matrix} and seed weight matrix {seed_weight_matrix}.")
                
                self.seed_weight_matrix = seed_weight_matrix
            
            self.weight_matrix = self.seed_weight_matrix.clone().requires_grad_(True)
            
            # Set the function to be optimised by the weights
            if optimisation_criterion is None:
                print("Optimisation criterion was unspecified. Defaulting to 'distance_weighted_communicability'.")
                self.optimisation_criterion_name = 'distance_weighted_communicability'
            else:
                self.optimisation_criterion_name = optimisation_criterion
            if optimisation_normalisation is None:
                print("Optimisation normalisation was unspecified. Defaulting to False.")
                self.optimisation_normalisation = False 
            else:
                self.optimisation_normalisation = optimisation_normalisation           

            # Set the optimisation criterion for the weights.
            if self.optimisation_criterion_name == 'distance_weighted_communicability':
                assert "omega" in optimisation_criterion_kwargs, "The 'omega' parameter must be specified for the 'distance_weighted_communicability' optimisation criterion."
                self.optimisation_criterion = DistanceWeightedCommunicability(normalisation = self.optimisation_normalisation, distance_matrix = self.distance_matrix, **optimisation_criterion_kwargs)
            elif self.optimisation_criterion_name == "weighted_distance":
                self.optimisation_criterion = WeightedDistance(normalisation = self.optimisation_normalisation, distance_matrix = self.distance_matrix, **optimisation_criterion_kwargs)
            else:
                raise NotImplementedError("The specified optimisation criterion is not yet supported.")


            # Set the lower and upper bounds for the weights.
            if weight_lower_bound is not None:
                if weight_lower_bound < 0.0:
                    raise ValueError(f"The weight_lower_bound must be non-negative. Recieved {self.weight_lower_bound}.")
                
                self.weight_lower_bound = weight_lower_bound
            else: 
                print("Weight lower bound was unspecified. Defaulting to 0.0.")
                self.weight_lower_bound = 0.0
            
            if weight_upper_bound is not None:
                if weight_upper_bound < 0.0:
                    raise ValueError(f"The weight_upper_bound must be non-negative. Recieved {self.weight_upper_bound}.")
            
                self.weight_upper_bound = weight_upper_bound
            else:
                print("Weight upper bound was unspecified. Defaulting to infinity.")
                self.weight_upper_bound = float('inf')

            # Initialise the optimiser for the weights.
            self.optimiser = optim.SGD([self.weight_matrix], lr=self.alpha, maximize=maximise_criterion)


    @jaxtyped(typechecker=typechecked)
    def binary_update(self, heterochronous_matrix: Optional[Float[torch.Tensor, "{self.num_nodes} {self.num_nodes}"]] = None) -> Tuple[Tuple[int, int], Float[torch.Tensor, "num_nodes num_nodes"]]:
        """
        Performs one update step of the adjacency matrix for the binary GNM.

        Parameters:
            - heterochronous_matrix (Pytorch tensor of shape (num_nodes, num_nodes), optional): The heterochronous development probability matrix. Defaults to None.

        Returns:
            - added_edges (Tuple[int, int]): The edge that was added to the adjacency matrix.
            - adjacency_matrix (Pytorch tensor of shape (num_nodes, num_nodes)): (A copy of) the updated adjacency matrix.
        """

        if heterochronous_matrix is None:
            heterochronous_matrix = torch.ones((self.num_nodes, self.num_nodes), dtype=self.seed_adjacency_matrix.dtype)
        
        matching_index_matrix = matching_index(self.adjacency_matrix)
        # Add on the prob_offset term to prevent zero to the power of negative number
        matching_index_matrix[matching_index_matrix == 0] += self.prob_offset
        
        if self.matching_relationship_type == 'powerlaw':
            matching_factor = matching_index_matrix.pow(self.gamma)
            heterochronous_factor = torch.exp(self.lambdah * heterochronous_matrix)
            #heterochronous_factor = heterochronous_matrix.pow(self.lambdah)
        elif self.matching_relationship_type == 'exponential':
            matching_factor = torch.exp(self.gamma * matching_index_matrix)
            heterochronous_factor = torch.exp(self.lambdah * heterochronous_matrix)
        
        
        # Calculate the unnormalised wiring probabilities for each edge.
        unnormalised_wiring_probabilities = heterochronous_factor * self.distance_factor * matching_factor 
        # Add on the prob_offset term to prevent division by zero
        unnormalised_wiring_probabilities += self.prob_offset
        # Set the probability for all existing connections to be zero
        unnormalised_wiring_probabilities = unnormalised_wiring_probabilities * (1 - self.adjacency_matrix)
        # Set the diagonal to zero to prevent self-connections
        unnormalised_wiring_probabilities.fill_diagonal_(0)
        # Normalize the wiring probability
        wiring_probability = unnormalised_wiring_probabilities / unnormalised_wiring_probabilities.sum()
        # Sample an edge to add
        edge_idx = torch.multinomial(wiring_probability.view(-1), num_samples=1).item()
        first_node = edge_idx // self.num_nodes
        second_node = edge_idx % self.num_nodes
        # Add the edge to the adjacency matrix
        self.adjacency_matrix[first_node, second_node] = 1
        self.adjacency_matrix[second_node, first_node] = 1
        # Record the added edge
        added_edges = (first_node, second_node)
        # Add the edge to the weight matrix if it exists
        if hasattr(self, 'weight_matrix'):
            self.weight_matrix.data[first_node, second_node] = 1
            self.weight_matrix.data[second_node, first_node] = 1

        # Return the added edge and (a copy of) the updated adjacency matrix
        return added_edges, self.adjacency_matrix.clone()
    
    @jaxtyped(typechecker=typechecked)
    def weighted_update(self) -> Float[torch.Tensor, "{self.num_nodes} {self.num_nodes}"]:
        """
        Performs one update step of the weight matrix for the weighted GNM.

        Returns:
            - weight_matrix (Pytorch tensor of shape (num_nodes, num_nodes)): (A copy of) the updated weight matrix.
        """

        
        # Perform the optimisation step on the weights.
        # Compute the loss
        loss = self.optimisation_criterion(self.weight_matrix)
        # Compute the gradients
        self.optimiser.zero_grad()
        loss.backward()
        # Update the weights
        self.optimiser.step()
        # Ensure the weight matrix is symmetric
        self.weight_matrix.data = 0.5*(self.weight_matrix.data + self.weight_matrix.data.T)
        # Clip the weights to the specified bounds
        self.weight_matrix.data = torch.clamp(self.weight_matrix.data, self.weight_lower_bound, self.weight_upper_bound)
        # Zero out all weights where the adjacency matrix is zero
        self.weight_matrix.data = self.weight_matrix.data * self.adjacency_matrix

        # Return the updated weight matrix
        return self.weight_matrix.detach().clone().cpu()
    
    @jaxtyped(typechecker=typechecked)
    def train_loop( 
        self,
        num_iterations: int,
        binary_updates_per_iteration: int = 1,
        weighted_updates_per_iteration: int = 1,
        heterochronous_matrix: Optional[Float[torch.Tensor, "{self.num_nodes} {self.num_nodes} {num_iterations*binary_updates_per_iteration}"]] = None
    ) -> Tuple[  
        List[Tuple[int, int]],
        Float[torch.Tensor, "{self.num_nodes} {self.num_nodes} {num_iterations*binary_updates_per_iteration}"],
        Optional[Float[torch.Tensor, "{self.num_nodes} {self.num_nodes} {num_iterations*weighted_updates_per_iteration}"]]
        ]:
        """
        Trains the network for a specified number of iterations.
        At each iteration, a number of binary updates and weighted updates are performed.

        Parameters:
            - num_iterations (int): The number of iterations to train the network for.
            - binary_updates_per_iteration (int): The number of binary updates to perform at each iteration. Defaults to 1.
            - weighted_updates_per_iteration (int): The number of weighted updates to perform at each iteration. Defaults to 1.
            - heterochronous_matrix (Pytorch tensor of shape (num_nodes, num_nodes, num_iterations*binary_updates_per_iteration), optional): The heterochronous development probability matrix. Defaults to None.
        
        Returns: 
            - added_edges (List[Tuple[int, int]]): The edges that were added to the adjacency matrix at each iteration.
            - adjacency_snapshots (Pytorch tensor of shape (num_nodes, num_nodes, num_iterations*binary_updates_per_iteration)): The adjacency matrices at each iteration of the binary updates.
            - weight_snapshots (Pytorch tensor of shape (num_nodes, num_nodes, num_iterations*weighted_updates_per_iteration)): The weight matrices at each iteration of the weighted updates.
        """

        added_edges_list = []
        adjacency_snapshots = torch.zeros((self.num_nodes, self.num_nodes, num_iterations*binary_updates_per_iteration))

        if weighted_updates_per_iteration != 0:
            assert hasattr(self, 'weight_matrix'), "Weighted updates per iteration was specified, but no alpha value was initialised."
            weight_snapshots = torch.zeros((self.num_nodes, self.num_nodes, num_iterations*weighted_updates_per_iteration))

        if heterochronous_matrix is None:
            heterochronous_matrix = torch.ones((self.num_nodes, self.num_nodes, num_iterations*binary_updates_per_iteration), dtype=self.seed_adjacency_matrix.dtype)

        for ii in range(num_iterations):
            
            for jj in range(binary_updates_per_iteration):
                added_edges, adjacency_matrix = self.binary_update(heterochronous_matrix[:,:,ii*binary_updates_per_iteration + jj])
                adjacency_snapshots[:,:,ii] = adjacency_matrix
                added_edges_list.append(added_edges)

            for jj in range(weighted_updates_per_iteration):
                weight_matrix = self.weighted_update()
                weight_snapshots[:,:,ii*weighted_updates_per_iteration + jj] = weight_matrix
        
        self.added_edges_list = added_edges_list
        self.adjacency_snapshots = adjacency_snapshots
        #self.weight_snapshots = weight_snapshots
        
        if weighted_updates_per_iteration == 0:
            return self.added_edges_list, self.adjacency_snapshots, None
        
        
        return self.added_edges_list, self.adjacency_snapshots, self.weight_snapshots
