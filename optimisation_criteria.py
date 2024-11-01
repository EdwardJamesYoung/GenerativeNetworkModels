import torch

from typing import Tuple, List
from jaxtyping import Float, Int, jaxtyped
from typeguard import typechecked
from beartype import beartype

from abc import ABC, abstractmethod

class OptimisationCriteria(ABC):
    def __init__(self, normalisation:bool = False):
        self.normalisation = normalisation

    @abstractmethod
    @jaxtyped
    @beartype
    def _unnormalised_call(self, weight_matrix:Float[torch.Tensor, "num_nodes num_nodes"]) -> Float[torch.Tensor, "num_nodes num_nodes"]:
        """
        Computes the unnormalised value of the optimisation criteria for a given weight matrix.
        
        Parameters:
        - weight_matrix (Pytorch tensor of shape (num_nodes, num_nodes)): The weight matrix of the network.
        
        Returns:
            - (Pytorch tensor of shape (num_nodes, num_nodes)): The unnormalised value of the optimisation criteria, before summation. 
        """
        pass

    @jaxtyped
    @beartype
    def __call__(self, weight_matrix:Float[torch.Tensor, "num_nodes num_nodes"]) -> Float[torch.Tensor, ""]:
        
        symmetrised_matrix = 0.5*(weight_matrix + weight_matrix.T)

        if self.normalisation == True:
            return torch.sum( self._unnormalised_call(symmetrised_matrix) )
        else:
            unnormalised_value = self._unnormalised_call(symmetrised_matrix)
            normalisation_term = unnormalised_value.max()
            return torch.sum( unnormalised_value/normalisation_term )


class DistanceWeightedCommunicability(OptimisationCriteria):
    def __init__(self, normalisation:bool, distance_matrix:Float[torch.Tensor, "num_nodes num_nodes"], omega:float = 1.0):
        super().__init__(normalisation=normalisation)
        self.distance_matrix = distance_matrix
        self.omega = omega
    
    def _unnormalised_call(self, weight_matrix:Float[torch.Tensor, "num_nodes num_nodes"]) -> Float[torch.Tensor, "num_nodes num_nodes"]:
        """
        Computes the distance-weighted communicability of a network with a given weight matrix.
        
        Parameters:
        - weight_matrix (Pytorch tensor of shape (num_nodes, num_nodes)): The weight matrix of the network.
        
        Returns:
        - distance_weighted_communicability (Pytorch tensor of shape (num_nodes, num_nodes)): The distance-weighted communicability matrix of the network.
        """
        # Compute the node strengths, with a small constant addition to prevent division by zero.
        node_strengths = 0.5* ( weight_matrix.sum(dim=0) + weight_matrix.sum(dim=1) ) + 1e-6
        # Compute the inverse square root of the node strengths
        inv_sqrt_node_strenghts = torch.diag( 1/torch.sqrt(node_strengths) )
        # Compute the normalised weight matrix
        normalised_weight_matrix = inv_sqrt_node_strenghts @ weight_matrix @ inv_sqrt_node_strenghts
        
        # Compute the communicability matrix
        communicability_matrix = torch.matrix_exp( normalised_weight_matrix )
        # Compute the distance-weighted communicability to the power of omega
        distance_weighted_communicability = torch.pow( communicability_matrix * self.distance_matrix , self.omega )
        
        # Return the cumulative distance-weighted communicability
        return distance_weighted_communicability
    