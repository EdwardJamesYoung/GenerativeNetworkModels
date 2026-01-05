from gnm.fitting.experiment_saving import *
from gnm.fitting.experiment_dataclasses import Experiment
from gnm import defaults, fitting, generative_rules, weight_criteria, evaluation
import torch

def _generate_basic_sweep():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    distance_matrix = defaults.get_distance_matrix(device=DEVICE)
    binary_consensus_network = defaults.get_binary_network(device=DEVICE)

    eta_values = torch.Tensor([1]) #torch.linspace(-5, -1, 1)
    gamma_values = torch.Tensor([-1])#torch.linspace(-0.5, 0.5, 1)
    num_connections = int( binary_consensus_network.sum().item() / 2 )

    binary_sweep_parameters = fitting.BinarySweepParameters(
        eta = eta_values,
        gamma = gamma_values,
        lambdah = torch.Tensor([0.0]),
        distance_relationship_type = ["powerlaw"],
        preferential_relationship_type = ["powerlaw"],
        heterochronicity_relationship_type = ["powerlaw"],
        generative_rule = [generative_rules.MatchingIndex()],
        num_iterations = [num_connections],
    )

    weighted_sweep_parameters = fitting.WeightedSweepParameters(
        alpha = [0.01],
        optimisation_criterion = [weight_criteria.DistanceWeightedCommunicability(distance_matrix=distance_matrix) ],
    )   

    num_simulations = 1

    sweep_config = fitting.SweepConfig(
        binary_sweep_parameters = binary_sweep_parameters,
        weighted_sweep_parameters = weighted_sweep_parameters,
        num_simulations = num_simulations,
        distance_matrix = [distance_matrix]    
    )

    criteria = [ evaluation.ClusteringKS(), evaluation.DegreeKS(), evaluation.EdgeLengthKS(distance_matrix) ]
    energy = evaluation.MaxCriteria( criteria )
    binary_evaluations = [energy]
    weighted_evaluations = [ evaluation.WeightedNodeStrengthKS(normalise=True), evaluation.WeightedClusteringKS() ]

    experiments = fitting.perform_sweep(sweep_config=sweep_config, 
                                    binary_evaluations=binary_evaluations, 
                                    real_binary_matrices=binary_consensus_network,
                                    weighted_evaluations=weighted_evaluations,
                                    save_model = False,
                                    save_run_history = False,
                                    verbose=True
    )

    assert len(experiments) > 0, 'Experiments not returned or sweep was run incorrectly during unit tests for experiment saving'
    return experiments

def test_experiment_evaluation_init():
    """Test ExperimentEvaluation initializes correctly."""
    eval_handler = ExperimentEvaluation(save=False)
    
    assert eval_handler is not None, "ExperimentEvaluation should initialize"
    assert eval_handler.save is False, "Save attribute should be set correctly"
    assert eval_handler.path == "generative_model_experiments", "Default path should be set"


def test_experiment_evaluation_init_custom_path():
    """Test ExperimentEvaluation initializes with custom path."""
    eval_handler = ExperimentEvaluation(path="custom_path", save=False)
    
    assert eval_handler.path == "custom_path", "Custom path should be set"


def test_save_experiments():
    """Test saving experiments to disk."""
    experiments = _generate_basic_sweep()
    eval_handler = ExperimentEvaluation(path="test_experiments", save=True)
    
    # Should not raise an error
    eval_handler.save_experiments(experiments)
    
    # Verify index file was updated
    assert len(eval_handler.index_file["experiment_configs"]) > 0, "Experiments should be saved to index"


def test_save_single_experiment():
    """Test saving a single experiment (not in a list)."""
    experiments = _generate_basic_sweep()
    eval_handler = ExperimentEvaluation(path="test_experiments", save=True)
    
    # Should handle single experiment without list wrapper
    eval_handler.save_experiments(experiments[0])
    
    assert len(eval_handler.index_file["experiment_configs"]) > 0, "Single experiment should be saved"


def test_list_experiment_parameters():
    """Test listing experiment parameters."""
    experiments = _generate_basic_sweep()
    eval_handler = ExperimentEvaluation(path="test_experiments", save=True)
    eval_handler.save_experiments(experiments)
    
    parameters = eval_handler.list_experiment_parameters()
    
    assert parameters is not None, "Parameters should be returned"
    assert "eta" in parameters, "eta should be in parameters"
    assert "gamma" in parameters, "gamma should be in parameters"


def test_get_dataframe_of_results():
    """Test getting results as a DataFrame."""
    experiments = _generate_basic_sweep()
    eval_handler = ExperimentEvaluation(path="test_experiments", save=True)
    eval_handler.save_experiments(experiments)
    
    df = eval_handler.get_dataframe_of_results(
        parameters=["eta", "gamma", "mean_of_max_ks_per_connectome"],
        save_dataframe=False,
    )
    
    assert df is not None, "DataFrame should be returned"
    assert "eta" in df.columns, "eta should be in DataFrame columns"
    assert "gamma" in df.columns, "gamma should be in DataFrame columns"
    assert len(df) > 0, "DataFrame should have rows"


def test_experiment_evaluation_no_save_mode():
    """Test ExperimentEvaluation in no-save mode returns None for queries."""
    eval_handler = ExperimentEvaluation(save=False)
    
    result = eval_handler.query_experiments(value=0.5, by="eta")
    
    assert result is None, "Should return None when save=False"
