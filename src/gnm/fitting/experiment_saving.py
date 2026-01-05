r"""Experiment saving and management for generative network model experiments.

This module provides tools for saving, loading, querying, and managing generative network
model experiments. It includes an index file system for tracking experiments and their
parameters, enabling efficient retrieval and analysis of experimental results.

The main components are:

- ExperimentEvaluation class for managing experiment storage and retrieval
- Methods for querying experiments by parameter values
- Tools for exporting results to DataFrames for further analysis
"""

import json
import os
from dataclasses import fields, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from warnings import warn

import numpy as np
import pandas as pd
import torch
from jaxtyping import jaxtyped
from typeguard import typechecked

from .experiment_dataclasses import (
    Experiment,
    BinarySweepParameters,
    WeightedSweepParameters,
)


class ExperimentEvaluation:
    r"""Manager for saving, loading, and querying generative network model experiments.

    This class provides a comprehensive interface for persisting experimental results
    and their associated metadata. It maintains an index file (JSON) that tracks all
    saved experiments and their parameters, enabling efficient queries and retrieval.

    The class supports:

    - Saving experiments with automatic parameter extraction
    - Querying experiments by parameter values
    - Deleting individual experiments
    - Exporting results to pandas DataFrames

    Attributes:
        save:
            Whether to persist experiments to disk. If False, experiments are processed
            but not saved.
        path:
            Directory path where experiment data and index file are stored.
        index_path:
            Full path to the JSON index file.
        variables_to_save:
            List of parameter names to track in the index file.
        index_file:
            The loaded index file data (dictionary).

    Examples:
        >>> from gnm.fitting import ExperimentEvaluation
        >>> # Create an experiment manager
        >>> manager = ExperimentEvaluation(path='my_experiments')
        >>> # Save experiments from a sweep
        >>> manager.save_experiments(experiments)
        >>> # Query experiments with specific parameters
        >>> matching = manager.query_experiments(value=0.5, by='eta')
        >>> # Get results as a DataFrame
        >>> df = manager.get_dataframe_of_results(
        ...     parameters=['eta', 'gamma', 'mean_of_max_ks_per_connectome']
        ... )

    See Also:
        - [`fitting.Experiment`][gnm.fitting.Experiment]: The dataclass representing an experiment
        - [`fitting.perform_sweep`][gnm.fitting.perform_sweep]: Function to run parameter sweeps
    """

    @jaxtyped(typechecker=typechecked)
    def __init__(
        self,
        path: Optional[str] = None,
        index_file_path: Optional[str] = None,
        variables_to_ignore: List[str] = [],
        save: bool = True,
    ):
        r"""
        Args:
            path:
                Directory path where experiments will be saved. Defaults to
                'generative_model_experiments' if not specified.
            index_file_path:
                Name of the JSON index file. Defaults to 'gnm_index.json'.
            variables_to_ignore:
                List of parameter names to exclude from tracking.
            save:
                Whether to persist experiments to disk. Set to False for
                in-memory processing only.
        """
        self.save = save

        if path is None:
            path = 'generative_model_experiments'
        
        if index_file_path is None:
            index_file_path = 'gnm_index.json'

        # create path to experiment data and index file if it doesn't exist already
        if not os.path.exists(path) and save:
            os.mkdir(path)

        self.path = path
        self.index_path = os.path.join(self.path, index_file_path)      

        # get the variables we want to save, i.e. alpha, gamma etc (some will be in list format)
        binary_variables_to_save = [f.name for f in fields(BinarySweepParameters)]
        weighted_variables_to_save = [f.name for f in fields(WeightedSweepParameters)]
        variables_to_save = binary_variables_to_save + weighted_variables_to_save
        self.variables_to_save = [i for i in variables_to_save if i not in variables_to_ignore]

        if self.save:
            self._refresh_index_file()

    def _refresh_index_file(self) -> None:
        r"""Reload the index file from disk into memory.

        Creates the index file if it does not exist, then loads its contents
        into the `index_file` attribute.
        """
        if not os.path.exists(self.index_path):
            self._make_index_file()

        with open(self.index_path, "r") as f:
            data = json.load(f)

        self.index_file = data

    def _make_index_file(self) -> None:
        r"""Create a new index file with initial metadata.

        Initializes a JSON file with the current date and an empty
        experiment configurations dictionary.
        """
        date = datetime.now()
        date_formatted = date.strftime("%d/%m/%Y")
        json_initial_data = {
            "date": date_formatted,
            "experiment_configs": {},
        }

        with open(self.index_path, "w") as f:
            json.dump(json_initial_data, f, indent=4)

        self._refresh_index_file()

    @jaxtyped(typechecker=typechecked)
    def save_experiments(self, experiments: Union[Experiment, List[Experiment]]) -> None:
        r"""Save a list of experiments to disk.

        Each experiment is saved and indexed for later retrieval. The experiments
        are instances of the Experiment dataclass containing run configurations
        and evaluation results.

        Args:
            experiments:
                A single Experiment or list of Experiment instances to save.

        See Also:
            - [`fitting.Experiment`][gnm.fitting.Experiment]: The experiment dataclass
            - [`ExperimentEvaluation.query_experiments`][gnm.fitting.ExperimentEvaluation.query_experiments]: Retrieve saved experiments
        """
        if not isinstance(experiments, list):
            experiments = [experiments]
        for experiment in experiments:
            self._save_experiment(experiment)

    @jaxtyped(typechecker=typechecked)
    def _save_experiment(
        self,
        experiment_dataclass: Experiment,
        experiment_name: str = "gnm_experiment",
    ) -> Optional[Dict[str, Any]]:
        r"""Extract information from experiment dataclass and save to index file.

        Processes an experiment's parameters and evaluation results, formats them
        for JSON serialization, and appends them to the index file.

        Args:
            experiment_dataclass:
                The experiment to save.
            experiment_name:
                Base name for the experiment entry. A random suffix is appended
                to ensure uniqueness.

        Returns:
            If save is False, returns the formatted configuration dictionary.
            Otherwise returns None after saving to disk.
        """
        if not self.save:
            warn(
                "Parameter Save is False - not saving experiment to disk or index file"
            )
            return

        binary_evaluations = experiment_dataclass.evaluation_results.binary_evaluations
        experiment_key = list(binary_evaluations.keys())[0]
        if len(binary_evaluations) > 1:
            warn(
                "Multiple binary evaluations found - only the first will be saved "
                "in the index file."
            )

        binary_evals = binary_evaluations[experiment_key]
        per_connectome_binary_evals = {
            i: np.round(binary_evals[i].cpu().numpy(), 4).tolist()
            for i in range(binary_evals.shape[0])
        }

        n_participants = binary_evals.shape[0]

        # may ignore weighted parameters if set to None
        all_config = asdict(experiment_dataclass.run_config.binary_parameters)

        all_config.update(
            {
                "n_participants": n_participants,
                "mean_of_max_ks_per_connectome": binary_evals.mean(axis=1)
                .cpu()
                .numpy()
                .tolist(),
                "std_of_max_ks_per_connectome": binary_evals.std(axis=1)
                .cpu()
                .numpy()
                .tolist(),
                "per_connectome_binary_evals": per_connectome_binary_evals,
            }
        )

        if experiment_dataclass.run_config.weighted_parameters is not None:
            all_config.update(asdict(experiment_dataclass.run_config.weighted_parameters))

            weighted_evals = experiment_dataclass.evaluation_results.weighted_evaluations
            weighted_experiment_key = list(weighted_evals.keys())[0]
            weighted_evals = weighted_evals[weighted_experiment_key]
            if len(weighted_evals) > 1:
                warn(
                    "Multiple weighted evaluations found - only the first will be "
                    "saved in the index file."
                )
            per_connectome_weighted_evals = {
                i: weighted_evals[i].cpu().numpy().tolist()
                for i in range(weighted_evals.shape[0])
            }
            all_config.update(
                {
                    "mean_of_weighted_ks_per_connectome": weighted_evals.mean(axis=1)
                    .cpu()
                    .numpy()
                    .tolist(),
                    "std_of_weighted_ks_per_connectome": weighted_evals.std(axis=1)
                    .cpu()
                    .numpy()
                    .tolist(),
                    "per_connectome_weighted_evals": per_connectome_weighted_evals,
                }
            )

        # De-tensor floating and int values for JSON serialization
        formatted_config = {}
        for key, value in all_config.items():
            if isinstance(value, torch.Tensor):
                formatted_config[key] = value.item()
            elif isinstance(value, dict):
                formatted_config[key] = value
            elif (
                not isinstance(value, (str, int, float, list))
                and hasattr(value, "__class__")
            ):
                try:
                    class_name = value.__class__.__name__
                    formatted_config[key] = class_name
                except Exception:
                    warn(
                        f"Attribute {value} could not be saved - no name or class "
                        "instance found."
                    )
            elif isinstance(value, list):
                formatted_config[key] = value
            elif isinstance(value, (int, float, str)):
                formatted_config[key] = value

        # Return names and values of parameters if save is False - mainly used for wandb
        if not self.save:
            return formatted_config

        # Add to JSON index file
        with open(os.path.join(self.path, "gnm_index.json"), "r") as f:
            data = json.load(f)

        def random_string(length: int = 6) -> str:
            r"""Generate a random alphanumeric string."""
            letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            return "".join(np.random.choice(list(letters), size=length))

        data["experiment_configs"][experiment_name + "_" + random_string()] = (
            formatted_config
        )

        with open(os.path.join(self.path, "gnm_index.json"), "w") as f:
            json.dump(data, f, indent=4)

        self._refresh_index_file()

    def view_experiments(self) -> None:
        r"""View experiments as a table and optionally save as CSV.

        Note:
            This method is not yet implemented.
        """
        pass

    def _sort_experiments(
        self,
        experiments: Dict[str, Dict[str, Any]],
        variable_to_sort_by: str,
        get_names_only: bool = False,
    ) -> Union[Dict[str, Any], List[str]]:
        r"""Sort experiments by a specified variable.

        Groups experiments by the type of their sorting variable (numbers, strings,
        lists) and sorts each group appropriately.

        Args:
            experiments:
                Dictionary mapping experiment names to their configuration data.
            variable_to_sort_by:
                The parameter name to sort by.
            get_names_only:
                If True, return only the sorted experiment names.
                If False, return a dictionary with names and their sort values.

        Returns:
            Either a list of experiment names (if get_names_only=True) or a
            dictionary mapping names to their sort values.
        """

        def combine_dictionary_by_key(
            list_of_dictionaries: List[Dict[str, Any]],
            key_value_items: Dict[str, Dict[str, Any]],
        ) -> Dict[str, Any]:
            r"""Combine dictionaries by extracting values for a specific key."""
            exp = {}
            for dictionary in list_of_dictionaries:
                for name in list(dictionary.keys()):
                    exp[name] = key_value_items[name][variable_to_sort_by]
            return exp

        experiment_names = list(experiments.keys())

        # Keys = experiment name, values = experiment values of the given variable
        sorting_dict = {
            experiment_name: value
            for experiment_name, value in zip(
                experiment_names,
                [experiments[name][variable_to_sort_by] for name in experiment_names],
            )
        }

        # Iterate to check types and separate by type
        sorting_dict_numbers = {}
        sorting_dict_strings = {}
        sorting_dict_lists = {}
        for key, value in sorting_dict.items():
            if isinstance(value, (int, float)):
                sorting_dict_numbers[key] = value
            elif isinstance(value, str):
                sorting_dict_strings[key] = value
            else:
                sorting_dict_lists[key] = value

        # Sort num, string dictionaries by values (sorted requires same datatype)
        sorting_dict_numbers = dict(
            sorted(sorting_dict_numbers.items(), key=lambda item: item[1])
        )
        sorting_dict_strings = dict(
            sorted(sorting_dict_strings.items(), key=lambda item: item[1])
        )

        # Create a new dictionary with experiments in sorted order
        sorted_experiments = combine_dictionary_by_key(
            [sorting_dict_numbers, sorting_dict_strings, sorting_dict_lists],
            experiments,
        )

        if get_names_only:
            sorted_experiments = list(sorted_experiments.keys())

        return sorted_experiments

    def clean_index_file(self) -> None:
        r"""Clean the index file by removing invalid or orphaned entries.

        Note:
            This method is not yet implemented.
        """
        pass

    def _ask_loop(self, question: str) -> bool:
        r"""Prompt the user for confirmation in a loop.

        Args:
            question:
                The question to display to the user.

        Returns:
            True if user confirms (y), False if user declines (n).
        """
        answer = None
        question = question + "\ny=confirm, n=exit\n> "
        while answer is None:
            user_input = input(question).lower()
            if user_input == "y":
                answer = True
            elif user_input == "n":
                answer = False
            else:
                print("Invalid response. Must be y for yes or n for no.")

        return answer

    @jaxtyped(typechecker=typechecked)
    def delete_experiment(
        self, experiment_name: str, ask_first: bool = True
    ) -> None:
        r"""Delete an experiment from the index file.

        Removes the specified experiment from the index file. Optionally
        prompts for confirmation before deletion.

        Args:
            experiment_name:
                The name of the experiment to delete.
            ask_first:
                If True, prompt for confirmation before deletion.
                Defaults to True.
        """
        if not self.save:
            warn("Parameter Save is False - no index file present so returning null")
            return

        if experiment_name not in self.index_file["experiment_configs"]:
            warn(f"Experiment {experiment_name} not found in index file, exiting.")
            return

        if ask_first:
            response = self._ask_loop(
                f"Are you sure you want to delete experiment {experiment_name}?"
            )
            if response is False:
                print("Aborting....")
                return

        del self.index_file["experiment_configs"][experiment_name]

        print(f"Experiment {experiment_name} deleted from index file.")

    def purge_index_file(self) -> None:
        r"""Remove all experiments from the index file.

        Note:
            This method is not yet implemented.
        """
        pass

    def _is_similar_wording(self, variable_word: str, verbose: bool = True) -> str:
        r"""Find the most similar variable name to a given word.

        Uses character frequency matching to suggest corrections for
        misspelled parameter names.

        Args:
            variable_word:
                The word to find a match for.
            verbose:
                If True, print the suggested match.

        Returns:
            The most similar variable name from the available parameters.
        """
        all_vars = self.variables_to_save

        char_frequency = {}
        for var in all_vars:
            letters_in_common = [
                character for character in variable_word if character in var
            ]
            char_frequency[var] = len(letters_in_common) / len(var)

        char_frequency = dict(
            sorted(char_frequency.items(), key=lambda item: item[1])
        )
        most_likely_word = list(char_frequency.keys())[-1]

        if verbose:
            print(f"Did you mean {most_likely_word}?")

        return most_likely_word

    @jaxtyped(typechecker=typechecked)
    def query_experiments(
        self,
        value: Optional[Any] = None,
        by: Optional[str] = None,
        limit: float = float("inf"),
        verbose: bool = True,
    ) -> Optional[List[Experiment]]:
        r"""Query the index file for experiments matching specified criteria.

        Searches through saved experiments to find those matching a specified
        parameter value. If no value is provided, returns all experiments
        sorted by the specified parameter.

        Args:
            value:
                The value to match for the specified parameter. If None,
                returns all experiments sorted by the `by` parameter.
            by:
                The parameter name to search or sort by.
            limit:
                Maximum number of experiments to return. Defaults to infinity.
            verbose:
                If True, print the number of matching experiments found.

        Returns:
            A list of Experiment data dictionaries matching the criteria,
            or None if save is False.

        Examples:
            >>> manager = ExperimentEvaluation(path='my_experiments')
            >>> # Find all experiments with eta = -2.0
            >>> experiments = manager.query_experiments(value=-2.0, by='eta')
            >>> # Get all experiments sorted by gamma
            >>> sorted_exps = manager.query_experiments(by='gamma')

        See Also:
            - [`ExperimentEvaluation.find_experiment_by_name`][gnm.fitting.ExperimentEvaluation.find_experiment_by_name]: Find experiments by name
        """
        if not self.save:
            warn("Parameter Save is False - no index file present so returning null")
            return None

        # Get all searchable variables
        all_experiments = self.index_file["experiment_configs"]
        if len(all_experiments) == 0:
            warn(f"No experiments saved in index file {self.index_file}")

        first_experiment = list(all_experiments.keys())[-1]
        first_experiment_data = all_experiments[first_experiment]
        searchable_variables = list(first_experiment_data.keys())

        # Make sure variable provided can be searched
        if by not in searchable_variables:
            print(
                f"Variable {by} not in searchable variables. "
                f"Must be one of {searchable_variables}"
            )
            self._is_similar_wording(by)
            return None

        # Sort by that variable and return list if no value to search for
        if value is None or len(all_experiments) == 1:
            experiments_sorted = self._sort_experiments(
                experiments=all_experiments,
                variable_to_sort_by=by,
                get_names_only=True,
            )
            return_files = self.open_experiments_by_name(experiments_sorted)
            return return_files

        # Iterate through index looking for experiments matching criteria
        to_return = []
        experiments_sorted = self._sort_experiments(
            experiments=all_experiments,
            variable_to_sort_by=by,
            get_names_only=False,
        )

        for experiment_name, experiment_value in experiments_sorted.items():
            if experiment_value == value:
                to_return.append(experiment_name)

        experiment_data_to_return = self.open_experiments_by_name(to_return)

        if verbose:
            print(
                f"\nFound {len(experiment_data_to_return)} item(s) matching: "
                f"{by} = {value}"
            )

        return experiment_data_to_return

    @jaxtyped(typechecker=typechecked)
    def find_experiment_by_name(
        self, experiment_names: Union[str, List[str]]
    ) -> List[Dict[str, Any]]:
        r"""Retrieve experiment data by experiment names.

        Looks up experiments in the index file by their names and returns
        their configuration data.

        Args:
            experiment_names:
                A single experiment name or list of names to look up.

        Returns:
            A list of experiment configuration dictionaries for the found
            experiments. Experiments named 'test_config' are skipped.

        See Also:
            - [`ExperimentEvaluation.query_experiments`][gnm.fitting.ExperimentEvaluation.query_experiments]: Query experiments by parameter values
        """
        if isinstance(experiment_names, str):
            experiment_names = [experiment_names]

        tmp_index = self.index_file["experiment_configs"]

        experiments_opened = []
        for name in experiment_names:
            if name == "test_config":
                continue

            if name in tmp_index:
                experiments_opened.append(tmp_index[name])
            else:
                warn(f"Experiment {name} not found in index file.")

        return experiments_opened

    def list_experiment_parameters(self) -> List[str]:
        r"""List all parameter names tracked in the index file.

        Prints and returns the list of parameter names from the first
        experiment configuration in the index file.

        Returns:
            A list of parameter names.
        """
        config = self.index_file["experiment_configs"]
        self.variables_to_save = list(config.values())[0].keys()
        print("Experiment Parameters:")
        for var in self.variables_to_save:
            print(f"   - {var}")
        return self.variables_to_save

    @jaxtyped(typechecker=typechecked)
    def get_dataframe_of_results(
        self,
        parameters: List[str] = ["eta", "gamma", "mean_of_max_ks_per_connectome"],
        save_dataframe: bool = True,
    ) -> pd.DataFrame:
        r"""Export experiment results to a pandas DataFrame.

        Compiles results from all experiments in the index file into a
        DataFrame with one row per connectome per experiment.

        Args:
            parameters:
                List of parameter names to include in the DataFrame.
                Defaults to ['eta', 'gamma', 'mean_of_max_ks_per_connectome'].
            save_dataframe:
                If True, save the DataFrame to a CSV file in the experiment
                directory.

        Returns:
            A DataFrame containing the specified parameters for all
            experiments and connectomes.

        Warning:
            Assumes the default Max KS Distance metric was used during
            evaluation.

        Examples:
            >>> manager = ExperimentEvaluation(path='my_experiments')
            >>> df = manager.get_dataframe_of_results(
            ...     parameters=['eta', 'gamma', 'mean_of_max_ks_per_connectome']
            ... )
            >>> print(df.head())

        See Also:
            - [`ExperimentEvaluation.query_experiments`][gnm.fitting.ExperimentEvaluation.query_experiments]: Query specific experiments
        """
        warn("Assumes default Max KS Distance metric was used during evaluation.")

        # Reload index file to get latest experiments
        self._refresh_index_file()

        all_experiments: Dict[str, Dict[str, Any]] = self.index_file[
            "experiment_configs"
        ]
        last_experiment_name = list(all_experiments.keys())[-1]
        last_experiment_data = all_experiments[last_experiment_name]

        for param in parameters:
            if param not in last_experiment_data.keys():
                warn(
                    f"Parameter {param} not found in experiment binary "
                    "parameters, skipping..."
                )
                parameters.remove(param)
                continue

        # Get n participants
        n_participants = last_experiment_data["n_participants"]

        # Set up empty df
        results_summary_df: Dict[str, List[Any]] = {"connectome_index": []}
        for param in parameters:
            results_summary_df[param] = []

        base_dict: Dict[str, List[Any]] = {param: [] for param in parameters}
        base_dict["connectome_index"] = []

        participant_indices = list(range(n_participants))

        # Iterate through experiment JSON data
        for experiment_name in all_experiments.keys():
            experiment = all_experiments[experiment_name]

            for param in parameters:
                param_values = experiment[param]

                if not isinstance(param_values, list):
                    param_values = [param_values] * n_participants

                if len(param_values) != n_participants:
                    warn(
                        f"Parameter {param} in experiment {experiment_name} has "
                        f"length {len(param_values)} but expected {n_participants}, "
                        "skipping..."
                    )
                    continue

                base_dict[param].extend(param_values)
            base_dict["connectome_index"].extend(participant_indices)

            # Append to main dict
            for key in base_dict.keys():
                results_summary_df[key].extend(base_dict[key])

        for key in results_summary_df.keys():
            print(f"Total entries for {key}: {len(results_summary_df[key])}")

        results_summary_df = pd.DataFrame(results_summary_df)

        if len(results_summary_df["connectome_index"]) == 0:
            warn("No results found to compile into DataFrame.")
            return pd.DataFrame()

        n_unique_connectomes = len(set(results_summary_df["connectome_index"]))

        print(f"Compiled results for {n_unique_connectomes} unique connectomes.")

        if save_dataframe:
            csv_path = os.path.join(self.path, "experiment_results_summary.csv")
            if os.path.exists(csv_path):
                if self._ask_loop(f"File {csv_path} already exists. Overwrite?"):
                    results_summary_df.to_csv(csv_path, index=False)
                    print(f"Saved results summary dataframe to {csv_path}")
                else:
                    new_name = input("In that case, enter new file name: ")
                    new_csv_path = os.path.join(self.path, new_name + ".csv")
                    results_summary_df.to_csv(new_csv_path, index=False)
                    print(f"Saved results summary dataframe to {new_csv_path}")

        return results_summary_df
