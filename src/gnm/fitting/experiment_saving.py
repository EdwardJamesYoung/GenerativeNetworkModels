import json
import pickle
from datetime import datetime
from warnings import warn
import uuid
from dataclasses import fields, asdict
from .experiment_dataclasses import (
    Experiment,
    BinarySweepParameters,
    WeightedSweepParameters
)
import os
import torch
from collections import defaultdict
import pandas as pd
import numpy as np

"""
This module provides functionality for managing and saving experiment data for generative network models. 
It includes the `ExperimentEvaluation` class, which handles saving experiments, managing an index file, 
querying experiments, and other related operations.

Classes:
    - ExperimentEvaluation: Handles saving, querying, and managing experiment data and configurations.

Usage:
    The `ExperimentEvaluation` class can be used to:
    - Save experiments and their configurations.
    - Query experiments based on specific variables and values.
    - Manage an index file that tracks experiment configurations.
    - Delete or clean up experiment data.

Example:
    evaluator = ExperimentEvaluation(path="experiment_data", index_file_path="index.json")
    evaluator.save_experiments([experiment1, experiment2])
    results = evaluator.query_experiments(value=0.5, by="alpha")
"""

class ExperimentEvaluation():
    """
    The `ExperimentEvaluation` class provides functionality for managing and saving experiment data for generative network models. 
    It handles creating directories, managing index files, saving experiment configurations, and querying experiments.
    
    Attributes:
        - path (str): Directory where experiment data is stored. Defaults to 'generative_model_experiments'.
        - index_path (str): Path to the index file that tracks experiment configurations.
        - variables_to_save (list): List of variables to save, excluding those specified in `variables_to_ignore`.
    
    Methods:
        - __init__(path=None, index_file_path=None, variables_to_ignore=[]): Initializes the class, sets up paths, and prepares the index file.
        - _refresh_index_file(): Reloads the index file from disk or creates it if it doesn't exist.
        - _make_index_file(): Creates a new index file with initial data.
        - save_experiments(experiments): Saves a list of `Experiment` objects to disk.
        - _save_experiment(experiment_dataclass, experiment_name='gnm_experiment'): Saves a single experiment and updates the index file.
        - view_experiments(): Placeholder for viewing experiments as a table or saving them as CSV.
        - _sort_experiments(experiments, variable_to_sort_by, get_names_only=False): Sorts experiments by a specified variable.
        - clean_index_file(): Placeholder for cleaning up the index file.
        - _ask_loop(question): Prompts the user for confirmation with a yes/no question.
        - delete_experiment(experiment_name, ask_first=True): Deletes an experiment and removes it from the index file.
        - purge_index_file(): Placeholder for purging the index file.
        - _is_similar_wording(variable_word, verbose=True): Suggests the most similar variable name if a given name is not found.
        - query_experiments(value=None, by=None, limit=float('inf'), verbose=True): Queries experiments based on a variable and value.
        - open_experiments_by_name(experiment_names): Opens experiments by their names and returns their data.
    
    Usage:
        evaluator = ExperimentEvaluation(path="experiment_data", index_file_path="index.json")
        evaluator.save_experiments([experiment1, experiment2])
        results = evaluator.query_experiments(value=0.5, by="alpha")
    """

    def __init__(self, path=None, index_file_path=None, variables_to_ignore=[], save=True):#
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

    def _refresh_index_file(self):
        if not os.path.exists(self.index_path):
            self._make_index_file()

        with open(self.index_path, "r") as f:
            data = json.load(f)
        
        self.index_file = data

    def _make_index_file(self):
        date = datetime.now()
        date_formatted = date.strftime("%d/%m/%Y")
        json_initial_data = {
            'date':date_formatted,
            'experiment_configs':{
            }
        }

        with open(self.index_path, "w") as f:
            json.dump(json_initial_data, f, indent=4)

        self._refresh_index_file()

    """
    Saves a list of experiments to disk. Each experiment is an instance of the Experiment dataclass.
    The experiments are saved in a specified directory, and the index file is updated accordingly.
    """
    def save_experiments(self, experiments:list[Experiment]):
        if not isinstance(experiments, list):
            experiments = [experiments]
        for experiment in experiments:
            self._save_experiment(experiment)

    """
    Extracts information from experiment dataclass and saves in json index file for later parsing
    """
    def _save_experiment(self, experiment_dataclass:Experiment, experiment_name='gnm_experiment'):
        if not self.save:
            warn('Parameter Save is False - not saving experiment to disk or index file')
            return
        

        binary_evaluations = experiment_dataclass.evaluation_results.binary_evaluations
        experiment_key = list(binary_evaluations.keys())[0]
        if len(binary_evaluations) > 1:
            warn('Multiple binary evaluations found - only the first will be saved in the index file.')
        
        binary_evals = binary_evaluations[experiment_key]
        per_connectome_binary_evals = {i: np.round(binary_evals[i].cpu().numpy(), 4).tolist() for i in range(binary_evals.shape[0])}

        n_participants = binary_evals.shape[0]

        # may ignore weighted parameters if set to None
        all_config = asdict(experiment_dataclass.run_config.binary_parameters)

        all_config.update({
            'n_participants': n_participants,
            'mean_of_max_ks_per_connectome': binary_evals.mean(axis=1).cpu().numpy().tolist(),
            'std_of_max_ks_per_connectome': binary_evals.std(axis=1).cpu().numpy().tolist(),
            'per_connectome_binary_evals': per_connectome_binary_evals
        })
        
        if experiment_dataclass.run_config.weighted_parameters is not None:
            all_config.update(asdict(experiment_dataclass.run_config.weighted_parameters))

            weighted_evals = experiment_dataclass.evaluation_results.weighted_evaluations
            weighted_experiment_key = list(weighted_evals.keys())[0]
            weighted_evals = weighted_evals[weighted_experiment_key]
            if len(weighted_evals) > 1:
                warn('Multiple weighted evaluations found - only the first will be saved in the index file.')
            per_connectome_weighted_evals = {i: weighted_evals[i].cpu().numpy().tolist() for i in range(weighted_evals.shape[0])}
            all_config.update({
                'mean_of_weighted_ks_per_connectome': weighted_evals.mean(axis=1).cpu().numpy().tolist(),
                'std_of_weighted_ks_per_connectome': weighted_evals.std(axis=1).cpu().numpy().tolist(),
                'per_connectome_weighted_evals': per_connectome_weighted_evals
            })
    
        # de-tensor floating and int values
        formatted_config = {}
        for key, value in all_config.items():
            if isinstance(value, torch.Tensor):
                formatted_config[key] = value.item()
            elif isinstance(value, dict):
                formatted_config[key] = value
            elif not isinstance(value, (str, int, float, list)) and hasattr(value, "__class__"):
                try:
                    class_name = value.__class__.__name__
                    formatted_config[key] = class_name
                except:
                    warn(f'Attribute {value} could not be saved - no name or class instance found.')
            elif isinstance(value, list):
                formatted_config[key] = value
            elif isinstance(value, (int, float, str)):
                formatted_config[key] = value

        # return names and values of parameters if save is False - mainly used for wandb
        if not self.save:
            return formatted_config

        self.index_file['experiment_configs'][experiment_name] = formatted_config
        
        # overwrite previous file
        with open(os.path.join(self.path, "gnm_index.json"), "w") as f:
            json.dump(self.index_file, f, indent=4)

        self._refresh_index_file()
        

    # view the experiments as a table and save as csv if you want
    def view_experiments():
        pass

    def _sort_experiments(self, experiments, variable_to_sort_by, get_names_only=False):

        def combine_dictionary_by_key(list_of_dictionaries, key_value_items):
            exp = {}
            for dictionary in list_of_dictionaries:
                for name in list(dictionary.keys()):
                    exp[name] = key_value_items[name][variable_to_sort_by]

            return exp

        experiment_names = list(experiments.keys())

        # keys = experiment name, values = experiment values of the given variable to sort by
        sorting_dict = {experiment_name:value for experiment_name, value in zip(experiment_names, [experiments[name][variable_to_sort_by] for name in experiment_names])}
        
        # # convert strings back to original 
        # sorting_dict = {experiment_name:eval(value) for experiment_name, value in sorting_dict.items()}

        # iterate to check types
        sorting_dict_numbers = {}
        sorting_dict_strings = {}
        sroting_dict_lists = {}
        for key, value in sorting_dict.items():
            if isinstance(value, int) or isinstance(value, float):
                sorting_dict_numbers[key] = value
            elif isinstance(value, str):
                sorting_dict_strings[key] = value
            else:
                sroting_dict_lists[key] = value

        
        # sort num, string dictionaries by values (sorted requires same datatype) - no point in sorting lists
        sorting_dict_numbers = dict(sorted(sorting_dict_numbers.items(), key=lambda item: item[1]))
        sorting_dict_strings = dict(sorted(sorting_dict_strings.items(), key=lambda item: item[1]))

        # create a new dictonary and add experiments based on the order of sorted_experiments
        # but with all the data included in experiments, rather than just the value used for sorting
        sorted_experiments = combine_dictionary_by_key([sorting_dict_numbers, sorting_dict_strings, sroting_dict_lists], experiments)
        
        if get_names_only:
            sorted_experiments = list(sorted_experiments.keys())
        
        return sorted_experiments
    
    def clean_index_file(self):
        pass

    def _ask_loop(self, question):
        answer = None
        question = question + '\ny=confirm, n=exit\n> '
        while answer is None:
            user_input = input(question).lower()
            if user_input == 'y':
                answer = True
            elif user_input == 'n':
                answer = False
            else:
                print('Invalid response. Must be y for yes or n for no.')

        return answer

    """
    Deletes an experiment from the index file and removes the corresponding file from disk.
    """
    def delete_experiment(self, experiment_name, ask_first=True):
        if not self.save:
            warn('Parameter Save is False - no index file present so returning null')
            return

        if not experiment_name in self.index_file['experiment_configs']:
            warn(f'Experiment {experiment_name} not found in index file, exiting.')

        if ask_first:
            response = self._ask_loop(f'Are you sure you want to delete experiment {experiment_name}?')
            if response == False:
                print('Aborting....')
                return

        del self.index_file['experiment_configs'][experiment_name]

        print(f'Experiment {experiment_name} deleted from index file.')

    def purge_index_file(self):
        pass

    def _is_similar_wording(self, variable_word, verbose=True):
        all_vars = self.variables_to_save

        char_frequency = {}
        for var in all_vars:
            letters_in_common = [character for character in variable_word if character in var]
            char_frequency[var] = len(letters_in_common) / len(var)

        char_frequency = dict(sorted(char_frequency.items(), key=lambda item: item[1]))
        most_likely_word = list(char_frequency.keys())[-1]

        if verbose:
            print(f'Did you mean {most_likely_word}?')
        
        return most_likely_word


    """
    Queries the index file for experiments matching a specified variable and value.
    Returns a list of experiment data files that match the criteria.
    """
    def query_experiments(
            self, 
            value=None, 
            by=None, 
            limit=float('inf'), 
            verbose=True
            ) -> list[Experiment]:
        if not self.save:
            warn('Parameter Save is False - no index file present so returning null')
            return

        # get all searchable variables
        all_experiments = self.index_file['experiment_configs']
        if len(all_experiments) == 0:
            warn(f'No experiments saved in index file {self.index_file}')

        first_experiment = list(all_experiments.keys())[-1]
        first_experiment_data = all_experiments[first_experiment]
        searchable_variables = list(first_experiment_data.keys())

        # make sure variable provided can be searched
        if by not in searchable_variables:
            print(f'Variable {by} not in searchable variables. Must be one of {searchable_variables}')
            self._is_similar_wording(by)
            return

        # sort by that variable and return list if no value to search for is specified
        if value is None or len(all_experiments) == 1:
            experiments_sorted = self._sort_experiments(experiments=all_experiments, variable_to_sort_by=by, get_names_only=True)
            return_files = self.open_experiments_by_name(experiments_sorted)
            return return_files 
        
        # iterate through index looking for experiments matching criteria
        to_return = []
        experiments_sorted = self._sort_experiments(experiments=all_experiments, variable_to_sort_by=by, get_names_only=False)

        for experiment_name, experiment_value in experiments_sorted.items():
            if experiment_value == value:
                to_return.append(experiment_name)
        
        experiment_data_to_return = self.open_experiments_by_name(to_return)

        if verbose:
            print(f'\nFound {len(experiment_data_to_return)} item(s) matching: {by} = {value}')
        
        return experiment_data_to_return
        
    """
    Opens experiments by their names and returns their data.
    If the name is 'test_config', it is skipped.
    """
    def find_experiment_by_name(self, experiment_names:list[str]):
            
        if isinstance(experiment_names, str):
            experiment_names = [experiment_names]

        tmp_index = self.index_file['experiment_configs']

        experiments_opened = []
        for name in experiment_names:
            if name == 'test_config': continue

            if name in tmp_index:
                experiments_opened.append(tmp_index[name])
            else:
                warn(f'Experiment {name} not found in index file.')

        return experiments_opened
    
    def list_experiment_parameters(self):
        config = self.index_file['experiment_configs']
        self.variables_to_save = list(config.values())[0].keys()
        print("Experiment Parameters:")
        for var in self.variables_to_save:
            print(f'   - {var}')
        return self.variables_to_save
    
    def get_dataframe_of_results(
        self,
        parameters: list[str] = ['eta', 'gamma', 'mean_of_max_ks_per_connectome'],
        save_dataframe: bool = True
        ) -> pd.DataFrame:
        
        warn("Assumes default Max KS Distance metric was used during evaluation.")

        # reload index file to get latest experiments
        self._refresh_index_file()
        
        all_experiments: dict = self.index_file['experiment_configs']
        last_experiment_name = list(all_experiments.keys())[-1]
        last_experiment_data = all_experiments[last_experiment_name]

        for param in parameters:
            if not param in last_experiment_data.keys():
                warn(f'Parameter {param} not found in experiment binary parameters, skipping...')
                parameters.remove(param)
                continue

        # get n participants
        n_participants = last_experiment_data['n_participants']

        # set up empty df
        results_summary_df = {'connectome_index': []}
        for param in parameters:
            results_summary_df[param] = []

        base_dict = {param: [] for param in parameters}
        base_dict['connectome_index'] = []

        participant_indices = list(range(n_participants))
        
        # iterate through experiment json data
        for experiment_name in all_experiments.keys():
            experiment = all_experiments[experiment_name]

            for param in parameters:
                param_values = experiment[param]

                if not isinstance(param_values, list):
                    param_values = [param_values] * n_participants

                if len(param_values) != n_participants:
                    warn(f'Parameter {param} in experiment {experiment_name} has length {len(param_values)} but expected {n_participants}, skipping...')
                    continue

                base_dict[param].extend(param_values)
            base_dict['connectome_index'].extend(participant_indices)
            
            # append to main dict
            for key in base_dict.keys():
                results_summary_df[key].extend(base_dict[key])

        for key in results_summary_df.keys():
            print(f'Total entries for {key}: {len(results_summary_df[key])}')

        results_summary_df = pd.DataFrame(results_summary_df)

        if len(results_summary_df['connectome_index']) == 0:
            warn("No results found to compile into DataFrame.")
            return pd.DataFrame()
        
        n_unique_connectomes = len(set(results_summary_df['connectome_index']))
        first_connectome_index = results_summary_df['connectome_index'][0]
        
        print(f"Compiled results for {n_unique_connectomes} unique connectomes.")

        if save_dataframe:
            csv_path = os.path.join(self.path, 'experiment_results_summary.csv')
            if os.path.exists(csv_path):
                if self._ask_loop(f'File {csv_path} already exists. Overwrite?'):
                    results_summary_df.to_csv(csv_path, index=False)
                    print(f'Saved results summary dataframe to {csv_path}')
                else:
                    new_name = input('In that case, enter new file name: ')
                    new_csv_path = os.path.join(self.path, new_name + '.csv')
                    results_summary_df.to_csv(new_csv_path, index=False)
                    print(f'Saved results summary dataframe to {new_csv_path}')

        return results_summary_df
