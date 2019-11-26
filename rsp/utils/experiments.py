"""
This library contains all utility functions to help you run your experiments.

Methods
-------
run_experiment:
    Run a single experiment with a specific solver and ExperimentParameters
run_experiment_agenda:
    Run a number of experiments defined in the ExperimentAgenda
run_specific_experiments_from_research_agenda
    Run only a few experiments fro a defined ExperimentAgenda
create_experiment_agenda
    Create an experiment agenda given ranges for paramters to be probed
span_n_grid
    Helper function to span the n-dimensional grid from parameter ranges
create_env_pair_for_experiment
    Create a pair of environments for the desired research. One environment has no malfunciton, the other one has
    exactly one malfunciton
save_experiment_agenda_to_file
    Save a generated experiment agenda to be used for later reruns
load_experiment_agenda_from_file
    Load a ExperimentAgenda
save_experiment_results_to_file
    Save the results of an experiment or a full experiment agenda
load_experiment_results_to_file
    Load the results form an experiment result file
"""

import errno
import os
import pprint
from typing import List, Tuple

import numpy as np
import pandas as pd
from flatland.envs.rail_env import RailEnv
from pandas import DataFrame, Series

from rsp.utils.data_types import ExperimentAgenda, ExperimentParameters, ParameterRanges
from rsp.utils.experiment_env_generators import create_flatland_environment, \
    create_flatland_environment_with_malfunction
from rsp.utils.experiment_solver import AbstractSolver

_pp = pprint.PrettyPrinter(indent=4)


def run_experiment(solver: AbstractSolver, experiment_parameters: ExperimentParameters, verbose=True,
                   force_only_one_trial=True) -> Series:
    """

    Run a single experiment with a given solver and ExperimentParameters
    Parameters
    ----------
    solver: AbstractSolver
        Solver from the class AbstractSolver that should be solving the experiments
    experiment_parameters: ExperimentParameters
        Parameter set of the data form ExperimentParameters

    Returns
    -------
    Returns a DataFram with the experiment results
    """

    # DataFrame to store all results of experiments
    experiment_results = pd.DataFrame(
        columns=['experiment_id', 'time_full', 'time_full_after_malfunction', 'time_delta_after_malfunction',
                 'solution_full', 'solution_delta', 'delta', 'size', 'n_agents', 'max_num_cities',
                 'max_rail_between_cities', 'max_rail_in_city'])

    # Run the sequence of experiment
    for trial in range(experiment_parameters.trials_in_experiment if not force_only_one_trial else 1):
        print("Running trial {} for experiment {}".format(trial + 1, experiment_parameters.experiment_id))
        if verbose:
            print("*** experiment paramters of trial {} for experiment {}".format(trial + 1,
                                                                                  experiment_parameters.experiment_id))
            _pp.pprint(experiment_parameters)

        # Create experiment environments
        static_rail_env, malfunction_rail_env = create_env_pair_for_experiment(experiment_parameters)

        env = malfunction_rail_env
        seed_value = experiment_parameters.seed_value

        # wrap reset params in this function, so we avoid copy-paste errors each time we have to reset the malfunction_rail_env
        def malfunction_env_reset():
            env.reset(False, False, False, seed_value)

        # Run experiments
        # TODO pass k (number of routing alternatives) explicitly
        current_results = solver.run_experiment_trial(static_rail_env=static_rail_env,
                                                      malfunction_rail_env=malfunction_rail_env,
                                                      malfunction_env_reset=malfunction_env_reset)
        # Store results
        experiment_result = {'experiment_id': experiment_parameters.experiment_id,
                             'time_full': current_results.time_full,
                             'time_full_after_malfunction': current_results.time_full_after_malfunction,
                             'time_delta_after_malfunction': current_results.time_delta_after_malfunction,
                             'solution_full': current_results.solution_full,
                             'solution_delta': current_results.solution_delta,
                             'delta': current_results.delta,
                             'size': experiment_parameters.width,
                             'n_agents': experiment_parameters.number_of_agents,
                             'max_num_cities': experiment_parameters.max_num_cities,
                             'max_rail_between_cities': experiment_parameters.max_rail_between_cities,
                             'max_rail_in_city': experiment_parameters.max_rail_in_city,
                             }
        experiment_results = experiment_results.append(experiment_result, ignore_index=True)
        if verbose:
            print("*** experiment result of trial {} for experiment {}".format(trial + 1,
                                                                               experiment_parameters.experiment_id))
            _pp.pprint(experiment_result)

    return experiment_results


def run_experiment_agenda(solver: AbstractSolver, experiment_agenda: ExperimentAgenda) -> DataFrame:
    """
     Run a given experiment_agenda with a suitable solver, return the results as a DataFrame

    Parameters
    ----------
    solver: AbstractSolver
        Solver from the class AbstractSolver that should be solving the experiments
    experiment_agenda: ExperimentAgenda
        List of ExperimentParameters

    Returns
    -------
    Returns a DataFrame with the results of all experiments in the agenda
    """

    # DataFrame to store all results of experiments
    experiment_results = pd.DataFrame(
        columns=['experiment_id', 'time_full', 'time_full_after_malfunction', 'time_delta_after_malfunction',
                 'solution_full', 'solution_delta', 'delta', 'size', 'n_agents', 'max_num_cities',
                 'max_rail_between_cities', 'max_rail_in_city'])

    # Run the sequence of experiment
    for current_experiment_parameters in experiment_agenda.experiments:
        experiment_results = experiment_results.append(
            run_experiment(solver=solver, experiment_parameters=current_experiment_parameters), ignore_index=True)
    return experiment_results


def run_specific_experiments_from_research_agenda(solver: AbstractSolver, experiment_agenda: ExperimentAgenda,
                                                  experiment_ids: List[int]) -> DataFrame:
    """

    Run a subset of experiments of a given agenda. This is useful when trying to find bugs in code.

    Parameters
    ----------
    solver: AbstractSolver
        AbstractSolver to be used for the experiments
    experiment_agenda: ExperimentAgenda
        Full list of experiments
    experiment_ids: List[int]
        List of experiment IDs we want to run

    Returns
    -------
    Returns a DataFrame with the results the desired experiments we ran

    """

    # DataFrame to store all results of experiments
    experiment_results = pd.DataFrame(
        columns=['experiment_id', 'time_full', 'time_full_after_malfunction', 'time_delta_after_malfunction',
                 'solution_full', 'solution_delta', 'delta', 'size', 'n_agents', 'max_num_cities',
                 'max_rail_between_cities', 'max_rail_in_city'])

    # Run the sequence of experiment
    for current_experiment_parameters in experiment_agenda.experiments:
        if current_experiment_parameters.experiment_id in experiment_ids:
            experiment_results = experiment_results.append(
                run_experiment(solver=solver, experiment_parameters=current_experiment_parameters), ignore_index=True)
    return experiment_results


def create_experiment_agenda(parameter_ranges: ParameterRanges, trials_per_experiment: int = 10) -> ExperimentAgenda:
    """
    Create an experiment agenda given a range of parameters defined as ParameterRanges

    Parameters
    ----------
    parameter_ranges: ParameterRanges
        Ranges of all the parameters we want to vary in our experiments
    trials_per_experiment: int
        Number of trials per parameter set we want to run

    Returns
    -------
    ExperimentAgenda built from the ParameterRanges
    """
    # Todo Check that parameters are correctly filled into ExperimentParameters
    # Todo add malfunction parameters correctly to ExperimentParameters
    number_of_dimensions = len(parameter_ranges)
    parameter_values = [[] for i in range(number_of_dimensions)]

    # Setup experiment parameters
    for dim_idx, dimensions in enumerate(parameter_ranges):
        if dimensions[-1] > 1:
            parameter_values[dim_idx] = np.arange(dimensions[0], dimensions[1],
                                                  np.abs(dimensions[1] - dimensions[0]) / dimensions[-1], dtype=int)
        else:
            parameter_values[dim_idx] = [dimensions[0]]
    full_param_set = span_n_grid([], parameter_values)
    experiment_list = []
    for param_id, parameter_set in enumerate(full_param_set):
        # TODO SIM-105 Fix the dependance on the order of parameters in generator maybe work with namedtuples?
        current_experiment = ExperimentParameters(experiment_id=param_id,
                                                  trials_in_experiment=trials_per_experiment,
                                                  number_of_agents=parameter_set[1],
                                                  width=parameter_set[0],
                                                  height=parameter_set[0],
                                                  seed_value=12,
                                                  max_num_cities=parameter_set[4],
                                                  grid_mode=True,
                                                  max_rail_between_cities=parameter_set[3],
                                                  max_rail_in_city=parameter_set[2],
                                                  earliest_malfunction=parameter_set[5],
                                                  malfunction_duration=parameter_set[6])
        experiment_list.append(current_experiment)
    experiment_agenda = ExperimentAgenda(experiments=experiment_list)
    print("Generated an agenda with {} experiments".format(len(experiment_list)))
    return experiment_agenda


def span_n_grid(collected_parameters: list, open_dimensions: list) -> list:
    """
    Recursive function to generate all combinations of parameters given the open_dimensions

    Parameters
    ----------
    collected_parameters: list
        The parameter sets filled so far in the recurions, starts out empty
    open_dimensions: list
        Parameter dimensions we have not yet included in the set

    Returns
    -------
    list of parameter sets for ExperimentAgenda

    """
    full_params = []
    if len(open_dimensions) == 0:
        return [collected_parameters]

    for parameter in open_dimensions[0]:
        full_params.extend(span_n_grid(collected_parameters + [parameter], open_dimensions[1:]))

    return full_params


def create_env_pair_for_experiment(params: ExperimentParameters) -> Tuple[RailEnv, RailEnv]:
    """
    # TODO: Write test to check that these envs are identical at step 0: https://gitlab.aicrowd.com/flatland/submission-scoring/issues/1

    Parameters
    ----------
    params: ExperimentParameters
        Parameter set that we pass to the constructor of the RailEenv

    Returns
    -------
    Tuple[RailEnv, RailEnv]
        First env is a static environment where no malfunction occurs
        Second env is an environment with exactly one malfunction
    """

    number_of_agents = params.number_of_agents
    width = params.width
    height = params.height
    seed_value = params.seed_value
    max_num_cities = params.max_num_cities
    grid_mode = params.grid_mode
    max_rails_between_cities = params.max_rail_between_cities
    max_rails_in_city = params.max_rail_in_city
    earliest_malfunction = params.earliest_malfunction
    malfunction_duration = params.malfunction_duration

    # Generate static environment for initial schedule generation
    env_static = create_flatland_environment(number_of_agents, width, height, seed_value, max_num_cities, grid_mode,
                                             max_rails_between_cities, max_rails_in_city)
    env_static.reset(random_seed=seed_value)

    # Generate dynamic environment with single malfunction
    env_malfunction = create_flatland_environment_with_malfunction(number_of_agents, width, height, seed_value,
                                                                   max_num_cities, grid_mode, max_rails_between_cities,
                                                                   max_rails_in_city, earliest_malfunction,
                                                                   malfunction_duration)
    env_malfunction.reset(random_seed=seed_value)
    return env_static, env_malfunction


def save_experiment_agenda_to_file(experiment_agenda: ExperimentAgenda, file_name: str):
    """
    Save a generated experiment agenda into a file for later use

    Parameters
    ----------
    experiment_agenda: ExperimentAgenda
        The experiment agenda we want to save
    file_name: str
        File name containing path and name of file we want to store the experiment agenda
    Returns
    -------

    """
    pass


def load_experiment_agenda_from_file(file_name: str) -> ExperimentAgenda:
    """
    Load agenda from file in order to rerun experiments.

    Parameters
    ----------
    file_name: str
        File name containing path to file that we want to load
    Returns
    -------
    ExperimentAgenda loaded from file
    """
    pass


def save_experiment_results_to_file(experiment_results: DataFrame, file_name: str):
    """
    Save the data frame with all the result from an experiment into a given file

    Parameters
    ----------
    experiment_results: DataFrame
        Data Frame containing all the experiment results
    file_name: str
        File name containing path and name of file we want to store the experiment results

    Returns
    -------

    """
    if not os.path.exists(os.path.dirname(file_name)):
        try:
            os.makedirs(os.path.dirname(file_name))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise exc
    experiment_results.to_json(file_name)


def load_experiment_results_to_file(file_name: str) -> DataFrame:
    """
    Load results as DataFrame to do further analysis

    Parameters
    ----------
    file_name: str
        File name containing path to file that we want to load

    Returns
    -------
    DataFrame containing the loaded experiment results
    """
    experiment_results = pd.read_json(file_name)
    return experiment_results
