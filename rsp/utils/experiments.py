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
import datetime
import errno
import os
import pickle
import pprint
from typing import List, Tuple

import numpy as np
import pandas as pd
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from pandas import DataFrame

from rsp.utils.data_types import ExperimentAgenda, ExperimentParameters, ParameterRanges
from rsp.utils.experiment_env_generators import create_flatland_environment, \
    create_flatland_environment_with_malfunction
from rsp.utils.experiment_solver import AbstractSolver

_pp = pprint.PrettyPrinter(indent=4)

COLUMNS = ['experiment_id',
           'time_full',
           'time_full_after_malfunction',
           'time_delta_after_malfunction',
           'solution_full',
           'solution_full_after_malfunction',
           'solution_delta_after_malfunction',
           'costs_full',
           'costs_full_after_malfunction',
           'costs_delta_after_malfunction',
           'delta',
           'size',
           'n_agents',
           'max_num_cities',
           'max_rail_between_cities',
           'max_rail_in_city']


def run_experiment(solver: AbstractSolver,
                   experiment_parameters: ExperimentParameters,
                   verbose=False,
                   force_only_one_trial=False) -> List:
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
    experiment_results = []
    # Run the sequence of experiment
    for trial in range(experiment_parameters.trials_in_experiment if not force_only_one_trial else 1):
        print("Running trial {} for experiment {}".format(trial + 1, experiment_parameters.experiment_id))
        if verbose:
            print("*** experiment parameters of trial {} for experiment {}".format(trial + 1,
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
        time_delta_after_m = current_results.time_delta_after_malfunction
        time_full_after_m = current_results.time_full_after_malfunction

        experiment_results.append({'experiment_id': experiment_parameters.experiment_id,
                             'time_full': current_results.time_full,
                             'time_full_after_malfunction': time_delta_after_m,
                             'time_delta_after_malfunction': time_full_after_m,
                             'solution_full': current_results.solution_full,
                             'solution_full_after_malfunction': current_results.solution_full_after_malfunction,
                             'solution_delta_after_malfunction': current_results.solution_delta_after_malfunction,
                             'costs_full': current_results.costs_full,
                             'costs_full_after_malfunction': current_results.costs_full_after_malfunction,
                             'costs_delta_after_malfunction': current_results.costs_delta_after_malfunction,
                             'delta': current_results.delta,
                             'size': experiment_parameters.width,
                             'n_agents': experiment_parameters.number_of_agents,
                             'max_num_cities': experiment_parameters.max_num_cities,
                             'max_rail_between_cities': experiment_parameters.max_rail_between_cities,
                             'max_rail_in_city': experiment_parameters.max_rail_in_city,
                             })

        if verbose:
            print("*** experiment result of trial {} for experiment {}".format(trial + 1,
                                                                               experiment_parameters.experiment_id))

            _pp.pprint({key: value for key, value in experiment_results[-1].items()
                        if not key.startswith('solution_') and not key == 'delta'})

            # Delta is all train run way points in the re-schedule that are not also in the schedule
            schedule_trainrunwaypoints = current_results.solution_full
            full_reschedule_trainrunwaypoints_dict = current_results.solution_full_after_malfunction
            delta: TrainrunDict = {
                agent_id: sorted(list(
                    set(full_reschedule_trainrunwaypoints_dict[agent_id]).difference(
                        set(schedule_trainrunwaypoints[agent_id]))),
                    key=lambda p: p.scheduled_at)
                for agent_id in schedule_trainrunwaypoints.keys()
            }
            delta_percentage = 100 * sum([len(delta[agent_id]) for agent_id in delta.keys()]) / sum(
                [len(full_reschedule_trainrunwaypoints_dict[agent_id]) for agent_id in
                 full_reschedule_trainrunwaypoints_dict.keys()])

            # Freeze is all train run way points in the schedule that are also in the re-schedule
            freeze: TrainrunDict = \
                {agent_id: sorted(list(
                    set(full_reschedule_trainrunwaypoints_dict[agent_id]).intersection(
                        set(schedule_trainrunwaypoints[agent_id]))),
                    key=lambda p: p.scheduled_at) for agent_id in delta.keys()}
            freeze_percentage = 100 * sum([len(freeze[agent_id]) for agent_id in freeze.keys()]) / sum(
                [len(schedule_trainrunwaypoints[agent_id]) for agent_id in schedule_trainrunwaypoints.keys()])

            print(
                f"**** freeze: {freeze_percentage}% of waypoints in the full schedule are the same in the full re-schedule")
            print(f"**** delta: {delta_percentage}% of waypoints in the re-schedule are the as in the initial schedule")

            all_full_reschedule_trainrunwaypoints = {
                full_reschedule_trainrunwaypoint
                for full_reschedule_trainrunwaypoints in full_reschedule_trainrunwaypoints_dict.values()
                for full_reschedule_trainrunwaypoint in full_reschedule_trainrunwaypoints
            }
            all_delta_reschedule_trainrunwaypoints = {
                full_reschedule_trainrunwaypoint
                for full_reschedule_trainrunwaypoints in current_results.solution_delta_after_malfunction.values()
                for full_reschedule_trainrunwaypoint in full_reschedule_trainrunwaypoints
            }

            full_delta_same_counts = len(
                all_full_reschedule_trainrunwaypoints.intersection(all_delta_reschedule_trainrunwaypoints))
            full_delta_same_percentage = 100 * full_delta_same_counts / len(all_full_reschedule_trainrunwaypoints)
            full_delta_new_counts = len(
                all_delta_reschedule_trainrunwaypoints.difference(all_full_reschedule_trainrunwaypoints))
            full_delta_stale_counts = len(
                all_full_reschedule_trainrunwaypoints.difference(all_delta_reschedule_trainrunwaypoints))
            print(
                f"**** full re-schedule -> delta re-schedule: "
                f"same {full_delta_same_percentage}% ({full_delta_same_counts})"
                f"(+{full_delta_new_counts}, -{full_delta_stale_counts}) waypoints")
            time_rescheduling_improve_perc = 100 * (time_delta_after_m - time_full_after_m) / time_full_after_m
            print(f"**** full re-schedule -> delta re-schedule: "
                  f"time {time_rescheduling_improve_perc:+2.1f}% "
                  f"{time_full_after_m}s -> {time_delta_after_m}s")

    return experiment_results


def run_experiment_agenda(solver: AbstractSolver, experiment_agenda: ExperimentAgenda, verbose: bool = False) -> str:
    """
     Run a given experiment_agenda with a suitable solver, return the name of the experiment folder

    Parameters
    ----------
    solver: AbstractSolver
        Solver from the class AbstractSolver that should be solving the experiments
    experiment_agenda: ExperimentAgenda
        List of ExperimentParameters
    verbose: bool
        Print additional information

    Returns
    -------
    Returns the name of the experiment folder
    """
    experiment_folder_name = create_experiment_folder_name(experiment_agenda.experiment_name)

    for current_experiment_parameters in experiment_agenda.experiments:
        experiment_result = run_experiment(solver=solver, experiment_parameters=current_experiment_parameters,
                                           verbose=verbose)
        filename = create_experiment_filename(experiment_folder_name, current_experiment_parameters.experiment_id)
        save_experiment_results_to_file(experiment_result, filename)

    return experiment_folder_name


def run_specific_experiments_from_research_agenda(solver: AbstractSolver, experiment_agenda: ExperimentAgenda,
                                                  experiment_ids: List[int], verbose: bool = False) -> str:
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
    verbose: bool
        Print additional information

    Returns
    -------
    Returns the name of the experiment folder

    """
    experiment_folder_name = create_experiment_folder_name(experiment_agenda.experiment_name)

    for current_experiment_parameters in experiment_agenda.experiments:
        if current_experiment_parameters.experiment_id in experiment_ids:
            experiment_result = run_experiment(solver=solver, experiment_parameters=current_experiment_parameters,
                                               verbose=verbose)
            filename = create_experiment_filename(experiment_folder_name, current_experiment_parameters.experiment_id)
            save_experiment_results_to_file(experiment_result, filename)

    return experiment_folder_name


def create_experiment_agenda(experiment_name: str, parameter_ranges: ParameterRanges, trials_per_experiment: int = 10) -> ExperimentAgenda:
    """
    Create an experiment agenda given a range of parameters defined as ParameterRanges

    Parameters
    ----------
    experiment_name: str
        Name of the experiment
    parameter_ranges: ParameterRanges
        Ranges of all the parameters we want to vary in our experiments
    trials_per_experiment: int
        Number of trials per parameter set we want to run

    Returns
    -------
    ExperimentAgenda built from the ParameterRanges
    """
    # TODO Check that parameters are correctly filled into ExperimentParameters
    # TODO add malfunction parameters correctly to ExperimentParameters
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
        # TODO can we use named structure in pandas?
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
    experiment_agenda = ExperimentAgenda(experiment_name=experiment_name, experiments=experiment_list)
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


def create_experiment_folder_name(experiment_name: str) -> str:
    datetime_string = datetime.datetime.now().strftime("%Y_%m_%dT%H_%M_%S")
    return"{}_{}".format(experiment_name, datetime_string)


def create_experiment_filename(experiment_folder_name: str, experiment_id: int) -> str:
    filename = "experiment_{}.json".format(experiment_id)
    return os.path.join(experiment_folder_name, filename)


def save_experiment_results_to_file(experiment_results: List, file_name: str):
    """
    Save the data frame with all the result from an experiment into a given file

    Parameters
    ----------
    experiment_results: List of experiment results
       List containing all the experiment results
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

    with open(file_name, 'wb') as handle:
        pickle.dump(experiment_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_experiment_results_from_file(file_name: str) -> List:
    """
    Load results as List to do further analysis

    Parameters
    ----------
    file_name: str
        File name containing path to file that we want to load

    Returns
    -------
    List containing the loaded experiment results
    """
    with open(file_name, 'rb') as handle:
        experiment_results = pickle.load(handle)
    return experiment_results


def load_experiment_results_from_folder(experiment_folder_name: str) -> DataFrame:
    """
    Load results as DataFrame to do further analysis

    Parameters
    ----------
    experiment_folder_name: str
        Folder name of experiment where all experiment files are stored

    Returns
    -------
    DataFrame containing the loaded experiment results
    """

    experiment_results = pd.DataFrame(columns=COLUMNS)

    files = os.listdir(experiment_folder_name)
    for file in files:
        file_name = os.path.join(experiment_folder_name, file)
        file_data = load_experiment_results_from_file(file_name)
        experiment_results = experiment_results.append(file_data, ignore_index=True)

    return experiment_results
