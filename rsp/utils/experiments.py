"""This library contains all utility functions to help you run your
experiments.

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
import multiprocessing
import os
import pickle
import pprint
import shutil
import sys
import time
import traceback
from functools import partial
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from flatland.envs.rail_env import RailEnv
from pandas import DataFrame

from rsp.rescheduling.rescheduling_analysis_utils import _analyze_paths
from rsp.rescheduling.rescheduling_analysis_utils import _analyze_times
from rsp.utils.data_types import COLUMNS
from rsp.utils.data_types import convert_experiment_results_to_data_frame
from rsp.utils.data_types import ExperimentAgenda
from rsp.utils.data_types import ExperimentParameters
from rsp.utils.data_types import ExperimentResults
from rsp.utils.data_types import ParameterRanges
from rsp.utils.data_types import SpeedData
from rsp.utils.experiment_env_generators import create_flatland_environment
from rsp.utils.experiment_env_generators import create_flatland_environment_with_malfunction
from rsp.utils.experiment_solver import AbstractSolver
from rsp.utils.file_utils import check_create_folder

_pp = pprint.PrettyPrinter(indent=4)


def run_experiment(solver: AbstractSolver,
                   experiment_parameters: ExperimentParameters,
                   show_results_without_details: bool = True,
                   rendering: bool = False,
                   verbose: bool = False,
                   debug: bool = False,
                   ) -> List[dict]:
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
    Returns a DataFrame with the experiment results
    """

    # DataFrame to store all results of experiments
    data_frame = []
    # Run the sequence of experiment
    for trial in range(experiment_parameters.trials_in_experiment):
        print("Running trial {} for experiment {}".format(trial + 1, experiment_parameters.experiment_id))
        start_trial = time.time()
        if show_results_without_details:
            print("*** experiment parameters of trial {} for experiment {}".format(trial + 1,
                                                                                   experiment_parameters.experiment_id))
            _pp.pprint(experiment_parameters)

        # Create experiment environments
        static_rail_env, malfunction_rail_env = create_env_pair_for_experiment(experiment_parameters, trial)

        env = malfunction_rail_env
        if rendering:
            from flatland.utils.rendertools import RenderTool, AgentRenderVariant
            env_renderer = RenderTool(env, gl="PILSVG",
                                      agent_render_variant=AgentRenderVariant.ONE_STEP_BEHIND,
                                      show_debug=False,
                                      screen_height=600,  # Adjust these parameters to fit your resolution
                                      screen_width=800)
            env_renderer.reset()
            env_renderer.render_env(show=True, show_observations=False, show_predictions=False)

        seed_value = experiment_parameters.seed_value

        # wrap reset params in this function, so we avoid copy-paste errors each time we have to reset the malfunction_rail_env
        def malfunction_env_reset():
            env.reset(False, False, False, seed_value)

        # Run experiments
        # TODO pass k (number of routing alternatives) explicitly
        current_results: ExperimentResults = solver.run_experiment_trial(static_rail_env=static_rail_env,
                                                                         malfunction_rail_env=malfunction_rail_env,
                                                                         malfunction_env_reset=malfunction_env_reset,
                                                                         verbose=verbose,
                                                                         debug=debug
                                                                         )
        if current_results is None:
            print(f"No malfunction for experiment {experiment_parameters.experiment_id}")
            return []
        # Store results
        data_frame.append(
            convert_experiment_results_to_data_frame(
                experiment_results=current_results,
                experiment_parameters=experiment_parameters
            )
        )

        if show_results_without_details:
            print("*** experiment result of trial {} for experiment {}".format(trial + 1,
                                                                               experiment_parameters.experiment_id))

            _pp.pprint({key: data_frame[-1][key]
                        for key in COLUMNS
                        if not key.startswith('solution_') and 'experiment_freeze' not in key and key != 'agents_paths_dict'
                        })

            _analyze_times(current_results)
            _analyze_paths(current_results)
        if rendering:
            from flatland.utils.rendertools import RenderTool, AgentRenderVariant
            env_renderer.close_window()
        trial_time = (time.time() - start_trial)
        print("Running trial {} for experiment {}: took {:5.3f}ms"
              .format(trial + 1, experiment_parameters.experiment_id, trial_time))
    return data_frame


def run_experiment_agenda(solver: AbstractSolver,
                          experiment_agenda: ExperimentAgenda,
                          run_experiments_parallel: bool = True,
                          show_results_without_details: bool = True,
                          verbose: bool = False) -> str:
    """Run a given experiment_agenda with a suitable solver, return the name of
    the experiment folder.

    Parameters
    ----------
    solver: AbstractSolver
        Solver from the class AbstractSolver that should be solving the experiments

    experiment_agenda: ExperimentAgenda
        List of ExperimentParameters
    run_experiments_parallel: bool
        run experiments in parallel
    show_results_without_details: bool
        Print results
    verbose: bool
        Print additional information

    Returns
    -------
    Returns the name of the experiment folder
    """
    experiment_folder_name = create_experiment_folder_name(experiment_agenda.experiment_name)
    save_experiment_agenda_to_file(experiment_folder_name, experiment_agenda)

    if run_experiments_parallel:
        pool = multiprocessing.Pool()
        run_and_save_one_experiment_partial = partial(run_and_save_one_experiment,
                                                      solver=solver,
                                                      verbose=verbose,
                                                      show_results_without_details=show_results_without_details,
                                                      experiment_folder_name=experiment_folder_name)
        pool.map(run_and_save_one_experiment_partial, experiment_agenda.experiments)
    else:
        for current_experiment_parameters in experiment_agenda.experiments:
            run_and_save_one_experiment(current_experiment_parameters, solver, verbose, show_results_without_details,
                                        experiment_folder_name)

    return experiment_folder_name


def run_and_save_one_experiment(current_experiment_parameters,
                                solver,
                                verbose,
                                show_results_without_details,
                                experiment_folder_name,
                                rendering: bool = False, ):
    try:
        filename = create_experiment_filename(experiment_folder_name, current_experiment_parameters.experiment_id)
        experiment_result = run_experiment(solver=solver,
                                           experiment_parameters=current_experiment_parameters,
                                           rendering=rendering,
                                           verbose=verbose,
                                           show_results_without_details=show_results_without_details)
        save_experiment_results_to_file(experiment_result, filename)
    except Exception as e:
        print("XXX failed " + filename + " " + str(e))
        traceback.print_exc(file=sys.stdout)


def run_specific_experiments_from_research_agenda(solver: AbstractSolver,
                                                  experiment_agenda: ExperimentAgenda,
                                                  experiment_ids: List[int],
                                                  run_experiments_parallel: bool = True,
                                                  show_results_without_details: bool = True,
                                                  rendering: bool = False,
                                                  verbose: bool = False) -> str:
    """Run a subset of experiments of a given agenda. This is useful when
    trying to find bugs in code.

    Parameters
    ----------
    solver: AbstractSolver
        AbstractSolver to be used for the experiments
    experiment_agenda: ExperimentAgenda
        Full list of experiments
    experiment_ids: List[int]
        List of experiment IDs we want to run
    run_experiments_parallel: bool
        run experiments in parallel
    show_results_without_details: bool
        Print results
    verbose: bool
        Print additional information

    Returns
    -------
    Returns the name of the experiment folder
    """
    experiment_folder_name = create_experiment_folder_name(experiment_agenda.experiment_name)

    filter_experiment_agenda_partial = partial(filter_experiment_agenda, experiment_ids=experiment_ids)
    experiments_filtered = filter(filter_experiment_agenda_partial, experiment_agenda.experiments)
    experiment_agenda_filtered = ExperimentAgenda(
        experiment_name=experiment_agenda.experiment_name,
        experiments=list(experiments_filtered)
    )
    save_experiment_agenda_to_file(experiment_folder_name, experiment_agenda_filtered)

    if run_experiments_parallel:
        pool = multiprocessing.Pool()
        run_and_save_one_experiment_partial = partial(run_and_save_one_experiment,
                                                      solver=solver,
                                                      verbose=verbose,
                                                      show_results_without_details=show_results_without_details,
                                                      experiment_folder_name=experiment_folder_name
                                                      )
        pool.map(run_and_save_one_experiment_partial, experiment_agenda_filtered.experiments)
    else:
        for current_experiment_parameters in experiment_agenda_filtered.experiments:
            run_and_save_one_experiment(current_experiment_parameters,
                                        solver,
                                        verbose,
                                        show_results_without_details,
                                        experiment_folder_name,
                                        rendering=rendering)

    return experiment_folder_name


def filter_experiment_agenda(current_experiment_parameters, experiment_ids) -> bool:
    return current_experiment_parameters.experiment_id in experiment_ids


def create_experiment_agenda(experiment_name: str,
                             parameter_ranges: ParameterRanges,
                             speed_data: SpeedData,
                             trials_per_experiment: int = 10,
                             vary_malfunction: int = 1,
                             vary_malfunction_step: int = 20
                             ) -> ExperimentAgenda:
    """Create an experiment agenda given a range of parameters defined as
    ParameterRanges.

    Parameters
    ----------

    experiment_name: str
        Name of the experiment
    parameter_ranges: ParameterRanges
        Ranges of all the parameters we want to vary in our experiments

    trials_per_experiment: int
        Number of trials per parameter set we want to run

    speed_data
        Dictionary containing all the desired speeds in the environment

    vary_malfunction
        Deprecated. Use malfunction range instead.
        Run the same experiment `vary_malfunction` times with ids <experiment_id>_<0...var_malfunction-1>

    vary_malfunction_step
        Deprecated. Use malfunction range instead.
        If the same experiment is run multiple times (`vary_malfunction > 1`), the earliest malfunction is set to
        `parameter_set[5] + i * vary_malfunction_step` at the `i`th iteration.

    Returns
    -------
    ExperimentAgenda built from the ParameterRanges
    """
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
        for i in range(vary_malfunction):
            earliest_malfunction = parameter_set[5] + i * vary_malfunction_step
            experiment_id = param_id
            if vary_malfunction > 1:
                experiment_id = f"{param_id}_{earliest_malfunction}"
            current_experiment = ExperimentParameters(experiment_id=experiment_id,
                                                      trials_in_experiment=trials_per_experiment,
                                                      number_of_agents=parameter_set[1],
                                                      speed_data=speed_data,
                                                      width=parameter_set[0],
                                                      height=parameter_set[0],
                                                      seed_value=12,
                                                      max_num_cities=parameter_set[4],
                                                      grid_mode=False,
                                                      max_rail_between_cities=parameter_set[3],
                                                      max_rail_in_city=parameter_set[2],
                                                      earliest_malfunction=earliest_malfunction,
                                                      malfunction_duration=parameter_set[6],
                                                      number_of_shortest_paths_per_agent=parameter_set[7])
            experiment_list.append(current_experiment)
    experiment_agenda = ExperimentAgenda(experiment_name=experiment_name, experiments=experiment_list)
    print("Generated an agenda with {} experiments".format(len(experiment_list)))
    return experiment_agenda


def span_n_grid(collected_parameters: list, open_dimensions: list) -> list:
    """Recursive function to generate all combinations of parameters given the
    open_dimensions.

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


def create_env_pair_for_experiment(params: ExperimentParameters, trial: int = 0) -> Tuple[RailEnv, RailEnv]:
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
    speed_data = params.speed_data

    # Generate static environment for initial schedule generation
    env_static = create_flatland_environment(number_of_agents=number_of_agents,
                                             width=width,
                                             height=height,
                                             seed_value=seed_value + trial,
                                             max_num_cities=max_num_cities,
                                             grid_mode=grid_mode,
                                             max_rails_between_cities=max_rails_between_cities,
                                             max_rails_in_city=max_rails_in_city,
                                             speed_data=speed_data)
    env_static.reset(random_seed=seed_value)

    # Generate dynamic environment with single malfunction
    env_malfunction = create_flatland_environment_with_malfunction(number_of_agents=number_of_agents,
                                                                   width=width,
                                                                   height=height,
                                                                   seed_value=seed_value + trial,
                                                                   max_num_cities=max_num_cities,
                                                                   grid_mode=grid_mode,
                                                                   max_rails_between_cities=max_rails_between_cities,
                                                                   max_rails_in_city=max_rails_in_city,
                                                                   malfunction_duration=malfunction_duration,
                                                                   earliest_malfunction=earliest_malfunction,
                                                                   speed_data=speed_data)
    env_malfunction.reset(random_seed=seed_value)
    return env_static, env_malfunction


def save_experiment_agenda_to_file(experiment_folder_name: str, experiment_agenda: ExperimentAgenda):
    """Save experiment agenda to the folder with the experiments.

    Parameters
    ----------
    experiment_folder_name: str
        Folder name of experiment where all experiment files and agenda are stored
    experiment_agenda: ExperimentAgenda
        The experiment agenda to save
    """
    file_name = os.path.join(experiment_folder_name, "experiment_agenda.pkl")
    check_create_folder(experiment_folder_name)
    with open(file_name, 'wb') as handle:
        pickle.dump(experiment_agenda, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_experiment_agenda_from_file(experiment_folder_name: str) -> ExperimentAgenda:
    """Save experiment agenda to the folder with the experiments.

    Parameters
    ----------
    experiment_folder_name: str
        Folder name of experiment where all experiment files and agenda are stored
    experiment_agenda: ExperimentAgenda
        The experiment agenda to save
    """
    file_name = os.path.join(experiment_folder_name, "experiment_agenda.pkl")
    with open(file_name, 'rb') as handle:
        file_data: ExperimentAgenda = pickle.load(handle)
        return file_data


def create_experiment_folder_name(experiment_name: str) -> str:
    datetime_string = datetime.datetime.now().strftime("%Y_%m_%dT%H_%M_%S")
    return "{}_{}".format(experiment_name, datetime_string)


def create_experiment_filename(experiment_folder_name: str, experiment_id: int) -> str:
    filename = "experiment_{}.pkl".format(experiment_id)
    return os.path.join(experiment_folder_name, filename)


def save_experiment_results_to_file(experiment_results: List, file_name: str):
    """Save the data frame with all the result from an experiment into a given
    file.

    Parameters
    ----------
    experiment_results: List of experiment results
       List containing all the experiment results
    file_name: str
        File name containing path and name of file we want to store the experiment results

    Returns
    -------
    """
    check_create_folder(os.path.dirname(file_name))

    with open(file_name, 'wb') as handle:
        pickle.dump(experiment_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_experiment_results_from_file(file_name: str) -> List:
    """Load results as List to do further analysis.

    Parameters
    ----------
    file_name: str
        File name containing path to file that we want to load

    Returns
    -------
    List containing the loaded experiment results
    """
    experiment_results = pd.DataFrame(columns=COLUMNS)

    with open(file_name, 'rb') as handle:
        file_data = pickle.load(handle)
    experiment_results = experiment_results.append(file_data, ignore_index=True)
    return experiment_results


def load_experiment_results_from_folder(experiment_folder_name: str) -> DataFrame:
    """Load results as DataFrame to do further analysis.

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
    for file in [file for file in files if 'agenda' not in file]:
        file_name = os.path.join(experiment_folder_name, file)
        if file_name.endswith('experiment_agenda.pkl'):
            continue
        with open(file_name, 'rb') as handle:
            file_data = pickle.load(handle)
        # TODO SIM-250 malfunction data files may be empty
        if len(file_data) > 0:
            experiment_results = experiment_results.append(file_data, ignore_index=True)

    return experiment_results


def delete_experiment_folder(experiment_folder_name: str):
    """Delete experiment folder.

    Parameters
    ----------
    experiment_folder_name: str
        Folder name of experiment where all experiment files are stored

    Returns
    -------
    """
    shutil.rmtree(experiment_folder_name)
