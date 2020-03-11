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
from typing import Optional
from typing import Tuple

import numpy as np
from flatland.envs.rail_env import RailEnv

from rsp.experiment_solvers.data_types import ScheduleAndMalfunction
from rsp.experiment_solvers.experiment_solver import ASPExperimentSolver
from rsp.route_dag.analysis.rescheduling_analysis_utils import _analyze_paths
from rsp.route_dag.analysis.rescheduling_analysis_utils import _analyze_times
from rsp.route_dag.analysis.rescheduling_verification_utils import plausibility_check_experiment_results
from rsp.utils.data_types import expand_experiment_results_for_analysis
from rsp.utils.data_types import ExperimentAgenda
from rsp.utils.data_types import ExperimentParameters
from rsp.utils.data_types import ExperimentResults
from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.data_types import ParameterRanges
from rsp.utils.data_types import SpeedData
from rsp.utils.experiment_env_generators import create_flatland_environment
from rsp.utils.experiment_env_generators import create_flatland_environment_with_malfunction
from rsp.utils.file_utils import check_create_folder
from rsp.utils.tee import reset_tee
from rsp.utils.tee import tee_stdout_to_file

_pp = pprint.PrettyPrinter(indent=4)

EXPERIMENT_DATA_DIRECTORY_NAME = "Data"
EXPERIMENT_ANALYSIS_DIRECTORY_NAME = "Analysis"


def run_experiment(solver: ASPExperimentSolver,
                   experiment_parameters: ExperimentParameters,
                   show_results_without_details: bool = True,
                   rendering: bool = False,
                   verbose: bool = False,
                   debug: bool = False,
                   ) -> ExperimentResults:
    """

    Run a single experiment with a given solver and ExperimentParameters
    Parameters
    ----------
    solver: ASPExperimentSolver
        Solver from the class ASPExperimentSolver that should be solving the experiments
    experiment_parameters: ExperimentParameters
        Parameter set of the data form ExperimentParameters

    Returns
    -------
    Returns a DataFrame with the experiment results
    """

    # Run the sequence of experiment
    print("Running experiment {}".format(experiment_parameters.experiment_id))
    start_time = time.time()
    if show_results_without_details:
        print("*** experiment parameters for experiment {}".format(experiment_parameters.experiment_id))
        _pp.pprint(experiment_parameters)

    # Create experiment environments
    static_rail_env, malfunction_rail_env = create_env_pair_for_experiment(experiment_parameters)

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

    # wrap reset params in this function, so we avoid copy-paste errors each time we have to reset the malfunction_rail_env
    def malfunction_env_reset():
        env.reset(False, False, False, experiment_parameters.flatland_seed_value)

    # Run experiments
    schedule_and_malfunction: ScheduleAndMalfunction = solver.gen_schedule_and_malfunction(
        static_rail_env=static_rail_env,
        malfunction_rail_env=malfunction_rail_env,
        malfunction_env_reset=malfunction_env_reset,
        experiment_parameters=experiment_parameters,
        verbose=verbose,
        debug=debug
    )
    experiment_results: ExperimentResults = solver._run_experiment_from_environment(
        schedule_and_malfunction=schedule_and_malfunction,
        malfunction_rail_env=malfunction_rail_env,
        malfunction_env_reset=malfunction_env_reset,
        experiment_parameters=experiment_parameters,
        verbose=verbose,
        debug=debug
    )
    if experiment_results is None:
        print(f"No malfunction for experiment {experiment_parameters.experiment_id}")
        return []

    if show_results_without_details:
        print("*** experiment result of experiment {}".format(experiment_parameters.experiment_id))

        experiment_results_analysis = expand_experiment_results_for_analysis(
            experiment_results=experiment_results)
        _analyze_times(experiment_results_analysis=experiment_results_analysis)
        _analyze_paths(experiment_results_analysis=experiment_results_analysis,
                       experiment_id=experiment_parameters.experiment_id)
    if rendering:
        from flatland.utils.rendertools import RenderTool, AgentRenderVariant
        env_renderer.close_window()
    elapsed_time = (time.time() - start_time)
    print("Running experiment {}: took {:5.3f}s"
          .format(experiment_parameters.experiment_id, elapsed_time))

    plausibility_check_experiment_results(experiment_results=experiment_results)
    return experiment_results


def _write_sha_txt(folder_name: str):
    """
    Write the current commit hash to a file "sha.txt" in the given folder
    Parameters
    ----------
    folder_name
    """
    import git
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    out_file = os.path.join(folder_name, 'sha.txt')
    print(f"writing {sha} to {out_file}")
    with open(out_file, 'w') as out:
        out.write(sha)


def run_and_save_one_experiment(current_experiment_parameters,
                                solver,
                                verbose,
                                show_results_without_details,
                                experiment_folder_name,
                                rendering: bool = False, ):
    try:
        filename = create_experiment_filename(experiment_folder_name, current_experiment_parameters.experiment_id)
        experiment_results: ExperimentResults = run_experiment(solver=solver,
                                                               experiment_parameters=current_experiment_parameters,
                                                               rendering=rendering,
                                                               verbose=verbose,
                                                               show_results_without_details=show_results_without_details)
        save_experiment_results_to_file(experiment_results, filename)
    except Exception as e:
        print("XXX failed " + filename + " " + str(e))
        traceback.print_exc(file=sys.stdout)


def run_experiment_agenda(experiment_agenda: ExperimentAgenda,
                          experiment_ids: Optional[List[int]] = None,
                          run_experiments_parallel: bool = True,
                          show_results_without_details: bool = True,
                          rendering: bool = False,
                          verbose: bool = False) -> (str, str):
    """Run a subset of experiments of a given agenda. This is useful when
    trying to find bugs in code.

    Parameters
    ----------
    experiment_agenda: ExperimentAgenda
        Full list of experiments
    experiment_ids: Optional[List[int]]
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
    experiment_base_directory = create_experiment_folder_name(experiment_agenda.experiment_name)
    experiment_data_directory = f'{experiment_base_directory}/{EXPERIMENT_DATA_DIRECTORY_NAME}'

    check_create_folder(experiment_base_directory)
    check_create_folder(experiment_data_directory)

    # tee stdout to log file
    stdout_orig = tee_stdout_to_file(log_file=os.path.join(experiment_data_directory, "log.txt"))

    if experiment_ids is not None:
        filter_experiment_agenda_partial = partial(filter_experiment_agenda, experiment_ids=experiment_ids)
        experiments_filtered = filter(filter_experiment_agenda_partial, experiment_agenda.experiments)
        experiment_agenda = ExperimentAgenda(
            experiment_name=experiment_agenda.experiment_name,
            experiments=list(experiments_filtered)
        )

    save_experiment_agenda_and_hash_to_file(experiment_data_directory, experiment_agenda)

    solver = ASPExperimentSolver()

    if run_experiments_parallel:
        pool = multiprocessing.Pool()
        run_and_save_one_experiment_partial = partial(run_and_save_one_experiment,
                                                      solver=solver,
                                                      verbose=verbose,
                                                      show_results_without_details=show_results_without_details,
                                                      experiment_folder_name=experiment_data_directory
                                                      )
        pool.map(run_and_save_one_experiment_partial, experiment_agenda.experiments)
    else:
        for current_experiment_parameters in experiment_agenda.experiments:
            run_and_save_one_experiment(current_experiment_parameters,
                                        solver,
                                        verbose,
                                        show_results_without_details,
                                        experiment_data_directory,
                                        rendering=rendering)

    # remove tee
    reset_tee(stdout_orig)
    return experiment_base_directory, experiment_data_directory


def filter_experiment_agenda(current_experiment_parameters, experiment_ids) -> bool:
    return current_experiment_parameters.experiment_id in experiment_ids


def create_experiment_agenda(experiment_name: str,
                             parameter_ranges: ParameterRanges,
                             speed_data: SpeedData,
                             experiments_per_grid_element: int = 10
                             ) -> ExperimentAgenda:
    """Create an experiment agenda given a range of parameters defined as
    ParameterRanges.

    Parameters
    ----------

    experiment_name: str
        Name of the experiment
    parameter_ranges: ParameterRanges
        Ranges of all the parameters we want to vary in our experiments
    experiments_per_grid_element: int
        Number of runs with different seed per parameter set we want to run
    speed_data
        Dictionary containing all the desired speeds in the environment


    Returns
    -------
    ExperimentAgenda built from the ParameterRanges
    """
    number_of_dimensions = len(parameter_ranges)
    parameter_values = [[] for _ in range(number_of_dimensions)]

    # Setup experiment parameters
    for dim_idx, dimensions in enumerate(parameter_ranges):
        if dimensions[-1] > 1:
            parameter_values[dim_idx] = np.arange(dimensions[0], dimensions[1],
                                                  np.abs(dimensions[1] - dimensions[0]) / dimensions[-1], dtype=int)
        else:
            parameter_values[dim_idx] = [dimensions[0]]
    full_param_set = span_n_grid([], parameter_values)
    experiment_list = []
    for grid_id, parameter_set in enumerate(full_param_set):
        for run_of_this_grid_element in range(experiments_per_grid_element):
            earliest_malfunction = parameter_set[5]
            experiment_id = grid_id * experiments_per_grid_element + run_of_this_grid_element
            current_experiment = ExperimentParameters(
                experiment_id=experiment_id,
                grid_id=grid_id,
                number_of_agents=parameter_set[1],
                speed_data=speed_data,
                width=parameter_set[0],
                height=parameter_set[0],
                flatland_seed_value=12 + run_of_this_grid_element,
                asp_seed_value=94,
                max_num_cities=parameter_set[4],
                grid_mode=False,
                max_rail_between_cities=parameter_set[3],
                max_rail_in_city=parameter_set[2],
                earliest_malfunction=earliest_malfunction,
                malfunction_duration=parameter_set[6],
                number_of_shortest_paths_per_agent=parameter_set[7],
                # route change is penalized the same as 60 seconds delay
                weight_route_change=60,
                weight_lateness_seconds=1,
                max_window_size_from_earliest=parameter_set[8],
            )

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
    flatland_seed_value = params.flatland_seed_value
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
                                             flatland_seed_value=flatland_seed_value,
                                             max_num_cities=max_num_cities,
                                             grid_mode=grid_mode,
                                             max_rails_between_cities=max_rails_between_cities,
                                             max_rails_in_city=max_rails_in_city,
                                             speed_data=speed_data)
    env_static.reset(random_seed=flatland_seed_value)

    # Generate dynamic environment with single malfunction
    env_malfunction = create_flatland_environment_with_malfunction(number_of_agents=number_of_agents,
                                                                   width=width,
                                                                   height=height,
                                                                   flatland_seed_value=flatland_seed_value,
                                                                   max_num_cities=max_num_cities,
                                                                   grid_mode=grid_mode,
                                                                   max_rails_between_cities=max_rails_between_cities,
                                                                   max_rails_in_city=max_rails_in_city,
                                                                   malfunction_duration=malfunction_duration,
                                                                   earliest_malfunction=earliest_malfunction,
                                                                   speed_data=speed_data)
    env_malfunction.reset(random_seed=flatland_seed_value)
    return env_static, env_malfunction


def save_experiment_agenda_and_hash_to_file(experiment_folder_name: str, experiment_agenda: ExperimentAgenda):
    """Save experiment agenda and current git hash to the folder with the
    experiments.

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

    # write current hash to sha.txt to experiment folder
    _write_sha_txt(experiment_folder_name)


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
    filename = "experiment_{:04d}.pkl".format(experiment_id)
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


def load_and_expand_experiment_results_from_folder(experiment_folder_name: str, experiment_ids: List[int] = None) -> \
        List[ExperimentResultsAnalysis]:
    """Load results as DataFrame to do further analysis.

    Parameters
    ----------
    experiment_folder_name: str
        Folder name of experiment where all experiment files are stored
    experiment_ids
        List of experiment ids which should be loaded, if None all experiments in experiment_folder are loaded

    Returns
    -------
    DataFrame containing the loaded experiment results
    """

    experiment_results_list = []

    files = os.listdir(experiment_folder_name)
    for file in [file for file in files if 'agenda' not in file]:
        file_name = os.path.join(experiment_folder_name, file)
        if file_name.endswith('experiment_agenda.pkl') or not file_name.endswith(".pkl"):
            continue

        # filter experiments according to defined experiment_ids
        if experiment_ids is not None and all(
                [not "experiment_{:04d}".format(exp_id) in file_name for exp_id in experiment_ids]):
            continue

        with open(file_name, 'rb') as handle:
            file_data = pickle.load(handle)
            experiment_results_list.append(expand_experiment_results_for_analysis(file_data))

    return experiment_results_list


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
