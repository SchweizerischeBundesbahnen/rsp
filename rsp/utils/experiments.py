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
save_experiment_agenda_and_hash_to_file
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
import threading
import time
import traceback
from functools import partial
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import tqdm as tqdm
from flatland.envs.rail_env import RailEnv
from pandas import DataFrame

from rsp.experiment_solvers.data_types import ScheduleAndMalfunction
from rsp.experiment_solvers.experiment_solver import ASPExperimentSolver
from rsp.route_dag.analysis.rescheduling_verification_utils import plausibility_check_experiment_results
from rsp.utils.data_types import convert_list_of_experiment_results_analysis_to_data_frame
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

EXPERIMENT_AGENDA_SUBDIRECTORY_NAME = "agenda"
EXPERIMENT_DATA_SUBDIRECTORY_NAME = "data"
EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME = "analysis"


def save_schedule_and_malfunction(schedule_and_malfunction: ScheduleAndMalfunction,
                                  experiment_agenda_directory: str,
                                  experiment_id: int):
    """Persist `ScheduleAndMalfunction` to a file.

    Parameters
    ----------
    schedule_and_malfunction
    experiment_agenda_directory
    experiment_id
    """
    schedule_and_malfunction_file_name = os.path.join(experiment_agenda_directory,
                                                      f"experiment_{experiment_id:03d}_schedule_and_malfunction.pkl")
    check_create_folder(experiment_agenda_directory)
    with open(schedule_and_malfunction_file_name, 'wb') as handle:
        pickle.dump(schedule_and_malfunction, handle, protocol=pickle.HIGHEST_PROTOCOL)


def exists_schedule_and_malfunction(experiment_agenda_directory: str, experiment_id: int) -> bool:
    """Does a persisted `ScheduleAndMalfunction` exist?

    Parameters
    ----------
    experiment_agenda_directory
    experiment_id

    Returns
    -------
    """
    schedule_and_malfunction_file_name = os.path.join(experiment_agenda_directory,
                                                      f"experiment_{experiment_id:03d}_schedule_and_malfunction.pkl")
    return os.path.isfile(schedule_and_malfunction_file_name)


def load_schedule_and_malfunction(experiment_agenda_directory: str, experiment_id: int) -> ScheduleAndMalfunction:
    """Load a persisted `ScheduleAndMalfunction` from a file.

    Parameters
    ----------
    experiment_agenda_directory
    experiment_id

    Returns
    -------
    """
    schedule_and_malfunction_file_name = os.path.join(experiment_agenda_directory,
                                                      f"experiment_{experiment_id:03d}_schedule_and_malfunction.pkl")
    with open(schedule_and_malfunction_file_name, 'rb') as handle:
        file_data: ExperimentAgenda = pickle.load(handle)
        return file_data


def run_experiment(solver: ASPExperimentSolver,
                   experiment_parameters: ExperimentParameters,
                   experiment_base_directory: str,
                   show_results_without_details: bool = True,
                   rendering: bool = False,
                   verbose: bool = False,
                   debug: bool = False
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
    experiment_agenda_directory = f'{experiment_base_directory}/{EXPERIMENT_AGENDA_SUBDIRECTORY_NAME}'
    check_create_folder(experiment_agenda_directory)

    if show_results_without_details:
        print("Running experiment {} in thread {}".format(experiment_parameters.experiment_id, threading.get_ident()))
    start_time = time.time()
    if show_results_without_details:
        print("*** experiment parameters for experiment {}".format(experiment_parameters.experiment_id))
        _pp.pprint(experiment_parameters)

    # B.1: load or re-generate?
    # we want to be able to reuse the same schedule and malfunction to be able to compare
    # identical re-scheduling problems between runs and to debug them
    # if the data already exists, load it and do not re-generate it
    if experiment_agenda_directory is not None and exists_schedule_and_malfunction(
            experiment_agenda_directory=experiment_agenda_directory,
            experiment_id=experiment_parameters.experiment_id):
        schedule_and_malfunction = load_schedule_and_malfunction(
            experiment_agenda_directory=experiment_agenda_directory,
            experiment_id=experiment_parameters.experiment_id)
        _, malfunction_rail_env = create_env_pair_for_experiment(experiment_parameters)
    else:
        malfunction_rail_env, schedule_and_malfunction = create_schedule_and_malfunction(
            debug=debug,
            experiment_parameters=experiment_parameters,
            rendering=rendering,
            solver=solver,
            verbose=verbose)
        if experiment_agenda_directory is not None:
            save_schedule_and_malfunction(
                schedule_and_malfunction=schedule_and_malfunction,
                experiment_agenda_directory=experiment_agenda_directory,
                experiment_id=experiment_parameters.experiment_id)

    if rendering:
        from flatland.utils.rendertools import RenderTool, AgentRenderVariant
        env_renderer = RenderTool(malfunction_rail_env, gl="PILSVG",
                                  agent_render_variant=AgentRenderVariant.ONE_STEP_BEHIND,
                                  show_debug=False,
                                  screen_height=600,  # Adjust these parameters to fit your resolution
                                  screen_width=800)
        env_renderer.reset()
        env_renderer.render_env(show=True, show_observations=False, show_predictions=False)

    # wrap reset params in this function, so we avoid copy-paste errors each time we have to reset the malfunction_rail_env
    def malfunction_env_reset():
        malfunction_rail_env.reset(False, False, False, experiment_parameters.flatland_seed_value)

    malfunction_env_reset()

    # B2: full and delta re-scheduling
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

    if rendering:
        from flatland.utils.rendertools import RenderTool, AgentRenderVariant
        env_renderer.close_window()
    elapsed_time = (time.time() - start_time)
    solver_time_full = experiment_results.results_full.solver_statistics["summary"]["times"]["total"]
    solver_time_full_after_malfunction = \
        experiment_results.results_full_after_malfunction.solver_statistics["summary"]["times"]["total"]
    solver_time_delta_after_malfunction = \
        experiment_results.results_delta_after_malfunction.solver_statistics["summary"]["times"]["total"]
    elapsed_overhead_time = (
            elapsed_time - solver_time_full -
            solver_time_full_after_malfunction -
            solver_time_delta_after_malfunction)
    if show_results_without_details:
        print(("Running experiment {}: took {:5.3f}s "
               "(sched: {:5.3f}s = {:5.2f}% / "
               "resched: {:5.3f}s = {:5.2f}% / "
               "resched-delta: {:5.3f}s = {:5.2f}% / "
               "remaining: {:5.3f}s = {:5.2f}%)  in thread {}")
              .format(experiment_parameters.experiment_id,
                      elapsed_time,
                      solver_time_full,
                      solver_time_full / elapsed_time * 100,
                      solver_time_full_after_malfunction,
                      solver_time_full_after_malfunction / elapsed_time * 100,
                      solver_time_delta_after_malfunction,
                      solver_time_delta_after_malfunction / elapsed_time * 100,
                      elapsed_overhead_time,
                      elapsed_overhead_time / elapsed_time * 100,
                      threading.get_ident()))

    plausibility_check_experiment_results(experiment_results=experiment_results)
    return experiment_results


def create_schedule_and_malfunction(
        experiment_parameters: ExperimentParameters,
        solver: ASPExperimentSolver,
        rendering: bool = False,
        verbose: bool = False,
        debug: bool = False
) -> Tuple[RailEnv, ScheduleAndMalfunction]:
    """B.1 Create schedule and malfunction from experiment parameters.

    Parameters
    ----------
    experiment_parameters
    solver
    debug
    rendering
    verbose

    Returns
    -------
    malfunction_rail_env, schedule_and_malfunction
    """
    static_rail_env, malfunction_rail_env = create_env_pair_for_experiment(experiment_parameters)
    if rendering:
        from flatland.utils.rendertools import RenderTool, AgentRenderVariant
        env_renderer = RenderTool(malfunction_rail_env, gl="PILSVG",
                                  agent_render_variant=AgentRenderVariant.ONE_STEP_BEHIND,
                                  show_debug=False,
                                  screen_height=600,  # Adjust these parameters to fit your resolution
                                  screen_width=800)
        env_renderer.reset()
        env_renderer.render_env(show=True, show_observations=False, show_predictions=False)

    # wrap reset params in this function, so we avoid copy-paste errors each time we have to reset the malfunction_rail_env
    def malfunction_env_reset():
        malfunction_rail_env.reset(False, False, False, experiment_parameters.flatland_seed_value)

    # Run experiments
    schedule_and_malfunction: ScheduleAndMalfunction = solver.gen_schedule_and_malfunction(
        static_rail_env=static_rail_env,
        malfunction_rail_env=malfunction_rail_env,
        malfunction_env_reset=malfunction_env_reset,
        experiment_parameters=experiment_parameters,
        verbose=verbose,
        debug=debug
    )
    return malfunction_rail_env, schedule_and_malfunction


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


def run_and_save_one_experiment(current_experiment_parameters: ExperimentParameters,
                                solver: ASPExperimentSolver,
                                verbose: bool,
                                show_results_without_details: bool,
                                experiment_base_directory: str,
                                rendering: bool = False) -> List[ExperimentResults]:
    """B. Run and save one experiment from experiment parameters.

    Parameters
    ----------
    current_experiment_parameters
    solver
    verbose
    show_results_without_details
    experiment_base_directory
    rendering
    """
    try:
        experiment_data_directory = f'{experiment_base_directory}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}'
        check_create_folder(experiment_data_directory)

        filename = create_experiment_filename(experiment_data_directory, current_experiment_parameters.experiment_id)
        experiment_results: ExperimentResults = run_experiment(solver=solver,
                                                               experiment_parameters=current_experiment_parameters,
                                                               rendering=rendering,
                                                               verbose=verbose,
                                                               experiment_base_directory=experiment_base_directory,
                                                               show_results_without_details=show_results_without_details)
        save_experiment_results_to_file(experiment_results, filename)
        return experiment_results
    except Exception as e:
        print("XXX failed " + filename + " " + str(e))
        traceback.print_exc(file=sys.stdout)


def run_experiment_agenda(experiment_agenda: ExperimentAgenda,
                          experiment_ids: Optional[List[int]] = None,
                          copy_agenda_from_base_directory: Optional[str] = None,
                          run_experiments_parallel: bool = True,
                          show_results_without_details: bool = True,
                          rendering: bool = False,
                          verbose: bool = False) -> (str, str, List[ExperimentResultsAnalysis]):
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
    copy_agenda_from_base_directory: bool
        copy schedule and malfunction data from this directory to the experiments folder if given
    rendering: bool


    Returns
    -------
    Returns the name of the experiment folder
    """
    experiment_base_directory = create_experiment_folder_name(experiment_agenda.experiment_name)
    experiment_data_directory = f'{experiment_base_directory}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}'
    experiment_agenda_directory = f'{experiment_base_directory}/{EXPERIMENT_AGENDA_SUBDIRECTORY_NAME}'

    check_create_folder(experiment_base_directory)
    check_create_folder(experiment_data_directory)
    check_create_folder(experiment_agenda_directory)

    # tee stdout to log file
    stdout_orig = tee_stdout_to_file(log_file=os.path.join(experiment_data_directory, "log.txt"))

    if copy_agenda_from_base_directory is not None:
        _copy_agenda_from_base_directory(copy_agenda_from_base_directory, experiment_agenda_directory)

    if experiment_ids is not None:
        filter_experiment_agenda_partial = partial(filter_experiment_agenda, experiment_ids=experiment_ids)
        experiments_filtered = filter(filter_experiment_agenda_partial, experiment_agenda.experiments)
        experiment_agenda = ExperimentAgenda(
            experiment_name=experiment_agenda.experiment_name,
            experiments=list(experiments_filtered)
        )

    save_experiment_agenda_and_hash_to_file(experiment_agenda_directory, experiment_agenda)

    solver = ASPExperimentSolver()

    if run_experiments_parallel:
        pool = multiprocessing.Pool()
        print(f"pool size {pool._processes} / {multiprocessing.cpu_count()} ({os.cpu_count()}) cpus")
        run_and_save_one_experiment_partial = partial(run_and_save_one_experiment,
                                                      solver=solver,
                                                      verbose=verbose,
                                                      show_results_without_details=show_results_without_details,
                                                      experiment_base_directory=experiment_base_directory
                                                      )
        experiment_results_list = [
            experiment_results
            for experiment_results
            in tqdm.tqdm(
                pool.imap_unordered(
                    run_and_save_one_experiment_partial,
                    experiment_agenda.experiments
                ),
                total=len(experiment_agenda.experiments))]
    else:
        experiment_results_list = [
            run_and_save_one_experiment(current_experiment_parameters=current_experiment_parameters,
                                        solver=solver,
                                        verbose=verbose,
                                        show_results_without_details=show_results_without_details,
                                        experiment_base_directory=experiment_base_directory,
                                        rendering=rendering)
            for current_experiment_parameters
            in tqdm.tqdm(experiment_agenda.experiments)
        ]

    # remove tee
    reset_tee(stdout_orig)
    return experiment_base_directory, experiment_data_directory, experiment_results_list


def _copy_agenda_from_base_directory(copy_agenda_from_base_directory: str, experiment_agenda_directory: str):
    """
    Copy agenda and schedule for re-use.
    Parameters
    ----------
    copy_agenda_from_base_directory
        base directory to copy from
    experiment_agenda_directory
        agenda subdirectory to copy to
    """
    copy_agenda_from_agenda_directory = os.path.join(copy_agenda_from_base_directory,
                                                     EXPERIMENT_AGENDA_SUBDIRECTORY_NAME)
    files = os.listdir(copy_agenda_from_agenda_directory)
    print(f"Copying agenda, schedule and malfunctions {copy_agenda_from_agenda_directory} -> {experiment_agenda_directory}")
    for file in [file for file in files]:
        shutil.copy2(os.path.join(copy_agenda_from_agenda_directory, file), experiment_agenda_directory)


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


def load_and_expand_experiment_results_from_data_folder(
        experiment_data_folder_name: str) -> List[ExperimentResultsAnalysis]:
    """Load results as DataFrame to do further analysis.

    Parameters
    ----------
    experiment_data_folder_name: str
        Folder name of experiment where all experiment files are stored

    Returns
    -------
    DataFrame containing the loaded experiment results
    """

    experiment_results_list = []

    files = os.listdir(experiment_data_folder_name)
    for file in [file for file in files if 'agenda' not in file]:
        file_name = os.path.join(experiment_data_folder_name, file)
        if not file_name.endswith(".pkl"):
            continue
        with open(file_name, 'rb') as handle:
            file_data = pickle.load(handle)
            experiment_results_list.append(expand_experiment_results_for_analysis(file_data))

    return experiment_results_list


def load_without_average(data_folder: str) -> DataFrame:
    """Load all data from the folder, expand and convert to data frame.

    Parameters
    ----------
    data_folder: str
        folder with pkl files.

    Returns
    -------
    DataFrame
    """
    experiment_results_list: List[ExperimentResultsAnalysis] = load_and_expand_experiment_results_from_data_folder(
        data_folder)
    experiment_data: DataFrame = convert_list_of_experiment_results_analysis_to_data_frame(experiment_results_list)
    return experiment_data


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
