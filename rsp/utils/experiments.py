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
import errno
import multiprocessing
import os
import pickle
import pprint
import shutil
from functools import partial
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from pandas import DataFrame

from rsp.utils.data_types import ExperimentAgenda
from rsp.utils.data_types import ExperimentParameters
from rsp.utils.data_types import ExperimentResults
from rsp.utils.data_types import ParameterRanges
from rsp.utils.data_types import SpeedData
from rsp.utils.experiment_env_generators import create_flatland_environment
from rsp.utils.experiment_env_generators import create_flatland_environment_with_malfunction
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
           'experiment_freeze',
           'malfunction',
           'size',
           'n_agents',
           'max_num_cities',
           'max_rail_between_cities',
           'max_rail_in_city']


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
        # Store results
        time_delta_after_m = current_results.time_delta_after_malfunction
        time_full_after_m = current_results.time_full_after_malfunction

        data_frame.append({'experiment_id': experiment_parameters.experiment_id,
                           'time_full': current_results.time_full,
                           'time_full_after_malfunction': time_delta_after_m,
                           'time_delta_after_malfunction': time_full_after_m,
                           'solution_full': current_results.solution_full,
                           'solution_full_after_malfunction': current_results.solution_full_after_malfunction,
                           'solution_delta_after_malfunction': current_results.solution_delta_after_malfunction,
                           'costs_full': current_results.costs_full,
                           'costs_full_after_malfunction': current_results.costs_full_after_malfunction,
                           'costs_delta_after_malfunction': current_results.costs_delta_after_malfunction,
                           'experiment_freeze': current_results.experiment_freeze,
                           'malfunction': current_results.malfunction,
                           'size': experiment_parameters.width,
                           'n_agents': experiment_parameters.number_of_agents,
                           'max_num_cities': experiment_parameters.max_num_cities,
                           'max_rail_between_cities': experiment_parameters.max_rail_between_cities,
                           'max_rail_in_city': experiment_parameters.max_rail_in_city})

        # TODO SIM-239 move to analysis toolkit!
        if show_results_without_details:
            print("*** experiment result of trial {} for experiment {}".format(trial + 1,
                                                                               experiment_parameters.experiment_id))

            _pp.pprint({key: data_frame[-1][key] for key in COLUMNS
                        if not key.startswith('solution_') and not key == 'experiment_freeze'
                        })

            _analyze_times(current_results)
            _analyze_paths(current_results, env)
        if rendering:
            from flatland.utils.rendertools import RenderTool, AgentRenderVariant
            env_renderer.close_window()
    return data_frame


# TODO print only or add to experiment results?
def _analyze_times(current_results: ExperimentResults):
    time_delta_after_m = current_results.time_delta_after_malfunction
    time_full_after_m = current_results.time_full_after_malfunction
    # Delta is all train run way points in the re-schedule that are not also in the schedule
    schedule_trainrunwaypoints = current_results.solution_full
    full_reschedule_trainrunwaypoints_dict = current_results.solution_full_after_malfunction
    schedule_full_reschedule_delta: TrainrunDict = {
        agent_id: sorted(list(
            set(full_reschedule_trainrunwaypoints_dict[agent_id]).difference(
                set(schedule_trainrunwaypoints[agent_id]))),
            key=lambda p: p.scheduled_at)
        for agent_id in schedule_trainrunwaypoints.keys()
    }
    schedule_full_reschedule_delta_percentage = \
        100 * sum([len(schedule_full_reschedule_delta[agent_id])
                   for agent_id in schedule_full_reschedule_delta.keys()]) / \
        sum([len(full_reschedule_trainrunwaypoints_dict[agent_id])
             for agent_id in full_reschedule_trainrunwaypoints_dict.keys()])
    # Freeze is all train run way points in the schedule that are also in the re-schedule
    schedule_full_reschedule_freeze: TrainrunDict = \
        {agent_id: sorted(list(
            set(full_reschedule_trainrunwaypoints_dict[agent_id]).intersection(
                set(schedule_trainrunwaypoints[agent_id]))),
            key=lambda p: p.scheduled_at) for agent_id in schedule_full_reschedule_delta.keys()}
    schedule_full_reschedule_freeze_percentage = 100 * sum(
        [len(schedule_full_reschedule_freeze[agent_id]) for agent_id in schedule_full_reschedule_freeze.keys()]) / sum(
        [len(schedule_trainrunwaypoints[agent_id]) for agent_id in schedule_trainrunwaypoints.keys()])

    # TODO SIM-151 do we need absolute counts as well as below?
    print(
        f"**** full schedule -> full re-schedule: {schedule_full_reschedule_freeze_percentage}%"
        " of waypoints in the full schedule stay the same in the full re-schedule")
    print(
        f"**** full schedule -> full re-schedule: {schedule_full_reschedule_delta_percentage}% "
        "of waypoints in the full re-schedule are different from the initial full schedule")
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
    time_rescheduling_speedup_factor = time_full_after_m / time_delta_after_m
    print(f"**** full re-schedule -> delta re-schedule: "
          f"time speed-up factor {time_rescheduling_speedup_factor:4.1f} "
          f"{time_full_after_m}s -> {time_delta_after_m}s")


# TODO SIM-151 print only or add to experiment results?
def _analyze_paths(experiment_results: ExperimentResults, env: RailEnv):
    schedule_trainruns = experiment_results.solution_full
    malfunction = experiment_results.malfunction
    agents_path_dict = experiment_results.agent_paths_dict

    print("**** number of remaining route alternatives after malfunction")
    for agent_id, schedule_trainrun in schedule_trainruns.items():
        _analyze_agent_path(agent_id, agents_path_dict, env, malfunction, schedule_trainrun)


def _analyze_agent_path(agent_id, agents_path_dict, env, malfunction, schedule_trainrun):
    # where are we at the malfunction?
    scheduled_already_done = None
    scheduled_remainder = None
    for index, trainrun_waypoint in enumerate(schedule_trainrun):
        if trainrun_waypoint.waypoint.position == env.agents[agent_id].target:
            scheduled_already_done = schedule_trainrun
            # already done
            break
        if trainrun_waypoint.scheduled_at >= malfunction.time_step:
            if trainrun_waypoint.scheduled_at == malfunction.time_step:
                scheduled_already_done = schedule_trainrun[:index + 1]
                scheduled_remainder = schedule_trainrun[index + 1:]
            else:
                scheduled_already_done = schedule_trainrun[:index]
                scheduled_remainder = schedule_trainrun[index:]
            break
    if scheduled_remainder is None:
        # agent has not started yet or is at the target already
        return
    remainder_waypoints_set = set(
        map(lambda trainrun_waypoint: trainrun_waypoint.waypoint, scheduled_remainder))

    nb_paths = 0
    very_verbose = False
    for path_index, agent_path in enumerate(agents_path_dict[agent_id]):
        after_malfunction = False
        reachable_after_malfunction = False
        for waypoint in agent_path:
            after_malfunction = waypoint in remainder_waypoints_set
            reachable_after_malfunction = \
                reachable_after_malfunction or (after_malfunction and waypoint in remainder_waypoints_set)
            if reachable_after_malfunction:
                nb_paths += 1
                break
        if very_verbose:
            print(f"agent {agent_id}: path {path_index} "
                  f"reachable_from_malfunction_point={after_malfunction}")
            print(f"   agent {agent_id}: at malfunction {malfunction}, scheduled_already_done={scheduled_already_done}")
            print(f"   agent {agent_id}: at malfunction {malfunction}, schedule_remainder={scheduled_remainder}")
            print(f"   agent {agent_id}: path {path_index} is {agent_path}")


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
    experiment_agenda_filtered = filter(filter_experiment_agenda_partial, experiment_agenda.experiments)
    save_experiment_agenda_to_file(experiment_folder_name, experiment_agenda_filtered)

    if run_experiments_parallel:
        pool = multiprocessing.Pool()
        run_and_save_one_experiment_partial = partial(run_and_save_one_experiment,
                                                      solver=solver,
                                                      verbose=verbose,
                                                      show_results_without_details=show_results_without_details,
                                                      experiment_folder_name=experiment_folder_name
                                                      )
        pool.map(run_and_save_one_experiment_partial, experiment_agenda_filtered)
    else:
        for current_experiment_parameters in experiment_agenda_filtered:
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
                             trials_per_experiment: int = 10) -> ExperimentAgenda:
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

    Returns
    -------
    ExperimentAgenda built from the ParameterRanges
    :param speed_data:
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
        current_experiment = ExperimentParameters(experiment_id=param_id,
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
                                                  earliest_malfunction=parameter_set[5],
                                                  malfunction_duration=parameter_set[6])
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
    if not os.path.exists(os.path.dirname(file_name)):
        try:
            os.makedirs(os.path.dirname(file_name))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise exc
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
    if not os.path.exists(os.path.dirname(file_name)):
        try:
            os.makedirs(os.path.dirname(file_name))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise exc

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
    for file in files:
        file_name = os.path.join(experiment_folder_name, file)
        if file_name.endswith('experiment_agenda.pkl'):
            continue
        with open(file_name, 'rb') as handle:
            file_data = pickle.load(handle)
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
