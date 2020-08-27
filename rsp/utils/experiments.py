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
import itertools
import logging
import multiprocessing
import os
import pickle
import platform
import pprint
import shutil
import sys
import threading
import time
import traceback
from functools import partial
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import tqdm as tqdm
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_shortest_paths import get_k_shortest_paths
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from pandas import DataFrame

from rsp.experiment_solvers.asp.asp_helper import _print_stats
from rsp.experiment_solvers.data_types import ExperimentMalfunction
from rsp.experiment_solvers.data_types import Infrastructure
from rsp.experiment_solvers.data_types import Schedule
from rsp.experiment_solvers.experiment_solver import asp_reschedule_wrapper
from rsp.experiment_solvers.experiment_solver import asp_schedule_wrapper
from rsp.logger import add_file_handler_to_rsp_logger
from rsp.logger import remove_file_handler_from_rsp_logger
from rsp.logger import rsp_logger
from rsp.schedule_problem_description.analysis.rescheduling_verification_utils import plausibility_check_experiment_results
from rsp.schedule_problem_description.analysis.route_dag_analysis import visualize_route_dag_constraints_simple_wrapper
from rsp.schedule_problem_description.data_types_and_utils import _get_topology_from_agents_path_dict
from rsp.schedule_problem_description.data_types_and_utils import get_paths_in_route_dag
from rsp.schedule_problem_description.data_types_and_utils import get_sources_for_topo
from rsp.schedule_problem_description.data_types_and_utils import ScheduleProblemDescription
from rsp.schedule_problem_description.data_types_and_utils import TopoDict
from rsp.schedule_problem_description.route_dag_constraints.delta_zero import delta_zero_for_all_agents
from rsp.schedule_problem_description.route_dag_constraints.perfect_oracle import perfect_oracle_for_all_agents
from rsp.schedule_problem_description.route_dag_constraints.route_dag_constraints_schedule import _get_route_dag_constraints_for_scheduling
from rsp.utils.data_types import convert_list_of_experiment_results_analysis_to_data_frame
from rsp.utils.data_types import expand_experiment_results_for_analysis
from rsp.utils.data_types import ExperimentAgenda
from rsp.utils.data_types import ExperimentParameters
from rsp.utils.data_types import ExperimentResults
from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.data_types import InfrastructureParameters
from rsp.utils.data_types import InfrastructureParametersRange
from rsp.utils.data_types import ParameterRangesAndSpeedData
from rsp.utils.data_types import ReScheduleParameters
from rsp.utils.data_types import ReScheduleParametersRange
from rsp.utils.data_types import ScheduleParameters
from rsp.utils.data_types import ScheduleParametersRange
from rsp.utils.data_types import SpeedData
from rsp.utils.experiment_env_generators import create_flatland_environment
from rsp.utils.file_utils import check_create_folder
from rsp.utils.file_utils import get_experiment_id_from_filename
from rsp.utils.file_utils import newline_and_flush_stdout_and_stderr
from rsp.utils.psutil_helpers import current_process_stats_human_readable
from rsp.utils.psutil_helpers import virtual_memory_human_readable

#  B008 Do not perform function calls in argument defaults.
#  The call is performed only once at function definition time.
#  All calls to your function will reuse the result of that definition-time function call.
#  If this is intended, ass ign the function call to a module-level variable and use that variable as a default value.
AVAILABLE_CPUS = os.cpu_count()

_pp = pprint.PrettyPrinter(indent=4)

EXPERIMENT_INFRA_SUBDIRECTORY_NAME = "infra"
EXPERIMENT_SCHEDULE_SUBDIRECTORY_NAME = "schedule"

EXPERIMENT_DATA_SUBDIRECTORY_NAME = "data"
EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME = "analysis"
EXPERIMENT_POTASSCO_SUBDIRECTORY_NAME = "potassco"


def _pickle_dump(obj: Any, file_name: str, folder: Optional[str] = None):
    file_path = file_name
    if folder is not None:
        file_path = os.path.join(folder, file_name)
    check_create_folder(folder)
    with open(file_path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _pickle_load(file_name: str, folder: Optional[str] = None):
    file_path = file_name
    if folder is not None:
        file_path = os.path.join(folder, file_name)
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)


def save_schedule(schedule: Schedule,
                  schedule_parameters: ScheduleParameters,
                  base_directory: str,
                  ):
    """Persist `Schedule` and `ScheduleParameters` to a file.

    Parameters
    ----------
    schedule_parameters
    schedule
    base_directory
    """
    folder = os.path.join(base_directory, EXPERIMENT_INFRA_SUBDIRECTORY_NAME, f"{schedule_parameters.infra_id:03d}", EXPERIMENT_SCHEDULE_SUBDIRECTORY_NAME,
                          f"{schedule_parameters.schedule_id:03d}")
    _pickle_dump(obj=schedule, folder=folder, file_name="schedule.pkl")
    _pickle_dump(obj=schedule_parameters, folder=folder, file_name="schedule_parameters.pkl")


def save_infrastructure(
        infrastructure: Infrastructure,
        infrastructure_parameters: InfrastructureParameters,
        base_directory: str
):
    """Persist `Infrastructure` to a file.
    Parameters
    ----------
    infrastructure_parameters
    infrastructure
    base_directory
    """
    folder = os.path.join(base_directory, EXPERIMENT_INFRA_SUBDIRECTORY_NAME, f"{infrastructure_parameters.infra_id:03d}")
    _pickle_dump(obj=infrastructure, folder=folder, file_name="infrastructure.pkl")
    _pickle_dump(obj=infrastructure_parameters, folder=folder, file_name="infrastructure_parameters.pkl")


def exists_schedule(base_directory: str, infra_id: int, schedule_id: int) -> bool:
    """Does a persisted `Schedule` exist?"""
    file_name = os.path.join(base_directory, EXPERIMENT_INFRA_SUBDIRECTORY_NAME, f"{infra_id:03d}", EXPERIMENT_SCHEDULE_SUBDIRECTORY_NAME, f"{schedule_id:03d}",
                             f"schedule.pkl")
    return os.path.isfile(file_name)


def exists_infrastructure(base_directory: str, infra_id: int) -> bool:
    """Does a persisted `Infrastructure` exist?"""
    file_name = os.path.join(base_directory, EXPERIMENT_INFRA_SUBDIRECTORY_NAME, f"{infra_id:03d}", "infrastructure.pkl")
    return os.path.isfile(file_name)


def exists_malfunction(base_directory: str, experiment_id: int) -> bool:
    """Does a persisted `ExperimentMalfunction` exist?
    Parameters
    ----------
    base_directory
    experiment_id
    Returns
    -------
    """
    file_name = os.path.join(base_directory, EXPERIMENT_INFRA_SUBDIRECTORY_NAME, f"{experiment_id:03d}", f"schedule.pkl")
    return os.path.isfile(file_name)


def load_infrastructure(base_directory: str, infra_id: int) -> Tuple[Infrastructure, InfrastructureParameters]:
    """Load a persisted `Infrastructure` from a file.
    Parameters
    ----------
    base_directory
    infra_id


    Returns
    -------
    """
    folder = os.path.join(base_directory, EXPERIMENT_INFRA_SUBDIRECTORY_NAME, f"{infra_id:03d}")
    infra = _pickle_load(folder=folder, file_name=f"infrastructure.pkl")
    infra_parameters = _pickle_load(folder=folder, file_name=f"infrastructure_parameters.pkl")
    return infra, infra_parameters


def load_schedule(base_directory: str, infra_id: int, schedule_id: int = 0) -> Tuple[Schedule, ScheduleParameters]:
    """Load a persisted `Schedule` from a file.
    Parameters
    ----------
    schedule_id
    base_directory
    infra_id


    Returns
    -------
    """
    folder = os.path.join(base_directory, EXPERIMENT_INFRA_SUBDIRECTORY_NAME, f"{infra_id:03d}", EXPERIMENT_SCHEDULE_SUBDIRECTORY_NAME,
                          f"{schedule_id:03d}")
    schedule = _pickle_load(folder=folder, file_name="schedule.pkl")
    schedule_parameters = _pickle_load(folder=folder, file_name="schedule_parameters.pkl")
    return schedule, schedule_parameters


def run_experiment_in_memory(
        schedule: Schedule,
        experiment_parameters: ExperimentParameters,
        infrastructure_topo_dict: TopoDict,
        verbose: bool = False,
        debug: bool = False,
        visualize_route_dag_constraints: bool = False
) -> ExperimentResults:
    """A.2 + B Runs the main part of the experiment: re-scheduling full and
    delta.

    Parameters
    ----------
    schedule
        operational schedule that where malfunction happened
    experiment_parameters
        hierarchical experiment parameters
    infrastructure_topo_dict
        the "full" topology for each agent
    verbose
        verbose logging
    debug
        debug logging
    visualize_route_dag_constraints
        save route dag constraints visualization for debugging

    Returns
    -------
    ExperimentResults
    """
    rsp_logger.info(f"start re-schedule full and delta for experiment {experiment_parameters.experiment_id}")
    schedule_problem, schedule_result = schedule
    schedule_trainruns: TrainrunDict = schedule_result.trainruns_dict

    # --------------------------------------------------------------------------------------
    # A.2 Determine malfunction (deterministically from experiment parameters)
    # --------------------------------------------------------------------------------------
    experiment_malfunction = gen_malfunction(
        earliest_malfunction=experiment_parameters.earliest_malfunction,
        malfunction_duration=experiment_parameters.malfunction_duration,
        malfunction_agent_id=experiment_parameters.malfunction_agend_id,
        schedule_trainruns=schedule.schedule_experiment_result.trainruns_dict
    )

    if debug:
        _visualize_route_dag_constraints_for_schedule_and_malfunction(
            schedule=schedule,
            experiment_malfunction=experiment_malfunction
        )

    # --------------------------------------------------------------------------------------
    # B.1. Re-schedule Full
    # --------------------------------------------------------------------------------------
    rsp_logger.info("2. reschedule full")
    # clone topos since propagation will modify them
    full_reschedule_topo_dict = {agent_id: topo.copy() for agent_id, topo in infrastructure_topo_dict.items()}
    full_reschedule_problem: ScheduleProblemDescription = delta_zero_for_all_agents(
        malfunction=experiment_malfunction,
        schedule_trainruns=schedule_trainruns,
        minimum_travel_time_dict=schedule_problem.minimum_travel_time_dict,
        latest_arrival=schedule_problem.max_episode_steps + experiment_malfunction.malfunction_duration,
        max_window_size_from_earliest=experiment_parameters.max_window_size_from_earliest,
        topo_dict=full_reschedule_topo_dict,
        weight_route_change=experiment_parameters.weight_route_change,
        weight_lateness_seconds=experiment_parameters.weight_lateness_seconds
    )

    # activate visualize_route_dag_constraints for debugging
    if visualize_route_dag_constraints:
        for agent_id in schedule_trainruns:
            visualize_route_dag_constraints_simple_wrapper(
                schedule_problem_description=full_reschedule_problem,
                trainrun_dict=None,
                experiment_malfunction=experiment_malfunction,
                agent_id=agent_id,
                file_name=f"rescheduling_neu_agent_{agent_id}.pdf",
            )

    full_reschedule_result = asp_reschedule_wrapper(
        reschedule_problem_description=full_reschedule_problem,
        debug=debug,
        asp_seed_value=experiment_parameters.schedule_parameters.asp_seed_value
    )

    full_reschedule_trainruns = full_reschedule_result.trainruns_dict

    if verbose:
        print(f"  **** full re-schedule_solution=\n{full_reschedule_trainruns}")

    # --------------------------------------------------------------------------------------
    # B.2 Re-Schedule Delta
    # --------------------------------------------------------------------------------------
    rsp_logger.info("3. reschedule delta")
    # clone topos since propagation will modify them
    delta_reschedule_topo_dict = {agent_id: topo.copy() for agent_id, topo in infrastructure_topo_dict.items()}
    delta_reschedule_problem = perfect_oracle_for_all_agents(
        full_reschedule_trainrun_dict=full_reschedule_trainruns,
        malfunction=experiment_malfunction,
        max_episode_steps=schedule_problem.max_episode_steps + experiment_malfunction.malfunction_duration,
        schedule_topo_dict=delta_reschedule_topo_dict,
        schedule_trainrun_dict=schedule_trainruns,
        minimum_travel_time_dict=schedule_problem.minimum_travel_time_dict,
        max_window_size_from_earliest=experiment_parameters.max_window_size_from_earliest,
        weight_route_change=experiment_parameters.weight_route_change,
        weight_lateness_seconds=experiment_parameters.weight_lateness_seconds
    )

    # activate visualize_route_dag_constraints for debugging
    if visualize_route_dag_constraints:
        for agent_id in schedule_trainruns:
            visualize_route_dag_constraints_simple_wrapper(
                schedule_problem_description=delta_reschedule_problem,
                trainrun_dict=None,
                experiment_malfunction=experiment_malfunction,
                agent_id=agent_id,
                file_name=f"delta_rescheduling_neu_agent_{agent_id}.pdf",
            )

    delta_reschedule_result = asp_reschedule_wrapper(
        reschedule_problem_description=delta_reschedule_problem,
        debug=debug,
        asp_seed_value=experiment_parameters.schedule_parameters.asp_seed_value
    )

    if verbose:
        print(f"  **** delta re-schedule solution")
        print(delta_reschedule_result.trainruns_dict)

    # --------------------------------------------------------------------------------------
    # B.3. Result
    # --------------------------------------------------------------------------------------
    current_results = ExperimentResults(
        experiment_parameters=experiment_parameters,
        malfunction=experiment_malfunction,
        problem_full=schedule_problem,
        problem_full_after_malfunction=full_reschedule_problem,
        problem_delta_after_malfunction=delta_reschedule_problem,
        results_full=schedule_result,
        results_full_after_malfunction=full_reschedule_result,
        results_delta_after_malfunction=delta_reschedule_result
    )
    rsp_logger.info(f"done re-schedule full and delta for experiment {experiment_parameters.experiment_id}")
    return current_results


def _visualize_route_dag_constraints_for_schedule_and_malfunction(
        schedule: Schedule,
        experiment_malfunction: ExperimentMalfunction):
    for agent_id in schedule.schedule_experiment_result.trainruns_dict:
        visualize_route_dag_constraints_simple_wrapper(
            schedule_problem_description=schedule.schedule_problem_description,
            trainrun_dict=None,
            experiment_malfunction=experiment_malfunction,
            agent_id=agent_id,
            file_name=f"schedule_alt_agent_{agent_id}.pdf"
        )


def _render_route_dags_from_data(experiment_base_directory: str, experiment_id: int):
    results_before: ExperimentResultsAnalysis = load_and_expand_experiment_results_from_data_folder(
        experiment_data_folder_name=experiment_base_directory + "/data",
        experiment_ids=[experiment_id]
    )[0]
    problem_full_after_malfunction: ScheduleProblemDescription = results_before.problem_full_after_malfunction
    for agent_id in problem_full_after_malfunction.route_dag_constraints_dict:
        visualize_route_dag_constraints_simple_wrapper(
            schedule_problem_description=problem_full_after_malfunction.schedule_problem_description,
            trainrun_dict=None,
            experiment_malfunction=problem_full_after_malfunction.experiment_malfunction,
            agent_id=agent_id,
            file_name=f"reschedule_alt_agent_{agent_id}.pdf"
        )


def _get_asp_solver_details_from_statistics(elapsed_time: float, statistics: Dict):
    return "{:5.3f}s = {:5.2f}%  ({:5.3f}s (Solving: {}s 1st Model: {}s Unsat: {}s)".format(
        statistics["summary"]["times"]["total"],
        statistics["summary"]["times"]["total"] / elapsed_time * 100,
        statistics["summary"]["times"]["total"],
        statistics["summary"]["times"]["solve"],
        statistics["summary"]["times"]["sat"],
        statistics["summary"]["times"]["unsat"],
    )


def gen_infrastructure(
        infra_parameters: InfrastructureParameters
) -> Infrastructure:
    """A.1.1 infrastructure generation."""
    rsp_logger.info(f"gen_infrastructure {infra_parameters}")
    infra = create_infrastructure_from_rail_env(
        env=create_env_from_experiment_parameters(infra_parameters),
        k=infra_parameters.number_of_shortest_paths_per_agent)
    rsp_logger.info(f"done gen_infrastructure {infra_parameters}")
    return infra


def gen_schedule(
        schedule_parameters: ScheduleParameters,
        infrastructure: Infrastructure,
        debug: bool = False

) -> Schedule:
    """A.1.2 Create schedule from parameter ranges.

    Parameters
    ----------
    infrastructure
    schedule_parameters
    debug

    Returns
    -------
    """
    rsp_logger.info(f"gen_schedule {schedule_parameters}")
    schedule_problem = create_schedule_problem_description_from_instructure(
        infrastructure=infrastructure,
        number_of_shortest_paths_per_agent_schedule=schedule_parameters.number_of_shortest_paths_per_agent_schedule
    )
    if debug:
        for agent_id, topo in schedule_problem.topo_dict.items():
            rsp_logger.info(f"    {agent_id} has {len(get_paths_in_route_dag(topo))} paths in scheduling")
            rsp_logger.info(f"    {agent_id} has {len(get_paths_in_route_dag(infrastructure.topo_dict[agent_id]))} paths in infrastructure")

    schedule_result = asp_schedule_wrapper(
        schedule_problem_description=schedule_problem,
        asp_seed_value=schedule_parameters.asp_seed_value,
        debug=debug
    )
    rsp_logger.info(f"done gen_schedule {schedule_parameters}")
    return Schedule(schedule_problem_description=schedule_problem, schedule_experiment_result=schedule_result)


def gen_malfunction(
        earliest_malfunction: int,
        malfunction_duration: int,
        schedule_trainruns: TrainrunDict,
        malfunction_agent_id: int,
) -> ExperimentMalfunction:
    """A.2.2. Create malfunction.

    Parameters
    ----------
    earliest_malfunction
    malfunction_duration
    malfunction_agent_id
    schedule_trainruns

    Returns
    -------
    """
    # --------------------------------------------------------------------------------------
    # 1. Generate malfuntion
    # --------------------------------------------------------------------------------------
    # The malfunction is chosen to start relative to the start time of the malfunction_agent_id
    # This relative malfunction time makes it easier to run malfunciton-time variation experiments
    malfunction_start = min(schedule_trainruns[malfunction_agent_id][0].scheduled_at + earliest_malfunction,
                            schedule_trainruns[malfunction_agent_id][-1].scheduled_at)
    malfunction = ExperimentMalfunction(
        time_step=malfunction_start,
        malfunction_duration=malfunction_duration,
        agent_id=malfunction_agent_id
    )
    return malfunction


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
    rsp_logger.info(f"writing {sha} to {out_file}")
    with open(out_file, 'w') as out:
        out.write(sha)


def run_experiment_from_to_file(
        experiment_parameters: ExperimentParameters,
        experiment_base_directory: str,
        experiment_output_directory: str,
        verbose: bool = False,
        debug: bool = False,
):
    """A.2 + B. Run and save one experiment from experiment parameters.
    Parameters
    ----------
    experiment_base_directory
        base for infrastructure and schedules
    experiment_parameters
        contains reference to infrastructure and schedules
    verbose
    experiment_output_directory
    debug
    """

    experiment_data_directory = f'{experiment_output_directory}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}'

    # add logging file handler in this thread
    stdout_log_file = os.path.join(experiment_data_directory, f"log.txt")
    stderr_log_file = os.path.join(experiment_data_directory, f"err.txt")
    stdout_log_fh = add_file_handler_to_rsp_logger(stdout_log_file, logging.INFO)
    stderr_log_fh = add_file_handler_to_rsp_logger(stderr_log_file, logging.ERROR)

    rsp_logger.info(f"start experiment {experiment_parameters.experiment_id}")
    try:

        check_create_folder(experiment_data_directory)
        filename = create_experiment_filename(experiment_data_directory, experiment_parameters.experiment_id)

        start_datetime_str = datetime.datetime.now().strftime("%H:%M:%S")
        rsp_logger.info(
            "Running experiment {} under pid {} at {}".format(experiment_parameters.experiment_id, os.getpid(),
                                                              start_datetime_str))
        start_time = time.time()

        rsp_logger.info("*** experiment parameters for experiment {}. {}"
                        .format(experiment_parameters.experiment_id,
                                _pp.pformat(experiment_parameters)))

        if experiment_base_directory is None or not exists_schedule(
                base_directory=experiment_base_directory,
                infra_id=experiment_parameters.infra_parameters.infra_id,
                schedule_id=experiment_parameters.schedule_parameters.schedule_id
        ):
            rsp_logger.warn(f"Could not find schedule for {experiment_parameters.experiment_id} in {experiment_base_directory}")
            return

        rsp_logger.info(f"load_schedule for {experiment_parameters.experiment_id}")
        schedule, schedule_parameters = load_schedule(
            base_directory=f"{experiment_base_directory}",
            infra_id=experiment_parameters.infra_parameters.infra_id,
            schedule_id=experiment_parameters.schedule_parameters.schedule_id
        )
        infrastructure, _ = load_infrastructure(
            base_directory=f"{experiment_base_directory}",
            infra_id=experiment_parameters.infra_parameters.infra_id
        )

        if debug:
            _render_route_dags_from_data(
                experiment_base_directory=experiment_output_directory,
                experiment_id=experiment_parameters.experiment_id
            )

        # B2: full and delta re-scheduling
        experiment_results: ExperimentResults = run_experiment_in_memory(
            schedule=schedule,
            experiment_parameters=experiment_parameters,
            infrastructure_topo_dict=infrastructure.topo_dict,
            verbose=verbose,
            debug=debug
        )
        if experiment_results is None:
            print(f"No malfunction for experiment {experiment_parameters.experiment_id}")
            return []

        elapsed_time = (time.time() - start_time)
        end_datetime_str = datetime.datetime.now().strftime("%H:%M:%S")
        s = (
            "Running experiment {}: took {:5.3f}s ({}--{}) (sched:  {} / re-sched full:  {} / re-sched delta:  {} / ").format(
            experiment_parameters.experiment_id,
            elapsed_time,
            start_datetime_str,
            end_datetime_str,
            _get_asp_solver_details_from_statistics(elapsed_time=elapsed_time,
                                                    statistics=experiment_results.results_full.solver_statistics),
            _get_asp_solver_details_from_statistics(elapsed_time=elapsed_time,
                                                    statistics=experiment_results.results_full_after_malfunction.solver_statistics),
            _get_asp_solver_details_from_statistics(elapsed_time=elapsed_time,
                                                    statistics=experiment_results.results_delta_after_malfunction.solver_statistics),
        )
        solver_time_full = experiment_results.results_full.solver_statistics["summary"]["times"]["total"]
        solver_time_full_after_malfunction = \
            experiment_results.results_full_after_malfunction.solver_statistics["summary"]["times"]["total"]
        solver_time_delta_after_malfunction = \
            experiment_results.results_delta_after_malfunction.solver_statistics["summary"]["times"]["total"]
        elapsed_overhead_time = (
                elapsed_time - solver_time_full -
                solver_time_full_after_malfunction -
                solver_time_delta_after_malfunction)
        s += "remaining: {:5.3f}s = {:5.2f}%)  in thread {}".format(
            elapsed_overhead_time,
            elapsed_overhead_time / elapsed_time * 100,
            threading.get_ident())
        rsp_logger.info(s)

        rsp_logger.info(virtual_memory_human_readable())
        rsp_logger.info(current_process_stats_human_readable())

        # TODO SIM-324 pull out validation steps
        plausibility_check_experiment_results(experiment_results=experiment_results)

        save_experiment_results_to_file(experiment_results, filename)
        return os.getpid()
    except Exception as e:
        rsp_logger.error("XXX failed " + filename + " " + str(e))
        traceback.print_exc(file=sys.stderr)
        return os.getpid()
    finally:
        remove_file_handler_from_rsp_logger(stdout_log_fh)
        remove_file_handler_from_rsp_logger(stderr_log_fh)
        rsp_logger.info(f"end experiment {experiment_parameters.experiment_id}")


def run_experiment_agenda(
        experiment_agenda: ExperimentAgenda,
        experiment_base_directory: str,
        experiment_output_base_directory: Optional[str] = None,
        filter_experiment_agenda: Callable[[ExperimentParameters], bool] = None,
        # take only half of avilable cpus so the machine stays responsive
        run_experiments_parallel: int = AVAILABLE_CPUS // 2,
        verbose: bool = False,
) -> str:
    """Run A.2 + B.
    Parameters
    ----------

    experiment_output_base_directory
    experiment_agenda: ExperimentAgenda
        Full list of experiments
    experiment_base_directory: str
        where are schedules etc?
    filter_experiment_agenda
        filter which experiment to run
    run_experiments_parallel: in
        run experiments in parallel
    verbose: bool
        Print additional information

    Returns
    -------
    Returns the name of the experiment base and data folders
    """
    if experiment_output_base_directory is None:
        experiment_output_base_directory = experiment_base_directory
    experiment_output_directory = f"{experiment_output_base_directory}/" + create_experiment_folder_name(experiment_agenda.experiment_name)
    experiment_data_directory = f'{experiment_output_directory}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}'

    check_create_folder(experiment_data_directory)

    if run_experiments_parallel <= 1:
        rsp_logger.warn(
            "Using only one process in pool might cause pool to stall sometimes. Use more than one process in pool?")

    # tee stdout to log file
    stdout_log_file = os.path.join(experiment_data_directory, "log.txt")
    stderr_log_file = os.path.join(experiment_data_directory, "err.txt")
    stdout_log_fh = add_file_handler_to_rsp_logger(stdout_log_file, logging.INFO)
    stderr_log_fh = add_file_handler_to_rsp_logger(stderr_log_file, logging.ERROR)
    try:
        if filter_experiment_agenda is not None:
            experiments_filtered = filter(filter_experiment_agenda, experiment_agenda.experiments)
            experiment_agenda = ExperimentAgenda(
                experiment_name=experiment_agenda.experiment_name,
                experiments=list(experiments_filtered)
            )

        rsp_logger.info(f"============================================================================================================")
        rsp_logger.info(f"RUNNING agenda {experiment_base_directory} -> {experiment_data_directory}")
        rsp_logger.info(f"============================================================================================================")
        for file_name in ["rsp/utils/global_constants.py"]:
            with open(file_name, "r") as content:
                rsp_logger.info(f"{file_name}: {content.read()}")
        rsp_logger.info(f"============================================================================================================")

        # use processes in pool only once because of https://github.com/potassco/clingo/issues/203
        # https://stackoverflow.com/questions/38294608/python-multiprocessing-pool-new-process-for-each-variable
        # N.B. even with parallelization degree 1, we want to run each experiment in a new process
        #      in order to get around https://github.com/potassco/clingo/issues/203
        pool = multiprocessing.Pool(
            processes=run_experiments_parallel,
            maxtasksperchild=1)
        rsp_logger.info(f"pool size {pool._processes} / {multiprocessing.cpu_count()} ({os.cpu_count()}) cpus on {platform.node()}")
        # nicer printing when tdqm print to stderr and we have logging to stdout shown in to the same console (IDE, separated in files)
        newline_and_flush_stdout_and_stderr()
        run_and_save_one_experiment_partial = partial(
            run_experiment_from_to_file,
            verbose=verbose,
            experiment_base_directory=experiment_base_directory,
            experiment_output_directory=experiment_output_directory,
        )

        for pid_done in tqdm.tqdm(
                pool.imap_unordered(
                    run_and_save_one_experiment_partial,
                    experiment_agenda.experiments
                ),
                total=len(experiment_agenda.experiments)):
            # unsafe use of inner API
            procs = [f"{str(proc)}={proc.pid}" for proc in pool._pool]
            rsp_logger.info(f'pid {pid_done} done. Pool: {procs}')

        # nicer printing when tdqm print to stderr and we have logging to stdout shown in to the same console (IDE)
        newline_and_flush_stdout_and_stderr()
        _print_error_summary(experiment_data_directory)
    finally:
        remove_file_handler_from_rsp_logger(stdout_log_fh)
        remove_file_handler_from_rsp_logger(stderr_log_fh)

    return experiment_output_directory


def _print_error_summary(experiment_data_directory):
    rsp_logger.info(f"loading and expanding experiment results from {experiment_data_directory}")

    print(f"=========================================================")
    print(f"ERROR SUMMARY")
    print(f"=========================================================")
    with open(os.path.join(experiment_data_directory, "err.txt"), "r") as file_in:
        content = file_in.read()
        print(content)
    print(f"=========================================================")
    print(f"END OF ERROR SUMMARY")
    print(f"=========================================================")
    print("\n\n\n\n")


def filter_experiment_agenda(current_experiment_parameters, experiment_ids) -> bool:
    return current_experiment_parameters.experiment_id in experiment_ids


def create_experiment_agenda_from_parameter_ranges_and_speed_data(
        experiment_name: str,
        parameter_ranges_and_speed_data: ParameterRangesAndSpeedData,
        flatland_seed: int = 12,
        experiments_per_grid_element: int = 1,
        debug: bool = False) -> ExperimentAgenda:
    """Create an experiment agenda given a range of parameters defined as
    ParameterRanges.
    Parameters
    ----------

    parameter_ranges_and_speed_data
    flatland_seed
    experiment_name: str
        Name of the experiment
    experiments_per_grid_element: int
        Number of runs with different seed per parameter set we want to run
    debug

    Returns
    -------
    ExperimentAgenda built from the ParameterRanges
    """
    parameter_ranges = parameter_ranges_and_speed_data.parameter_ranges
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
            experiment_id = grid_id * experiments_per_grid_element + run_of_this_grid_element
            # 0: size_range
            # 1: agent_range
            # 2: in_city_rail_range
            # 3: out_city_rail_range
            # 4: city_range
            # 5: earliest_malfunction
            # 6: malfunction_duration
            # 7: number_of_shortest_paths_per_agent
            # 8: max_window_size_from_earliest
            # 9: asp_seed_value
            # 10: weight_route_change
            # 11: weight_lateness_seconds
            current_experiment = ExperimentParameters(
                experiment_id=experiment_id,
                grid_id=grid_id,

                infra_parameters=InfrastructureParameters(
                    infra_id=grid_id,
                    speed_data=parameter_ranges_and_speed_data.speed_data,
                    width=parameter_set[0],
                    height=parameter_set[0],
                    flatland_seed_value=flatland_seed + run_of_this_grid_element,
                    max_num_cities=parameter_set[4],
                    # Do we need to have this true?
                    grid_mode=False,
                    max_rail_between_cities=parameter_set[3],
                    max_rail_in_city=parameter_set[2],
                    number_of_agents=parameter_set[1],
                    number_of_shortest_paths_per_agent=parameter_set[7]
                ),
                schedule_parameters=ScheduleParameters(
                    infra_id=grid_id,
                    schedule_id=grid_id,
                    asp_seed_value=parameter_set[9],
                    number_of_shortest_paths_per_agent_schedule=1
                ),

                earliest_malfunction=parameter_set[5],
                malfunction_duration=parameter_set[6],
                malfunction_agend_id=0,
                weight_route_change=parameter_set[10],
                weight_lateness_seconds=parameter_set[11],
                max_window_size_from_earliest=parameter_set[8],
            )

            experiment_list.append(current_experiment)
    experiment_agenda = ExperimentAgenda(experiment_name=experiment_name, experiments=experiment_list)
    rsp_logger.info("Generated an agenda with {} experiments".format(len(experiment_list)))
    return experiment_agenda


def create_infrastructure_and_schedule_from_ranges(
        infrastructure_parameters_range: InfrastructureParametersRange,
        schedule_parameters_range: ScheduleParametersRange,
        base_directory: str,
        speed_data: SpeedData,
        grid_mode: bool = True
) -> List[ScheduleParameters]:
    list_of_infrastructure_parameters = expand_infrastructure_parameter_range_and_generate_infrastructure(
        infrastructure_parameter_range=infrastructure_parameters_range,
        base_directory=base_directory,
        speed_data=speed_data,
        grid_mode=grid_mode
    )
    list_of_schedule_parameters: List[ScheduleParameters] = list(itertools.chain.from_iterable([
        expand_schedule_parameter_range_and_generate_schedule(
            schedule_parameters_range=schedule_parameters_range,
            base_directory=base_directory,
            infra_id=infrastructure_parameters.infra_id
        )
        for infrastructure_parameters in list_of_infrastructure_parameters
    ]))
    return list_of_schedule_parameters


def list_infrastructure_and_schedule_params_from_base_directory(
        base_directory: str,
        infra_ids: List[int] = None,
        debug: bool = False
) -> Tuple[List[InfrastructureParameters], Dict[int, List[ScheduleParameters]]]:
    infra_schedule_dict = {}
    infra_parameters_list = []
    nb_infras = len(os.listdir(f'{base_directory}/infra/'))
    for infra_id in range(nb_infras):
        if infra_ids is not None and infra_id not in infra_ids:
            continue
        infra, infra_parameters = load_infrastructure(
            base_directory=base_directory,
            infra_id=infra_id
        )
        if debug:
            for agent_id, topo in infra.topo_dict.items():
                print(f"    {agent_id} has {len(get_paths_in_route_dag(topo))} paths in infra {infra_id}")
        infra_parameters_list.append(infra_parameters)
        schedule_dir = f'{base_directory}/infra/{infra_id:03d}/schedule'
        if not os.path.isdir(schedule_dir):
            continue
        nb_schedules = len(os.listdir(schedule_dir))
        for schedule_id in range(nb_schedules):
            schedule, schedule_parameters = load_schedule(
                base_directory=base_directory,
                infra_id=infra_id,
                schedule_id=schedule_id
            )
            if debug:
                for agent_id, topo in schedule.schedule_problem_description.topo_dict.topo_dict.items():
                    print(
                        f"    {agent_id} has {len(get_paths_in_route_dag(topo))} paths in infra {infra_id} / schedule {schedule_id}")
            infra_schedule_dict.setdefault(infra_parameters.infra_id, []).append(schedule_parameters)
    return infra_parameters_list, infra_schedule_dict


def create_experiment_agenda_from_infrastructure_and_schedule_ranges(
        experiment_name: str,
        reschedule_parameters_range: ReScheduleParametersRange,
        infra_parameters_list: List[InfrastructureParameters],
        infra_schedule_dict: Dict[InfrastructureParameters, List[ScheduleParameters]],
        experiments_per_grid_element: int = 1,
):
    list_of_re_schedule_parameters = [ReScheduleParameters(*expanded) for expanded in expand_range_to_parameter_set(reschedule_parameters_range)]
    infra_parameters_dict = {infra_parameters.infra_id: infra_parameters for infra_parameters in infra_parameters_list}
    experiments = []

    # we want arity not to be the same at the same level, therefore, we increment counters
    experiment_id = 0
    grid_id = 0
    for infra_id, list_of_schedule_parameters in infra_schedule_dict.items():
        for schedule_parameters in list_of_schedule_parameters:
            for re_schedule_parameters in list_of_re_schedule_parameters:
                for _ in range(experiments_per_grid_element):
                    experiments.append(
                        ExperimentParameters(
                            experiment_id=experiment_id,

                            schedule_parameters=schedule_parameters,
                            infra_parameters=infra_parameters_dict[infra_id],

                            grid_id=grid_id,
                            earliest_malfunction=re_schedule_parameters.earliest_malfunction,
                            malfunction_duration=re_schedule_parameters.malfunction_duration,
                            malfunction_agend_id=re_schedule_parameters.malfunction_agend_id,
                            weight_route_change=re_schedule_parameters.weight_route_change,
                            weight_lateness_seconds=re_schedule_parameters.weight_lateness_seconds,
                            max_window_size_from_earliest=re_schedule_parameters.max_window_size_from_earliest,
                        )
                    )
                    experiment_id += 1
            grid_id += 1
    return ExperimentAgenda(
        experiment_name=experiment_name,
        experiments=experiments
    )


def expand_range_to_parameter_set(
        parameter_ranges: List[Tuple[int, int, int]],
        debug: bool = False
) -> List[List[int]]:
    """Expand parameter ranges.

    Parameters
    ----------


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
    return full_param_set


def expand_infrastructure_parameter_range(
        infrastructure_parameter_range: InfrastructureParametersRange,
        speed_data: SpeedData,
        grid_mode: bool = True
) -> List[InfrastructureParameters]:
    expanded = expand_range_to_parameter_set(infrastructure_parameter_range)
    return [
        InfrastructureParameters(*([infra_id] + params[:4] + [grid_mode] + params[4:7] + [speed_data, params[7]]))
        for infra_id, params in enumerate(expanded)
    ]


def expand_schedule_parameter_range(schedule_parameter_range: ScheduleParametersRange, infra_id: int) -> List[ScheduleParameters]:
    expanded = expand_range_to_parameter_set(schedule_parameter_range)
    return [
        ScheduleParameters(*([infra_id, schedule_id] + params))
        for schedule_id, params in enumerate(expanded)
    ]


def expand_infrastructure_parameter_range_and_generate_infrastructure(
        infrastructure_parameter_range: InfrastructureParametersRange,
        base_directory: str,
        speed_data: SpeedData,
        grid_mode: bool = True
) -> List[InfrastructureParameters]:
    list_of_infra_parameters = expand_infrastructure_parameter_range(
        infrastructure_parameter_range=infrastructure_parameter_range,
        grid_mode=grid_mode,
        speed_data=speed_data
    )
    for infra_parameters in list_of_infra_parameters:
        if exists_infrastructure(base_directory=base_directory, infra_id=infra_parameters.infra_id):
            rsp_logger.info(f"skipping gen infrastructure for [{infra_parameters.infra_id}] {infra_parameters} -> infrastructure already exists")
            continue
        infra = gen_infrastructure(infra_parameters=infra_parameters)
        save_infrastructure(infrastructure=infra, infrastructure_parameters=infra_parameters, base_directory=base_directory)
    return list_of_infra_parameters


def expand_schedule_parameter_range_and_generate_schedule(
        schedule_parameters_range: ScheduleParametersRange,
        base_directory: str,
        infra_id: int) -> List[ScheduleParameters]:
    list_of_schedule_parameters = expand_schedule_parameter_range(
        schedule_parameter_range=schedule_parameters_range,
        infra_id=infra_id)
    infra, infra_parameters = load_infrastructure(
        base_directory=base_directory,
        infra_id=infra_id
    )
    for schedule_parameters in list_of_schedule_parameters:
        if exists_schedule(base_directory=base_directory, infra_id=infra_id, schedule_id=schedule_parameters.schedule_id):
            rsp_logger.info(f"skipping gen schedule for [{infra_id}/{schedule_parameters.schedule_id}] {infra_parameters} {schedule_parameters} "
                            f"-> schedule already exists")
            continue
        rsp_logger.info(f"gen schedule for [{infra_id}/{schedule_parameters.schedule_id}] {infra_parameters} {schedule_parameters}")
        schedule = gen_schedule(infrastructure=infra, schedule_parameters=schedule_parameters)
        save_schedule(
            schedule=schedule,
            schedule_parameters=schedule_parameters,
            base_directory=base_directory
        )
        _print_stats(schedule.schedule_experiment_result.solver_statistics)
    return list_of_schedule_parameters


def span_n_grid(collected_parameters: List, open_dimensions: List) -> list:
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


def create_env_from_experiment_parameters(params: InfrastructureParameters) -> RailEnv:
    """
    Parameters
    ----------
    params: ExperimentParameters2
        Parameter set that we pass to the constructor of the RailEenv
    Returns
    -------
    RailEnv
        Static environment where no malfunction occurs
    """

    number_of_agents = params.number_of_agents
    width = params.width
    height = params.height
    flatland_seed_value = int(params.flatland_seed_value)
    max_num_cities = params.max_num_cities
    grid_mode = params.grid_mode
    max_rails_between_cities = params.max_rail_between_cities
    max_rails_in_city = params.max_rail_in_city
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
    return env_static


def create_infrastructure_from_rail_env(env: RailEnv, k: int):
    rsp_logger.info("create_infrastructure_from_rail_env")
    agents_paths_dict = {
        # TODO https://gitlab.aicrowd.com/flatland/flatland/issues/302: add method to FLATland to create of k shortest paths for all agents
        i: get_k_shortest_paths(env,
                                agent.initial_position,
                                agent.initial_direction,
                                agent.target,
                                k)
        for i, agent in enumerate(env.agents)
    }
    rsp_logger.info("create_infrastructure_from_rail_env: shortest paths done")
    minimum_travel_time_dict = {agent.handle: int(np.ceil(1 / agent.speed_data['speed']))
                                for agent in env.agents}
    topo_dict = _get_topology_from_agents_path_dict(agents_paths_dict)
    return Infrastructure(
        topo_dict=topo_dict,
        minimum_travel_time_dict=minimum_travel_time_dict,
        max_episode_steps=env._max_episode_steps
    )


def create_schedule_problem_description_from_instructure(
        infrastructure: Infrastructure,
        number_of_shortest_paths_per_agent_schedule: int
) -> ScheduleProblemDescription:
    # deep copy dict
    topo_dict = {
        agent_id: topo.copy()
        for agent_id, topo in infrastructure.topo_dict.items()
    }
    # reduce topo_dict to number_of_shortest_paths_per_agent_schedule
    for _, topo in topo_dict.items():
        paths = get_paths_in_route_dag(topo)
        paths = paths[:number_of_shortest_paths_per_agent_schedule]
        remaining_vertices = {vertex for path in paths for vertex in path}
        topo.remove_nodes_from(set(topo.nodes).difference(remaining_vertices))

    schedule_problem_description = ScheduleProblemDescription(
        route_dag_constraints_dict={
            agent_id: _get_route_dag_constraints_for_scheduling(
                minimum_travel_time=infrastructure.minimum_travel_time_dict[agent_id],
                topo=topo_dict[agent_id],
                source_waypoint=next(get_sources_for_topo(topo_dict[agent_id])),
                latest_arrival=infrastructure.max_episode_steps)
            for agent_id, topo in topo_dict.items()},
        minimum_travel_time_dict=infrastructure.minimum_travel_time_dict,
        topo_dict=topo_dict,
        max_episode_steps=infrastructure.max_episode_steps,
        route_section_penalties={agent_id: {} for agent_id in topo_dict.keys()},
        weight_lateness_seconds=1
    )
    return schedule_problem_description


def save_experiment_agenda_and_hash_to_file(experiment_agenda_folder_name: str, experiment_agenda: ExperimentAgenda):
    """Save experiment agenda and current git hash to the folder with the
    experiments.
    Parameters
    ----------
    experiment_agenda_folder_name: str
        Folder name of experiment where all experiment files and agenda are stored
    experiment_agenda: ExperimentAgenda
        The experiment agenda to save
    """
    file_name = os.path.join(experiment_agenda_folder_name, "experiment_agenda.pkl")
    check_create_folder(experiment_agenda_folder_name)
    with open(file_name, 'wb') as handle:
        pickle.dump(experiment_agenda, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # write current hash to sha.txt to experiment folder
    _write_sha_txt(experiment_agenda_folder_name)


def load_experiment_agenda_from_file(experiment_folder_name: str) -> ExperimentAgenda:
    """Save experiment agenda to the folder with the experiments.
    Parameters
    ----------
    experiment_folder_name: str
        Folder name of experiment where all experiment files and agenda are stored
    """
    file_name = os.path.join(experiment_folder_name, "experiment_agenda.pkl")
    with open(file_name, 'rb') as handle:
        file_data: ExperimentAgenda = pickle.load(handle)
        return file_data


def save_parameter_ranges_and_speed_data(experiment_agenda_folder_name: str, parameter_ranges_and_speed_data: ParameterRangesAndSpeedData):
    """
    Save experiment parameters and speed data to allow for easier modification after reloading
    Parameters
    ----------
    experiment_agenda_folder_name
        Folder to store parameter ranges and speed data
    parameter_ranges_and_speed_data
        Data to store

    Returns
    -------

    """
    if parameter_ranges_and_speed_data is None:
        return

    file_name = os.path.join(experiment_agenda_folder_name, "parameter_ranges_and_speed_data.pkl")
    check_create_folder(experiment_agenda_folder_name)
    with open(file_name, 'wb') as handle:
        pickle.dump(parameter_ranges_and_speed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_experiment_folder_name(experiment_name: str) -> str:
    datetime_string = datetime.datetime.now().strftime("%Y_%m_%dT%H_%M_%S")
    return "{}_{}".format(experiment_name, datetime_string)


def create_experiment_filename(experiment_data_folder_name: str, experiment_id: int) -> str:
    datetime_string = datetime.datetime.now().strftime("%Y_%m_%dT%H_%M_%S")
    filename = "experiment_{:04d}_{}.pkl".format(experiment_id, datetime_string)
    return os.path.join(experiment_data_folder_name, filename)


def save_experiment_results_to_file(experiment_results: ExperimentResults, file_name: str):
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
        experiment_data_folder_name: str,
        experiment_ids: List[int] = None,
        nonify_problem_and_results: bool = False,
) -> List[ExperimentResultsAnalysis]:
    """Load results as DataFrame to do further analysis.
    Parameters
    ----------
    experiment_data_folder_name: str
        Folder name of experiment where all experiment files are stored
    experiment_ids
        List of experiment ids which should be loaded, if None all experiments in experiment_folder are loaded
    nonify_problem_and_results
        in order to save space, set results_* and problem_* fields to None. This may cause not all code to work any more.
        TODO SIM-418 cleanup of this workaround: what would be a good compromise between typing and memory usage?
    Returns
    -------
    DataFrame containing the loaded experiment results
    """

    experiment_results_list = []

    files = os.listdir(experiment_data_folder_name)
    rsp_logger.info(f"loading and expanding experiment results from {experiment_data_folder_name}")
    # nicer printing when tdqm print to stderr and we have logging to stdout shown in to the same console (IDE, separated in files)
    newline_and_flush_stdout_and_stderr()
    for file in tqdm.tqdm([file for file in files if 'agenda' not in file]):
        file_name = os.path.join(experiment_data_folder_name, file)
        if not file_name.endswith(".pkl"):
            continue

        # filter experiments according to defined experiment_ids
        exp_id = get_experiment_id_from_filename(file_name)
        if experiment_ids is not None and exp_id not in experiment_ids:
            continue
        with open(file_name, 'rb') as handle:
            file_data: ExperimentResults = pickle.load(handle)
        experiment_results_list.append(expand_experiment_results_for_analysis(
            file_data,
            nonify_problem_and_results=nonify_problem_and_results))
    # nicer printing when tdqm print to stderr and we have logging to stdout shown in to the same console (IDE, separated in files)
    newline_and_flush_stdout_and_stderr()
    rsp_logger.info(f" -> loading and expanding experiment results from {experiment_data_folder_name} done")
    return experiment_results_list


def load_experiment_result_without_expanding(
        experiment_data_folder_name: str,
        experiment_id: int,
) -> Tuple[ExperimentResults, str]:
    """

    Parameters
    ----------
    experiment_data_folder_name
    experiment_id

    Returns
    -------

    """
    files = os.listdir(experiment_data_folder_name)
    rsp_logger.info(f"loading experiment results from {experiment_data_folder_name}")
    # nicer printing when tdqm print to stderr and we have logging to stdout shown in to the same console (IDE, separated in files)
    for file in tqdm.tqdm([file for file in files if 'agenda' not in file]):
        file_name = os.path.join(experiment_data_folder_name, file)
        if not file_name.endswith(".pkl"):
            continue

        # filter experiments according to defined experiment_ids
        exp_id = get_experiment_id_from_filename(file_name)
        if exp_id != experiment_id:
            continue
        with open(file_name, 'rb') as handle:
            experiment_result: ExperimentResults = pickle.load(handle)
        return experiment_result, file_name


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


# -----------------------
# Agenda tweaking methods
# -----------------------

def tweak_name(
        agenda_null: ExperimentAgenda,
        alt_index: Optional[int],
        experiment_name: str) -> ExperimentAgenda:
    """Produce a new `ExperimentAgenda` under a "tweaked" name.

    Parameters
    ----------
    agenda_null
    alt_index
    experiment_name

    Returns
    -------
    """
    suffix = _make_suffix(alt_index)
    return ExperimentAgenda(
        experiment_name=f"{experiment_name}_{suffix}",
        experiments=agenda_null.experiments
    )


def _make_suffix(alt_index: Optional[int]) -> str:
    """Make suffix for experiment name: either "null" if `alt_index` is `None`,
    else `alt{alt_index:03d}`

    Parameters
    ----------
    alt_index

    Returns
    -------
    """
    suffix = "null"
    if alt_index is not None:
        suffix = f"alt{alt_index:03d}"
    return suffix


def hypothesis_gen_infrastructure_and_schedule_full_agenda(
        parameter_ranges_and_speed_data: ParameterRangesAndSpeedData,
        base_directory: str,
        flatland_seed: int = 12,
        experiments_per_grid_element: int = 1,
        experiment_name: str = "exp_hypothesis_one"):
    """A.1 + A.2.

    Parameters
    ----------
    parameter_ranges_and_speed_data
    base_directory
    flatland_seed
    experiments_per_grid_element
    experiment_name
    """
    rsp_logger.info("GEN INFRASTRUCTURE AND SCHEDULE")

    experiment_agenda: ExperimentAgenda = create_experiment_agenda_from_parameter_ranges_and_speed_data(
        experiment_name=experiment_name,
        parameter_ranges_and_speed_data=parameter_ranges_and_speed_data,
        flatland_seed=flatland_seed,
        experiments_per_grid_element=experiments_per_grid_element)

    for experiment_parameters in experiment_agenda.experiments:
        rsp_logger.info(f"create_schedule_and_malfunction for {experiment_parameters.experiment_id}")

        infra = gen_infrastructure(infra_parameters=experiment_parameters.infra_parameters)

        schedule = gen_schedule(
            infrastructure=infra,
            schedule_parameters=experiment_parameters.schedule_parameters)

        save_infrastructure(
            infrastructure=infra,
            base_directory=base_directory,
            infrastructure_parameters=experiment_parameters.infra_parameters
        )

        save_schedule(
            schedule=schedule,
            schedule_parameters=experiment_parameters.schedule_parameters,
            base_directory=base_directory
        )
