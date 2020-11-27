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
save_experiment_results_to_file
    Save the results of an experiment or a full experiment agenda
load_experiment_results_to_file
    Load the results form an experiment result file
"""
import _pickle
import datetime
import glob
import itertools
import logging
import multiprocessing
import os
import pickle
import platform
import pprint
import shutil
import threading
import time
from functools import partial
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import tqdm as tqdm
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_shortest_paths import get_k_shortest_paths
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from pandas import DataFrame
from rsp.scheduling.asp.asp_helper import _print_stats
from rsp.scheduling.asp_wrapper import asp_reschedule_wrapper
from rsp.scheduling.asp_wrapper import asp_schedule_wrapper
from rsp.scheduling.schedule import Schedule
from rsp.scheduling.scheduling_problem import _get_topology_from_agents_path_dict
from rsp.scheduling.scheduling_problem import get_paths_in_route_dag
from rsp.scheduling.scheduling_problem import get_sources_for_topo
from rsp.scheduling.scheduling_problem import path_stats
from rsp.scheduling.scheduling_problem import ScheduleProblemDescription
from rsp.scheduling.scheduling_problem import TopoDict
from rsp.step_01_planning.agenda_expansion import expand_infrastructure_parameter_range
from rsp.step_01_planning.agenda_expansion import expand_schedule_parameter_range
from rsp.step_01_planning.experiment_parameters_and_ranges import ExperimentAgenda
from rsp.step_01_planning.experiment_parameters_and_ranges import ExperimentParameters
from rsp.step_01_planning.experiment_parameters_and_ranges import InfrastructureParameters
from rsp.step_01_planning.experiment_parameters_and_ranges import InfrastructureParametersRange
from rsp.step_01_planning.experiment_parameters_and_ranges import ScheduleParameters
from rsp.step_01_planning.experiment_parameters_and_ranges import ScheduleParametersRange
from rsp.step_01_planning.experiment_parameters_and_ranges import SpeedData
from rsp.step_02_setup.data_types import ExperimentMalfunction
from rsp.step_02_setup.data_types import Infrastructure
from rsp.step_02_setup.route_dag_constraints_schedule import _get_route_dag_constraints_for_scheduling
from rsp.step_03_run.experiment_results import ExperimentResults
from rsp.step_03_run.experiment_results import plausibility_check_experiment_results
from rsp.step_03_run.experiment_results_analysis import convert_list_of_experiment_results_analysis_to_data_frame
from rsp.step_03_run.experiment_results_analysis import expand_experiment_results_for_analysis
from rsp.step_03_run.experiment_results_analysis import ExperimentResultsAnalysis
from rsp.step_03_run.experiment_results_analysis import plausibility_check_experiment_results_analysis
from rsp.step_03_run.experiment_results_analysis import temporary_backwards_compatibility_scope
from rsp.step_03_run.experiment_results_analysis_online_unrestricted import convert_list_of_experiment_results_analysis_online_unrestricted_to_data_frame
from rsp.step_03_run.experiment_results_analysis_online_unrestricted import expand_experiment_results_online_unrestricted
from rsp.step_03_run.experiment_results_analysis_online_unrestricted import ExperimentResultsAnalysisOnlineUnrestricted
from rsp.step_03_run.scopers.scoper_offline_delta import scoper_offline_delta_for_all_agents
from rsp.step_03_run.scopers.scoper_offline_delta_weak import scoper_offline_delta_weak_for_all_agents
from rsp.step_03_run.scopers.scoper_offline_fully_restricted import scoper_offline_fully_restricted_for_all_agents
from rsp.step_03_run.scopers.scoper_online_random import scoper_online_random_for_all_agents
from rsp.step_03_run.scopers.scoper_online_route_restricted import scoper_online_route_restricted_for_all_agents
from rsp.step_03_run.scopers.scoper_online_transmission_chains import scoper_online_transmission_chains_for_all_agents
from rsp.step_03_run.scopers.scoper_online_unrestricted import scoper_online_unrestricted_for_all_agents
from rsp.step_04_analysis.detailed_experiment_analysis.route_dag_analysis import visualize_route_dag_constraints_simple_wrapper
from rsp.utils.experiment_env_generators import create_flatland_environment
from rsp.utils.file_utils import check_create_folder
from rsp.utils.file_utils import get_experiment_id_from_filename
from rsp.utils.file_utils import newline_and_flush_stdout_and_stderr
from rsp.utils.global_constants import GLOBAL_CONSTANTS
from rsp.utils.global_constants import GlobalConstants
from rsp.utils.global_data_configuration import BASELINE_DATA_FOLDER
from rsp.utils.psutil_helpers import current_process_stats_human_readable
from rsp.utils.psutil_helpers import virtual_memory_human_readable
from rsp.utils.rename_unpickler import RenameUnpickler
from rsp.utils.rsp_logger import add_file_handler_to_rsp_logger
from rsp.utils.rsp_logger import remove_file_handler_from_rsp_logger
from rsp.utils.rsp_logger import rsp_logger

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


def _pickle_dump(obj: Any, file_name: str, folder: Optional[str] = None):
    file_path = file_name
    if folder is not None:
        file_path = os.path.join(folder, file_name)
        check_create_folder(folder)
    else:
        check_create_folder(os.path.dirname(file_name))
    with open(file_path, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _pickle_load(file_name: str, folder: Optional[str] = None):
    file_path = file_name
    if folder is not None:
        file_path = os.path.join(folder, file_name)
    with open(file_path, "rb") as handle:
        try:
            return RenameUnpickler(handle).load()
        except _pickle.UnpicklingError as e:
            rsp_logger.error(f"Failed unpickling {file_path}")
            rsp_logger.error(e, exc_info=True)
            raise e


def save_schedule(
    schedule: Schedule, schedule_parameters: ScheduleParameters, base_directory: str,
):
    """Persist `Schedule` and `ScheduleParameters` to a file.

    Parameters
    ----------
    schedule_parameters
    schedule
    base_directory
    """
    folder = os.path.join(
        base_directory,
        EXPERIMENT_INFRA_SUBDIRECTORY_NAME,
        f"{schedule_parameters.infra_id:03d}",
        EXPERIMENT_SCHEDULE_SUBDIRECTORY_NAME,
        f"{schedule_parameters.schedule_id:03d}",
    )
    _pickle_dump(obj=schedule, folder=folder, file_name="schedule.pkl")
    _pickle_dump(obj=schedule_parameters, folder=folder, file_name="schedule_parameters.pkl")


def save_infrastructure(infrastructure: Infrastructure, infrastructure_parameters: InfrastructureParameters, base_directory: str):
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
    file_name = os.path.join(
        base_directory, EXPERIMENT_INFRA_SUBDIRECTORY_NAME, f"{infra_id:03d}", EXPERIMENT_SCHEDULE_SUBDIRECTORY_NAME, f"{schedule_id:03d}", f"schedule.pkl"
    )
    return os.path.isfile(file_name)


def exists_infrastructure(base_directory: str, infra_id: int) -> bool:
    """Does a persisted `Infrastructure` exist?"""
    file_name = os.path.join(base_directory, EXPERIMENT_INFRA_SUBDIRECTORY_NAME, f"{infra_id:03d}", "infrastructure.pkl")
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
    folder = os.path.join(base_directory, EXPERIMENT_INFRA_SUBDIRECTORY_NAME, f"{infra_id:03d}", EXPERIMENT_SCHEDULE_SUBDIRECTORY_NAME, f"{schedule_id:03d}")
    schedule = _pickle_load(folder=folder, file_name="schedule.pkl")
    schedule_parameters = _pickle_load(folder=folder, file_name="schedule_parameters.pkl")
    return schedule, schedule_parameters


def run_experiment_in_memory(
    schedule: Schedule,
    experiment_parameters: ExperimentParameters,
    infrastructure_topo_dict: TopoDict,
    # TODO we should use logging debug levels instead
    debug: bool = False,
    online_unrestricted_only: bool = False,
) -> ExperimentResults:
    """A.2 + B Runs the main part of the experiment: re-scheduling full and
    delta perfect/naive.

    Parameters
    ----------
    schedule
        operational schedule that where malfunction happened
    experiment_parameters
        hierarchical experiment parameters
    infrastructure_topo_dict
        the "full" topology for each agent
    debug
        debug logging

    Returns
    -------
    ExperimentResults
    """
    rsp_logger.info(f"run_experiment_in_memory for  {experiment_parameters.experiment_id} with GLOBAL_CONSTANTS={GLOBAL_CONSTANTS._constants}")
    rsp_logger.info(f"1. gen malfunction for  {experiment_parameters.experiment_id}")
    schedule_problem, schedule_result = schedule
    schedule_trainruns: TrainrunDict = schedule_result.trainruns_dict

    # --------------------------------------------------------------------------------------
    # A.2 Determine malfunction (deterministically from experiment parameters)
    # --------------------------------------------------------------------------------------
    experiment_malfunction = gen_malfunction(
        earliest_malfunction=experiment_parameters.re_schedule_parameters.earliest_malfunction,
        malfunction_duration=experiment_parameters.re_schedule_parameters.malfunction_duration,
        malfunction_agent_id=experiment_parameters.re_schedule_parameters.malfunction_agent_id,
        schedule_trainruns=schedule.schedule_experiment_result.trainruns_dict,
    )
    malfunction_agent_trainrun = schedule_trainruns[experiment_malfunction.agent_id]
    rsp_logger.info(f"{experiment_malfunction} for scheduled start {malfunction_agent_trainrun[0]} and arrival {malfunction_agent_trainrun[-1]}")

    rescheduling_topo_dict = _make_restricted_topo(
        infrastructure_topo_dict=infrastructure_topo_dict,
        number_of_shortest_paths=experiment_parameters.re_schedule_parameters.number_of_shortest_paths_per_agent,
    )

    # TODO SIM-774 streamline 5 stages according to overleaf; introduce planning stage; split experiments.py!!!
    # --------------------------------------------------------------------------------------
    # B.1. Re-schedule Full
    # --------------------------------------------------------------------------------------
    rsp_logger.info("2. reschedule full")
    # clone topos since propagation will modify them
    online_unrestricted_topo_dict = {agent_id: topo.copy() for agent_id, topo in rescheduling_topo_dict.items()}
    problem_online_unrestricted: ScheduleProblemDescription = scoper_online_unrestricted_for_all_agents(
        malfunction=experiment_malfunction,
        schedule_trainruns=schedule_trainruns,
        minimum_travel_time_dict=schedule_problem.minimum_travel_time_dict,
        latest_arrival=schedule_problem.max_episode_steps + experiment_malfunction.malfunction_duration,
        max_window_size_from_earliest=experiment_parameters.re_schedule_parameters.max_window_size_from_earliest,
        topo_dict_=online_unrestricted_topo_dict,
        weight_route_change=experiment_parameters.re_schedule_parameters.weight_route_change,
        weight_lateness_seconds=experiment_parameters.re_schedule_parameters.weight_lateness_seconds,
    )

    results_online_unrestricted = asp_reschedule_wrapper(
        reschedule_problem_description=problem_online_unrestricted,
        schedule=schedule_trainruns,
        debug=debug,
        asp_seed_value=experiment_parameters.schedule_parameters.asp_seed_value,
    )

    online_unrestricted_trainruns = results_online_unrestricted.trainruns_dict

    costs_ = results_online_unrestricted.solver_statistics["summary"]["costs"][0]
    rsp_logger.info(f" full re-schedule has costs {costs_}")

    if online_unrestricted_only:
        return ExperimentResults(
            experiment_parameters=experiment_parameters,
            malfunction=experiment_malfunction,
            problem_schedule=schedule_problem,
            problem_online_unrestricted=problem_online_unrestricted,
            problem_offline_delta=None,
            problem_offline_delta_weak=None,
            problem_offline_fully_restricted=None,
            problem_online_route_restricted=None,
            problem_online_transmission_chains_fully_restricted=None,
            problem_online_transmission_chains_route_restricted=None,
            results_schedule=schedule_result,
            results_online_unrestricted=results_online_unrestricted,
            results_offline_delta=None,
            results_offline_delta_weak=None,
            results_offline_fully_restricted=None,
            results_online_route_restricted=None,
            results_online_transmission_chains_fully_restricted=None,
            results_online_transmission_chains_route_restricted=None,
            predicted_changed_agents_online_transmission_chains_fully_restricted=None,
            predicted_changed_agents_online_transmission_chains_route_restricted=None,
            **{f"problem_online_random_{i}": None for i in range(GLOBAL_CONSTANTS.NB_RANDOM)},
            **{f"results_online_random_{i}": None for i in range(GLOBAL_CONSTANTS.NB_RANDOM)},
            **{f"predicted_changed_agents_online_random_{i}": None for i in range(GLOBAL_CONSTANTS.NB_RANDOM)},
        )

    # --------------------------------------------------------------------------------------
    # B.2.a Lower bound: Re-Schedule Delta Perfect
    # --------------------------------------------------------------------------------------
    rsp_logger.info("3a. reschedule delta perfect (lower bound)")
    # clone topos since propagation will modify them
    offline_delta_topo_dict = {agent_id: topo.copy() for agent_id, topo in rescheduling_topo_dict.items()}
    problem_offline_delta = scoper_offline_delta_for_all_agents(
        online_unrestricted_trainrun_dict=online_unrestricted_trainruns,
        malfunction=experiment_malfunction,
        max_episode_steps=schedule_problem.max_episode_steps + experiment_malfunction.malfunction_duration,
        offline_delta_topo_dict_=offline_delta_topo_dict,
        schedule_trainrun_dict=schedule_trainruns,
        minimum_travel_time_dict=schedule_problem.minimum_travel_time_dict,
        max_window_size_from_earliest=experiment_parameters.re_schedule_parameters.max_window_size_from_earliest,
        weight_route_change=experiment_parameters.re_schedule_parameters.weight_route_change,
        weight_lateness_seconds=experiment_parameters.re_schedule_parameters.weight_lateness_seconds,
    )

    results_offline_delta = asp_reschedule_wrapper(
        reschedule_problem_description=problem_offline_delta,
        schedule=schedule_trainruns,
        debug=debug,
        asp_seed_value=experiment_parameters.schedule_parameters.asp_seed_value,
    )

    # --------------------------------------------------------------------------------------
    # B.2.a Above Lower bound: Re-Schedule Delta Weak
    # --------------------------------------------------------------------------------------

    rsp_logger.info("3a. reschedule delta Weak (above lower bound)")
    # clone topos since propagation will modify them
    offline_delta_weak_topo_dict = {agent_id: topo.copy() for agent_id, topo in rescheduling_topo_dict.items()}
    problem_offline_delta_weak = scoper_offline_delta_weak_for_all_agents(
        online_unrestricted_trainrun_dict=online_unrestricted_trainruns,
        online_unrestricted_problem=problem_online_unrestricted,
        malfunction=experiment_malfunction,
        latest_arrival=schedule_problem.max_episode_steps + experiment_malfunction.malfunction_duration,
        topo_dict_=offline_delta_weak_topo_dict,
        schedule_trainrun_dict=schedule_trainruns,
        minimum_travel_time_dict=schedule_problem.minimum_travel_time_dict,
        max_window_size_from_earliest=experiment_parameters.re_schedule_parameters.max_window_size_from_earliest,
        weight_route_change=experiment_parameters.re_schedule_parameters.weight_route_change,
        weight_lateness_seconds=experiment_parameters.re_schedule_parameters.weight_lateness_seconds,
    )

    results_offline_delta_weak = asp_reschedule_wrapper(
        reschedule_problem_description=problem_offline_delta_weak,
        schedule=schedule_trainruns,
        debug=debug,
        asp_seed_value=experiment_parameters.schedule_parameters.asp_seed_value,
    )

    # --------------------------------------------------------------------------------------
    # B.2.b Lower bound: Re-Schedule Delta trivially_perfect
    # --------------------------------------------------------------------------------------
    rsp_logger.info("3b. reschedule delta trivially_perfect (lower bound)")
    # clone topos since propagation will modify them
    delta_trivially_perfect_reschedule_topo_dict = {agent_id: topo.copy() for agent_id, topo in rescheduling_topo_dict.items()}
    problem_offline_fully_restricted = scoper_offline_fully_restricted_for_all_agents(
        online_unrestricted_trainrun_dict=online_unrestricted_trainruns,
        malfunction=experiment_malfunction,
        max_episode_steps=schedule_problem.max_episode_steps + experiment_malfunction.malfunction_duration,
        offline_fully_restricted_topo_dict_=delta_trivially_perfect_reschedule_topo_dict,
        schedule_trainrun_dict=schedule_trainruns,
        minimum_travel_time_dict=schedule_problem.minimum_travel_time_dict,
        max_window_size_from_earliest=experiment_parameters.re_schedule_parameters.max_window_size_from_earliest,
        weight_route_change=experiment_parameters.re_schedule_parameters.weight_route_change,
        weight_lateness_seconds=experiment_parameters.re_schedule_parameters.weight_lateness_seconds,
    )

    results_offline_fully_restricted = asp_reschedule_wrapper(
        reschedule_problem_description=problem_offline_fully_restricted,
        schedule=schedule_trainruns,
        debug=debug,
        asp_seed_value=experiment_parameters.schedule_parameters.asp_seed_value,
    )

    # --------------------------------------------------------------------------------------
    # B.2.c Some restriction
    # --------------------------------------------------------------------------------------
    rsp_logger.info("4. reschedule no rerouting")
    # clone topos since propagation will modify them
    delta_no_rerouting_reschedule_topo_dict = {agent_id: topo.copy() for agent_id, topo in rescheduling_topo_dict.items()}
    problem_online_route_restricted = scoper_online_route_restricted_for_all_agents(
        online_unrestricted_trainrun_dict=online_unrestricted_trainruns,
        online_unrestricted_problem=problem_online_unrestricted,
        malfunction=experiment_malfunction,
        max_episode_steps=schedule_problem.max_episode_steps + experiment_malfunction.malfunction_duration,
        # pytorch convention for in-place operations: postfixed with underscore.
        topo_dict_=delta_no_rerouting_reschedule_topo_dict,
        schedule_trainrun_dict=schedule_trainruns,
        minimum_travel_time_dict=schedule_problem.minimum_travel_time_dict,
        max_window_size_from_earliest=experiment_parameters.re_schedule_parameters.max_window_size_from_earliest,
        weight_route_change=experiment_parameters.re_schedule_parameters.weight_route_change,
        weight_lateness_seconds=experiment_parameters.re_schedule_parameters.weight_lateness_seconds,
    )

    results_online_route_restricted = asp_reschedule_wrapper(
        reschedule_problem_description=problem_online_route_restricted,
        schedule=schedule_trainruns,
        debug=debug,
        asp_seed_value=experiment_parameters.schedule_parameters.asp_seed_value,
    )
    # --------------------------------------------------------------------------------------
    # B.2.d Upper bound: online predictor
    # --------------------------------------------------------------------------------------
    rsp_logger.info("5a. reschedule delta online transmission chains: upper bound")
    # clone topos since propagation will modify them
    online_transmission_chains_fully_restricted_topo_dict = {agent_id: topo.copy() for agent_id, topo in rescheduling_topo_dict.items()}
    (
        problem_online_transmission_chains_fully_restricted,
        predicted_changed_agents_online_transmission_chains_fully_restricted_predicted,
    ) = scoper_online_transmission_chains_for_all_agents(
        online_unrestricted_problem=problem_online_unrestricted,
        malfunction=experiment_malfunction,
        latest_arrival=schedule_problem.max_episode_steps + experiment_malfunction.malfunction_duration,
        # pytorch convention for in-place operations: postfixed with underscore.
        delta_online_topo_dict_to_=online_transmission_chains_fully_restricted_topo_dict,
        schedule_trainrun_dict=schedule_trainruns,
        minimum_travel_time_dict=schedule_problem.minimum_travel_time_dict,
        max_window_size_from_earliest=experiment_parameters.re_schedule_parameters.max_window_size_from_earliest,
        weight_route_change=experiment_parameters.re_schedule_parameters.weight_route_change,
        weight_lateness_seconds=experiment_parameters.re_schedule_parameters.weight_lateness_seconds,
        time_flexibility=False,
    )

    results_online_transmission_chains_fully_restricted = asp_reschedule_wrapper(
        reschedule_problem_description=problem_online_transmission_chains_fully_restricted,
        schedule=schedule_trainruns,
        debug=debug,
        asp_seed_value=experiment_parameters.schedule_parameters.asp_seed_value,
    )

    # --------------------------------------------------------------------------------------
    # B.2.d Upper bound: online_no_time_flexibility predictor
    # --------------------------------------------------------------------------------------
    rsp_logger.info("5b. reschedule delta online_no_time_flexibility transmission chains: upper bound")
    # clone topos since propagation will modify them
    online_transmission_chains_route_restricted_topo_dict = {agent_id: topo.copy() for agent_id, topo in rescheduling_topo_dict.items()}
    (
        problem_online_transmission_chains_route_restricted,
        predicted_changed_agents_online_transmission_chains_route_restricted_predicted,
    ) = scoper_online_transmission_chains_for_all_agents(
        online_unrestricted_problem=problem_online_unrestricted,
        malfunction=experiment_malfunction,
        latest_arrival=schedule_problem.max_episode_steps + experiment_malfunction.malfunction_duration,
        # pytorch convention for in-place operations: postfixed with underscore.
        delta_online_topo_dict_to_=online_transmission_chains_route_restricted_topo_dict,
        schedule_trainrun_dict=schedule_trainruns,
        minimum_travel_time_dict=schedule_problem.minimum_travel_time_dict,
        max_window_size_from_earliest=experiment_parameters.re_schedule_parameters.max_window_size_from_earliest,
        weight_route_change=experiment_parameters.re_schedule_parameters.weight_route_change,
        weight_lateness_seconds=experiment_parameters.re_schedule_parameters.weight_lateness_seconds,
        time_flexibility=True,
    )

    results_online_transmission_chains_route_restricted = asp_reschedule_wrapper(
        reschedule_problem_description=problem_online_transmission_chains_route_restricted,
        schedule=schedule_trainruns,
        debug=debug,
        asp_seed_value=experiment_parameters.schedule_parameters.asp_seed_value,
    )

    # --------------------------------------------------------------------------------------
    # B.2.e Sanity check: random predictor
    # if that also reduces solution time, our problem is not hard enough, showing the problem is not trivial
    # --------------------------------------------------------------------------------------
    rsp_logger.info("6. reschedule delta random naive: upper bound")
    randoms = []
    for _ in range(GLOBAL_CONSTANTS.NB_RANDOM):
        # clone topos since propagation will modify them
        online_random_topo_dict = {agent_id: topo.copy() for agent_id, topo in rescheduling_topo_dict.items()}
        problem_online_random, predicted_changed_agents_online_random = scoper_online_random_for_all_agents(
            online_unrestricted_problem=problem_online_unrestricted,
            malfunction=experiment_malfunction,
            # TODO document? will it be visible in ground times?
            latest_arrival=(schedule_problem.max_episode_steps + experiment_malfunction.malfunction_duration),
            # pytorch convention for in-place operations: postfixed with underscore.
            delta_random_topo_dict_to_=online_random_topo_dict,
            schedule_trainrun_dict=schedule_trainruns,
            minimum_travel_time_dict=schedule_problem.minimum_travel_time_dict,
            # TODO document? will it be visible in ground times?
            max_window_size_from_earliest=experiment_parameters.re_schedule_parameters.max_window_size_from_earliest,
            weight_route_change=experiment_parameters.re_schedule_parameters.weight_route_change,
            weight_lateness_seconds=experiment_parameters.re_schedule_parameters.weight_lateness_seconds,
            changed_running_agents_online=len(predicted_changed_agents_online_transmission_chains_fully_restricted_predicted),
        )
        results_online_random = asp_reschedule_wrapper(
            reschedule_problem_description=problem_online_random,
            schedule=schedule_trainruns,
            debug=debug,
            asp_seed_value=experiment_parameters.schedule_parameters.asp_seed_value,
        )
        randoms.append((problem_online_random, results_online_random, predicted_changed_agents_online_random))

    # --------------------------------------------------------------------------------------
    # B.3. Result
    # --------------------------------------------------------------------------------------
    rsp_logger.info("7. gathering results")
    current_results = ExperimentResults(
        experiment_parameters=experiment_parameters,
        malfunction=experiment_malfunction,
        problem_schedule=schedule_problem,
        problem_online_unrestricted=problem_online_unrestricted,
        problem_offline_delta=problem_offline_delta,
        problem_offline_delta_weak=problem_offline_delta_weak,
        problem_offline_fully_restricted=problem_offline_fully_restricted,
        problem_online_route_restricted=problem_online_route_restricted,
        problem_online_transmission_chains_fully_restricted=problem_online_transmission_chains_fully_restricted,
        problem_online_transmission_chains_route_restricted=problem_online_transmission_chains_route_restricted,
        results_schedule=schedule_result,
        results_online_unrestricted=results_online_unrestricted,
        results_offline_delta=results_offline_delta,
        results_offline_delta_weak=results_offline_delta_weak,
        results_offline_fully_restricted=results_offline_fully_restricted,
        results_online_route_restricted=results_online_route_restricted,
        results_online_transmission_chains_fully_restricted=results_online_transmission_chains_fully_restricted,
        results_online_transmission_chains_route_restricted=results_online_transmission_chains_route_restricted,
        predicted_changed_agents_online_transmission_chains_fully_restricted=predicted_changed_agents_online_transmission_chains_fully_restricted_predicted,
        predicted_changed_agents_online_transmission_chains_route_restricted=predicted_changed_agents_online_transmission_chains_route_restricted_predicted,
        **{f"problem_online_random_{i}": randoms[i][0] for i in range(GLOBAL_CONSTANTS.NB_RANDOM)},
        **{f"results_online_random_{i}": randoms[i][1] for i in range(GLOBAL_CONSTANTS.NB_RANDOM)},
        **{f"predicted_changed_agents_online_random_{i}": randoms[i][2] for i in range(GLOBAL_CONSTANTS.NB_RANDOM)},
    )
    rsp_logger.info(f"done re-schedule full and delta naive/perfect for experiment {experiment_parameters.experiment_id}")
    return current_results


def _make_restricted_topo(infrastructure_topo_dict: TopoDict, number_of_shortest_paths: int):
    topo_dict = {agent_id: topo.copy() for agent_id, topo in infrastructure_topo_dict.items()}
    nb_paths_before = []
    nb_paths_after = []
    for _, topo in topo_dict.items():
        paths = get_paths_in_route_dag(topo)
        nodes_to_keep = [node for path in paths[:number_of_shortest_paths] for node in path]
        nodes_to_remove = {node for node in topo.nodes if node not in nodes_to_keep}
        topo.remove_nodes_from(nodes_to_remove)
        nb_paths_before.append(len(paths))
        nb_paths_after.append(len(get_paths_in_route_dag(topo)))
    rsp_logger.info(
        f"make restricted topo for re-scheduling with number_of_shortest_paths{number_of_shortest_paths}: "
        f"{path_stats(nb_paths_before)} -> {path_stats(nb_paths_after)}"
    )
    return topo_dict


def _render_route_dags_from_data(experiment_base_directory: str, experiment_id: int):
    results_before, _ = load_and_expand_experiment_results_from_data_folder(
        experiment_data_folder_name=experiment_base_directory + "/data", experiment_ids=[experiment_id]
    )[0]
    problem_online_unrestricted: ScheduleProblemDescription = results_before.problem_online_unrestricted
    for agent_id in problem_online_unrestricted.route_dag_constraints_dict:
        visualize_route_dag_constraints_simple_wrapper(
            schedule_problem_description=problem_online_unrestricted.schedule_problem_description,
            trainrun_dict=None,
            experiment_malfunction=problem_online_unrestricted.experiment_malfunction,
            agent_id=agent_id,
            file_name=f"reschedule_alt_agent_{agent_id}.pdf",
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


def gen_infrastructure(infra_parameters: InfrastructureParameters) -> Infrastructure:
    """A.1.1 infrastructure generation."""
    rsp_logger.info(f"gen_infrastructure {infra_parameters}")
    infra = create_infrastructure_from_rail_env(
        env=create_env_from_experiment_parameters(infra_parameters), k=infra_parameters.number_of_shortest_paths_per_agent
    )
    rsp_logger.info(f"done gen_infrastructure {infra_parameters}")
    return infra


def gen_schedule(schedule_parameters: ScheduleParameters, infrastructure: Infrastructure, debug: bool = False) -> Schedule:
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
        infrastructure=infrastructure, number_of_shortest_paths_per_agent_schedule=schedule_parameters.number_of_shortest_paths_per_agent_schedule
    )
    if debug:
        for agent_id, topo in schedule_problem.topo_dict.items():
            rsp_logger.info(f"    {agent_id} has {len(get_paths_in_route_dag(topo))} paths in scheduling")
            rsp_logger.info(f"    {agent_id} has {len(get_paths_in_route_dag(infrastructure.topo_dict[agent_id]))} paths in infrastructure")

    schedule_result = asp_schedule_wrapper(schedule_problem_description=schedule_problem, asp_seed_value=schedule_parameters.asp_seed_value, debug=debug)
    rsp_logger.info(f"done gen_schedule {schedule_parameters}")
    return Schedule(schedule_problem_description=schedule_problem, schedule_experiment_result=schedule_result)


def gen_malfunction(
    earliest_malfunction: int, malfunction_duration: int, schedule_trainruns: TrainrunDict, malfunction_agent_id: int,
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
    # The malfunction must happen during the scheduled run of the agent, therefore it must happen before the scheduled end!
    malfunction_start = min(
        schedule_trainruns[malfunction_agent_id][0].scheduled_at + earliest_malfunction, schedule_trainruns[malfunction_agent_id][-1].scheduled_at - 1
    )
    malfunction = ExperimentMalfunction(time_step=malfunction_start, malfunction_duration=malfunction_duration, agent_id=malfunction_agent_id)
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
    out_file = os.path.join(folder_name, "sha.txt")
    rsp_logger.info(f"writing {sha} to {out_file}")
    with open(out_file, "w") as out:
        out.write(sha)


def run_experiment_from_to_file(
    experiment_parameters: ExperimentParameters,
    experiment_base_directory: str,
    experiment_output_directory: str,
    global_constants: GlobalConstants,
    csv_only: bool = False,
    debug: bool = False,
    online_unrestricted_only: bool = False,
    raise_exceptions: bool = False,
):
    """A.2 + B. Run and save one experiment from experiment parameters.
    Parameters
    ----------
    experiment_base_directory
        base for infrastructure and schedules
    experiment_parameters
        contains reference to infrastructure and schedules
    experiment_output_directory
    debug
    """
    rsp_logger.info(f"run_experiment_from_to_file with {global_constants}")
    # N.B. this works since we ensure that every experiment runs in its own process!
    GLOBAL_CONSTANTS.set_defaults(constants=global_constants)

    experiment_data_directory = f"{experiment_output_directory}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}"

    # add logging file handler in this thread
    stdout_log_file = os.path.join(experiment_data_directory, f"log.txt")
    stderr_log_file = os.path.join(experiment_data_directory, f"err.txt")
    stdout_log_fh = add_file_handler_to_rsp_logger(stdout_log_file, logging.INFO)
    stderr_log_fh = add_file_handler_to_rsp_logger(stderr_log_file, logging.ERROR)

    rsp_logger.info(f"start experiment {experiment_parameters.experiment_id}")
    try:

        check_create_folder(experiment_data_directory)

        start_datetime_str = datetime.datetime.now().strftime("%H:%M:%S")
        rsp_logger.info("Running experiment {} under pid {} at {}".format(experiment_parameters.experiment_id, os.getpid(), start_datetime_str))
        start_time = time.time()

        rsp_logger.info("*** experiment parameters for experiment {}. {}".format(experiment_parameters.experiment_id, _pp.pformat(experiment_parameters)))

        if experiment_base_directory is None or not exists_schedule(
            base_directory=experiment_base_directory,
            infra_id=experiment_parameters.infra_parameters.infra_id,
            schedule_id=experiment_parameters.schedule_parameters.schedule_id,
        ):
            rsp_logger.warn(f"Could not find schedule for {experiment_parameters.experiment_id} in {experiment_base_directory}")
            return

        rsp_logger.info(f"load_schedule for {experiment_parameters.experiment_id}")
        schedule, schedule_parameters = load_schedule(
            base_directory=f"{experiment_base_directory}",
            infra_id=experiment_parameters.infra_parameters.infra_id,
            schedule_id=experiment_parameters.schedule_parameters.schedule_id,
        )
        infrastructure, _ = load_infrastructure(base_directory=f"{experiment_base_directory}", infra_id=experiment_parameters.infra_parameters.infra_id)

        if debug:
            _render_route_dags_from_data(experiment_base_directory=experiment_output_directory, experiment_id=experiment_parameters.experiment_id)

        # B2: full and delta perfect re-scheduling
        experiment_results: ExperimentResults = run_experiment_in_memory(
            schedule=schedule,
            experiment_parameters=experiment_parameters,
            infrastructure_topo_dict=infrastructure.topo_dict,
            debug=debug,
            online_unrestricted_only=online_unrestricted_only,
        )

        if experiment_results is None:
            print(f"No malfunction for experiment {experiment_parameters.experiment_id}")
            return []

        elapsed_time = time.time() - start_time
        end_datetime_str = datetime.datetime.now().strftime("%H:%M:%S")
        if not online_unrestricted_only:
            s = ("Running experiment {}: took {:5.3f}s ({}--{}) (sched:  {} / re-sched full:  {} / re-sched delta perfect:  {} / ").format(
                experiment_parameters.experiment_id,
                elapsed_time,
                start_datetime_str,
                end_datetime_str,
                _get_asp_solver_details_from_statistics(elapsed_time=elapsed_time, statistics=experiment_results.results_schedule.solver_statistics),
                _get_asp_solver_details_from_statistics(elapsed_time=elapsed_time, statistics=experiment_results.results_online_unrestricted.solver_statistics),
                _get_asp_solver_details_from_statistics(elapsed_time=elapsed_time, statistics=experiment_results.results_offline_delta.solver_statistics),
            )
            solver_time_schedule = experiment_results.results_schedule.solver_statistics["summary"]["times"]["total"]
            solver_statistics_times_total_online_unrestricted = experiment_results.results_online_unrestricted.solver_statistics["summary"]["times"]["total"]
            solver_time_offline_delta = experiment_results.results_offline_delta.solver_statistics["summary"]["times"]["total"]
            elapsed_overhead_time = elapsed_time - solver_time_schedule - solver_statistics_times_total_online_unrestricted - solver_time_offline_delta
            s += "remaining: {:5.3f}s = {:5.2f}%)  in thread {}".format(
                elapsed_overhead_time, elapsed_overhead_time / elapsed_time * 100, threading.get_ident()
            )
            rsp_logger.info(s)

        rsp_logger.info(virtual_memory_human_readable())
        rsp_logger.info(current_process_stats_human_readable())

        # fail fast!
        if not online_unrestricted_only:
            plausibility_check_experiment_results(experiment_results=experiment_results)
            plausibility_check_experiment_results_analysis(
                experiment_results_analysis=expand_experiment_results_for_analysis(experiment_results=experiment_results)
            )
        filename = create_experiment_filename(experiment_data_directory, experiment_parameters.experiment_id)
        save_experiment_results_to_file(
            experiment_results=experiment_results, file_name=filename, csv_only=csv_only, online_unrestricted_only=online_unrestricted_only
        )

        return os.getpid()
    except Exception as e:
        rsp_logger.error(e, exc_info=True)
        rsp_logger.error(
            f"XXX failed experiment_id={experiment_parameters.experiment_id} in {experiment_data_directory}, "
            f"infra_id={experiment_parameters.infra_parameters.infra_id}, "
            f"schedule_id={experiment_parameters.schedule_parameters.schedule_id}"
        )
        if raise_exceptions:
            raise e
        return os.getpid()
    finally:
        remove_file_handler_from_rsp_logger(stdout_log_fh)
        remove_file_handler_from_rsp_logger(stderr_log_fh)
        rsp_logger.info(f"end experiment {experiment_parameters.experiment_id}")


def load_and_filter_experiment_results_analysis(
    experiment_base_directory: str = BASELINE_DATA_FOLDER,
    experiments_of_interest: List[int] = None,
    from_cache: bool = False,
    from_individual_csv: bool = True,
    local_filter_experiment_results_analysis_data_frame: Callable[[DataFrame], DataFrame] = None,
) -> DataFrame:
    if from_cache:
        experiment_data_filtered = pd.read_csv(f"{experiment_base_directory}.csv")
    else:
        if from_individual_csv:
            experiment_data: pd.DataFrame = load_data_from_individual_csv_in_data_folder(
                experiment_data_folder_name=f"{experiment_base_directory}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}", experiment_ids=experiments_of_interest
            )
        else:
            _, experiment_results_analysis_list = load_and_expand_experiment_results_from_data_folder(
                experiment_data_folder_name=f"{experiment_base_directory}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}", experiment_ids=experiments_of_interest,
            )
            experiment_data: pd.DataFrame = convert_list_of_experiment_results_analysis_to_data_frame(experiment_results_analysis_list)

        if local_filter_experiment_results_analysis_data_frame is not None:
            experiment_data_filtered = local_filter_experiment_results_analysis_data_frame(experiment_data)
            print(f"removed {len(experiment_data) - len(experiment_data_filtered)}/{len(experiment_data)} rows")
        else:
            experiment_data_filtered = experiment_data
        experiment_data_filtered.to_csv(f"{experiment_base_directory}.csv")
    return experiment_data_filtered


def run_experiment_agenda(
    experiment_base_directory: str,
    experiment_agenda: ExperimentAgenda = None,
    experiment_output_directory: str = None,
    filter_experiment_agenda: Callable[[ExperimentParameters], bool] = None,
    # take only half of avilable cpus so the machine stays responsive
    run_experiments_parallel: int = AVAILABLE_CPUS // 2,
    csv_only: bool = False,
    online_unrestricted_only: bool = False,
) -> str:
    """Run A.2 + B. Presupposes infras and schedules
    Parameters
    ----------
    experiment_output_directory
        if passed, agenda in this directory must be the same as the one passed
    experiment_agenda: ExperimentAgenda
        Full list of experiments
    experiment_base_directory: str
        where are schedules etc?
    filter_experiment_agenda
        filter which experiment to run
    run_experiments_parallel: in
        run experiments in parallel
    run_analysis
    online_unrestricted_only
    csv_only

    Returns
    -------
    Returns the name of the experiment base and data folders
    """
    assert (
        experiment_agenda is not None or experiment_output_directory is not None
    ), "Either experiment_agenda or experiment_output_directory must be specified."
    if experiment_output_directory is None:
        experiment_output_directory = f"{experiment_base_directory}/" + create_experiment_folder_name(experiment_agenda.experiment_name)
        check_create_folder(experiment_output_directory)

    experiment_data_directory = f"{experiment_output_directory}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}"

    if exists_experiment_agenda(experiment_output_directory):

        rsp_logger.info(f"============================================================================================================")
        rsp_logger.info(f"loading agenda <- {experiment_output_directory}")
        rsp_logger.info(f"============================================================================================================")

        experiment_agenda_from_file = load_experiment_agenda_from_file(experiment_folder_name=experiment_output_directory)

        if experiment_agenda is not None:
            assert experiment_agenda_from_file == experiment_agenda
        experiment_agenda = experiment_agenda_from_file
    elif experiment_agenda is not None:
        save_experiment_agenda_and_hash_to_file(output_base_folder=experiment_output_directory, experiment_agenda=experiment_agenda)
    else:
        raise Exception("Either experiment_agenda or experiment_output_directory with experiment_agenda.pkl must be passed.")
    assert experiment_agenda is not None

    check_create_folder(experiment_data_directory)

    if run_experiments_parallel <= 1:
        rsp_logger.warn("Using only one process in pool might cause pool to stall sometimes. Use more than one process in pool?")

    # tee stdout to log file
    stdout_log_file = os.path.join(experiment_data_directory, "log.txt")
    stderr_log_file = os.path.join(experiment_data_directory, "err.txt")
    stdout_log_fh = add_file_handler_to_rsp_logger(stdout_log_file, logging.INFO)
    stderr_log_fh = add_file_handler_to_rsp_logger(stderr_log_file, logging.ERROR)
    try:

        if filter_experiment_agenda is not None:
            len_before_filtering = len(experiment_agenda.experiments)
            rsp_logger.info(f"============================================================================================================")
            rsp_logger.info(f"filtering agenda by passed filter {filter_experiment_agenda} <- {experiment_output_directory}")
            rsp_logger.info(f"============================================================================================================")
            experiments_filtered = filter(filter_experiment_agenda, experiment_agenda.experiments)
            experiment_agenda = ExperimentAgenda(
                experiment_name=experiment_agenda.experiment_name, global_constants=experiment_agenda.global_constants, experiments=list(experiments_filtered)
            )
            rsp_logger.info(
                f"after applying filter, there are {len(experiment_agenda.experiments)} experiments out of {len_before_filtering}: \n" + str(experiment_agenda)
            )

        rsp_logger.info(f"============================================================================================================")
        rsp_logger.info(f"filtering agenda by experiments not run yet <- {experiment_output_directory}")
        rsp_logger.info(f"============================================================================================================")
        len_before_filtering = len(experiment_agenda.experiments)
        experiment_agenda = ExperimentAgenda(
            experiment_name=experiment_agenda.experiments,
            experiments=[
                experiment
                for experiment in experiment_agenda.experiments
                if load_experiments_results(experiment_data_folder_name=experiment_data_directory, experiment_id=experiment.experiment_id) is None
            ],
            global_constants=experiment_agenda.global_constants,
        )
        rsp_logger.info(
            f"after filtering out experiments already run from {experiment_output_directory}, "
            f"there are {len(experiment_agenda.experiments)} experiments out of {len_before_filtering}: \n" + str(experiment_agenda)
        )

        rsp_logger.info(f"============================================================================================================")
        rsp_logger.info(f"RUNNING agenda -> {experiment_data_directory} ({len(experiment_agenda.experiments)} experiments)")
        rsp_logger.info(f"============================================================================================================")
        rsp_logger.info(f"experiment_agenda.global_constants={experiment_agenda.global_constants}")
        rsp_logger.info(f"============================================================================================================")

        # use processes in pool only once because of https://github.com/potassco/clingo/issues/203
        # https://stackoverflow.com/questions/38294608/python-multiprocessing-pool-new-process-for-each-variable
        # N.B. even with parallelization degree 1, we want to run each experiment in a new process
        #      in order to get around https://github.com/potassco/clingo/issues/203
        pool = multiprocessing.Pool(processes=run_experiments_parallel, maxtasksperchild=1)
        rsp_logger.info(f"pool size {pool._processes} / {multiprocessing.cpu_count()} ({os.cpu_count()}) cpus on {platform.node()}")
        # nicer printing when tdqm print to stderr and we have logging to stdout shown in to the same console (IDE, separated in files)
        newline_and_flush_stdout_and_stderr()

        run_and_save_one_experiment_partial = partial(
            run_experiment_from_to_file,
            experiment_base_directory=experiment_base_directory,
            experiment_output_directory=experiment_output_directory,
            csv_only=csv_only,
            global_constants=experiment_agenda.global_constants,
            online_unrestricted_only=online_unrestricted_only,
        )

        for pid_done in tqdm.tqdm(
            pool.imap_unordered(run_and_save_one_experiment_partial, experiment_agenda.experiments), total=len(experiment_agenda.experiments)
        ):
            # unsafe use of inner API
            procs = [f"{str(proc)}={proc.pid}" for proc in pool._pool]
            rsp_logger.info(f"pid {pid_done} done. Pool: {procs}")

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


def create_infrastructure_and_schedule_from_ranges(
    infrastructure_parameters_range: InfrastructureParametersRange,
    schedule_parameters_range: ScheduleParametersRange,
    base_directory: str,
    speed_data: SpeedData,
    grid_mode: bool = False,
    run_experiments_parallel: int = 5,
) -> List[ScheduleParameters]:

    # expand infrastructure parameters and generate infrastructure
    list_of_infrastructure_parameters = expand_infrastructure_parameter_range_and_generate_infrastructure(
        infrastructure_parameter_range=infrastructure_parameters_range, base_directory=base_directory, speed_data=speed_data, grid_mode=grid_mode
    )

    # expand schedule parameters and get list of those missing
    list_of_schedule_parameters_to_generate: List[ScheduleParameters] = list(
        itertools.chain.from_iterable(
            [
                expand_schedule_parameter_range_and_get_those_not_existing_yet(
                    schedule_parameters_range=schedule_parameters_range, base_directory=base_directory, infra_id=infrastructure_parameters.infra_id
                )
                for infrastructure_parameters in list_of_infrastructure_parameters
            ]
        )
    )

    # generate schedules in parallel
    pool = multiprocessing.Pool(processes=run_experiments_parallel, maxtasksperchild=1)
    gen_and_save_schedule_partial = partial(gen_and_save_schedule, base_directory=base_directory)
    for done in tqdm.tqdm(
        pool.imap_unordered(gen_and_save_schedule_partial, list_of_schedule_parameters_to_generate), total=len(list_of_schedule_parameters_to_generate)
    ):
        rsp_logger.info(f"done: {done}")

    # expand schedule parameters and get full list
    list_of_schedule_parameters: List[ScheduleParameters] = list(
        itertools.chain.from_iterable(
            [
                expand_schedule_parameter_range(schedule_parameter_range=schedule_parameters_range, infra_id=infrastructure_parameters.infra_id)
                for infrastructure_parameters in list_of_infrastructure_parameters
            ]
        )
    )
    return list_of_schedule_parameters


def list_infrastructure_and_schedule_params_from_base_directory(
    base_directory: str, filter_experiment_agenda: Callable[[int, int], bool] = None, debug: bool = False
) -> Tuple[List[InfrastructureParameters], Dict[int, List[Tuple[ScheduleParameters, Schedule]]]]:
    infra_schedule_dict = {}
    infra_parameters_list = []
    nb_infras = len(os.listdir(f"{base_directory}/infra/"))
    for infra_id in range(nb_infras):
        infra, infra_parameters = load_infrastructure(base_directory=base_directory, infra_id=infra_id)
        if debug:
            for agent_id, topo in infra.topo_dict.items():
                print(f"    {agent_id} has {len(get_paths_in_route_dag(topo))} paths in infra {infra_id}")
        infra_parameters_list.append(infra_parameters)
        schedule_dir = f"{base_directory}/infra/{infra_id:03d}/schedule"
        if not os.path.isdir(schedule_dir):
            continue
        schedule_ids = [int(s) for s in os.listdir(schedule_dir)]
        for schedule_id in schedule_ids:
            if filter_experiment_agenda is not None and not filter_experiment_agenda(infra_id, schedule_id):
                continue
            schedule, schedule_parameters = load_schedule(base_directory=base_directory, infra_id=infra_id, schedule_id=schedule_id)
            infra_schedule_dict.setdefault(infra_parameters.infra_id, []).append((schedule_parameters, schedule))
    return infra_parameters_list, infra_schedule_dict


def expand_infrastructure_parameter_range_and_generate_infrastructure(
    infrastructure_parameter_range: InfrastructureParametersRange, base_directory: str, speed_data: SpeedData, grid_mode: bool = True
) -> List[InfrastructureParameters]:
    list_of_infra_parameters = expand_infrastructure_parameter_range(
        infrastructure_parameter_range=infrastructure_parameter_range, grid_mode=grid_mode, speed_data=speed_data
    )
    for infra_parameters in list_of_infra_parameters:
        if exists_infrastructure(base_directory=base_directory, infra_id=infra_parameters.infra_id):
            rsp_logger.info(f"skipping gen infrastructure for [{infra_parameters.infra_id}] {infra_parameters} -> infrastructure already exists")
            _, infra_parameters_from_file = load_infrastructure(base_directory=base_directory, infra_id=infra_parameters.infra_id)
            assert (
                infra_parameters == infra_parameters_from_file
            ), f"infra parameters not the same for  [{infra_parameters.infra_id}]: expected {infra_parameters}, found {infra_parameters_from_file} in file"
            continue
        infra = gen_infrastructure(infra_parameters=infra_parameters)
        save_infrastructure(infrastructure=infra, infrastructure_parameters=infra_parameters, base_directory=base_directory)
    return list_of_infra_parameters


def gen_and_save_schedule(schedule_parameters: ScheduleParameters, base_directory: str):
    infra_id = schedule_parameters.infra_id
    infra, infra_parameters = load_infrastructure(base_directory=base_directory, infra_id=infra_id)
    rsp_logger.info(f"gen schedule for [{infra_id}/{schedule_parameters.schedule_id}] {infra_parameters} {schedule_parameters}")
    schedule = gen_schedule(infrastructure=infra, schedule_parameters=schedule_parameters)
    save_schedule(schedule=schedule, schedule_parameters=schedule_parameters, base_directory=base_directory)
    _print_stats(schedule.schedule_experiment_result.solver_statistics)
    return schedule_parameters


def expand_schedule_parameter_range_and_get_those_not_existing_yet(
    schedule_parameters_range: ScheduleParametersRange, base_directory: str, infra_id: int
) -> List[ScheduleParameters]:
    list_of_schedule_parameters = expand_schedule_parameter_range(schedule_parameter_range=schedule_parameters_range, infra_id=infra_id)
    infra, infra_parameters = load_infrastructure(base_directory=base_directory, infra_id=infra_id)
    list_of_schedule_parameters_to_generate = []
    for schedule_parameters in list_of_schedule_parameters:
        if exists_schedule(base_directory=base_directory, infra_id=infra_id, schedule_id=schedule_parameters.schedule_id):
            rsp_logger.info(
                f"skipping gen schedule for [infra {infra_id}/schedule {schedule_parameters.schedule_id}] {infra_parameters} {schedule_parameters} "
                f"-> schedule already exists"
            )
            _, schedule_parameters_from_file = load_schedule(base_directory=base_directory, infra_id=infra_id, schedule_id=schedule_parameters.schedule_id)
            assert schedule_parameters_from_file == schedule_parameters, (
                f"schedule parameters [infra {infra_id}/schedule {schedule_parameters.schedule_id}] not the same: "
                f"expected  {schedule_parameters}, found {schedule_parameters_from_file} in file"
            )

            continue
        list_of_schedule_parameters_to_generate.append(schedule_parameters)
    return list_of_schedule_parameters_to_generate


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
    env_static = create_flatland_environment(
        number_of_agents=number_of_agents,
        width=width,
        height=height,
        flatland_seed_value=flatland_seed_value,
        max_num_cities=max_num_cities,
        grid_mode=grid_mode,
        max_rails_between_cities=max_rails_between_cities,
        max_rails_in_city=max_rails_in_city,
        speed_data=speed_data,
    )
    return env_static


def create_infrastructure_from_rail_env(env: RailEnv, k: int):
    rsp_logger.info("create_infrastructure_from_rail_env")
    agents_paths_dict = {
        # TODO https://gitlab.aicrowd.com/flatland/flatland/issues/302: add method to FLATland to create of k shortest paths for all agents
        i: get_k_shortest_paths(env, agent.initial_position, agent.initial_direction, agent.target, k)
        for i, agent in enumerate(env.agents)
    }
    rsp_logger.info("create_infrastructure_from_rail_env: shortest paths done")
    minimum_travel_time_dict = {agent.handle: int(np.ceil(1 / agent.speed_data["speed"])) for agent in env.agents}
    topo_dict = _get_topology_from_agents_path_dict(agents_paths_dict)
    return Infrastructure(topo_dict=topo_dict, minimum_travel_time_dict=minimum_travel_time_dict, max_episode_steps=env._max_episode_steps)


def create_schedule_problem_description_from_instructure(
    infrastructure: Infrastructure, number_of_shortest_paths_per_agent_schedule: int
) -> ScheduleProblemDescription:
    # deep copy dict
    topo_dict = {agent_id: topo.copy() for agent_id, topo in infrastructure.topo_dict.items()}
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
                latest_arrival=infrastructure.max_episode_steps,
            )
            for agent_id, topo in topo_dict.items()
        },
        minimum_travel_time_dict=infrastructure.minimum_travel_time_dict,
        topo_dict=topo_dict,
        max_episode_steps=infrastructure.max_episode_steps,
        route_section_penalties={agent_id: {} for agent_id in topo_dict.keys()},
        weight_lateness_seconds=1,
    )
    return schedule_problem_description


def save_experiment_agenda_and_hash_to_file(output_base_folder: str, experiment_agenda: ExperimentAgenda):
    """Save experiment agenda and current git hash to the folder with the
    experiments.
    Parameters
    ----------
    output_base_folder: str
        Folder name of experiment where all experiment files and agenda are stored
    experiment_agenda: ExperimentAgenda
        The experiment agenda to save
    """
    _pickle_dump(obj=experiment_agenda, file_name="experiment_agenda.pkl", folder=output_base_folder)

    # write current hash to sha.txt to experiment folder
    _write_sha_txt(output_base_folder)


def load_experiment_agenda_from_file(experiment_folder_name: str) -> ExperimentAgenda:
    """Save experiment agenda to the folder with the experiments.
    Parameters
    ----------
    experiment_folder_name: str
        Folder name of experiment where all experiment files and agenda are stored
    """
    return _pickle_load(file_name="experiment_agenda.pkl", folder=experiment_folder_name)


def exists_experiment_agenda(experiment_folder_name: str) -> bool:
    """Does a `ExperimentAgenda` exist?"""
    file_name = os.path.join(experiment_folder_name, "experiment_agenda.pkl")
    return os.path.isfile(file_name)


def create_experiment_folder_name(experiment_name: str) -> str:
    datetime_string = datetime.datetime.now().strftime("%Y_%m_%dT%H_%M_%S")
    return "{}_{}".format(experiment_name, datetime_string)


def create_experiment_filename(experiment_data_folder_name: str, experiment_id: int) -> str:
    datetime_string = datetime.datetime.now().strftime("%Y_%m_%dT%H_%M_%S")
    filename = "experiment_{:04d}_{}.pkl".format(experiment_id, datetime_string)
    return os.path.join(experiment_data_folder_name, filename)


def save_experiment_results_to_file(experiment_results: ExperimentResults, file_name: str, csv_only: bool = False, online_unrestricted_only: bool = False):
    """Save the data frame with all the result from an experiment into a given
    file.
    Parameters
    ----------
    experiment_results: List of experiment results
       List containing all the experiment results
    file_name: str
        File name containing path and name of file we want to store the experiment results
    csv_only:bool
        write only csv or also pkl?
    Returns
    -------
    """
    if not csv_only:
        _pickle_dump(obj=experiment_results, file_name=file_name)
    if online_unrestricted_only:
        experiment_data = convert_list_of_experiment_results_analysis_online_unrestricted_to_data_frame(
            [expand_experiment_results_online_unrestricted(experiment_results)]
        )
    else:
        experiment_data: pd.DataFrame = convert_list_of_experiment_results_analysis_to_data_frame([expand_experiment_results_for_analysis(experiment_results)])
    experiment_data.to_csv(file_name.replace(".pkl", ".csv"))


def load_experiments_results(experiment_data_folder_name: str, experiment_id: int) -> Optional[ExperimentResults]:
    """
    Load experiment results from single file.
    Parameters
    ----------
    experiment_data_folder_name
    experiment_id

    Returns
    -------
    Content if file exists (and is unique). `None` else.
    """
    file_names = glob.glob(f"{experiment_data_folder_name}/experiment_{experiment_id:04d}_*.pkl")
    if len(file_names) != 1:
        return None
    return _pickle_load(file_names[0])


def load_and_expand_experiment_results_from_data_folder(
    experiment_data_folder_name: str, experiment_ids: List[int] = None, re_save_csv_after_expansion: bool = False, online_unrestricted_only: bool = False,
) -> Tuple[List[ExperimentResults], List[Union[ExperimentResultsAnalysis, ExperimentResultsAnalysisOnlineUnrestricted]]]:
    """Load results as DataFrame to do further analysis.
    Parameters
    ----------
    experiment_data_folder_name: str
        Folder name of experiment where all experiment files are stored
    experiment_ids
        List of experiment ids which should be loaded, if None all experiments in experiment_folder are loaded
    Returns
    -------
    DataFrame containing the loaded experiment results
    """

    experiment_results_list_analysis = []
    experiment_results_list = []

    files = os.listdir(experiment_data_folder_name)
    rsp_logger.info(f"loading and expanding experiment results from {experiment_data_folder_name}")
    # nicer printing when tdqm print to stderr and we have logging to stdout shown in to the same console (IDE, separated in files)
    newline_and_flush_stdout_and_stderr()
    for file in tqdm.tqdm([file for file in files if "agenda" not in file]):
        file_name = os.path.join(experiment_data_folder_name, file)
        if not file_name.endswith(".pkl"):
            continue

        # filter experiments according to defined experiment_ids
        exp_id = get_experiment_id_from_filename(file_name)
        if experiment_ids is not None and exp_id not in experiment_ids:
            continue
        try:
            file_data: ExperimentResults = _pickle_load(file_name=file_name)
            experiment_results_list.append(file_data)
            if online_unrestricted_only:
                results_for_analysis = expand_experiment_results_online_unrestricted(file_data)
                experiment_results_list_analysis.append(results_for_analysis)
                if re_save_csv_after_expansion:
                    experiment_data: pd.DataFrame = convert_list_of_experiment_results_analysis_online_unrestricted_to_data_frame([results_for_analysis])
                    experiment_data.to_csv(file_name.replace(".pkl", ".csv"))
            else:
                results_for_analysis = expand_experiment_results_for_analysis(file_data)
                experiment_results_list_analysis.append(results_for_analysis)
                if re_save_csv_after_expansion:
                    # ensure it is nonified
                    experiment_data: pd.DataFrame = convert_list_of_experiment_results_analysis_to_data_frame([results_for_analysis])
                    experiment_data.to_csv(file_name.replace(".pkl", ".csv"))
        except Exception as e:
            rsp_logger.warn(f"skipping {file} because of {e}")

    # nicer printing when tdqm print to stderr and we have logging to stdout shown in to the same console (IDE, separated in files)
    newline_and_flush_stdout_and_stderr()
    rsp_logger.info(f" -> done loading and expanding experiment results from {experiment_data_folder_name} done")

    return experiment_results_list, experiment_results_list_analysis


def load_and_filter_experiment_results_analysis_online_unrestricted(
    experiment_base_directory: str = BASELINE_DATA_FOLDER,
    experiments_of_interest: List[int] = None,
    from_cache: bool = False,
    from_individual_csv: bool = True,
    local_filter_experiment_results_analysis_data_frame: Callable[[DataFrame], DataFrame] = None,
) -> DataFrame:
    if from_cache:
        experiment_data_filtered = pd.read_csv(f"{experiment_base_directory}.csv")
    else:
        if from_individual_csv:
            experiment_data: pd.DataFrame = load_data_from_individual_csv_in_data_folder(
                experiment_data_folder_name=f"{experiment_base_directory}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}",
                experiment_ids=experiments_of_interest,
                online_unrestricted_only=True,
            )
        else:
            _, experiment_results_analysis_list = load_and_expand_experiment_results_from_data_folder(
                experiment_data_folder_name=f"{experiment_base_directory}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}",
                experiment_ids=experiments_of_interest,
                online_unrestricted_only=True,
            )
            experiment_data: pd.DataFrame = convert_list_of_experiment_results_analysis_online_unrestricted_to_data_frame(experiment_results_analysis_list)

        if local_filter_experiment_results_analysis_data_frame is not None:
            experiment_data_filtered = local_filter_experiment_results_analysis_data_frame(experiment_data)
            print(f"removed {len(experiment_data) - len(experiment_data_filtered)}/{len(experiment_data)} rows")
        else:
            experiment_data_filtered = experiment_data
        experiment_data_filtered.to_csv(f"{experiment_base_directory}.csv")
    return experiment_data_filtered


def load_data_from_individual_csv_in_data_folder(
    experiment_data_folder_name: str, experiment_ids: List[int] = None, online_unrestricted_only: bool = False
) -> DataFrame:
    """Load results as DataFrame to do further analysis.
    Parameters
    ----------
    experiment_data_folder_name: str
        Folder name of experiment where all experiment files are stored
    experiment_ids
        List of experiment ids which should be loaded, if None all experiments in experiment_folder are loaded
    Returns
    -------
    DataFrame containing the loaded experiment results
    """

    list_of_frames = []
    files = os.listdir(experiment_data_folder_name)
    rsp_logger.info(f"loading individual csv experiment results from {experiment_data_folder_name}")
    # nicer printing when tdqm print to stderr and we have logging to stdout shown in to the same console (IDE, separated in files)
    newline_and_flush_stdout_and_stderr()

    for file in tqdm.tqdm([file for file in files if "agenda" not in file]):
        file_name = os.path.join(experiment_data_folder_name, file)
        if not file_name.endswith(".csv"):
            continue

        # filter experiments according to defined experiment_ids
        exp_id = get_experiment_id_from_filename(file_name)
        if experiment_ids is not None and exp_id not in experiment_ids:
            continue
        try:
            list_of_frames.append(pd.read_csv(file_name))
        except Exception as e:
            rsp_logger.warn(f"skipping {file} because of {e}")

    # nicer printing when tdqm print to stderr and we have logging to stdout shown in to the same console (IDE, separated in files)
    newline_and_flush_stdout_and_stderr()
    rsp_logger.info(f" -> done loading individual csv results from {experiment_data_folder_name} done")

    experiment_data = pd.concat(list_of_frames)

    if not online_unrestricted_only:
        temporary_backwards_compatibility_scope(experiment_data)
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
