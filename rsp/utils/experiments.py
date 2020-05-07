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
import platform
import pprint
import re
import shutil
import sys
import threading
import time
import traceback
from functools import partial
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import tqdm as tqdm
from flatland.action_plan.action_plan import ControllerFromTrainruns
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from pandas import DataFrame

from rsp.experiment_solvers.asp.asp_helper import _print_stats
from rsp.experiment_solvers.data_types import fake_solver_statistics
from rsp.experiment_solvers.data_types import ScheduleAndMalfunction
from rsp.experiment_solvers.data_types import SchedulingExperimentResult
from rsp.experiment_solvers.experiment_solver import ASPExperimentSolver
from rsp.experiment_solvers.trainrun_utils import verify_trainruns_dict
from rsp.flatland_controller.ckua_schedule_generator import ckua_generate_schedule
from rsp.flatland_integration.flatland_replay_utils import create_controller_from_trainruns_and_malfunction
from rsp.flatland_integration.flatland_replay_utils import replay
from rsp.logger import rsp_logger
from rsp.route_dag.analysis.rescheduling_verification_utils import plausibility_check_experiment_results
from rsp.route_dag.analysis.route_dag_analysis import visualize_route_dag_constraints_simple_wrapper
from rsp.route_dag.generators.route_dag_generator_schedule import schedule_problem_description_from_rail_env
from rsp.route_dag.route_dag import ScheduleProblemDescription
from rsp.utils.data_types import convert_list_of_experiment_results_analysis_to_data_frame
from rsp.utils.data_types import expand_experiment_results_for_analysis
from rsp.utils.data_types import ExperimentAgenda
from rsp.utils.data_types import ExperimentParameters
from rsp.utils.data_types import ExperimentResults
from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.data_types import ParameterRangesAndSpeedData
from rsp.utils.experiment_env_generators import create_flatland_environment
from rsp.utils.experiment_env_generators import create_flatland_environment_with_malfunction
from rsp.utils.file_utils import check_create_folder
from rsp.utils.file_utils import get_experiment_id_from_filename
from rsp.utils.file_utils import newline_and_flush_stdout_and_stderr
from rsp.utils.psutil_helpers import current_process_stats_human_readable
from rsp.utils.psutil_helpers import virtual_memory_human_readable
from rsp.utils.tee import reset_tee
from rsp.utils.tee import tee_stdout_stderr_to_file

#  B008 Do not perform function calls in argument defaults.
#  The call is performed only once at function definition time.
#  All calls to your function will reuse the result of that definition-time function call.
#  If this is intended, ass ign the function call to a module-level variable and use that variable as a default value.
AVAILABLE_CPUS = os.cpu_count()

_pp = pprint.PrettyPrinter(indent=4)

EXPERIMENT_AGENDA_SUBDIRECTORY_NAME = "agenda"
EXPERIMENT_DATA_SUBDIRECTORY_NAME = "data"
EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME = "analysis"
EXPERIMENT_POTASSCO_SUBDIRECTORY_NAME = "potassco"


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
        file_data: ScheduleAndMalfunction = pickle.load(handle)
        return file_data


def run_experiment(solver: ASPExperimentSolver,  # noqa: C901
                   experiment_parameters: ExperimentParameters,
                   experiment_base_directory: str,
                   show_results_without_details: bool = True,
                   rendering: bool = False,
                   verbose: bool = False,
                   debug: bool = False,
                   gen_only: bool = False,
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

    start_datetime_str = datetime.datetime.now().strftime("%H:%M:%S")
    if show_results_without_details:
        rsp_logger.info("Running experiment {} under pid {} at {}".format(experiment_parameters.experiment_id, os.getpid(), start_datetime_str))
    start_time = time.time()

    if show_results_without_details:
        rsp_logger.info("*** experiment parameters for experiment {}. {}"
                        .format(experiment_parameters.experiment_id,
                                _pp.pformat(experiment_parameters)))

    # A.2: load or re-generate?
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

        if debug:
            _render_route_dags_from_data(experiment_base_directory=experiment_base_directory, experiment_id=experiment_parameters.experiment_id)
            _visualize_route_dag_constraints_for_schedule_and_malfunction(schedule_and_malfunction=schedule_and_malfunction)

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
    if gen_only:
        elapsed_time = (time.time() - start_time)
        _print_stats(schedule_and_malfunction.schedule_experiment_result.solver_statistics)
        solver_time_full = schedule_and_malfunction.schedule_experiment_result.solver_statistics["summary"]["times"]["total"]
        rsp_logger.info("Generating schedule {}: took {:5.3f}s (sched: {:5.3f}s = {:5.2f}%".format(
            experiment_parameters.experiment_id,
            elapsed_time, solver_time_full,
            solver_time_full / elapsed_time * 100))
        return ExperimentResults(
            experiment_parameters=experiment_parameters,
            malfunction=schedule_and_malfunction.experiment_malfunction,
            problem_full=schedule_and_malfunction.schedule_problem_description,
            problem_full_after_malfunction=None,
            problem_delta_after_malfunction=None,
            results_full=schedule_and_malfunction.schedule_experiment_result,
            results_full_after_malfunction=None,
            results_delta_after_malfunction=None
        )

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

    if show_results_without_details:
        elapsed_time = (time.time() - start_time)
        end_datetime_str = datetime.datetime.now().strftime("%H:%M:%S")
        s = (
            "Running experiment {}: took {:5.3f}s ({}--{}) (sched:  {} / re-sched full:  {} / re-sched delta:  {} / ").format(
            experiment_parameters.experiment_id,
            elapsed_time,
            start_datetime_str,
            end_datetime_str,
            _get_asp_solver_details_from_statistics(elapsed_time=elapsed_time, statistics=experiment_results.results_full.solver_statistics),
            _get_asp_solver_details_from_statistics(elapsed_time=elapsed_time, statistics=experiment_results.results_full_after_malfunction.solver_statistics),
            _get_asp_solver_details_from_statistics(elapsed_time=elapsed_time, statistics=experiment_results.results_delta_after_malfunction.solver_statistics),
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
        print(s)

        rsp_logger.info(virtual_memory_human_readable())
        rsp_logger.info(current_process_stats_human_readable())

    # TODO SIM-324 pull out validation steps
    plausibility_check_experiment_results(experiment_results=experiment_results)
    return experiment_results


def _visualize_route_dag_constraints_for_schedule_and_malfunction(schedule_and_malfunction: ScheduleAndMalfunction):
    for agent_id in schedule_and_malfunction.schedule_experiment_result.trainruns_dict:
        visualize_route_dag_constraints_simple_wrapper(
            schedule_problem_description=schedule_and_malfunction.schedule_problem_description,
            trainrun_dict=None,
            experiment_malfunction=schedule_and_malfunction.experiment_malfunction,
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


def create_schedule_and_malfunction(
        experiment_parameters: ExperimentParameters,
        solver: ASPExperimentSolver,
        rendering: bool = False,
        verbose: bool = False,
        debug: bool = False
) -> Tuple[RailEnv, ScheduleAndMalfunction]:
    """A.2 Create schedule and malfunction from experiment parameters.
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

    return gen_schedule_and_malfunction(
        experiment_parameters=experiment_parameters,
        static_rail_env=static_rail_env,
        malfunction_rail_env=malfunction_rail_env,
        malfunction_env_reset=malfunction_env_reset,
        solver=solver,
        verbose=verbose,
        debug=debug)


def gen_schedule_and_malfunction(
        experiment_parameters: ExperimentParameters,
        static_rail_env: RailEnv,
        malfunction_rail_env: RailEnv,
        malfunction_env_reset: Callable,
        solver: ASPExperimentSolver,
        verbose: bool = False,
        debug: bool = False

):
    """A.2 Create schedule and malfunction from experiment parameters.

    Parameters
    ----------
    experiment_parameters
    static_rail_env
    malfunction_rail_env
    malfunction_env_reset
    solver
    verbose
    debug

    Returns
    -------
    """
    # TODO SIM-443 pull out switch out
    SWITCH_CKUA = False
    if SWITCH_CKUA:
        tc_schedule_problem = schedule_problem_description_from_rail_env(
            env=static_rail_env,
            k=experiment_parameters.number_of_shortest_paths_per_agent
        )
        trainrun_dict, elapsed_time = ckua_generate_schedule(
            env=static_rail_env,
            random_seed=experiment_parameters.flatland_seed_value,
            rendering=False,
            show=False
        )
        verify_trainruns_dict(
            env=static_rail_env,
            trainrun_dict=trainrun_dict,
            expected_route_dag_constraints=tc_schedule_problem.route_dag_constraints_dict
        )
        schedule_result = SchedulingExperimentResult(
            total_reward=-np.inf,
            solve_time=-np.inf,
            optimization_costs=-np.inf,
            build_problem_time=-np.inf,
            nb_conflicts=-np.inf,
            trainruns_dict=trainrun_dict,
            route_dag_constraints=tc_schedule_problem.route_dag_constraints_dict,
            solver_statistics=fake_solver_statistics(elapsed_time),
            solver_result={},
            solver_configuration={},
            solver_seed=experiment_parameters.asp_seed_value,
            solver_program=None
        )
    else:
        tc_schedule_problem, schedule_result = solver.gen_schedule(
            static_rail_env=static_rail_env,
            experiment_parameters=experiment_parameters,
            verbose=verbose,
            debug=debug
        )
    malfunction = gen_malfunction(
        malfunction_env_reset=malfunction_env_reset,
        malfunction_rail_env=malfunction_rail_env,
        schedule_trainruns=schedule_result.trainruns_dict,
        verbose=verbose)
    schedule_and_malfunction = ScheduleAndMalfunction(tc_schedule_problem, schedule_result, malfunction)
    return malfunction_rail_env, schedule_and_malfunction


def gen_malfunction(malfunction_rail_env: RailEnv,
                    malfunction_env_reset,
                    schedule_trainruns: TrainrunDict,
                    verbose: bool = False
                    ):
    """A.2.2. Create malfunction.

    Parameters
    ----------
    malfunction_rail_env
    malfunction_env_reset
    schedule_trainruns
    verbose

    Returns
    -------
    """
    # --------------------------------------------------------------------------------------
    # 1. Generate malfuntion
    # --------------------------------------------------------------------------------------
    malfunction_env_reset()
    controller_from_train_runs: ControllerFromTrainruns = create_controller_from_trainruns_and_malfunction(
        trainrun_dict=schedule_trainruns,
        env=malfunction_rail_env)
    malfunction_env_reset()
    malfunction = replay(
        controller_from_train_runs=controller_from_train_runs,
        env=malfunction_rail_env,
        stop_on_malfunction=True,
        solver_name="ASP")
    malfunction_env_reset()
    # replay may return None (if the given malfunction does not happen during the agents time in the grid
    if malfunction is None:
        raise Exception("Could not produce a malfunction")
    if verbose:
        print(f"  **** malfunction={malfunction}")
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


def run_and_save_one_experiment(current_experiment_parameters: ExperimentParameters,
                                solver: ASPExperimentSolver,
                                verbose: bool,
                                show_results_without_details: bool,
                                experiment_base_directory: str,
                                rendering: bool = False,
                                gen_only: bool = False):
    """B. Run and save one experiment from experiment parameters.
    Parameters
    ----------
    current_experiment_parameters
    solver
    verbose
    show_results_without_details
    experiment_base_directory
    rendering
    gen_only
    """
    rsp_logger.info(f"start experiment {current_experiment_parameters.experiment_id}")
    experiment_data_directory = f'{experiment_base_directory}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}'

    # tee stdout to thread-specific log file
    tee_orig = tee_stdout_stderr_to_file(
        stdout_log_file=os.path.join(experiment_data_directory, f"log_{os.getpid()}.txt"),
        stderr_log_file=os.path.join(experiment_data_directory, f"err_{os.getpid()}.txt")
    )
    try:

        check_create_folder(experiment_data_directory)

        filename = create_experiment_filename(experiment_data_directory, current_experiment_parameters.experiment_id)
        experiment_results: ExperimentResults = run_experiment(solver=solver,
                                                               experiment_parameters=current_experiment_parameters,
                                                               rendering=rendering,
                                                               verbose=verbose,
                                                               experiment_base_directory=experiment_base_directory,
                                                               show_results_without_details=show_results_without_details,
                                                               gen_only=gen_only)
        save_experiment_results_to_file(experiment_results, filename)
        return os.getpid()
    except Exception as e:
        rsp_logger.error("XXX failed " + filename + " " + str(e))
        traceback.print_exc(file=sys.stderr)
        return os.getpid()
    finally:
        # remove tees
        reset_tee(*tee_orig)
        rsp_logger.info(f"end experiment {current_experiment_parameters.experiment_id}")


def run_experiment_agenda(experiment_agenda: ExperimentAgenda,
                          experiment_ids: Optional[List[int]] = None,
                          copy_agenda_from_base_directory: Optional[str] = None,
                          run_experiments_parallel: int = AVAILABLE_CPUS // 2,  # take only half of avilable cpus so the machine stays responsive
                          show_results_without_details: bool = True,
                          rendering: bool = False,
                          verbose: bool = False,
                          gen_only: bool = False
                          ) -> (str, str):
    """Run B. a subset of experiments of a given agenda. This is useful when
    trying to find bugs in code.
    Parameters
    ----------
    experiment_agenda: ExperimentAgenda
        Full list of experiments
    experiment_ids: Optional[List[int]]
        List of experiment IDs we want to run
    run_experiments_parallel: in
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
    Returns the name of the experiment base and data folders
    """
    experiment_base_directory = create_experiment_folder_name(experiment_agenda.experiment_name)
    experiment_data_directory = f'{experiment_base_directory}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}'
    experiment_agenda_directory = f'{experiment_base_directory}/{EXPERIMENT_AGENDA_SUBDIRECTORY_NAME}'

    check_create_folder(experiment_base_directory)
    check_create_folder(experiment_data_directory)
    check_create_folder(experiment_agenda_directory)

    if run_experiments_parallel <= 1:
        rsp_logger.warn("Using only one process in pool might cause pool to stall sometimes. Use more than one process in pool?")

    # tee stdout to log file
    tee_orig = tee_stdout_stderr_to_file(
        stdout_log_file=os.path.join(experiment_data_directory, "log.txt"),
        stderr_log_file=os.path.join(experiment_data_directory, "err.txt"),
    )

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
        run_and_save_one_experiment,
        solver=solver,
        verbose=verbose,
        show_results_without_details=show_results_without_details,
        experiment_base_directory=experiment_base_directory,
        gen_only=gen_only
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

    # nicer printing when tdqm print to stderr and we have logging to stdout shown in to the same console (IDE, separated in files)
    newline_and_flush_stdout_and_stderr()
    _print_log_files_from_experiment_data_directory(experiment_data_directory)

    # remove tees
    reset_tee(*tee_orig)
    return experiment_base_directory, experiment_data_directory


# TODO SIM-411 adapt to logger
def _print_log_files_from_experiment_data_directory(experiment_data_directory):
    log_files = os.listdir(experiment_data_directory)
    rsp_logger.info(f"loading and expanding experiment results from {experiment_data_directory}")
    error_summay = []
    for file in [file for file in log_files if file.startswith("log_")]:
        print("\n\n\n\n")
        print(f"=========================================================")
        print(f"output of {file}")
        print(f"=========================================================")
        with open(os.path.join(experiment_data_directory, file), "r") as file_in:
            content = file_in.read()
            print(content)
            error_summay += re.findall("XXX.*$", content, re.MULTILINE)
    print("\n\n\n\n")

    print(f"=========================================================")
    print(f"ERROR SUMMARY")
    print(f"=========================================================")
    for err in error_summay:
        print(err)
    print(f"=========================================================")
    print(f"END OF ERROR SUMMARY")
    print(f"=========================================================")
    print("\n\n\n\n")


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
    print(os.path.abspath(os.curdir))
    files = os.listdir(copy_agenda_from_agenda_directory)
    rsp_logger.info(f"Copying agenda, schedule and malfunctions {copy_agenda_from_agenda_directory} "
                    f"-> {experiment_agenda_directory}")
    for file in [file for file in files]:
        shutil.copy2(os.path.join(copy_agenda_from_agenda_directory, file), experiment_agenda_directory)


def filter_experiment_agenda(current_experiment_parameters, experiment_ids) -> bool:
    return current_experiment_parameters.experiment_id in experiment_ids


def create_experiment_agenda(experiment_name: str,
                             parameter_ranges_and_speed_data: ParameterRangesAndSpeedData,
                             experiments_per_grid_element: int = 10,
                             debug: bool = False
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
    parameter_ranges = parameter_ranges_and_speed_data.parameter_ranges
    number_of_dimensions = len(parameter_ranges)
    parameter_values = [[] for _ in range(number_of_dimensions)]

    # Setup experiment parameters
    for dim_idx, dimensions in enumerate(parameter_ranges):
        if dimensions[-1] > 1:
            if debug:
                print(f"{dimensions[0]} {dimensions[1]} {np.abs(dimensions[1] - dimensions[0]) / dimensions[-1]}")
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
                number_of_agents=parameter_set[1],
                speed_data=parameter_ranges_and_speed_data.speed_data,
                width=parameter_set[0],
                height=parameter_set[0],
                flatland_seed_value=12 + run_of_this_grid_element,
                asp_seed_value=parameter_set[9],
                max_num_cities=parameter_set[4],
                grid_mode=True,
                max_rail_between_cities=parameter_set[3],
                max_rail_in_city=parameter_set[2],
                earliest_malfunction=parameter_set[5],
                malfunction_duration=parameter_set[6],
                number_of_shortest_paths_per_agent=parameter_set[7],
                weight_route_change=parameter_set[10],
                weight_lateness_seconds=parameter_set[11],
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
    datetime_string = datetime.datetime.now().strftime("%Y_%m_%dT%H_%M_%S")
    filename = "experiment_{:04d}_{}.pkl".format(experiment_id, datetime_string)
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


def load_and_expand_experiment_results_from_data_folder(experiment_data_folder_name: str,
                                                        experiment_ids: List[int] = None) -> \
        List[ExperimentResultsAnalysis]:
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
            file_data = pickle.load(handle)
            experiment_results_list.append(expand_experiment_results_for_analysis(file_data))
    # nicer printing when tdqm print to stderr and we have logging to stdout shown in to the same console (IDE, separated in files)
    newline_and_flush_stdout_and_stderr()
    rsp_logger.info(f" -> loading and expanding experiment results from {experiment_data_folder_name} done")
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
