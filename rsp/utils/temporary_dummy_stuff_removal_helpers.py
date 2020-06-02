import os
import pickle
from typing import Tuple

import tqdm

from rsp.experiment_solvers.data_types import ScheduleAndMalfunction
from rsp.logger import rsp_logger
from rsp.utils.data_types import ExperimentResults
from rsp.utils.experiments import load_schedule_and_malfunction
from rsp.utils.experiments import save_experiment_results_to_file
from rsp.utils.experiments import save_schedule_and_malfunction
from rsp.utils.file_utils import get_experiment_id_from_filename


def remove_dummy_stuff_from_experiment_results_file(experiment_data_folder_name: str,
                                                    experiment_id: int):
    experiment_results, file_name = load_experiment_result_without_expanding(experiment_data_folder_name, experiment_id)

    for topo_dict in [
        experiment_results.problem_full.topo_dict,
        experiment_results.problem_full_after_malfunction.topo_dict,
        experiment_results.problem_delta_after_malfunction.topo_dict,
    ]:
        _remove_dummy_stuff_from_topo_dict(topo_dict=topo_dict)
    for route_dag_constraints_dict in [
        experiment_results.problem_full.route_dag_constraints_dict,
        experiment_results.problem_full_after_malfunction.route_dag_constraints_dict,
        experiment_results.problem_delta_after_malfunction.route_dag_constraints_dict,
        experiment_results.results_full.route_dag_constraints,
        experiment_results.results_full_after_malfunction.route_dag_constraints,
        experiment_results.results_delta_after_malfunction.route_dag_constraints,
    ]:
        _remove_dummy_stuff_from_route_dag_constraints_dict(route_dag_constraints_dict=route_dag_constraints_dict)
    for trainrun_dict in [
        experiment_results.results_full.trainruns_dict,
        experiment_results.results_full_after_malfunction.trainruns_dict,
        experiment_results.results_delta_after_malfunction.trainruns_dict,
    ]:
        _remove_dummy_stuff_from_trainrun_dict(trainrun_dict=trainrun_dict)

    save_experiment_results_to_file(experiment_results=experiment_results, file_name=file_name)


def load_experiment_result_without_expanding(experiment_data_folder_name, experiment_id) -> Tuple[ExperimentResults, str]:
    files = os.listdir(experiment_data_folder_name)
    rsp_logger.info(f"loading and expanding experiment results from {experiment_data_folder_name}")
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


def remove_dummy_stuff_from_schedule_and_malfunction_pickle(
        experiment_agenda_directory: str,
        experiment_id: int):
    schedule_and_malfunction: ScheduleAndMalfunction = load_schedule_and_malfunction(experiment_agenda_directory=experiment_agenda_directory,
                                                                                     experiment_id=experiment_id)
    topo_dict = schedule_and_malfunction.schedule_problem_description.topo_dict
    _remove_dummy_stuff_from_topo_dict(topo_dict)
    for route_dag_constraints_dict in [
        schedule_and_malfunction.schedule_problem_description.route_dag_constraints_dict,
        schedule_and_malfunction.schedule_experiment_result.route_dag_constraints
    ]:
        _remove_dummy_stuff_from_route_dag_constraints_dict(route_dag_constraints_dict)
    trainrun_dict = schedule_and_malfunction.schedule_experiment_result.trainruns_dict
    _remove_dummy_stuff_from_trainrun_dict(trainrun_dict)
    save_schedule_and_malfunction(schedule_and_malfunction=schedule_and_malfunction, experiment_agenda_directory=experiment_agenda_directory,
                                  experiment_id=experiment_id)


def _remove_dummy_stuff_from_topo_dict(topo_dict):
    for _, topo in topo_dict.items():
        dummy_nodes = [v for v in topo.nodes if v.direction == 5]
        topo.remove_nodes_from(dummy_nodes)


def _remove_dummy_stuff_from_trainrun_dict(trainrun_dict):
    for agent_id in trainrun_dict:
        trainrun = trainrun_dict[agent_id]
        trainrun_tweaked = [trainrun_waypoint for trainrun_waypoint in trainrun if trainrun_waypoint.waypoint.direction != 5]
        trainrun_dict[agent_id] = trainrun_tweaked


def _remove_dummy_stuff_from_route_dag_constraints_dict(route_dag_constraints_dict):
    for _, constraints in route_dag_constraints_dict.items():
        dummy_nodes_earliest_latest = [v for v in constraints.freeze_earliest.keys() if v.direction == 5]
        for v in dummy_nodes_earliest_latest:
            del constraints.freeze_earliest[v]
            del constraints.freeze_latest[v]
        dummy_nodes_visit = [v for v in constraints.freeze_visit if v.direction == 5]
        for v in dummy_nodes_visit:
            constraints.freeze_visit.remove(v)
