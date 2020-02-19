from typing import Set

from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.rescheduling.rescheduling_analysis_utils import _extract_number_of_path_alternatives
from rsp.rescheduling.rescheduling_analysis_utils import _extract_path_search_space
from rsp.utils.data_types import ExperimentResults


def plausibility_check_experiment_results(experiment_results: ExperimentResults,
                                          experiment_id: int):
    """Verify the following experiment expectations:
       1. a) same waypoint in schedule and re-schedule -> waypoint also in delta re-schedule
          b) same waypoint and time in schedule and re-schedule -> same waypoint and ant time also in delta-re-schedule

    """
    experiment_freeze_delta_afer_malfunction = experiment_results.experiment_freeze_delta_after_malfunction
    experiment_freeze_full_after_malfunction = experiment_results.experiment_freeze_full_after_malfunction
    topo_dict = experiment_results.topo_dict

    # 1. plausibility check
    # a) same waypoint in schedule and re-schedule -> waypoint also in delta re-schedule
    # b) same waypoint and time in schedule and re-schedule -> same waypoint and ant time also in delta-re-schedule
    for agent_id in experiment_freeze_delta_afer_malfunction:
        # b) S0[x] == S[x]) ==> S'[x]: path and time
        schedule: Set[TrainrunWaypoint] = frozenset(experiment_results.solution_full[agent_id])
        reschedule_full: Set[TrainrunWaypoint] = frozenset(experiment_results.solution_full_after_malfunction[agent_id])
        reschedule_delta: Set[TrainrunWaypoint] = frozenset(experiment_results.solution_full[agent_id])
        assert schedule.intersection(reschedule_full).issubset(reschedule_delta)

        # a) S0[x] == S[x]) ==> S'[x]: path
        waypoints_schedule: Set[Waypoint] = {
            trainrun_waypoint.waypoint
            for trainrun_waypoint in experiment_results.solution_full[agent_id]
        }
        waypoints_reschedule_full: Set[Waypoint] = {
            trainrun_waypoint.waypoint
            for trainrun_waypoint in experiment_results.solution_full_after_malfunction[agent_id]
        }
        waypoints_reschedule_delta: Set[Waypoint] = {
            trainrun_waypoint.waypoint
            for trainrun_waypoint in experiment_results.solution_full[agent_id]
        }
        assert waypoints_schedule.intersection(waypoints_reschedule_full).issubset(waypoints_reschedule_delta)

    # 2. plausibility test: number of alternatives should be decreasing
    all_nb_alternatives_rsp_delta, all_nb_alternatives_rsp_full, all_nb_alternatives_schedule = _extract_number_of_path_alternatives(
        topo_dict, experiment_freeze_delta_afer_malfunction,
        experiment_freeze_full_after_malfunction)

    for agent_id in experiment_freeze_delta_afer_malfunction:
        nb_alternatives_schedule = all_nb_alternatives_schedule[agent_id]
        nb_alternatives_rsp_full = all_nb_alternatives_rsp_full[agent_id]
        nb_alternatives_rsp_delta = all_nb_alternatives_rsp_delta[agent_id]
        assert nb_alternatives_schedule >= nb_alternatives_rsp_full
        assert nb_alternatives_rsp_full >= nb_alternatives_rsp_delta

    assert len(all_nb_alternatives_schedule) == len(topo_dict.keys())
    assert len(all_nb_alternatives_rsp_full) == len(topo_dict.keys())
    assert len(all_nb_alternatives_rsp_delta) == len(topo_dict.keys())

    path_search_space_rsp_delta, path_search_space_rsp_full, path_search_space_schedule = _extract_path_search_space(
        experiment_results)
    assert path_search_space_schedule >= path_search_space_rsp_full
    assert path_search_space_rsp_full >= path_search_space_rsp_delta
    return path_search_space_rsp_delta, path_search_space_rsp_full, path_search_space_schedule
