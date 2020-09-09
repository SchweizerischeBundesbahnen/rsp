from typing import Set

from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.utils.data_types import ExperimentResults


def plausibility_check_experiment_results(experiment_results: ExperimentResults):
    """Verify the following experiment expectations:

    1. a) same waypoint in schedule and re-schedule -> waypoint also in delta re-schedule
       b) same waypoint and time in schedule and re-schedule -> same waypoint and ant time also in delta-re-schedule
    """
    route_dag_constraints_delta_afer_malfunction = experiment_results.results_delta_perfect_after_malfunction.route_dag_constraints

    # 1. plausibility check
    # a) same waypoint in schedule and re-schedule -> waypoint also in delta re-schedule
    # b) same waypoint and time in schedule and re-schedule -> same waypoint and ant time also in delta-re-schedule
    for agent_id in route_dag_constraints_delta_afer_malfunction:
        # b) S0[x] == S[x]) ==> S'[x]: path and time
        schedule: Set[TrainrunWaypoint] = frozenset(experiment_results.results_full.trainruns_dict[agent_id])
        reschedule_full: Set[TrainrunWaypoint] = frozenset(
            experiment_results.results_full_after_malfunction.trainruns_dict[agent_id])
        reschedule_delta: Set[TrainrunWaypoint] = frozenset(
            experiment_results.results_delta_perfect_after_malfunction.trainruns_dict[agent_id])
        assert schedule.intersection(reschedule_full).issubset(reschedule_delta)

        # a) S0[x] == S[x]) ==> S'[x]: path
        waypoints_schedule: Set[Waypoint] = {
            trainrun_waypoint.waypoint
            for trainrun_waypoint in experiment_results.results_full.trainruns_dict[agent_id]
        }
        waypoints_reschedule_full: Set[Waypoint] = {
            trainrun_waypoint.waypoint
            for trainrun_waypoint in experiment_results.results_full_after_malfunction.trainruns_dict[agent_id]
        }
        waypoints_reschedule_delta: Set[Waypoint] = {
            trainrun_waypoint.waypoint
            for trainrun_waypoint in experiment_results.results_delta_perfect_after_malfunction.trainruns_dict[agent_id]
        }
        assert waypoints_schedule.intersection(waypoints_reschedule_full).issubset(waypoints_reschedule_delta)

    # 2. plausibility test: number of alternatives should be decreasing from full to delta
    for agent_id in route_dag_constraints_delta_afer_malfunction:
        node_set_full_after_malfunction = set(experiment_results.problem_full_after_malfunction.topo_dict[agent_id].nodes)
        node_set_delta_perfect_after_malfunction = set(experiment_results.problem_delta_perfect_after_malfunction.topo_dict[agent_id].nodes)
        if not node_set_full_after_malfunction.issuperset(node_set_delta_perfect_after_malfunction):
            assert node_set_full_after_malfunction.issuperset(node_set_delta_perfect_after_malfunction), \
                f"{agent_id}: not all delta nodes are also in full: only delta " \
                f"{node_set_delta_perfect_after_malfunction.difference(node_set_full_after_malfunction)}, " \
                f"malfunction={experiment_results.malfunction}"
