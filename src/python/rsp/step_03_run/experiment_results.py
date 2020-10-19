from typing import NamedTuple
from typing import Set

from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint
from rsp.scheduling.schedule import SchedulingExperimentResult
from rsp.scheduling.scheduling_problem import ScheduleProblemDescription
from rsp.step_01_planning.experiment_parameters_and_ranges import ExperimentParameters
from rsp.step_02_setup.data_types import ExperimentMalfunction
from rsp.utils.global_constants import NB_RANDOM

ExperimentResults = NamedTuple(
    "ExperimentResults",
    [
        ("experiment_parameters", ExperimentParameters),
        ("malfunction", ExperimentMalfunction),
        ("problem_full", ScheduleProblemDescription),
        ("problem_full_after_malfunction", ScheduleProblemDescription),
        ("problem_delta_trivially_perfect_after_malfunction", ScheduleProblemDescription),
        ("problem_delta_perfect_after_malfunction", ScheduleProblemDescription),
        ("problem_delta_no_rerouting_after_malfunction", ScheduleProblemDescription),
        ("problem_delta_online_after_malfunction", ScheduleProblemDescription),
        ("problem_delta_online_no_time_flexibility_after_malfunction", ScheduleProblemDescription),
        ("results_full", SchedulingExperimentResult),
        ("results_full_after_malfunction", SchedulingExperimentResult),
        ("results_delta_trivially_perfect_after_malfunction", SchedulingExperimentResult),
        ("results_delta_perfect_after_malfunction", SchedulingExperimentResult),
        ("results_delta_no_rerouting_after_malfunction", SchedulingExperimentResult),
        ("results_delta_online_after_malfunction", SchedulingExperimentResult),
        ("results_delta_online_no_time_flexibility_after_malfunction", SchedulingExperimentResult),
        ("predicted_changed_agents_online_after_malfunction", Set[int]),
        ("predicted_changed_agents_online_no_time_flexibility_after_malfunction", Set[int]),
    ]
    + [(f"problem_delta_random_{i}_after_malfunction", ScheduleProblemDescription) for i in range(NB_RANDOM)]
    + [(f"results_delta_random_{i}_after_malfunction", SchedulingExperimentResult) for i in range(NB_RANDOM)]
    + [(f"predicted_changed_agents_random_{i}_after_malfunction", Set[int]) for i in range(NB_RANDOM)],
)


def plausibility_check_experiment_results(experiment_results: ExperimentResults):
    """Verify the following experiment expectations:

    1. a) same waypoint in schedule and re-schedule -> waypoint also in scope perfect re-schedule
       b) same waypoint and time in schedule and re-schedule -> same waypoint and ant time also in re-schedule delta perfect
    2. number of routing alternatives should be decreasing from full to delta
    """
    route_dag_constraints_delta_perfect_after_malfunction = experiment_results.results_delta_perfect_after_malfunction.route_dag_constraints

    # 1.
    for agent_id in route_dag_constraints_delta_perfect_after_malfunction:
        # b) S0[x] == S[x]) ==> S'[x]: path and time
        schedule: Set[TrainrunWaypoint] = frozenset(experiment_results.results_full.trainruns_dict[agent_id])
        reschedule_full: Set[TrainrunWaypoint] = frozenset(experiment_results.results_full_after_malfunction.trainruns_dict[agent_id])
        reschedule_delta_perfect: Set[TrainrunWaypoint] = frozenset(experiment_results.results_delta_perfect_after_malfunction.trainruns_dict[agent_id])
        assert schedule.intersection(reschedule_full).issubset(reschedule_delta_perfect)

        # a) S0[x] == S[x]) ==> S'[x]: path
        waypoints_schedule: Set[Waypoint] = {trainrun_waypoint.waypoint for trainrun_waypoint in experiment_results.results_full.trainruns_dict[agent_id]}
        waypoints_reschedule_full: Set[Waypoint] = {
            trainrun_waypoint.waypoint for trainrun_waypoint in experiment_results.results_full_after_malfunction.trainruns_dict[agent_id]
        }
        waypoints_reschedule_delta_perfect: Set[Waypoint] = {
            trainrun_waypoint.waypoint for trainrun_waypoint in experiment_results.results_delta_perfect_after_malfunction.trainruns_dict[agent_id]
        }
        assert waypoints_schedule.intersection(waypoints_reschedule_full).issubset(waypoints_reschedule_delta_perfect)

    # 2. plausibility test: number of alternatives should be decreasing from full to delta
    for agent_id in route_dag_constraints_delta_perfect_after_malfunction:
        node_set_full_after_malfunction = set(experiment_results.problem_full_after_malfunction.topo_dict[agent_id].nodes)
        node_set_delta_perfect_after_malfunction = set(experiment_results.problem_delta_perfect_after_malfunction.topo_dict[agent_id].nodes)
        if not node_set_full_after_malfunction.issuperset(node_set_delta_perfect_after_malfunction):
            assert node_set_full_after_malfunction.issuperset(node_set_delta_perfect_after_malfunction), (
                f"{agent_id}: not all nodes from delta perfect are also in full: only delta perfect "
                f"{node_set_delta_perfect_after_malfunction.difference(node_set_full_after_malfunction)}, "
                f"malfunction={experiment_results.malfunction}"
            )
