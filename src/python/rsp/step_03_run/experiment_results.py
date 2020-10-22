from typing import NamedTuple
from typing import Set

from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint
from rsp.scheduling.schedule import SchedulingExperimentResult
from rsp.scheduling.scheduling_problem import ScheduleProblemDescription
from rsp.step_01_planning.experiment_parameters_and_ranges import ExperimentParameters
from rsp.step_02_setup.data_types import ExperimentMalfunction
from rsp.utils.global_constants import GLOBAL_CONSTANTS

ExperimentResults = NamedTuple(
    "ExperimentResults",
    [
        ("experiment_parameters", ExperimentParameters),
        ("malfunction", ExperimentMalfunction),
        ("problem_schedule", ScheduleProblemDescription),
        ("problem_online_unrestricted", ScheduleProblemDescription),
        ("problem_online_fully_restricted", ScheduleProblemDescription),
        ("problem_offline_delta", ScheduleProblemDescription),
        ("problem_online_route_restricted", ScheduleProblemDescription),
        ("problem_online_transmission_chains_fully_restricted", ScheduleProblemDescription),
        ("problem_online_transmission_chains_route_restricted", ScheduleProblemDescription),
        ("results_schedule", SchedulingExperimentResult),
        ("results_online_unrestricted", SchedulingExperimentResult),
        ("results_online_fully_restricted", SchedulingExperimentResult),
        ("results_offline_delta", SchedulingExperimentResult),
        ("results_online_route_restricted", SchedulingExperimentResult),
        ("results_online_transmission_chains_fully_restricted", SchedulingExperimentResult),
        ("results_online_transmission_chains_route_restricted", SchedulingExperimentResult),
        ("predicted_changed_agents_online_transmission_chains_fully_restricted", Set[int]),
        ("predicted_changed_agents_online_transmission_chains_route_restricted", Set[int]),
    ]
    + [(f"problem_online_random_{i}", ScheduleProblemDescription) for i in range(GLOBAL_CONSTANTS.NB_RANDOM)]
    + [(f"results_online_random_{i}", SchedulingExperimentResult) for i in range(GLOBAL_CONSTANTS.NB_RANDOM)]
    + [(f"predicted_changed_agents_online_random_{i}", Set[int]) for i in range(GLOBAL_CONSTANTS.NB_RANDOM)],
)


def plausibility_check_experiment_results(experiment_results: ExperimentResults):
    """Verify the following experiment expectations:

    1. a) same waypoint in schedule and re-schedule -> waypoint also in scope perfect re-schedule
       b) same waypoint and time in schedule and re-schedule -> same waypoint and ant time also in re-schedule delta perfect
    2. number of routing alternatives should be decreasing from full to delta
    """
    route_dag_constraints_offline_delta = experiment_results.results_offline_delta.route_dag_constraints

    # 1.
    for agent_id in route_dag_constraints_offline_delta:
        # b) S0[x] == S[x]) ==> S'[x]: path and time
        schedule: Set[TrainrunWaypoint] = frozenset(experiment_results.results_schedule.trainruns_dict[agent_id])
        online_unrestricted: Set[TrainrunWaypoint] = frozenset(experiment_results.results_online_unrestricted.trainruns_dict[agent_id])
        offline_delta: Set[TrainrunWaypoint] = frozenset(experiment_results.results_offline_delta.trainruns_dict[agent_id])
        assert schedule.intersection(online_unrestricted).issubset(offline_delta)

        # a) S0[x] == S[x]) ==> S'[x]: path
        waypoints_schedule: Set[Waypoint] = {trainrun_waypoint.waypoint for trainrun_waypoint in experiment_results.results_schedule.trainruns_dict[agent_id]}
        waypoints_online_unrestricted: Set[Waypoint] = {
            trainrun_waypoint.waypoint for trainrun_waypoint in experiment_results.results_online_unrestricted.trainruns_dict[agent_id]
        }
        waypoints_offline_delta: Set[Waypoint] = {
            trainrun_waypoint.waypoint for trainrun_waypoint in experiment_results.results_offline_delta.trainruns_dict[agent_id]
        }
        assert waypoints_schedule.intersection(waypoints_online_unrestricted).issubset(waypoints_offline_delta)

    # 2. plausibility test: number of alternatives should be decreasing from full to delta
    for agent_id in route_dag_constraints_offline_delta:
        node_set_online_unrestricted = set(experiment_results.problem_online_unrestricted.topo_dict[agent_id].nodes)
        node_set_offline_delta = set(experiment_results.problem_offline_delta.topo_dict[agent_id].nodes)
        if not node_set_online_unrestricted.issuperset(node_set_offline_delta):
            assert node_set_online_unrestricted.issuperset(node_set_offline_delta), (
                f"{agent_id}: not all nodes from delta perfect are also in full: only delta perfect "
                f"{node_set_offline_delta.difference(node_set_online_unrestricted)}, "
                f"malfunction={experiment_results.malfunction}"
            )
