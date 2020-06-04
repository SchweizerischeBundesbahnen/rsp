from typing import Dict
from typing import Optional
from typing import Tuple

from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.route_dag.route_dag import get_sinks_for_topo
from rsp.route_dag.route_dag import get_sources_for_topo
from rsp.route_dag.route_dag import RouteDAGConstraintsDict
from rsp.route_dag.route_dag import ScheduleProblemDescription
from rsp.route_dag.route_dag import TopoDict
from rsp.utils.data_types import ExperimentMalfunction
from rsp.utils.data_types_converters_and_validators import extract_resource_occupations
from rsp.utils.data_types_converters_and_validators import verify_schedule_as_resource_occupations
from rsp.utils.global_constants import RELEASE_TIME


def get_sum_running_times_trainruns_dict(trainruns_dict: TrainrunDict):
    return sum([
        agent_path[-1].scheduled_at - agent_path[0].scheduled_at
        for agent_id, agent_path in trainruns_dict.items()])


def get_delay_trainruns_dict(trainruns_dict_schedule: TrainrunDict, trainruns_dict_reschedule: TrainrunDict):
    return sum([
        max(trainruns_dict_reschedule[agent_id][-1].scheduled_at - trainruns_dict_schedule[agent_id][-1].scheduled_at,
            0)
        for agent_id in trainruns_dict_reschedule])


def verify_trainrun_dict_for_schedule_problem(
        schedule_problem: ScheduleProblemDescription,
        trainrun_dict: TrainrunDict,
        expected_malfunction: Optional[ExperimentMalfunction] = None,
        expected_route_dag_constraints: Optional[RouteDAGConstraintsDict] = None
):
    """Verify the consistency of a train run.
    1. ensure train runs are scheduled ascending, the train run is non-circular and respects the train's constant speed.
    2. verify mutual exclusion
    3. check that the paths lead from the desired start and goal
    4. check that the transitions are valid FLATland transitions according to the grid
    5. verify expected malfunction (if given)
    6. verfy freezes are respected (if given)
    Parameters
    ----------
    schedule_problem
    trainrun_dict
    expected_malfunction
    expected_route_dag_constraints
    Returns
    -------
    """
    minimum_runningtime_dict = schedule_problem.minimum_travel_time_dict
    initial_positions = {agent_id: next(get_sources_for_topo(topo)).position for agent_id, topo in schedule_problem.topo_dict.items()}
    initial_directions = {agent_id: next(get_sources_for_topo(topo)).direction for agent_id, topo in schedule_problem.topo_dict.items()}
    targets = {agent_id: next(get_sinks_for_topo(topo)).position for agent_id, topo in schedule_problem.topo_dict.items()}

    # 1. ensure train runs are scheduled ascending, the train run is non-circular and respects the train's constant speed.
    # 2. verify mutual exclusion
    # 3. check that the paths lead from the desired start and goal
    verify_trainrun_dict_simple(initial_directions=initial_directions,
                                initial_positions=initial_positions,
                                minimum_runningtime_dict=minimum_runningtime_dict,
                                targets=targets,
                                trainrun_dict=trainrun_dict)

    # 4. check that the transitions are valid FLATland transitions according to the topo_dict
    _verify_trainruns_4_consistency_with_flatland_moves(topo_dict=schedule_problem.topo_dict, trainrun_dict=trainrun_dict)

    # 5. verify expected malfunction
    if expected_malfunction:
        _verify_trainruns_5_malfunction(expected_malfunction=expected_malfunction,
                                        trainrun_dict=trainrun_dict,
                                        minimum_runningtime_dict=minimum_runningtime_dict)

    # 6. verify freezes are respected
    if expected_route_dag_constraints:
        _verify_trainruns_6_freeze(expected_route_dag_constraints, trainrun_dict)


def verify_trainrun_dict_simple(trainrun_dict: TrainrunDict,
                                minimum_runningtime_dict: Dict[int, int],
                                initial_positions: Dict[int, Tuple[int, int]],
                                initial_directions: Dict[int, int],
                                targets: Dict[int, Tuple[int, int]]):
    """
    1. ensure train runs are scheduled ascending, the train run is non-circular and respects the train's constant speed.
    2. verify mutual exclusion (with hard-coded release time 1)
    3. check that the paths lead from the desired start and goal
    4. check that the trainrun has no cycle (in waypoints)
    Parameters
    ----------
    trainrun_dict
    minimum_runningtime_dict
    initial_positions
    initial_directions
    targets
    """
    # 1. ensure train runs are scheduled ascending, the train run is non-circular and respects the train's constant speed.
    _verify_trainruns_1_path_consistency(trainrun_dict=trainrun_dict, minimum_runningtime_dict=minimum_runningtime_dict)
    # 2. verify mutual exclusion
    _verify_trainruns_2_mutual_exclusion(trainrun_dict)
    # 3. check that the paths lead from the desired start and goal
    _verify_trainruns_3_source_target(trainrun_dict=trainrun_dict,
                                      initial_positions=initial_positions,
                                      initial_directions=initial_directions,
                                      targets=targets)

    for agent_id, trainrun in trainrun_dict.items():
        waypoints = [trainrun_waypoint.waypoint for trainrun_waypoint in trainrun]
        no_cycle = len(waypoints) == len(set(waypoints))
        assert no_cycle, f"cycle detected for agent {agent_id} \nduplicates={set([x for x in waypoints if waypoints.count(x) > 1])}\ntrainrun={trainrun}"


def _verify_trainruns_5_malfunction(expected_malfunction: ExperimentMalfunction, trainrun_dict: TrainrunDict, minimum_runningtime_dict: Dict[int, int]):
    malfunction_agent_path = trainrun_dict[expected_malfunction.agent_id]
    # malfunction must not start before the agent is in the grid
    assert malfunction_agent_path[0].scheduled_at + 1 <= expected_malfunction.time_step
    previous_time = malfunction_agent_path[0].scheduled_at + 1
    agent_minimum_running_time = minimum_runningtime_dict[expected_malfunction.agent_id]
    for waypoint_index, trainrun_waypoint in enumerate(malfunction_agent_path):
        if trainrun_waypoint.scheduled_at > expected_malfunction.time_step:
            lower_bound_for_scheduled_at = previous_time + agent_minimum_running_time + expected_malfunction.malfunction_duration
            assert trainrun_waypoint.scheduled_at >= lower_bound_for_scheduled_at, \
                f"malfunction={expected_malfunction}, " + \
                f"but found {malfunction_agent_path[max(0, waypoint_index - 1)]}{trainrun_waypoint}"
            break


def _verify_trainruns_6_freeze(expected_route_dag_constraints, trainruns_dict):
    for agent_id, route_dag_constraints in expected_route_dag_constraints.items():
        waypoint_dict: Dict[Waypoint, int] = {
            trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at
            for trainrun_waypoint in trainruns_dict[agent_id]
        }

        # is freeze_visit respected?
        for waypoint in route_dag_constraints.freeze_visit:
            assert waypoint in waypoint_dict

        # is freeze_earliest respected?
        for waypoint, scheduled_at in route_dag_constraints.freeze_earliest.items():
            if waypoint in waypoint_dict:
                actual_scheduled_at = waypoint_dict[waypoint]
                assert actual_scheduled_at >= scheduled_at, \
                    f"expected {actual_scheduled_at} <= {scheduled_at} " + \
                    f"for {waypoint} of agent {agent_id}"

        # is freeze_latest respected?
        for waypoint, scheduled_at in route_dag_constraints.freeze_latest.items():
            if waypoint in waypoint_dict:
                actual_scheduled_at = waypoint_dict[waypoint]
                assert actual_scheduled_at <= scheduled_at, \
                    f"expected {actual_scheduled_at} <= {scheduled_at} " + \
                    f"for {waypoint} of agent {agent_id}"

        # is freeze_banned respected?
        for waypoint in route_dag_constraints.freeze_banned:
            assert waypoint not in waypoint_dict


def _verify_trainruns_4_consistency_with_flatland_moves(topo_dict: TopoDict, trainrun_dict):
    for agent_id, trainrun_sparse in trainrun_dict.items():
        previous_trainrun_waypoint: Optional[TrainrunWaypoint] = None
        for trainrun_waypoint in trainrun_sparse:
            if previous_trainrun_waypoint is not None:
                assert (previous_trainrun_waypoint, trainrun_waypoint.waypoint) in topo_dict[agent_id].edges, \
                    f"invalid move for agent {agent_id}: {previous_trainrun_waypoint} -> {trainrun_waypoint}, " \
                    f"expected one of {topo_dict[agent_id].neighbors(previous_trainrun_waypoint)}"


def _verify_trainruns_3_source_target(trainrun_dict: TrainrunDict,
                                      initial_positions: Dict[int, Tuple[int, int]],
                                      initial_directions: Dict[int, int],
                                      targets: Dict[int, Tuple[int, int]]):
    for agent_id, trainrun in trainrun_dict.items():
        initial_trainrun_waypoint = trainrun[0]
        assert initial_trainrun_waypoint.waypoint.position == initial_positions[agent_id], \
            f"agent {agent_id} does not start in expected initial position, found {initial_trainrun_waypoint}, expected {initial_positions[agent_id]}"
        assert initial_trainrun_waypoint.waypoint.direction == initial_directions[agent_id], \
            f"agent {agent_id} does not start in expected initial direction, found {initial_trainrun_waypoint}, expected {initial_directions[agent_id]}"
        # target trainrun waypoint
        final_trainrun_waypoint = trainrun_dict[agent_id][-1]
        assert final_trainrun_waypoint.waypoint.position == targets[agent_id], \
            f"agent {agent_id} does not end in expected target position, found {final_trainrun_waypoint}, expected{targets[agent_id]}"


def _verify_trainruns_2_mutual_exclusion(trainrun_dict: TrainrunDict):
    schedule_as_resource_occupations = extract_resource_occupations(schedule=trainrun_dict, release_time=RELEASE_TIME)
    verify_schedule_as_resource_occupations(schedule_as_resource_occupations=schedule_as_resource_occupations, release_time=RELEASE_TIME)


def _verify_trainruns_1_path_consistency(trainrun_dict: TrainrunDict, minimum_runningtime_dict: Dict[int, int]):
    for agent_id, trainrun_sparse in trainrun_dict.items():
        minimum_running_time_per_cell = minimum_runningtime_dict[agent_id]
        assert minimum_running_time_per_cell >= 1

        previous_trainrun_waypoint: Optional[TrainrunWaypoint] = None
        previous_waypoints = set()
        for trainrun_waypoint in trainrun_sparse:
            # 1.a) ensure schedule is ascending and respects the train's constant speed
            if previous_trainrun_waypoint is not None:
                assert trainrun_waypoint.scheduled_at >= previous_trainrun_waypoint.scheduled_at + minimum_running_time_per_cell, \
                    f"agent {agent_id} inconsistency: to {trainrun_waypoint} " + \
                    f"from {previous_trainrun_waypoint} " + \
                    f"minimum running time={minimum_running_time_per_cell}"
            # 1.b) ensure train run is non-circular
            assert trainrun_waypoint not in previous_waypoints
            previous_trainrun_waypoint = trainrun_waypoint
            previous_waypoints.add(trainrun_waypoint.waypoint)
