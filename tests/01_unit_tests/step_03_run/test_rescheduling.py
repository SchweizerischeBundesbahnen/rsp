from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from flatland.envs.malfunction_generators import Malfunction
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint
from rsp.scheduling.asp.asp_problem_description import ASPProblemDescription
from rsp.scheduling.asp_wrapper import asp_reschedule_wrapper
from rsp.scheduling.propagate import verify_consistency_of_route_dag_constraints_for_agent
from rsp.scheduling.scheduling_problem import get_sinks_for_topo
from rsp.scheduling.scheduling_problem import get_sources_for_topo
from rsp.scheduling.scheduling_problem import RouteDAGConstraintsDict
from rsp.scheduling.scheduling_problem import ScheduleProblemDescription
from rsp.scheduling.scheduling_problem import TopoDict
from rsp.step_01_planning.experiment_parameters_and_ranges import ExperimentParameters
from rsp.step_01_planning.experiment_parameters_and_ranges import InfrastructureParameters
from rsp.step_01_planning.experiment_parameters_and_ranges import ScheduleParameters
from rsp.step_02_setup.data_types import ExperimentMalfunction
from rsp.step_03_run.experiments import create_env_from_experiment_parameters
from rsp.step_03_run.experiments import create_infrastructure_from_rail_env
from rsp.step_03_run.experiments import create_schedule_problem_description_from_instructure
from rsp.step_03_run.experiments import gen_infrastructure
from rsp.step_03_run.route_dag_constraints.scoper_perfect import scoper_perfect_for_all_agents
from rsp.step_03_run.route_dag_constraints.scoper_zero import delta_zero_for_all_agents
from rsp.utils.global_constants import RELEASE_TIME
from rsp.utils.resource_occupation import extract_resource_occupations
from rsp.utils.resource_occupation import verify_schedule_as_resource_occupations


def verify_trainrun_dict_for_schedule_problem(
    schedule_problem: ScheduleProblemDescription,
    trainrun_dict: TrainrunDict,
    expected_malfunction: Optional[ExperimentMalfunction] = None,
    expected_route_dag_constraints: Optional[RouteDAGConstraintsDict] = None,
):
    """Verify the consistency rules of a train run:
    1. ensure train runs are scheduled ascending, the train run is non-circular and respects the train's constant speed.
    2. verify mutual exclusion
    3. check that the paths lead from the desired start and goal
    4. check that the transitions are valid FLATland transitions according to the grid
    5. verify expected malfunction (if given)
    6. verify freezes are respected (if given)
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
    verify_trainrun_dict_simple(
        initial_directions=initial_directions,
        initial_positions=initial_positions,
        minimum_runningtime_dict=minimum_runningtime_dict,
        targets=targets,
        trainrun_dict=trainrun_dict,
    )

    # 4. check that the transitions are valid transitions according to the topo_dict
    _verify_trainruns_rule_4_consistency_with_topology(topo_dict=schedule_problem.topo_dict, trainrun_dict=trainrun_dict)

    # 5. verify expected malfunction
    if expected_malfunction:
        _verify_trainruns_rule_5_malfunction(
            expected_malfunction=expected_malfunction, trainrun_dict=trainrun_dict, minimum_runningtime_dict=minimum_runningtime_dict
        )

    # 6. verify freezes are respected
    if expected_route_dag_constraints:
        _verify_trainruns_rule_6_freeze(expected_route_dag_constraints, trainrun_dict)


def verify_trainrun_dict_simple(
    trainrun_dict: TrainrunDict,
    minimum_runningtime_dict: Dict[int, int],
    initial_positions: Dict[int, Tuple[int, int]],
    initial_directions: Dict[int, int],
    targets: Dict[int, Tuple[int, int]],
):
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
    _verify_trainruns_rule_1_path_consistency(trainrun_dict=trainrun_dict, minimum_runningtime_dict=minimum_runningtime_dict)
    # 2. verify mutual exclusion
    _verify_trainruns_rule_2_mutual_exclusion(trainrun_dict)
    # 3. check that the paths lead from the desired start and goal
    _verify_trainruns_rule_3_source_target(
        trainrun_dict=trainrun_dict, initial_positions=initial_positions, initial_directions=initial_directions, targets=targets
    )

    for agent_id, trainrun in trainrun_dict.items():
        waypoints = [trainrun_waypoint.waypoint for trainrun_waypoint in trainrun]
        no_cycle = len(waypoints) == len(set(waypoints))
        assert no_cycle, f"cycle detected for agent {agent_id} \nduplicates={set([x for x in waypoints if waypoints.count(x) > 1])}\ntrainrun={trainrun}"


def _verify_trainruns_rule_5_malfunction(expected_malfunction: ExperimentMalfunction, trainrun_dict: TrainrunDict, minimum_runningtime_dict: Dict[int, int]):
    """Train run consistency rule 5: verify expected malfunction (if given)"""
    malfunction_agent_path = trainrun_dict[expected_malfunction.agent_id]
    # malfunction must not start before the agent is in the grid
    assert malfunction_agent_path[0].scheduled_at + 1 <= expected_malfunction.time_step
    previous_time = malfunction_agent_path[0].scheduled_at + 1
    agent_minimum_running_time = minimum_runningtime_dict[expected_malfunction.agent_id]
    for waypoint_index, trainrun_waypoint in enumerate(malfunction_agent_path):
        if trainrun_waypoint.scheduled_at > expected_malfunction.time_step:
            lower_bound_for_scheduled_at = previous_time + agent_minimum_running_time + expected_malfunction.malfunction_duration
            assert trainrun_waypoint.scheduled_at >= lower_bound_for_scheduled_at, (
                f"malfunction={expected_malfunction}, " + f"but found {malfunction_agent_path[max(0, waypoint_index - 1)]}{trainrun_waypoint}"
            )
            break


def _verify_trainruns_rule_6_freeze(expected_route_dag_constraints, trainruns_dict):
    """Train run consistency rule 6: verify freezes are respected (if given)"""
    for agent_id, route_dag_constraints in expected_route_dag_constraints.items():
        waypoint_dict: Dict[Waypoint, int] = {trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at for trainrun_waypoint in trainruns_dict[agent_id]}

        # is earliest respected?
        for waypoint, scheduled_at in route_dag_constraints.earliest.items():
            if waypoint in waypoint_dict:
                actual_scheduled_at = waypoint_dict[waypoint]
                assert actual_scheduled_at >= scheduled_at, f"expected {actual_scheduled_at} <= {scheduled_at} " + f"for {waypoint} of agent {agent_id}"

        # is latest respected?
        for waypoint, scheduled_at in route_dag_constraints.latest.items():
            if waypoint in waypoint_dict:
                actual_scheduled_at = waypoint_dict[waypoint]
                assert actual_scheduled_at <= scheduled_at, f"expected {actual_scheduled_at} <= {scheduled_at} " + f"for {waypoint} of agent {agent_id}"


def _verify_trainruns_rule_4_consistency_with_topology(topo_dict: TopoDict, trainrun_dict):
    """Train run consistency rule 4: check that the transitions are valid
    FLATland transitions according to the topology."""
    for agent_id, trainrun_sparse in trainrun_dict.items():
        previous_trainrun_waypoint: Optional[TrainrunWaypoint] = None
        for trainrun_waypoint in trainrun_sparse:
            if previous_trainrun_waypoint is not None:
                assert (previous_trainrun_waypoint, trainrun_waypoint.waypoint) in topo_dict[agent_id].edges, (
                    f"invalid move for agent {agent_id}: {previous_trainrun_waypoint} -> {trainrun_waypoint}, "
                    f"expected one of {topo_dict[agent_id].neighbors(previous_trainrun_waypoint)}"
                )


def _verify_trainruns_rule_3_source_target(
    trainrun_dict: TrainrunDict, initial_positions: Dict[int, Tuple[int, int]], initial_directions: Dict[int, int], targets: Dict[int, Tuple[int, int]]
):
    """Train run consistency rule 3: check that the paths lead from the desired
    start and goal."""
    for agent_id, trainrun in trainrun_dict.items():
        initial_trainrun_waypoint = trainrun[0]
        assert (
            initial_trainrun_waypoint.waypoint.position == initial_positions[agent_id]
        ), f"agent {agent_id} does not start in expected initial position, found {initial_trainrun_waypoint}, expected {initial_positions[agent_id]}"
        assert (
            initial_trainrun_waypoint.waypoint.direction == initial_directions[agent_id]
        ), f"agent {agent_id} does not start in expected initial direction, found {initial_trainrun_waypoint}, expected {initial_directions[agent_id]}"
        # target trainrun waypoint
        final_trainrun_waypoint = trainrun_dict[agent_id][-1]
        assert (
            final_trainrun_waypoint.waypoint.position == targets[agent_id]
        ), f"agent {agent_id} does not end in expected target position, found {final_trainrun_waypoint}, expected{targets[agent_id]}"


def _verify_trainruns_rule_2_mutual_exclusion(trainrun_dict: TrainrunDict):
    """Train run consistency rule 2: mutual exclusion."""
    schedule_as_resource_occupations = extract_resource_occupations(schedule=trainrun_dict, release_time=RELEASE_TIME)
    verify_schedule_as_resource_occupations(schedule_as_resource_occupations=schedule_as_resource_occupations, release_time=RELEASE_TIME)


def _verify_trainruns_rule_1_path_consistency(trainrun_dict: TrainrunDict, minimum_runningtime_dict: Dict[int, int]):
    """Train run consistency rule 1: ensure train runs are scheduled ascending,
    the train run is non-circular and respects the train's constant speed."""
    for agent_id, trainrun_sparse in trainrun_dict.items():
        minimum_running_time_per_cell = minimum_runningtime_dict[agent_id]
        assert minimum_running_time_per_cell >= 1

        previous_trainrun_waypoint: Optional[TrainrunWaypoint] = None
        previous_waypoints = set()
        for trainrun_waypoint in trainrun_sparse:
            # 1.a) ensure schedule is ascending and respects the train's constant speed
            if previous_trainrun_waypoint is not None:
                assert trainrun_waypoint.scheduled_at >= previous_trainrun_waypoint.scheduled_at + minimum_running_time_per_cell, (
                    f"agent {agent_id} inconsistency: to {trainrun_waypoint} "
                    + f"from {previous_trainrun_waypoint} "
                    + f"minimum running time={minimum_running_time_per_cell}"
                )
            # 1.b) ensure train run is non-circular
            assert trainrun_waypoint not in previous_waypoints
            previous_trainrun_waypoint = trainrun_waypoint
            previous_waypoints.add(trainrun_waypoint.waypoint)


def get_delay_trainruns_dict(trainruns_dict_schedule: TrainrunDict, trainruns_dict_reschedule: TrainrunDict):
    return sum(
        [
            max(trainruns_dict_reschedule[agent_id][-1].scheduled_at - trainruns_dict_schedule[agent_id][-1].scheduled_at, 0)
            for agent_id in trainruns_dict_reschedule
        ]
    )


test_parameters = ExperimentParameters(
    experiment_id=0,
    grid_id=0,
    infra_id_schedule_id=0,
    infra_parameters=InfrastructureParameters(
        infra_id=0,
        width=30,
        height=30,
        number_of_agents=2,
        flatland_seed_value=12,
        max_num_cities=20,
        grid_mode=True,
        max_rail_between_cities=2,
        max_rail_in_city=6,
        speed_data={1: 1.0},
        number_of_shortest_paths_per_agent=10,
    ),
    schedule_parameters=ScheduleParameters(infra_id=0, schedule_id=0, asp_seed_value=94, number_of_shortest_paths_per_agent_schedule=1),
    earliest_malfunction=20,
    malfunction_duration=20,
    malfunction_agent_id=0,
    weight_route_change=1,
    weight_lateness_seconds=1,
    max_window_size_from_earliest=np.inf,
)


# ---------------------------------------------------------------------------------------------------------------------
# Tests full re-scheduling
# ---------------------------------------------------------------------------------------------------------------------
def test_rescheduling_no_bottleneck():
    static_env = create_env_from_experiment_parameters(params=test_parameters.infra_parameters)

    expected_grid = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 16386, 1025, 1025, 1025, 4608, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16386, 1025, 1025, 1025, 4608, 0, 0, 0, 0],
        [
            0,
            16386,
            1025,
            5633,
            17411,
            3089,
            1025,
            1025,
            1025,
            1097,
            5633,
            17411,
            1025,
            1025,
            1025,
            1025,
            1025,
            1025,
            1025,
            5633,
            17411,
            3089,
            1025,
            1025,
            1025,
            1097,
            5633,
            17411,
            1025,
            4608,
        ],
        [
            0,
            49186,
            1025,
            1097,
            3089,
            5633,
            1025,
            1025,
            1025,
            17411,
            1097,
            3089,
            1025,
            1025,
            1025,
            1025,
            1025,
            1025,
            1025,
            1097,
            3089,
            5633,
            1025,
            1025,
            1025,
            17411,
            1097,
            3089,
            1025,
            37408,
        ],
        [0, 32800, 0, 0, 0, 72, 5633, 1025, 17411, 2064, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 5633, 1025, 17411, 2064, 0, 0, 0, 32800],
        [0, 32800, 0, 0, 0, 0, 72, 1025, 2064, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 1025, 2064, 0, 0, 0, 0, 32800],
        [0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800],
        [0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800],
        [0, 32872, 4608, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16386, 34864],
        [16386, 34864, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800, 32800],
        [32800, 32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800, 32800],
        [32800, 32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800, 32800],
        [32800, 49186, 2064, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 37408],
        [32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800],
        [32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800],
        [32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800],
        [32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800],
        [32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800],
        [
            72,
            33897,
            1025,
            5633,
            17411,
            1025,
            1025,
            1025,
            1025,
            1025,
            5633,
            17411,
            1025,
            1025,
            1025,
            1025,
            1025,
            1025,
            1025,
            5633,
            17411,
            1025,
            1025,
            1025,
            1025,
            1025,
            5633,
            17411,
            1025,
            34864,
        ],
        [
            0,
            72,
            1025,
            1097,
            3089,
            5633,
            1025,
            1025,
            1025,
            17411,
            1097,
            3089,
            1025,
            1025,
            1025,
            1025,
            1025,
            1025,
            1025,
            1097,
            3089,
            5633,
            1025,
            1025,
            1025,
            17411,
            1097,
            3089,
            1025,
            2064,
        ],
        [0, 0, 0, 0, 0, 72, 1025, 1025, 1025, 2064, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 1025, 1025, 1025, 2064, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    assert static_env.rail.grid.tolist() == expected_grid

    fake_schedule = {
        0: [
            TrainrunWaypoint(scheduled_at=18, waypoint=Waypoint(position=(8, 23), direction=1)),
            TrainrunWaypoint(scheduled_at=19, waypoint=Waypoint(position=(8, 24), direction=1)),
            TrainrunWaypoint(scheduled_at=20, waypoint=Waypoint(position=(8, 25), direction=1)),
            TrainrunWaypoint(scheduled_at=21, waypoint=Waypoint(position=(8, 26), direction=1)),
            TrainrunWaypoint(scheduled_at=22, waypoint=Waypoint(position=(8, 27), direction=1)),
            TrainrunWaypoint(scheduled_at=23, waypoint=Waypoint(position=(8, 28), direction=1)),
            TrainrunWaypoint(scheduled_at=24, waypoint=Waypoint(position=(8, 29), direction=1)),
            TrainrunWaypoint(scheduled_at=25, waypoint=Waypoint(position=(9, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=26, waypoint=Waypoint(position=(10, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=27, waypoint=Waypoint(position=(11, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=28, waypoint=Waypoint(position=(12, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=29, waypoint=Waypoint(position=(13, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=30, waypoint=Waypoint(position=(14, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=31, waypoint=Waypoint(position=(15, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=32, waypoint=Waypoint(position=(16, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=33, waypoint=Waypoint(position=(17, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=34, waypoint=Waypoint(position=(18, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=35, waypoint=Waypoint(position=(19, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=36, waypoint=Waypoint(position=(20, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=37, waypoint=Waypoint(position=(21, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=38, waypoint=Waypoint(position=(22, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=39, waypoint=Waypoint(position=(23, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=40, waypoint=Waypoint(position=(23, 28), direction=3)),
            TrainrunWaypoint(scheduled_at=41, waypoint=Waypoint(position=(23, 27), direction=3)),
            TrainrunWaypoint(scheduled_at=42, waypoint=Waypoint(position=(24, 27), direction=2)),
            TrainrunWaypoint(scheduled_at=43, waypoint=Waypoint(position=(24, 26), direction=3)),
            TrainrunWaypoint(scheduled_at=44, waypoint=Waypoint(position=(24, 25), direction=3)),
            TrainrunWaypoint(scheduled_at=45, waypoint=Waypoint(position=(24, 24), direction=3)),
            TrainrunWaypoint(scheduled_at=46, waypoint=Waypoint(position=(24, 23), direction=3)),
        ],
        1: [
            TrainrunWaypoint(scheduled_at=1, waypoint=Waypoint(position=(23, 23), direction=1)),
            TrainrunWaypoint(scheduled_at=2, waypoint=Waypoint(position=(23, 24), direction=1)),
            TrainrunWaypoint(scheduled_at=3, waypoint=Waypoint(position=(23, 25), direction=1)),
            TrainrunWaypoint(scheduled_at=4, waypoint=Waypoint(position=(23, 26), direction=1)),
            TrainrunWaypoint(scheduled_at=5, waypoint=Waypoint(position=(23, 27), direction=1)),
            TrainrunWaypoint(scheduled_at=6, waypoint=Waypoint(position=(23, 28), direction=1)),
            TrainrunWaypoint(scheduled_at=7, waypoint=Waypoint(position=(23, 29), direction=1)),
            TrainrunWaypoint(scheduled_at=8, waypoint=Waypoint(position=(22, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=9, waypoint=Waypoint(position=(21, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=10, waypoint=Waypoint(position=(20, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=11, waypoint=Waypoint(position=(19, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=12, waypoint=Waypoint(position=(18, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=13, waypoint=Waypoint(position=(17, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=14, waypoint=Waypoint(position=(16, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=15, waypoint=Waypoint(position=(15, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=16, waypoint=Waypoint(position=(14, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=17, waypoint=Waypoint(position=(13, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=18, waypoint=Waypoint(position=(12, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=19, waypoint=Waypoint(position=(11, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=20, waypoint=Waypoint(position=(10, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=21, waypoint=Waypoint(position=(9, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=22, waypoint=Waypoint(position=(8, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=23, waypoint=Waypoint(position=(7, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=24, waypoint=Waypoint(position=(7, 28), direction=3)),
            TrainrunWaypoint(scheduled_at=25, waypoint=Waypoint(position=(7, 27), direction=3)),
            TrainrunWaypoint(scheduled_at=26, waypoint=Waypoint(position=(7, 26), direction=3)),
            TrainrunWaypoint(scheduled_at=27, waypoint=Waypoint(position=(7, 25), direction=3)),
            TrainrunWaypoint(scheduled_at=28, waypoint=Waypoint(position=(7, 24), direction=3)),
            TrainrunWaypoint(scheduled_at=29, waypoint=Waypoint(position=(7, 23), direction=3)),
        ],
    }

    k = 10
    schedule_problem = schedule_problem = create_schedule_problem_description_from_instructure(
        infrastructure=create_infrastructure_from_rail_env(static_env, k=k), number_of_shortest_paths_per_agent_schedule=k
    )
    verify_trainrun_dict_for_schedule_problem(schedule_problem=schedule_problem, trainrun_dict=fake_schedule)

    fake_malfunction = ExperimentMalfunction(time_step=19, agent_id=0, malfunction_duration=20)

    reschedule_problem_description: ScheduleProblemDescription = delta_zero_for_all_agents(
        malfunction=fake_malfunction,
        schedule_trainruns=fake_schedule,
        minimum_travel_time_dict=schedule_problem.minimum_travel_time_dict,
        topo_dict_=schedule_problem.topo_dict,
        latest_arrival=static_env._max_episode_steps,
        weight_lateness_seconds=1,
        weight_route_change=1,
    )
    freeze_dict: RouteDAGConstraintsDict = reschedule_problem_description.route_dag_constraints_dict

    for agent_id, _ in freeze_dict.items():
        verify_consistency_of_route_dag_constraints_for_agent(
            agent_id=agent_id, route_dag_constraints=freeze_dict[agent_id], topo=schedule_problem.topo_dict[agent_id]
        )

    schedule_problem = schedule_problem = create_schedule_problem_description_from_instructure(
        infrastructure=create_infrastructure_from_rail_env(static_env, k), number_of_shortest_paths_per_agent_schedule=k
    )

    full_reschedule_result = asp_reschedule_wrapper(
        reschedule_problem_description=delta_zero_for_all_agents(
            malfunction=fake_malfunction,
            schedule_trainruns=fake_schedule,
            minimum_travel_time_dict=schedule_problem.minimum_travel_time_dict,
            latest_arrival=static_env._max_episode_steps,
            topo_dict_=schedule_problem.topo_dict,
            weight_lateness_seconds=1,
            weight_route_change=1,
        ),
        schedule=fake_schedule,
        asp_seed_value=94,
    )
    full_reschedule_trainruns: TrainrunDict = full_reschedule_result.trainruns_dict

    # agent 0: scheduled arrival was 46, new arrival is 66 -> penalty = 20 (equals malfunction delay)
    # agent 1: scheduled arrival was 29, new arrival is 29 -> penalty = 0
    actual_costs = full_reschedule_result.optimization_costs

    assert actual_costs == fake_malfunction.malfunction_duration, f"actual costs {actual_costs}"

    assert full_reschedule_trainruns[0][-1].scheduled_at == 66
    assert full_reschedule_trainruns[1][-1].scheduled_at == 29


def test_rescheduling_bottleneck():
    static_env = create_env_from_experiment_parameters(params=test_parameters.infra_parameters)

    expected_grid = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 16386, 1025, 1025, 1025, 4608, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16386, 1025, 1025, 1025, 4608, 0, 0, 0, 0],
        [
            0,
            16386,
            1025,
            5633,
            17411,
            3089,
            1025,
            1025,
            1025,
            1097,
            5633,
            17411,
            1025,
            1025,
            1025,
            1025,
            1025,
            1025,
            1025,
            5633,
            17411,
            3089,
            1025,
            1025,
            1025,
            1097,
            5633,
            17411,
            1025,
            4608,
        ],
        [
            0,
            49186,
            1025,
            1097,
            3089,
            5633,
            1025,
            1025,
            1025,
            17411,
            1097,
            3089,
            1025,
            1025,
            1025,
            1025,
            1025,
            1025,
            1025,
            1097,
            3089,
            5633,
            1025,
            1025,
            1025,
            17411,
            1097,
            3089,
            1025,
            37408,
        ],
        [0, 32800, 0, 0, 0, 72, 5633, 1025, 17411, 2064, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 5633, 1025, 17411, 2064, 0, 0, 0, 32800],
        [0, 32800, 0, 0, 0, 0, 72, 1025, 2064, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 1025, 2064, 0, 0, 0, 0, 32800],
        [0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800],
        [0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800],
        [0, 32872, 4608, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16386, 34864],
        [16386, 34864, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800, 32800],
        [32800, 32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800, 32800],
        [32800, 32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800, 32800],
        [32800, 49186, 2064, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 37408],
        [32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800],
        [32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800],
        [32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800],
        [32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800],
        [32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800],
        [
            72,
            33897,
            1025,
            5633,
            17411,
            1025,
            1025,
            1025,
            1025,
            1025,
            5633,
            17411,
            1025,
            1025,
            1025,
            1025,
            1025,
            1025,
            1025,
            5633,
            17411,
            1025,
            1025,
            1025,
            1025,
            1025,
            5633,
            17411,
            1025,
            34864,
        ],
        [
            0,
            72,
            1025,
            1097,
            3089,
            5633,
            1025,
            1025,
            1025,
            17411,
            1097,
            3089,
            1025,
            1025,
            1025,
            1025,
            1025,
            1025,
            1025,
            1097,
            3089,
            5633,
            1025,
            1025,
            1025,
            17411,
            1097,
            3089,
            1025,
            2064,
        ],
        [0, 0, 0, 0, 0, 72, 1025, 1025, 1025, 2064, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 1025, 1025, 1025, 2064, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    assert static_env.rail.grid.tolist() == expected_grid

    fake_schedule = {
        0: [
            TrainrunWaypoint(scheduled_at=18, waypoint=Waypoint(position=(8, 23), direction=1)),
            TrainrunWaypoint(scheduled_at=19, waypoint=Waypoint(position=(8, 24), direction=1)),
            TrainrunWaypoint(scheduled_at=20, waypoint=Waypoint(position=(8, 25), direction=1)),
            TrainrunWaypoint(scheduled_at=21, waypoint=Waypoint(position=(8, 26), direction=1)),
            TrainrunWaypoint(scheduled_at=22, waypoint=Waypoint(position=(8, 27), direction=1)),
            TrainrunWaypoint(scheduled_at=23, waypoint=Waypoint(position=(8, 28), direction=1)),
            TrainrunWaypoint(scheduled_at=24, waypoint=Waypoint(position=(8, 29), direction=1)),
            TrainrunWaypoint(scheduled_at=25, waypoint=Waypoint(position=(9, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=26, waypoint=Waypoint(position=(10, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=27, waypoint=Waypoint(position=(11, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=28, waypoint=Waypoint(position=(12, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=29, waypoint=Waypoint(position=(13, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=30, waypoint=Waypoint(position=(14, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=31, waypoint=Waypoint(position=(15, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=32, waypoint=Waypoint(position=(16, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=33, waypoint=Waypoint(position=(17, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=34, waypoint=Waypoint(position=(18, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=35, waypoint=Waypoint(position=(19, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=36, waypoint=Waypoint(position=(20, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=37, waypoint=Waypoint(position=(21, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=38, waypoint=Waypoint(position=(22, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=39, waypoint=Waypoint(position=(23, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=40, waypoint=Waypoint(position=(23, 28), direction=3)),
            TrainrunWaypoint(scheduled_at=41, waypoint=Waypoint(position=(23, 27), direction=3)),
            TrainrunWaypoint(scheduled_at=42, waypoint=Waypoint(position=(24, 27), direction=2)),
            TrainrunWaypoint(scheduled_at=43, waypoint=Waypoint(position=(24, 26), direction=3)),
            TrainrunWaypoint(scheduled_at=44, waypoint=Waypoint(position=(24, 25), direction=3)),
            TrainrunWaypoint(scheduled_at=45, waypoint=Waypoint(position=(24, 24), direction=3)),
            TrainrunWaypoint(scheduled_at=46, waypoint=Waypoint(position=(24, 23), direction=3)),
        ],
        1: [
            TrainrunWaypoint(scheduled_at=1, waypoint=Waypoint(position=(23, 23), direction=1)),
            TrainrunWaypoint(scheduled_at=2, waypoint=Waypoint(position=(23, 24), direction=1)),
            TrainrunWaypoint(scheduled_at=3, waypoint=Waypoint(position=(23, 25), direction=1)),
            TrainrunWaypoint(scheduled_at=4, waypoint=Waypoint(position=(23, 26), direction=1)),
            TrainrunWaypoint(scheduled_at=5, waypoint=Waypoint(position=(23, 27), direction=1)),
            TrainrunWaypoint(scheduled_at=6, waypoint=Waypoint(position=(23, 28), direction=1)),
            TrainrunWaypoint(scheduled_at=7, waypoint=Waypoint(position=(23, 29), direction=1)),
            TrainrunWaypoint(scheduled_at=8, waypoint=Waypoint(position=(22, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=9, waypoint=Waypoint(position=(21, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=10, waypoint=Waypoint(position=(20, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=11, waypoint=Waypoint(position=(19, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=12, waypoint=Waypoint(position=(18, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=13, waypoint=Waypoint(position=(17, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=14, waypoint=Waypoint(position=(16, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=15, waypoint=Waypoint(position=(15, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=16, waypoint=Waypoint(position=(14, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=17, waypoint=Waypoint(position=(13, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=18, waypoint=Waypoint(position=(12, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=19, waypoint=Waypoint(position=(11, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=20, waypoint=Waypoint(position=(10, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=21, waypoint=Waypoint(position=(9, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=22, waypoint=Waypoint(position=(8, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=23, waypoint=Waypoint(position=(7, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=24, waypoint=Waypoint(position=(7, 28), direction=3)),
            TrainrunWaypoint(scheduled_at=25, waypoint=Waypoint(position=(7, 27), direction=3)),
            TrainrunWaypoint(scheduled_at=26, waypoint=Waypoint(position=(7, 26), direction=3)),
            TrainrunWaypoint(scheduled_at=27, waypoint=Waypoint(position=(7, 25), direction=3)),
            TrainrunWaypoint(scheduled_at=28, waypoint=Waypoint(position=(7, 24), direction=3)),
            TrainrunWaypoint(scheduled_at=29, waypoint=Waypoint(position=(7, 23), direction=3)),
        ],
    }
    fake_malfunction = ExperimentMalfunction(time_step=14, agent_id=1, malfunction_duration=20)
    expected_reschedule = {
        0: [
            TrainrunWaypoint(scheduled_at=15 + 3, waypoint=Waypoint(position=(8, 23), direction=1)),
            TrainrunWaypoint(scheduled_at=16 + 3, waypoint=Waypoint(position=(8, 24), direction=1)),
            TrainrunWaypoint(scheduled_at=17 + 3, waypoint=Waypoint(position=(8, 25), direction=1)),
            TrainrunWaypoint(scheduled_at=18 + 3, waypoint=Waypoint(position=(8, 26), direction=1)),
            TrainrunWaypoint(scheduled_at=19 + 3, waypoint=Waypoint(position=(8, 27), direction=1)),
            TrainrunWaypoint(scheduled_at=20 + 3, waypoint=Waypoint(position=(8, 28), direction=1)),
            TrainrunWaypoint(scheduled_at=21 + 3, waypoint=Waypoint(position=(8, 29), direction=1)),
            TrainrunWaypoint(scheduled_at=22 + 3, waypoint=Waypoint(position=(9, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=23 + 3, waypoint=Waypoint(position=(10, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=24 + 3, waypoint=Waypoint(position=(11, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=25 + 3, waypoint=Waypoint(position=(12, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=26 + 3, waypoint=Waypoint(position=(13, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=27 + 3, waypoint=Waypoint(position=(13, 28), direction=3)),
            TrainrunWaypoint(scheduled_at=28 + 3, waypoint=Waypoint(position=(14, 28), direction=2)),
            TrainrunWaypoint(scheduled_at=29 + 3, waypoint=Waypoint(position=(15, 28), direction=2)),
            TrainrunWaypoint(scheduled_at=30 + 3, waypoint=Waypoint(position=(16, 28), direction=2)),
            TrainrunWaypoint(scheduled_at=31 + 3, waypoint=Waypoint(position=(17, 28), direction=2)),
            TrainrunWaypoint(scheduled_at=32 + 3, waypoint=Waypoint(position=(17, 29), direction=1)),
            TrainrunWaypoint(scheduled_at=33 + 3, waypoint=Waypoint(position=(18, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=34 + 3, waypoint=Waypoint(position=(19, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=35 + 3, waypoint=Waypoint(position=(20, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=36 + 3, waypoint=Waypoint(position=(21, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=37 + 3, waypoint=Waypoint(position=(22, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=38 + 3, waypoint=Waypoint(position=(23, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=39 + 3, waypoint=Waypoint(position=(24, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=40 + 3, waypoint=Waypoint(position=(24, 28), direction=3)),
            TrainrunWaypoint(scheduled_at=41 + 3, waypoint=Waypoint(position=(24, 27), direction=3)),
            TrainrunWaypoint(scheduled_at=42 + 3, waypoint=Waypoint(position=(24, 26), direction=3)),
            TrainrunWaypoint(scheduled_at=43 + 3, waypoint=Waypoint(position=(24, 25), direction=3)),
            TrainrunWaypoint(scheduled_at=44 + 3, waypoint=Waypoint(position=(24, 24), direction=3)),
            TrainrunWaypoint(scheduled_at=45 + 3, waypoint=Waypoint(position=(24, 23), direction=3)),
        ],
        1: [
            TrainrunWaypoint(scheduled_at=1, waypoint=Waypoint(position=(23, 23), direction=1)),
            TrainrunWaypoint(scheduled_at=2, waypoint=Waypoint(position=(23, 24), direction=1)),
            TrainrunWaypoint(scheduled_at=3, waypoint=Waypoint(position=(23, 25), direction=1)),
            TrainrunWaypoint(scheduled_at=4, waypoint=Waypoint(position=(23, 26), direction=1)),
            TrainrunWaypoint(scheduled_at=5, waypoint=Waypoint(position=(23, 27), direction=1)),
            TrainrunWaypoint(scheduled_at=6, waypoint=Waypoint(position=(23, 28), direction=1)),
            TrainrunWaypoint(scheduled_at=7, waypoint=Waypoint(position=(23, 29), direction=1)),
            TrainrunWaypoint(scheduled_at=8, waypoint=Waypoint(position=(22, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=9, waypoint=Waypoint(position=(21, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=10, waypoint=Waypoint(position=(20, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=11, waypoint=Waypoint(position=(19, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=12, waypoint=Waypoint(position=(18, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=13, waypoint=Waypoint(position=(17, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=14, waypoint=Waypoint(position=(16, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=35, waypoint=Waypoint(position=(15, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=36, waypoint=Waypoint(position=(14, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=37, waypoint=Waypoint(position=(13, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=38, waypoint=Waypoint(position=(12, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=39, waypoint=Waypoint(position=(11, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=40, waypoint=Waypoint(position=(10, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=41, waypoint=Waypoint(position=(9, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=42, waypoint=Waypoint(position=(8, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=43, waypoint=Waypoint(position=(7, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=44, waypoint=Waypoint(position=(7, 28), direction=3)),
            TrainrunWaypoint(scheduled_at=45, waypoint=Waypoint(position=(7, 27), direction=3)),
            TrainrunWaypoint(scheduled_at=46, waypoint=Waypoint(position=(7, 26), direction=3)),
            TrainrunWaypoint(scheduled_at=47, waypoint=Waypoint(position=(7, 25), direction=3)),
            TrainrunWaypoint(scheduled_at=48, waypoint=Waypoint(position=(7, 24), direction=3)),
            TrainrunWaypoint(scheduled_at=49, waypoint=Waypoint(position=(7, 23), direction=3)),
        ],
    }

    # we derive the re-schedule problem from the schedule problem
    k = 10
    infrastructure = create_infrastructure_from_rail_env(static_env, k=k)
    schedule_problem = create_schedule_problem_description_from_instructure(infrastructure=infrastructure, number_of_shortest_paths_per_agent_schedule=10)
    verify_trainrun_dict_for_schedule_problem(schedule_problem=schedule_problem, trainrun_dict=fake_schedule)
    reschedule_problem: ScheduleProblemDescription = delta_zero_for_all_agents(
        malfunction=fake_malfunction,
        schedule_trainruns=fake_schedule,
        minimum_travel_time_dict={agent.handle: int(np.ceil(1 / agent.speed_data["speed"])) for agent in static_env.agents},
        topo_dict_=infrastructure.topo_dict,
        latest_arrival=static_env._max_episode_steps,
        weight_lateness_seconds=1,
        weight_route_change=1,
    )
    freeze_dict: RouteDAGConstraintsDict = reschedule_problem.route_dag_constraints_dict

    for trainrun_waypoint in [
        TrainrunWaypoint(scheduled_at=1, waypoint=Waypoint(position=(23, 23), direction=1)),
        TrainrunWaypoint(scheduled_at=2, waypoint=Waypoint(position=(23, 24), direction=1)),
        TrainrunWaypoint(scheduled_at=3, waypoint=Waypoint(position=(23, 25), direction=1)),
        TrainrunWaypoint(scheduled_at=4, waypoint=Waypoint(position=(23, 26), direction=1)),
        TrainrunWaypoint(scheduled_at=5, waypoint=Waypoint(position=(23, 27), direction=1)),
        TrainrunWaypoint(scheduled_at=6, waypoint=Waypoint(position=(23, 28), direction=1)),
        TrainrunWaypoint(scheduled_at=7, waypoint=Waypoint(position=(23, 29), direction=1)),
        TrainrunWaypoint(scheduled_at=8, waypoint=Waypoint(position=(22, 29), direction=0)),
        TrainrunWaypoint(scheduled_at=9, waypoint=Waypoint(position=(21, 29), direction=0)),
        TrainrunWaypoint(scheduled_at=10, waypoint=Waypoint(position=(20, 29), direction=0)),
        TrainrunWaypoint(scheduled_at=11, waypoint=Waypoint(position=(19, 29), direction=0)),
        TrainrunWaypoint(scheduled_at=12, waypoint=Waypoint(position=(18, 29), direction=0)),
        TrainrunWaypoint(scheduled_at=13, waypoint=Waypoint(position=(17, 29), direction=0)),
        TrainrunWaypoint(scheduled_at=14, waypoint=Waypoint(position=(16, 29), direction=0)),
    ]:
        assert trainrun_waypoint.scheduled_at == freeze_dict[1].earliest[trainrun_waypoint.waypoint]
        assert trainrun_waypoint.scheduled_at == freeze_dict[1].latest[trainrun_waypoint.waypoint]

    for trainrun_waypoint in [TrainrunWaypoint(scheduled_at=35, waypoint=Waypoint(position=(15, 29), direction=0))]:
        assert trainrun_waypoint.scheduled_at == freeze_dict[1].earliest[trainrun_waypoint.waypoint]

    for agent_id, _ in freeze_dict.items():
        verify_consistency_of_route_dag_constraints_for_agent(
            agent_id=agent_id, route_dag_constraints=freeze_dict[agent_id], topo=reschedule_problem.topo_dict[agent_id]
        )

    full_reschedule_result = asp_reschedule_wrapper(reschedule_problem_description=reschedule_problem, schedule=fake_schedule, asp_seed_value=94)
    full_reschedule_trainruns: Dict[int, List[TrainrunWaypoint]] = full_reschedule_result.trainruns_dict

    assert full_reschedule_trainruns[0][-1].scheduled_at == 48, f"found {full_reschedule_trainruns[0][-1].scheduled_at}"
    assert full_reschedule_trainruns[1][-1].scheduled_at == 49, f"found {full_reschedule_trainruns[1][-1].scheduled_at}"

    # agent 0: scheduled arrival was 46, new arrival is 48 -> penalty = 2
    # agent 1: scheduled arrival was 29, new arrival is 49 -> penalty = 20 = delay
    actual_costs = full_reschedule_result.optimization_costs

    expected_delay_wr_schedule = 22
    assert expected_delay_wr_schedule == get_delay_trainruns_dict(fake_schedule, expected_reschedule)
    print(fake_schedule)
    print(expected_reschedule)
    actual_delay_wr_schedule = get_delay_trainruns_dict(fake_schedule, full_reschedule_trainruns)
    assert actual_delay_wr_schedule == expected_delay_wr_schedule, f"actual delay {actual_delay_wr_schedule}, expected {expected_delay_wr_schedule}"

    expected_rerouting_penalty = 1
    expected_costs = expected_delay_wr_schedule + expected_rerouting_penalty
    assert actual_costs == expected_costs, f"actual costs {actual_costs} from solver, expected {expected_costs}"


# ---------------------------------------------------------------------------------------------------------------------
# Tests re-scheduling
# ---------------------------------------------------------------------------------------------------------------------


def test_rescheduling_delta_perfect_no_bottleneck():
    """Train 1 has already passed the bottlneck when train 0 gets stuck in
    malfunction."""
    fake_malfunction = ExperimentMalfunction(time_step=19, agent_id=0, malfunction_duration=20)

    fake_schedule = {
        0: [
            TrainrunWaypoint(scheduled_at=18, waypoint=Waypoint(position=(8, 23), direction=1)),
            TrainrunWaypoint(scheduled_at=19, waypoint=Waypoint(position=(8, 24), direction=1)),
            TrainrunWaypoint(scheduled_at=20, waypoint=Waypoint(position=(8, 25), direction=1)),
            TrainrunWaypoint(scheduled_at=21, waypoint=Waypoint(position=(8, 26), direction=1)),
            TrainrunWaypoint(scheduled_at=22, waypoint=Waypoint(position=(8, 27), direction=1)),
            TrainrunWaypoint(scheduled_at=23, waypoint=Waypoint(position=(8, 28), direction=1)),
            TrainrunWaypoint(scheduled_at=24, waypoint=Waypoint(position=(8, 29), direction=1)),
            TrainrunWaypoint(scheduled_at=25, waypoint=Waypoint(position=(9, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=26, waypoint=Waypoint(position=(10, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=27, waypoint=Waypoint(position=(11, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=28, waypoint=Waypoint(position=(12, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=29, waypoint=Waypoint(position=(13, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=30, waypoint=Waypoint(position=(14, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=31, waypoint=Waypoint(position=(15, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=32, waypoint=Waypoint(position=(16, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=33, waypoint=Waypoint(position=(17, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=34, waypoint=Waypoint(position=(18, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=35, waypoint=Waypoint(position=(19, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=36, waypoint=Waypoint(position=(20, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=37, waypoint=Waypoint(position=(21, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=38, waypoint=Waypoint(position=(22, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=39, waypoint=Waypoint(position=(23, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=40, waypoint=Waypoint(position=(23, 28), direction=3)),
            TrainrunWaypoint(scheduled_at=41, waypoint=Waypoint(position=(23, 27), direction=3)),
            TrainrunWaypoint(scheduled_at=42, waypoint=Waypoint(position=(24, 27), direction=2)),
            TrainrunWaypoint(scheduled_at=43, waypoint=Waypoint(position=(24, 26), direction=3)),
            TrainrunWaypoint(scheduled_at=44, waypoint=Waypoint(position=(24, 25), direction=3)),
            TrainrunWaypoint(scheduled_at=45, waypoint=Waypoint(position=(24, 24), direction=3)),
            TrainrunWaypoint(scheduled_at=46, waypoint=Waypoint(position=(24, 23), direction=3)),
        ],
        1: [
            TrainrunWaypoint(scheduled_at=1, waypoint=Waypoint(position=(23, 23), direction=1)),
            TrainrunWaypoint(scheduled_at=2, waypoint=Waypoint(position=(23, 24), direction=1)),
            TrainrunWaypoint(scheduled_at=3, waypoint=Waypoint(position=(23, 25), direction=1)),
            TrainrunWaypoint(scheduled_at=4, waypoint=Waypoint(position=(23, 26), direction=1)),
            TrainrunWaypoint(scheduled_at=5, waypoint=Waypoint(position=(23, 27), direction=1)),
            TrainrunWaypoint(scheduled_at=6, waypoint=Waypoint(position=(23, 28), direction=1)),
            TrainrunWaypoint(scheduled_at=7, waypoint=Waypoint(position=(23, 29), direction=1)),
            TrainrunWaypoint(scheduled_at=8, waypoint=Waypoint(position=(22, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=9, waypoint=Waypoint(position=(21, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=10, waypoint=Waypoint(position=(20, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=11, waypoint=Waypoint(position=(19, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=12, waypoint=Waypoint(position=(18, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=13, waypoint=Waypoint(position=(17, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=14, waypoint=Waypoint(position=(16, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=15, waypoint=Waypoint(position=(15, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=16, waypoint=Waypoint(position=(14, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=17, waypoint=Waypoint(position=(13, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=18, waypoint=Waypoint(position=(12, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=19, waypoint=Waypoint(position=(11, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=20, waypoint=Waypoint(position=(10, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=21, waypoint=Waypoint(position=(9, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=22, waypoint=Waypoint(position=(8, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=23, waypoint=Waypoint(position=(7, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=24, waypoint=Waypoint(position=(7, 28), direction=3)),
            TrainrunWaypoint(scheduled_at=25, waypoint=Waypoint(position=(7, 27), direction=3)),
            TrainrunWaypoint(scheduled_at=26, waypoint=Waypoint(position=(7, 26), direction=3)),
            TrainrunWaypoint(scheduled_at=27, waypoint=Waypoint(position=(7, 25), direction=3)),
            TrainrunWaypoint(scheduled_at=28, waypoint=Waypoint(position=(7, 24), direction=3)),
            TrainrunWaypoint(scheduled_at=29, waypoint=Waypoint(position=(7, 23), direction=3)),
        ],
    }

    full_reschedule_trainruns = {
        0: [
            TrainrunWaypoint(scheduled_at=18, waypoint=Waypoint(position=(8, 23), direction=1)),
            TrainrunWaypoint(scheduled_at=19, waypoint=Waypoint(position=(8, 24), direction=1)),
            TrainrunWaypoint(scheduled_at=40, waypoint=Waypoint(position=(8, 25), direction=1)),
            TrainrunWaypoint(scheduled_at=41, waypoint=Waypoint(position=(8, 26), direction=1)),
            TrainrunWaypoint(scheduled_at=42, waypoint=Waypoint(position=(8, 27), direction=1)),
            TrainrunWaypoint(scheduled_at=43, waypoint=Waypoint(position=(8, 28), direction=1)),
            TrainrunWaypoint(scheduled_at=44, waypoint=Waypoint(position=(8, 29), direction=1)),
            TrainrunWaypoint(scheduled_at=45, waypoint=Waypoint(position=(9, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=46, waypoint=Waypoint(position=(10, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=47, waypoint=Waypoint(position=(11, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=48, waypoint=Waypoint(position=(12, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=49, waypoint=Waypoint(position=(13, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=50, waypoint=Waypoint(position=(14, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=51, waypoint=Waypoint(position=(15, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=52, waypoint=Waypoint(position=(16, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=53, waypoint=Waypoint(position=(17, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=54, waypoint=Waypoint(position=(18, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=55, waypoint=Waypoint(position=(19, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=56, waypoint=Waypoint(position=(20, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=57, waypoint=Waypoint(position=(21, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=58, waypoint=Waypoint(position=(22, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=59, waypoint=Waypoint(position=(23, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=60, waypoint=Waypoint(position=(23, 28), direction=3)),
            TrainrunWaypoint(scheduled_at=61, waypoint=Waypoint(position=(23, 27), direction=3)),
            TrainrunWaypoint(scheduled_at=62, waypoint=Waypoint(position=(24, 27), direction=2)),
            TrainrunWaypoint(scheduled_at=63, waypoint=Waypoint(position=(24, 26), direction=3)),
            TrainrunWaypoint(scheduled_at=64, waypoint=Waypoint(position=(24, 25), direction=3)),
            TrainrunWaypoint(scheduled_at=65, waypoint=Waypoint(position=(24, 24), direction=3)),
            TrainrunWaypoint(scheduled_at=66, waypoint=Waypoint(position=(24, 23), direction=3)),
        ],
        1: [
            TrainrunWaypoint(scheduled_at=1, waypoint=Waypoint(position=(23, 23), direction=1)),
            TrainrunWaypoint(scheduled_at=2, waypoint=Waypoint(position=(23, 24), direction=1)),
            TrainrunWaypoint(scheduled_at=3, waypoint=Waypoint(position=(23, 25), direction=1)),
            TrainrunWaypoint(scheduled_at=4, waypoint=Waypoint(position=(23, 26), direction=1)),
            TrainrunWaypoint(scheduled_at=5, waypoint=Waypoint(position=(23, 27), direction=1)),
            TrainrunWaypoint(scheduled_at=6, waypoint=Waypoint(position=(23, 28), direction=1)),
            TrainrunWaypoint(scheduled_at=7, waypoint=Waypoint(position=(23, 29), direction=1)),
            TrainrunWaypoint(scheduled_at=8, waypoint=Waypoint(position=(22, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=9, waypoint=Waypoint(position=(21, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=10, waypoint=Waypoint(position=(20, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=11, waypoint=Waypoint(position=(19, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=12, waypoint=Waypoint(position=(18, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=13, waypoint=Waypoint(position=(17, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=14, waypoint=Waypoint(position=(16, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=15, waypoint=Waypoint(position=(15, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=16, waypoint=Waypoint(position=(14, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=17, waypoint=Waypoint(position=(13, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=18, waypoint=Waypoint(position=(12, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=19, waypoint=Waypoint(position=(11, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=20, waypoint=Waypoint(position=(10, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=21, waypoint=Waypoint(position=(9, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=22, waypoint=Waypoint(position=(8, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=23, waypoint=Waypoint(position=(7, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=24, waypoint=Waypoint(position=(7, 28), direction=3)),
            TrainrunWaypoint(scheduled_at=25, waypoint=Waypoint(position=(7, 27), direction=3)),
            TrainrunWaypoint(scheduled_at=26, waypoint=Waypoint(position=(7, 26), direction=3)),
            TrainrunWaypoint(scheduled_at=27, waypoint=Waypoint(position=(7, 25), direction=3)),
            TrainrunWaypoint(scheduled_at=28, waypoint=Waypoint(position=(7, 24), direction=3)),
            TrainrunWaypoint(scheduled_at=29, waypoint=Waypoint(position=(7, 23), direction=3)),
        ],
    }

    # train 0 arrives at 46 (schedule) + 20 delay
    # train 1 arrives at 29 (schedule and re-eschedule)
    # agent 0: scheduled arrival was 46, new arrival is 45 -> penalty = 0 (no negative delay!)
    # agent 1: scheduled arrival was 29, new arrival is 49 -> penalty = 20 = delay
    _verify_rescheduling_delta_perfect(
        fake_malfunction=fake_malfunction,
        fake_schedule=fake_schedule,
        fake_full_reschedule_trainruns=full_reschedule_trainruns,
        expected_arrivals={0: 46 + 20, 1: 29},
        expected_delay=20,
    )


def test_rescheduling_delta_perfect_bottleneck():
    """Train 0 get's stuck in a bottlneck.

    Train 1 runs in opposite direction -> has to wait.
    """
    fake_malfunction = ExperimentMalfunction(time_step=19, agent_id=0, malfunction_duration=20)

    fake_schedule = {
        0: [
            TrainrunWaypoint(scheduled_at=1, waypoint=Waypoint(position=(8, 23), direction=1)),
            TrainrunWaypoint(scheduled_at=2, waypoint=Waypoint(position=(8, 24), direction=1)),
            TrainrunWaypoint(scheduled_at=3, waypoint=Waypoint(position=(8, 25), direction=1)),
            TrainrunWaypoint(scheduled_at=4, waypoint=Waypoint(position=(8, 26), direction=1)),
            TrainrunWaypoint(scheduled_at=5, waypoint=Waypoint(position=(8, 27), direction=1)),
            TrainrunWaypoint(scheduled_at=6, waypoint=Waypoint(position=(8, 28), direction=1)),
            TrainrunWaypoint(scheduled_at=7, waypoint=Waypoint(position=(8, 29), direction=1)),
            TrainrunWaypoint(scheduled_at=8, waypoint=Waypoint(position=(9, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=9, waypoint=Waypoint(position=(10, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=10, waypoint=Waypoint(position=(11, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=11, waypoint=Waypoint(position=(12, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=12, waypoint=Waypoint(position=(13, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=13, waypoint=Waypoint(position=(14, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=14, waypoint=Waypoint(position=(15, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=15, waypoint=Waypoint(position=(16, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=16, waypoint=Waypoint(position=(17, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=17, waypoint=Waypoint(position=(18, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=18, waypoint=Waypoint(position=(19, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=19, waypoint=Waypoint(position=(20, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=20, waypoint=Waypoint(position=(21, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=21, waypoint=Waypoint(position=(22, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=22, waypoint=Waypoint(position=(23, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=23, waypoint=Waypoint(position=(24, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=24, waypoint=Waypoint(position=(24, 28), direction=3)),
            TrainrunWaypoint(scheduled_at=25, waypoint=Waypoint(position=(24, 27), direction=3)),
            TrainrunWaypoint(scheduled_at=26, waypoint=Waypoint(position=(24, 26), direction=3)),
            TrainrunWaypoint(scheduled_at=27, waypoint=Waypoint(position=(24, 25), direction=3)),
            TrainrunWaypoint(scheduled_at=28, waypoint=Waypoint(position=(24, 24), direction=3)),
            TrainrunWaypoint(scheduled_at=29, waypoint=Waypoint(position=(24, 23), direction=3)),
        ],
        1: [
            TrainrunWaypoint(scheduled_at=18, waypoint=Waypoint(position=(23, 23), direction=1)),
            TrainrunWaypoint(scheduled_at=19, waypoint=Waypoint(position=(23, 24), direction=1)),
            TrainrunWaypoint(scheduled_at=20, waypoint=Waypoint(position=(23, 25), direction=1)),
            TrainrunWaypoint(scheduled_at=21, waypoint=Waypoint(position=(23, 26), direction=1)),
            TrainrunWaypoint(scheduled_at=22, waypoint=Waypoint(position=(23, 27), direction=1)),
            TrainrunWaypoint(scheduled_at=23, waypoint=Waypoint(position=(23, 28), direction=1)),
            TrainrunWaypoint(scheduled_at=24, waypoint=Waypoint(position=(23, 29), direction=1)),
            TrainrunWaypoint(scheduled_at=25, waypoint=Waypoint(position=(22, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=26, waypoint=Waypoint(position=(21, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=27, waypoint=Waypoint(position=(20, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=28, waypoint=Waypoint(position=(19, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=29, waypoint=Waypoint(position=(18, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=30, waypoint=Waypoint(position=(17, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=31, waypoint=Waypoint(position=(16, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=32, waypoint=Waypoint(position=(15, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=33, waypoint=Waypoint(position=(14, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=34, waypoint=Waypoint(position=(13, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=35, waypoint=Waypoint(position=(12, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=36, waypoint=Waypoint(position=(11, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=37, waypoint=Waypoint(position=(10, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=38, waypoint=Waypoint(position=(9, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=39, waypoint=Waypoint(position=(8, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=40, waypoint=Waypoint(position=(8, 28), direction=3)),
            TrainrunWaypoint(scheduled_at=41, waypoint=Waypoint(position=(8, 27), direction=3)),
            TrainrunWaypoint(scheduled_at=42, waypoint=Waypoint(position=(8, 26), direction=3)),
            TrainrunWaypoint(scheduled_at=43, waypoint=Waypoint(position=(7, 26), direction=0)),
            TrainrunWaypoint(scheduled_at=44, waypoint=Waypoint(position=(7, 25), direction=3)),
            TrainrunWaypoint(scheduled_at=45, waypoint=Waypoint(position=(7, 24), direction=3)),
            TrainrunWaypoint(scheduled_at=46, waypoint=Waypoint(position=(7, 23), direction=3)),
        ],
    }
    fake_full_reschedule_trainruns = {
        0: [
            TrainrunWaypoint(scheduled_at=1, waypoint=Waypoint(position=(8, 23), direction=1)),
            TrainrunWaypoint(scheduled_at=2, waypoint=Waypoint(position=(8, 24), direction=1)),
            TrainrunWaypoint(scheduled_at=3, waypoint=Waypoint(position=(8, 25), direction=1)),
            TrainrunWaypoint(scheduled_at=4, waypoint=Waypoint(position=(8, 26), direction=1)),
            TrainrunWaypoint(scheduled_at=5, waypoint=Waypoint(position=(8, 27), direction=1)),
            TrainrunWaypoint(scheduled_at=6, waypoint=Waypoint(position=(8, 28), direction=1)),
            TrainrunWaypoint(scheduled_at=7, waypoint=Waypoint(position=(8, 29), direction=1)),
            TrainrunWaypoint(scheduled_at=8, waypoint=Waypoint(position=(9, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=9, waypoint=Waypoint(position=(10, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=10, waypoint=Waypoint(position=(11, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=11, waypoint=Waypoint(position=(12, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=12, waypoint=Waypoint(position=(13, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=13, waypoint=Waypoint(position=(14, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=14, waypoint=Waypoint(position=(15, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=15, waypoint=Waypoint(position=(16, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=16, waypoint=Waypoint(position=(17, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=17, waypoint=Waypoint(position=(18, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=18, waypoint=Waypoint(position=(19, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=19, waypoint=Waypoint(position=(20, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=40, waypoint=Waypoint(position=(21, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=41, waypoint=Waypoint(position=(22, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=42, waypoint=Waypoint(position=(23, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=43, waypoint=Waypoint(position=(24, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=44, waypoint=Waypoint(position=(24, 28), direction=3)),
            TrainrunWaypoint(scheduled_at=45, waypoint=Waypoint(position=(24, 27), direction=3)),
            TrainrunWaypoint(scheduled_at=46, waypoint=Waypoint(position=(24, 26), direction=3)),
            TrainrunWaypoint(scheduled_at=47, waypoint=Waypoint(position=(24, 25), direction=3)),
            TrainrunWaypoint(scheduled_at=48, waypoint=Waypoint(position=(24, 24), direction=3)),
            TrainrunWaypoint(scheduled_at=49, waypoint=Waypoint(position=(24, 23), direction=3)),
        ],
        1: [
            TrainrunWaypoint(scheduled_at=18, waypoint=Waypoint(position=(23, 23), direction=1)),
            TrainrunWaypoint(scheduled_at=19, waypoint=Waypoint(position=(23, 24), direction=1)),
            TrainrunWaypoint(scheduled_at=20, waypoint=Waypoint(position=(23, 25), direction=1)),
            TrainrunWaypoint(scheduled_at=21, waypoint=Waypoint(position=(23, 26), direction=1)),
            TrainrunWaypoint(scheduled_at=22, waypoint=Waypoint(position=(23, 27), direction=1)),
            TrainrunWaypoint(scheduled_at=23, waypoint=Waypoint(position=(23, 28), direction=1)),
            TrainrunWaypoint(scheduled_at=44, waypoint=Waypoint(position=(23, 29), direction=1)),
            TrainrunWaypoint(scheduled_at=45, waypoint=Waypoint(position=(22, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=46, waypoint=Waypoint(position=(21, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=47, waypoint=Waypoint(position=(20, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=48, waypoint=Waypoint(position=(19, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=49, waypoint=Waypoint(position=(18, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=50, waypoint=Waypoint(position=(17, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=51, waypoint=Waypoint(position=(16, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=52, waypoint=Waypoint(position=(15, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=53, waypoint=Waypoint(position=(14, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=54, waypoint=Waypoint(position=(13, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=55, waypoint=Waypoint(position=(12, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=56, waypoint=Waypoint(position=(11, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=57, waypoint=Waypoint(position=(10, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=58, waypoint=Waypoint(position=(9, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=59, waypoint=Waypoint(position=(8, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=60, waypoint=Waypoint(position=(8, 28), direction=3)),
            TrainrunWaypoint(scheduled_at=61, waypoint=Waypoint(position=(8, 27), direction=3)),
            TrainrunWaypoint(scheduled_at=62, waypoint=Waypoint(position=(8, 26), direction=3)),
            TrainrunWaypoint(scheduled_at=63, waypoint=Waypoint(position=(7, 26), direction=0)),
            TrainrunWaypoint(scheduled_at=64, waypoint=Waypoint(position=(7, 25), direction=3)),
            TrainrunWaypoint(scheduled_at=65, waypoint=Waypoint(position=(7, 24), direction=3)),
            TrainrunWaypoint(scheduled_at=66, waypoint=Waypoint(position=(7, 23), direction=3)),
        ],
    }

    # train 0 arrives at 29 (schedule) + 20 delay in re-schedule (full and delta perfect) -> 20
    # train 1 arrives at 46 (schedule) 66 (in re-eschedule full and delta perfect) -> 20
    # (it has to wait for the other train to leave a bottleneck in opposite direction
    _verify_rescheduling_delta_perfect(
        fake_malfunction=fake_malfunction,
        fake_schedule=fake_schedule,
        fake_full_reschedule_trainruns=fake_full_reschedule_trainruns,
        expected_arrivals={0: 29 + 20, 1: 46 + 20},
        expected_delay=40,
    )


def _verify_rescheduling_delta_perfect(
    fake_malfunction: ExperimentMalfunction, fake_schedule: TrainrunDict, fake_full_reschedule_trainruns: TrainrunDict, expected_arrivals, expected_delay
):
    fake_malfunction, schedule_problem = _dummy_test_case(fake_malfunction)
    delta_perfect_reschedule_problem: ScheduleProblemDescription = scoper_perfect_for_all_agents(
        full_reschedule_trainrun_dict=fake_full_reschedule_trainruns,
        malfunction=fake_malfunction,
        max_episode_steps=schedule_problem.schedule_problem_description.max_episode_steps,
        delta_perfect_reschedule_topo_dict_=schedule_problem.schedule_problem_description.topo_dict,
        schedule_trainrun_dict=fake_schedule,
        minimum_travel_time_dict=schedule_problem.schedule_problem_description.minimum_travel_time_dict,
        weight_lateness_seconds=1,
        weight_route_change=1,
    )
    delta_perfect_reschedule_result = asp_reschedule_wrapper(
        reschedule_problem_description=delta_perfect_reschedule_problem, schedule=fake_schedule, asp_seed_value=94
    )
    delta_perfect_reschedule_trainruns = delta_perfect_reschedule_result.trainruns_dict
    for train, expected_arrival in expected_arrivals.items():
        delta_perfect_reschedule_train_arrival = delta_perfect_reschedule_trainruns[train][-1]
        assert (
            delta_perfect_reschedule_train_arrival.scheduled_at == expected_arrival
        ), f"train {train} found {delta_perfect_reschedule_train_arrival.scheduled_at} arrival but expected {expected_arrival}"

    delay_in_delta_perfect_reschedule = get_delay_trainruns_dict(fake_schedule, delta_perfect_reschedule_trainruns)
    assert delay_in_delta_perfect_reschedule == expected_delay, f"found {delay_in_delta_perfect_reschedule}, expected={expected_delay}"
    delay_in_full_reschedule = get_delay_trainruns_dict(fake_schedule, fake_full_reschedule_trainruns)
    assert delay_in_full_reschedule == expected_delay, f"found {delay_in_full_reschedule}, expected {expected_delay}"
    asp_costs = delta_perfect_reschedule_result.optimization_costs

    expected_asp_costs = expected_delay
    assert asp_costs == expected_asp_costs, f"found asp_costs={asp_costs}, expected={expected_asp_costs}"


# ---------------------------------------------------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------------------------------------------------


def _dummy_test_case(fake_malfunction: Malfunction):
    schedule_problem = ASPProblemDescription.factory_scheduling(
        schedule_problem_description=create_schedule_problem_description_from_instructure(
            infrastructure=gen_infrastructure(infra_parameters=test_parameters.infra_parameters),
            number_of_shortest_paths_per_agent_schedule=test_parameters.infra_parameters.number_of_shortest_paths_per_agent,
        )
    )

    return fake_malfunction, schedule_problem
