import pprint
from typing import Dict
from typing import List

import numpy as np
from flatland.envs.malfunction_generators import Malfunction
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.experiment_solvers.asp.asp_problem_description import ASPProblemDescription
from rsp.experiment_solvers.experiment_solver import asp_reschedule_wrapper
from rsp.experiment_solvers.trainrun_utils import get_delay_trainruns_dict
from rsp.experiment_solvers.trainrun_utils import verify_trainrun_dict_for_schedule_problem
from rsp.schedule_problem_description.data_types_and_utils import RouteDAGConstraintsDict
from rsp.schedule_problem_description.data_types_and_utils import ScheduleProblemDescription
from rsp.schedule_problem_description.route_dag_constraints.delta_zero import delta_zero_for_all_agents
from rsp.schedule_problem_description.route_dag_constraints.perfect_oracle import perfect_oracle_for_all_agents
from rsp.schedule_problem_description.route_dag_constraints.propagate import verify_consistency_of_route_dag_constraints_for_agent
from rsp.utils.data_types import ExperimentMalfunction
from rsp.utils.data_types import ExperimentParameters
from rsp.utils.data_types import InfrastructureParameters
from rsp.utils.data_types import ScheduleParameters
from rsp.utils.experiments import create_env_from_experiment_parameters
from rsp.utils.experiments import create_infrastructure_from_rail_env
from rsp.utils.experiments import create_schedule_problem_description_from_instructure
from rsp.utils.experiments import gen_infrastructure

_pp = pprint.PrettyPrinter(indent=4)

test_parameters = ExperimentParameters(
    experiment_id=0,
    grid_id=0,
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
        number_of_shortest_paths_per_agent=10

    ),
    schedule_parameters=ScheduleParameters(
        infra_id=0,
        schedule_id=0,
        asp_seed_value=94,
        number_of_shortest_paths_per_agent_schedule=1
    ),

    earliest_malfunction=20,
    malfunction_duration=20,
    malfunction_agend_id=0,
    weight_route_change=1,
    weight_lateness_seconds=1,
    max_window_size_from_earliest=np.inf
)


# ---------------------------------------------------------------------------------------------------------------------
# Tests full re-scheduling
# ---------------------------------------------------------------------------------------------------------------------
def test_rescheduling_no_bottleneck():
    static_env = create_env_from_experiment_parameters(params=test_parameters.infra_parameters)

    expected_grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 16386, 1025, 1025, 1025, 4608, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16386, 1025, 1025,
                      1025, 4608, 0, 0, 0, 0],
                     [0, 16386, 1025, 5633, 17411, 3089, 1025, 1025, 1025, 1097, 5633, 17411, 1025, 1025, 1025, 1025,
                      1025, 1025, 1025, 5633, 17411, 3089, 1025, 1025, 1025, 1097, 5633, 17411, 1025, 4608],
                     [0, 49186, 1025, 1097, 3089, 5633, 1025, 1025, 1025, 17411, 1097, 3089, 1025, 1025, 1025, 1025,
                      1025, 1025, 1025, 1097, 3089, 5633, 1025, 1025, 1025, 17411, 1097, 3089, 1025, 37408],
                     [0, 32800, 0, 0, 0, 72, 5633, 1025, 17411, 2064, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 5633, 1025,
                      17411, 2064, 0, 0, 0, 32800],
                     [0, 32800, 0, 0, 0, 0, 72, 1025, 2064, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 1025, 2064, 0, 0,
                      0, 0, 32800],
                     [0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800],
                     [0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800],
                     [0, 32872, 4608, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16386,
                      34864],
                     [16386, 34864, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      32800, 32800],
                     [32800, 32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      32800, 32800],
                     [32800, 32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      32800, 32800],
                     [32800, 49186, 2064, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72,
                      37408],
                     [32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      32800],
                     [32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      32800],
                     [32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      32800],
                     [32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      32800],
                     [32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      32800],
                     [72, 33897, 1025, 5633, 17411, 1025, 1025, 1025, 1025, 1025, 5633, 17411, 1025, 1025, 1025, 1025,
                      1025, 1025, 1025, 5633, 17411, 1025, 1025, 1025, 1025, 1025, 5633, 17411, 1025, 34864],
                     [0, 72, 1025, 1097, 3089, 5633, 1025, 1025, 1025, 17411, 1097, 3089, 1025, 1025, 1025, 1025, 1025,
                      1025, 1025, 1097, 3089, 5633, 1025, 1025, 1025, 17411, 1097, 3089, 1025, 2064],
                     [0, 0, 0, 0, 0, 72, 1025, 1025, 1025, 2064, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 1025, 1025, 1025,
                      2064, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
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
            TrainrunWaypoint(scheduled_at=46, waypoint=Waypoint(position=(24, 23), direction=3))],
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
            TrainrunWaypoint(scheduled_at=29, waypoint=Waypoint(position=(7, 23), direction=3))]}

    k = 10
    schedule_problem = schedule_problem = create_schedule_problem_description_from_instructure(
        infrastructure=create_infrastructure_from_rail_env(static_env, k=k),
        number_of_shortest_paths_per_agent_schedule=k)
    verify_trainrun_dict_for_schedule_problem(schedule_problem=schedule_problem, trainrun_dict=fake_schedule)

    fake_malfunction = ExperimentMalfunction(time_step=19, agent_id=0, malfunction_duration=20)

    reschedule_problem_description: ScheduleProblemDescription = delta_zero_for_all_agents(
        malfunction=fake_malfunction,
        schedule_trainruns=fake_schedule,
        minimum_travel_time_dict=schedule_problem.minimum_travel_time_dict,
        topo_dict=schedule_problem.topo_dict,
        latest_arrival=static_env._max_episode_steps
    )
    freeze_dict: RouteDAGConstraintsDict = reschedule_problem_description.route_dag_constraints_dict

    for agent_id, _ in freeze_dict.items():
        verify_consistency_of_route_dag_constraints_for_agent(
            agent_id=agent_id,
            route_dag_constraints=freeze_dict[agent_id],
            topo=schedule_problem.topo_dict[agent_id]
        )

    schedule_problem = schedule_problem = create_schedule_problem_description_from_instructure(
        infrastructure=create_infrastructure_from_rail_env(static_env, k),
        number_of_shortest_paths_per_agent_schedule=k
    )

    full_reschedule_result = asp_reschedule_wrapper(
        reschedule_problem_description=delta_zero_for_all_agents(
            malfunction=fake_malfunction,
            schedule_trainruns=fake_schedule,
            minimum_travel_time_dict=schedule_problem.minimum_travel_time_dict,
            latest_arrival=static_env._max_episode_steps,
            topo_dict=schedule_problem.topo_dict
        ),
        asp_seed_value=94
    )
    full_reschedule_trainruns: TrainrunDict = full_reschedule_result.trainruns_dict

    # agent 0: scheduled arrival was 46, new arrival is 66 -> penalty = 0 (equals malfunction delay)
    # agent 1: scheduled arrival was 29, new arrival is 29 -> penalty = 0
    actual_costs = full_reschedule_result.optimization_costs

    assert actual_costs == 0, f"actual costs {actual_costs}"

    assert full_reschedule_trainruns[0][-1].scheduled_at == 66
    assert full_reschedule_trainruns[1][-1].scheduled_at == 29


def test_rescheduling_bottleneck():
    static_env = create_env_from_experiment_parameters(params=test_parameters.infra_parameters)

    expected_grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 16386, 1025, 1025, 1025, 4608, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16386, 1025, 1025,
                      1025, 4608, 0, 0, 0, 0],
                     [0, 16386, 1025, 5633, 17411, 3089, 1025, 1025, 1025, 1097, 5633, 17411, 1025, 1025, 1025, 1025,
                      1025, 1025, 1025, 5633, 17411, 3089, 1025, 1025, 1025, 1097, 5633, 17411, 1025, 4608],
                     [0, 49186, 1025, 1097, 3089, 5633, 1025, 1025, 1025, 17411, 1097, 3089, 1025, 1025, 1025, 1025,
                      1025, 1025, 1025, 1097, 3089, 5633, 1025, 1025, 1025, 17411, 1097, 3089, 1025, 37408],
                     [0, 32800, 0, 0, 0, 72, 5633, 1025, 17411, 2064, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 5633, 1025,
                      17411, 2064, 0, 0, 0, 32800],
                     [0, 32800, 0, 0, 0, 0, 72, 1025, 2064, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 1025, 2064, 0, 0,
                      0, 0, 32800],
                     [0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800],
                     [0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800],
                     [0, 32872, 4608, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16386,
                      34864],
                     [16386, 34864, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      32800, 32800],
                     [32800, 32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      32800, 32800],
                     [32800, 32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      32800, 32800],
                     [32800, 49186, 2064, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72,
                      37408],
                     [32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      32800],
                     [32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      32800],
                     [32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      32800],
                     [32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      32800],
                     [32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      32800],
                     [72, 33897, 1025, 5633, 17411, 1025, 1025, 1025, 1025, 1025, 5633, 17411, 1025, 1025, 1025, 1025,
                      1025, 1025, 1025, 5633, 17411, 1025, 1025, 1025, 1025, 1025, 5633, 17411, 1025, 34864],
                     [0, 72, 1025, 1097, 3089, 5633, 1025, 1025, 1025, 17411, 1097, 3089, 1025, 1025, 1025, 1025, 1025,
                      1025, 1025, 1097, 3089, 5633, 1025, 1025, 1025, 17411, 1097, 3089, 1025, 2064],
                     [0, 0, 0, 0, 0, 72, 1025, 1025, 1025, 2064, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 1025, 1025, 1025,
                      2064, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
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
            TrainrunWaypoint(scheduled_at=46, waypoint=Waypoint(position=(24, 23), direction=3))],
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
            TrainrunWaypoint(scheduled_at=29, waypoint=Waypoint(position=(7, 23), direction=3))]}
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
            TrainrunWaypoint(scheduled_at=45 + 3, waypoint=Waypoint(position=(24, 23), direction=3))],
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
            TrainrunWaypoint(scheduled_at=49, waypoint=Waypoint(position=(7, 23), direction=3))]}

    # we derive the re-schedule problem from the schedule problem
    k = 10
    infrastructure = create_infrastructure_from_rail_env(static_env, k=k)
    schedule_problem = create_schedule_problem_description_from_instructure(
        infrastructure=infrastructure,
        number_of_shortest_paths_per_agent_schedule=10
    )
    verify_trainrun_dict_for_schedule_problem(schedule_problem=schedule_problem, trainrun_dict=fake_schedule)
    reschedule_problem: ScheduleProblemDescription = delta_zero_for_all_agents(
        malfunction=fake_malfunction,
        schedule_trainruns=fake_schedule,
        minimum_travel_time_dict={agent.handle: int(np.ceil(1 / agent.speed_data['speed']))
                                  for agent in static_env.agents},
        topo_dict=infrastructure.topo_dict,
        latest_arrival=static_env._max_episode_steps
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
        TrainrunWaypoint(scheduled_at=14, waypoint=Waypoint(position=(16, 29), direction=0))
    ]:
        assert trainrun_waypoint.scheduled_at == freeze_dict[1].earliest[trainrun_waypoint.waypoint]
        assert trainrun_waypoint.scheduled_at == freeze_dict[1].latest[trainrun_waypoint.waypoint]

    for trainrun_waypoint in [TrainrunWaypoint(scheduled_at=35, waypoint=Waypoint(position=(15, 29), direction=0))]:
        assert trainrun_waypoint.scheduled_at == freeze_dict[1].earliest[trainrun_waypoint.waypoint]

    for agent_id, _ in freeze_dict.items():
        verify_consistency_of_route_dag_constraints_for_agent(
            agent_id=agent_id,
            route_dag_constraints=freeze_dict[agent_id],
            topo=reschedule_problem.topo_dict[agent_id])

    full_reschedule_result = asp_reschedule_wrapper(
        reschedule_problem_description=reschedule_problem,
        asp_seed_value=94
    )
    full_reschedule_trainruns: Dict[int, List[TrainrunWaypoint]] = full_reschedule_result.trainruns_dict

    assert full_reschedule_trainruns[0][-1].scheduled_at == 48, f"found {full_reschedule_trainruns[0][-1].scheduled_at}"
    assert full_reschedule_trainruns[1][-1].scheduled_at == 49, f"found {full_reschedule_trainruns[1][-1].scheduled_at}"

    # agent 0: scheduled arrival was 46, new arrival is 45 -> penalty = 0 (no negative delay!)
    # agent 1: scheduled arrival was 29, new arrival is 49 -> penalty = 20 = delay
    actual_costs = full_reschedule_result.optimization_costs

    expected_delay_wr_schedule = 22
    assert expected_delay_wr_schedule == get_delay_trainruns_dict(fake_schedule, expected_reschedule)
    print(fake_schedule)
    print(expected_reschedule)
    actual_delay_wr_schedule = get_delay_trainruns_dict(fake_schedule, full_reschedule_trainruns)
    assert actual_delay_wr_schedule == expected_delay_wr_schedule, \
        f"actual delay {actual_delay_wr_schedule}, expected {expected_delay_wr_schedule}"

    # the solver returns the delay with respect to the defined earliest time (which is 20 after the scheduled arrival)
    expected_rerouting_penalty = 1
    expected_costs = expected_delay_wr_schedule - 20 + expected_rerouting_penalty
    assert actual_costs == expected_costs, f"actual costs {actual_costs} from solver, expected {expected_costs}"


# ---------------------------------------------------------------------------------------------------------------------
# Tests delta re-scheduling
# ---------------------------------------------------------------------------------------------------------------------

def test_rescheduling_delta_no_bottleneck():
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
            TrainrunWaypoint(scheduled_at=46, waypoint=Waypoint(position=(24, 23), direction=3))],
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
            TrainrunWaypoint(scheduled_at=29, waypoint=Waypoint(position=(7, 23), direction=3))]}

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
            TrainrunWaypoint(scheduled_at=66, waypoint=Waypoint(position=(24, 23), direction=3))],
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
            TrainrunWaypoint(scheduled_at=29, waypoint=Waypoint(position=(7, 23), direction=3))]}

    # train 0 arrives at 46 (schedule) + 20 delay
    # train 1 arrives at 29 (schedule and re-eschedule)
    # agent 0: scheduled arrival was 46, new arrival is 45 -> penalty = 0 (no negative delay!)
    # agent 1: scheduled arrival was 29, new arrival is 49 -> penalty = 20 = delay
    _verify_rescheduling_delta(fake_malfunction=fake_malfunction, fake_schedule=fake_schedule,
                               fake_full_reschedule_trainruns=full_reschedule_trainruns,
                               expected_arrivals={0: 46 + 20, 1: 29}, expected_delay=20)


def test_rescheduling_delta_bottleneck():
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
            TrainrunWaypoint(scheduled_at=29, waypoint=Waypoint(position=(24, 23), direction=3))],
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
            TrainrunWaypoint(scheduled_at=46, waypoint=Waypoint(position=(7, 23), direction=3))]}
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
            TrainrunWaypoint(scheduled_at=49, waypoint=Waypoint(position=(24, 23), direction=3))],
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
            TrainrunWaypoint(scheduled_at=66, waypoint=Waypoint(position=(7, 23), direction=3))]}

    # train 0 arrives at 29 (schedule) + 20 delay in re-schedule (full and delta) -> 20
    # train 1 arrives at 46 (schedule) 66 (in re-eschedule full and delta) -> 20
    # (it has to wait for the other train to leave a bottleneck in opposite direction
    _verify_rescheduling_delta(fake_malfunction=fake_malfunction,
                               fake_schedule=fake_schedule,
                               fake_full_reschedule_trainruns=fake_full_reschedule_trainruns,
                               expected_arrivals={0: 29 + 20, 1: 46 + 20}, expected_delay=40)


def _verify_rescheduling_delta(fake_malfunction: ExperimentMalfunction,
                               fake_schedule: TrainrunDict,
                               fake_full_reschedule_trainruns: TrainrunDict,
                               expected_arrivals, expected_delay):
    fake_malfunction, schedule_problem = _dummy_test_case(fake_malfunction)
    delta_reschedule_problem: ScheduleProblemDescription = perfect_oracle_for_all_agents(
        full_reschedule_trainrun_dict=fake_full_reschedule_trainruns,
        malfunction=fake_malfunction,
        max_episode_steps=schedule_problem.schedule_problem_description.max_episode_steps,
        schedule_topo_dict=schedule_problem.schedule_problem_description.topo_dict,
        schedule_trainrun_dict=fake_schedule,
        minimum_travel_time_dict=schedule_problem.schedule_problem_description.minimum_travel_time_dict
    )
    delta_reschedule_result = asp_reschedule_wrapper(
        reschedule_problem_description=delta_reschedule_problem,
        asp_seed_value=94
    )
    delta_reschedule_trainruns = delta_reschedule_result.trainruns_dict
    for train, expected_arrival in expected_arrivals.items():
        delta_reschedule_train_arrival = delta_reschedule_trainruns[train][-1]
        assert delta_reschedule_train_arrival.scheduled_at == expected_arrival, \
            f"train {train} found {delta_reschedule_train_arrival.scheduled_at} arrival but expected {expected_arrival}"

    delay_in_delta_reschedule = get_delay_trainruns_dict(fake_schedule, delta_reschedule_trainruns)
    assert delay_in_delta_reschedule == expected_delay, f"found {delay_in_delta_reschedule}, expected={expected_delay}"
    delay_in_full_reschedule = get_delay_trainruns_dict(fake_schedule, fake_full_reschedule_trainruns)
    assert delay_in_full_reschedule == expected_delay, f"found {delay_in_full_reschedule}, expected {expected_delay}"
    asp_costs = delta_reschedule_result.optimization_costs

    # the delta model does not count the malfunction as costs
    expected_asp_costs = expected_delay - fake_malfunction.malfunction_duration
    assert asp_costs == expected_asp_costs, f"found asp_costs={asp_costs}, expected={expected_asp_costs}"


# ---------------------------------------------------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------------------------------------------------


def _dummy_test_case(fake_malfunction: Malfunction):
    schedule_problem = ASPProblemDescription.factory_scheduling(
        schedule_problem_description=create_schedule_problem_description_from_instructure(
            infrastructure=gen_infrastructure(infra_parameters=test_parameters.infra_parameters),
            number_of_shortest_paths_per_agent_schedule=test_parameters.infra_parameters.number_of_shortest_paths_per_agent
        ))

    return fake_malfunction, schedule_problem
