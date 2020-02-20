import networkx as nx
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.experiment_solvers.asp.asp_problem_description import ASPProblemDescription
from rsp.experiment_solvers.asp.asp_solve_problem import solve_problem
from rsp.route_dag.generators.route_dag_generator_schedule import _get_freeze_for_scheduling
from rsp.route_dag.route_dag import MAGIC_DIRECTION_FOR_SOURCE_TARGET
from rsp.route_dag.route_dag import ScheduleProblemDescription


def test_costs_forced_rerouting_one_agent():
    """Force re-routing by routing penalty."""
    topo = nx.DiGraph()
    dummy_source = Waypoint(position=(0, 0), direction=MAGIC_DIRECTION_FOR_SOURCE_TARGET)
    dummy_target = Waypoint(position=(3, 1), direction=MAGIC_DIRECTION_FOR_SOURCE_TARGET)

    # first path
    edge_on_first_path = (Waypoint(position=(0, 0), direction=int(Grid4TransitionsEnum.SOUTH)),
                          Waypoint(position=(1, 0), direction=int(Grid4TransitionsEnum.SOUTH)))
    topo.add_edge(dummy_source,
                  Waypoint(position=(0, 0), direction=int(Grid4TransitionsEnum.SOUTH)))
    topo.add_edge(*edge_on_first_path)
    topo.add_edge(Waypoint(position=(1, 0), direction=int(Grid4TransitionsEnum.SOUTH)),
                  Waypoint(position=(2, 0), direction=int(Grid4TransitionsEnum.SOUTH)))
    topo.add_edge(Waypoint(position=(2, 0), direction=int(Grid4TransitionsEnum.SOUTH)),
                  Waypoint(position=(3, 0), direction=int(Grid4TransitionsEnum.SOUTH)))
    topo.add_edge(Waypoint(position=(3, 0), direction=int(Grid4TransitionsEnum.SOUTH)),
                  Waypoint(position=(3, 1), direction=int(Grid4TransitionsEnum.EAST)))
    topo.add_edge(Waypoint(position=(3, 1), direction=int(Grid4TransitionsEnum.EAST)),
                  dummy_target)

    # second path
    topo.add_edge(dummy_source,
                  Waypoint(position=(0, 0), direction=int(Grid4TransitionsEnum.EAST)))
    edge_on_second_path = (Waypoint(position=(0, 0), direction=int(Grid4TransitionsEnum.EAST)),
                           Waypoint(position=(0, 1), direction=int(Grid4TransitionsEnum.EAST)))
    topo.add_edge(*edge_on_second_path)
    topo.add_edge(Waypoint(position=(0, 1), direction=int(Grid4TransitionsEnum.EAST)),
                  Waypoint(position=(1, 1), direction=int(Grid4TransitionsEnum.SOUTH)))
    topo.add_edge(Waypoint(position=(1, 1), direction=int(Grid4TransitionsEnum.SOUTH)),
                  Waypoint(position=(2, 1), direction=int(Grid4TransitionsEnum.SOUTH)))
    topo.add_edge(Waypoint(position=(2, 1), direction=int(Grid4TransitionsEnum.SOUTH)),
                  Waypoint(position=(3, 1), direction=int(Grid4TransitionsEnum.SOUTH)))
    topo.add_edge(Waypoint(position=(3, 1), direction=int(Grid4TransitionsEnum.SOUTH)),
                  dummy_target)

    for penalized_edge in [edge_on_first_path, edge_on_second_path]:
        topo_dict = {0: topo}
        minimum_travel_time_dict = {0: 2}
        latest_arrival = 300

        schedule_problem_description = ScheduleProblemDescription(
            route_dag_constraints_dict=_get_freeze_for_scheduling(
                topo_dict=topo_dict,
                dummy_source_dict={0: dummy_source},
                minimum_travel_time_dict=minimum_travel_time_dict,
                latest_arrival=latest_arrival
            ),
            minimum_travel_time_dict=minimum_travel_time_dict,
            topo_dict=topo_dict,
            max_episode_steps=latest_arrival,
            route_section_penalties={0: {penalized_edge: 1}})

        reschedule_problem: ASPProblemDescription = ASPProblemDescription.factory_rescheduling(
            tc=schedule_problem_description
        )
        solution, _ = solve_problem(problem=reschedule_problem)
        assert solution.optimization_costs == 0
        print(solution.trainruns_dict[0])
        assert penalized_edge[1] not in {
            trainrun_waypoint.waypoint for trainrun_waypoint in
            solution.trainruns_dict[0]
        }


def test_costs_forced_delay_two_agents():
    """Force delay and re-routing because of bottleneck."""
