import pprint

import networkx as nx
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.experiment_solvers.asp.asp_problem_description import ASPProblemDescription
from rsp.experiment_solvers.asp.asp_solve_problem import solve_problem
from rsp.schedule_problem_description.data_types_and_utils import ScheduleProblemDescription
from rsp.schedule_problem_description.route_dag_constraints.route_dag_constraints_schedule import _get_route_dag_constraints_for_scheduling
from rsp.utils.data_types import experiment_freeze_dict_pretty_print
from rsp.utils.global_constants import RELEASE_TIME

_pp = pprint.PrettyPrinter(indent=4)


def test_costs_forced_rerouting_one_agent():
    """Force re-routing by routing penalty."""

    topo, edge_on_first_path, edge_on_second_path, source_waypoint, target_waypoint = _make_topo()

    for index, penalized_edge in enumerate([edge_on_first_path, edge_on_second_path]):
        topo_dict = {0: topo}
        minimum_travel_time_dict = {0: 2}
        latest_arrival = 300

        schedule_problem_description = ScheduleProblemDescription(
            route_dag_constraints_dict={0: _get_route_dag_constraints_for_scheduling(
                topo=topo_dict[0],
                source_waypoint=source_waypoint,
                minimum_travel_time=minimum_travel_time_dict[0],
                latest_arrival=latest_arrival,
            )},
            minimum_travel_time_dict=minimum_travel_time_dict,
            topo_dict=topo_dict,
            max_episode_steps=latest_arrival,
            route_section_penalties={0: {penalized_edge: 1}},
            weight_lateness_seconds=1
        )

        reschedule_problem: ASPProblemDescription = ASPProblemDescription.factory_rescheduling(
            schedule_problem_description=schedule_problem_description
        )
        solution, _ = solve_problem(problem=reschedule_problem)
        print(solution.trainruns_dict[0])
        experiment_freeze_dict_pretty_print(schedule_problem_description.route_dag_constraints_dict)
        assert solution.optimization_costs == 0, f"found {solution.optimization_costs} for test {index}"

        assert penalized_edge[1] not in {
            trainrun_waypoint.waypoint
            for trainrun_waypoint in
            solution.trainruns_dict[0]
        }


def test_costs_forced_delay_one_agent():
    """Force delay."""

    topo, edge_on_first_path, edge_on_second_path, source_waypoint, target_waypoint = _make_topo()

    for penalized_edge in [edge_on_first_path, edge_on_second_path]:
        topo_dict = {0: topo}
        minimum_travel_time_dict = {0: 2}
        latest_arrival = 300

        route_dag_constraints = _get_route_dag_constraints_for_scheduling(
            topo=topo_dict[0],
            source_waypoint=source_waypoint,
            minimum_travel_time=minimum_travel_time_dict[0],
            latest_arrival=latest_arrival)
        forced_delay = 5
        route_dag_constraints.freeze_earliest[source_waypoint] = forced_delay
        schedule_problem_description = ScheduleProblemDescription(
            route_dag_constraints_dict={0: route_dag_constraints},
            minimum_travel_time_dict=minimum_travel_time_dict,
            topo_dict=topo_dict,
            max_episode_steps=latest_arrival,
            route_section_penalties={0: {penalized_edge: 1}},
            weight_lateness_seconds=1
        )

        reschedule_problem: ASPProblemDescription = ASPProblemDescription.factory_rescheduling(
            schedule_problem_description=schedule_problem_description
        )
        solution, asp_solution = solve_problem(problem=reschedule_problem)

        print("####train runs dict")
        print(_pp.pformat(solution.trainruns_dict))

        print("####asp solution lateness")
        print(asp_solution.extract_list_of_lates())

        print("####asp solution penalties")
        print(asp_solution.extract_list_of_active_penalty())

        assert solution.optimization_costs == forced_delay, f"actual={solution.optimization_costs}, expected={forced_delay}"
        print(solution.trainruns_dict[0])
        assert penalized_edge[1] not in {
            trainrun_waypoint.waypoint for trainrun_waypoint in
            solution.trainruns_dict[0]
        }


def test_costs_forced_delay_two_agents():
    """Two trains were schedule to run one after the other on the same path.

    After the delay, they arrive at the same time. They enter by the
    same grid cell / resource, therefore one has to pass after the
    other.
    """
    topo, edge_on_first_path, edge_on_second_path, source_waypoint, target_waypoint = _make_topo()
    for penalized_edge in [edge_on_first_path, edge_on_second_path]:
        topo_dict = {0: topo, 1: topo}
        minimum_travel_time = 3
        minimum_travel_time_dict = {0: minimum_travel_time, 1: minimum_travel_time}
        latest_arrival = 300

        route_dag_constraints_dict = {agent_id: _get_route_dag_constraints_for_scheduling(
            topo=topo,
            source_waypoint=source_waypoint,
            minimum_travel_time=minimum_travel_time_dict[agent_id],
            latest_arrival=latest_arrival)
            for agent_id, topo in topo_dict.items()
        }
        edge_penalty = 5
        schedule_problem_description = ScheduleProblemDescription(
            route_dag_constraints_dict,
            minimum_travel_time_dict=minimum_travel_time_dict,
            topo_dict=topo_dict,
            max_episode_steps=latest_arrival,
            route_section_penalties={0: {penalized_edge: edge_penalty}, 1: {penalized_edge: edge_penalty}},
            weight_lateness_seconds=1
        )

        reschedule_problem: ASPProblemDescription = ASPProblemDescription.factory_rescheduling(
            schedule_problem_description=schedule_problem_description
        )
        solution, asp_solution = solve_problem(problem=reschedule_problem)

        print("####train runs dict")
        print(_pp.pformat(solution.trainruns_dict))

        print("####asp solution lateness")
        print(asp_solution.extract_list_of_lates())

        print("####asp solution penalties")
        print(asp_solution.extract_list_of_active_penalty())

        for agent_id in [0, 1]:
            assert penalized_edge[1] not in {
                trainrun_waypoint.waypoint for trainrun_waypoint in
                solution.trainruns_dict[agent_id]
            }
        #  the expected costs are only the delay (which is minimum_travel_time + 1 for release time)
        expected_costs = minimum_travel_time + RELEASE_TIME
        assert solution.optimization_costs == expected_costs, f"actual costs {solution.optimization_costs}, expected {expected_costs}"
        assert len(asp_solution.extract_list_of_lates()) == expected_costs
        assert len(asp_solution.extract_list_of_active_penalty()) == 0


def test_costs_forced_rerouting_two_agents():
    """Two trains were schedule to run one after the other on the same path."""
    topo1, edge_on_first_path1, _, source_waypoint1, _ = _make_topo2(dummy_offset=55)
    topo2, edge_on_first_path2, _, source_waypoint2, _ = _make_topo2(dummy_offset=56)

    topo_dict = {0: topo1, 1: topo2}
    minimum_travel_time = 3
    minimum_travel_time_dict = {0: minimum_travel_time, 1: minimum_travel_time}
    latest_arrival = 300

    for edge_penalty in range(1, 8):
        route_dag_constraints_dict = {
            agent_id: _get_route_dag_constraints_for_scheduling(
                topo=topo,
                source_waypoint=source_waypoint1 if agent_id == 0 else source_waypoint2,
                minimum_travel_time=minimum_travel_time_dict[agent_id],
                latest_arrival=latest_arrival
            )
            for agent_id, topo in topo_dict.items()}
        schedule_problem_description = ScheduleProblemDescription(
            route_dag_constraints_dict,
            minimum_travel_time_dict=minimum_travel_time_dict,
            topo_dict=topo_dict,
            max_episode_steps=latest_arrival,
            route_section_penalties={0: {edge_on_first_path1: edge_penalty},
                                     1: {edge_on_first_path2: edge_penalty}},
            weight_lateness_seconds=1
        )

        reschedule_problem: ASPProblemDescription = ASPProblemDescription.factory_rescheduling(
            schedule_problem_description=schedule_problem_description
        )
        solution, asp_solution = solve_problem(problem=reschedule_problem)

        print("####train runs dict")
        print(_pp.pformat(solution.trainruns_dict))

        print("####asp solution lateness")
        print(asp_solution.extract_list_of_lates())

        print("####asp solution penalties")
        print(asp_solution.extract_list_of_active_penalty())

        #  divide is the expected costs are only the delay (which is minimum_travel_time + 1 for release time)
        divide = minimum_travel_time + 1
        if edge_penalty < divide:
            # the edge penalty is lower than the divide -> one train will take the penalty
            assert solution.optimization_costs == edge_penalty
            assert len(asp_solution.extract_list_of_active_penalty()) == 1
            assert len(asp_solution.extract_list_of_lates()) == 0
        elif edge_penalty == divide:
            # the edge penalty is lower than the divide -> one train will take the penalty or wait, it's the same
            assert solution.optimization_costs == divide
            assert (
                           len(asp_solution.extract_list_of_active_penalty()) == 0 and
                           len(asp_solution.extract_list_of_lates()) == divide
                   ) or (
                           len(asp_solution.extract_list_of_active_penalty()) == 1 and
                           len(asp_solution.extract_list_of_lates()) == 0
                   )
        elif edge_penalty > divide:
            # the edge penalty is higher -> one train will wait
            assert solution.optimization_costs == divide
            assert len(asp_solution.extract_list_of_active_penalty()) == 0
            assert len(asp_solution.extract_list_of_lates()) == divide


def _make_topo() -> nx.DiGraph:
    """
    (0,0)
     |
    (1,0) -> (1,1)
     |         |
    (2,0)    (2,1)
     |         |
    (3,0)    (3,1)
     |         |
    (4,0) -> (4,1)
               |
             (5,1)


    Returns
    -------

    """
    source_waypoint = Waypoint(position=(0, 0), direction=int(Grid4TransitionsEnum.SOUTH))
    target_waypoint = Waypoint(position=(1 + 4, 1), direction=int(Grid4TransitionsEnum.SOUTH))

    topo = nx.DiGraph()

    # first path
    topo.add_edge(source_waypoint,
                  Waypoint(position=(1 + 0, 0), direction=int(Grid4TransitionsEnum.SOUTH)))
    edge_on_first_path = (Waypoint(position=(1 + 0, 0), direction=int(Grid4TransitionsEnum.SOUTH)),
                          Waypoint(position=(1 + 1, 0), direction=int(Grid4TransitionsEnum.SOUTH)))
    topo.add_edge(*edge_on_first_path)
    topo.add_edge(Waypoint(position=(1 + 1, 0), direction=int(Grid4TransitionsEnum.SOUTH)),
                  Waypoint(position=(1 + 2, 0), direction=int(Grid4TransitionsEnum.SOUTH)))
    topo.add_edge(Waypoint(position=(1 + 2, 0), direction=int(Grid4TransitionsEnum.SOUTH)),
                  Waypoint(position=(1 + 3, 0), direction=int(Grid4TransitionsEnum.SOUTH)))
    topo.add_edge(Waypoint(position=(1 + 3, 0), direction=int(Grid4TransitionsEnum.SOUTH)),
                  Waypoint(position=(1 + 3, 1), direction=int(Grid4TransitionsEnum.EAST)))
    topo.add_edge(Waypoint(position=(1 + 3, 1), direction=int(Grid4TransitionsEnum.EAST)),
                  target_waypoint)
    # second path
    topo.add_edge(source_waypoint,
                  Waypoint(position=(1 + 0, 0), direction=int(Grid4TransitionsEnum.EAST)))
    edge_on_second_path = (Waypoint(position=(1 + 0, 0), direction=int(Grid4TransitionsEnum.EAST)),
                           Waypoint(position=(1 + 0, 1), direction=int(Grid4TransitionsEnum.EAST)))
    topo.add_edge(*edge_on_second_path)
    topo.add_edge(Waypoint(position=(1 + 0, 1), direction=int(Grid4TransitionsEnum.EAST)),
                  Waypoint(position=(1 + 1, 1), direction=int(Grid4TransitionsEnum.SOUTH)))
    topo.add_edge(Waypoint(position=(1 + 1, 1), direction=int(Grid4TransitionsEnum.SOUTH)),
                  Waypoint(position=(1 + 2, 1), direction=int(Grid4TransitionsEnum.SOUTH)))
    topo.add_edge(Waypoint(position=(1 + 2, 1), direction=int(Grid4TransitionsEnum.SOUTH)),
                  Waypoint(position=(1 + 3, 1), direction=int(Grid4TransitionsEnum.SOUTH)))
    topo.add_edge(Waypoint(position=(1 + 3, 1), direction=int(Grid4TransitionsEnum.SOUTH)),
                  target_waypoint)
    return topo, edge_on_first_path, edge_on_second_path, source_waypoint, target_waypoint


def _make_topo2(dummy_offset: int) -> nx.DiGraph:
    """
          X
    (0,0)  (0,1)
     |         |
    (1,0)    (1,1)
     |         |
    (2,0)    (2,1)
     |         |
    (3,0)  (3,1)
         X

    Parameters
    ----------
    dummy_offset
        change the postion of the dummy source and sink to make them not use the same resource.

    Returns
    -------

    """
    source_waypoint = Waypoint(position=(0 + dummy_offset, 0 + dummy_offset), direction=int(Grid4TransitionsEnum.SOUTH))
    target_waypoint = Waypoint(position=(3 + dummy_offset, 0 + dummy_offset), direction=int(Grid4TransitionsEnum.SOUTH))

    topo = nx.DiGraph()

    # first path
    topo.add_edge(source_waypoint,
                  Waypoint(position=(0, 0), direction=int(Grid4TransitionsEnum.SOUTH)))
    edge_on_left_path = (Waypoint(position=(0, 0), direction=int(Grid4TransitionsEnum.SOUTH)),
                         Waypoint(position=(1, 0), direction=int(Grid4TransitionsEnum.SOUTH)))
    topo.add_edge(*edge_on_left_path)
    topo.add_edge(Waypoint(position=(1, 0), direction=int(Grid4TransitionsEnum.SOUTH)),
                  Waypoint(position=(2, 0), direction=int(Grid4TransitionsEnum.SOUTH)))
    topo.add_edge(Waypoint(position=(2, 0), direction=int(Grid4TransitionsEnum.SOUTH)),
                  Waypoint(position=(3, 0), direction=int(Grid4TransitionsEnum.SOUTH)))
    topo.add_edge(Waypoint(position=(3, 0), direction=int(Grid4TransitionsEnum.SOUTH)),
                  target_waypoint)
    # second path
    topo.add_edge(source_waypoint,
                  Waypoint(position=(0, 1), direction=int(Grid4TransitionsEnum.SOUTH)))
    edge_on_right_path = (Waypoint(position=(0, 1), direction=int(Grid4TransitionsEnum.SOUTH)),
                          Waypoint(position=(1, 1), direction=int(Grid4TransitionsEnum.SOUTH)))
    topo.add_edge(*edge_on_right_path)
    topo.add_edge(Waypoint(position=(1, 1), direction=int(Grid4TransitionsEnum.SOUTH)),
                  Waypoint(position=(2, 1), direction=int(Grid4TransitionsEnum.SOUTH)))
    topo.add_edge(Waypoint(position=(2, 1), direction=int(Grid4TransitionsEnum.SOUTH)),
                  Waypoint(position=(3, 1), direction=int(Grid4TransitionsEnum.SOUTH)))
    topo.add_edge(Waypoint(position=(3, 1), direction=int(Grid4TransitionsEnum.SOUTH)),
                  target_waypoint)
    return topo, edge_on_left_path, edge_on_right_path, source_waypoint, target_waypoint
