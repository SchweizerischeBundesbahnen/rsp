import pprint

import networkx as nx
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.experiment_solvers.asp.asp_problem_description import ASPProblemDescription
from rsp.experiment_solvers.asp.asp_solve_problem import solve_problem
from rsp.route_dag.generators.route_dag_generator_schedule import _get_freeze_for_scheduling
from rsp.route_dag.route_dag import MAGIC_DIRECTION_FOR_SOURCE_TARGET
from rsp.route_dag.route_dag import ScheduleProblemDescription

_pp = pprint.PrettyPrinter(indent=4)


def test_costs_forced_rerouting_one_agent():
    """Force re-routing by routing penalty."""

    topo, edge_on_first_path, edge_on_second_path, dummy_source, dummy_target = _make_topo()

    for penalized_edge in [edge_on_first_path, edge_on_second_path]:
        topo_dict = {0: topo}
        minimum_travel_time_dict = {0: 2}
        latest_arrival = 300

        schedule_problem_description = ScheduleProblemDescription(
            route_dag_constraints_dict=_get_freeze_for_scheduling(
                topo_dict=topo_dict,
                dummy_source_dict={0: dummy_source},
                minimum_travel_time_dict=minimum_travel_time_dict,
                latest_arrival=latest_arrival,
            ),
            minimum_travel_time_dict=minimum_travel_time_dict,
            topo_dict=topo_dict,
            max_episode_steps=latest_arrival,
            route_section_penalties={0: {penalized_edge: 1}},
            weight_lateness_seconds=1
        )

        reschedule_problem: ASPProblemDescription = ASPProblemDescription.factory_rescheduling(
            tc=schedule_problem_description
        )
        solution, _ = solve_problem(problem=reschedule_problem)
        assert solution.optimization_costs == 0
        print(solution.trainruns_dict[0])
        assert penalized_edge[1] not in {
            trainrun_waypoint.waypoint
            for trainrun_waypoint in
            solution.trainruns_dict[0]
        }


def test_costs_forced_delay_one_agent():
    """Force delay."""

    topo, edge_on_first_path, edge_on_second_path, dummy_source, dummy_target = _make_topo()

    for penalized_edge in [edge_on_first_path, edge_on_second_path]:
        topo_dict = {0: topo}
        minimum_travel_time_dict = {0: 2}
        latest_arrival = 300

        route_dag_constraints_dict = _get_freeze_for_scheduling(topo_dict=topo_dict,
                                                                dummy_source_dict={0: dummy_source},
                                                                minimum_travel_time_dict=minimum_travel_time_dict,
                                                                latest_arrival=latest_arrival)
        forced_delay = 5
        route_dag_constraints_dict[0].freeze_earliest[dummy_source] = forced_delay
        schedule_problem_description = ScheduleProblemDescription(
            route_dag_constraints_dict,
            minimum_travel_time_dict=minimum_travel_time_dict,
            topo_dict=topo_dict,
            max_episode_steps=latest_arrival,
            route_section_penalties={0: {penalized_edge: 1}},
            weight_lateness_seconds=1
        )

        reschedule_problem: ASPProblemDescription = ASPProblemDescription.factory_rescheduling(
            tc=schedule_problem_description
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
    topo, edge_on_first_path, edge_on_second_path, dummy_source, dummy_target = _make_topo()
    for penalized_edge in [edge_on_first_path, edge_on_second_path]:
        topo_dict = {0: topo, 1: topo}
        minimum_travel_time = 3
        minimum_travel_time_dict = {0: minimum_travel_time, 1: minimum_travel_time}
        latest_arrival = 300

        route_dag_constraints_dict = _get_freeze_for_scheduling(
            topo_dict=topo_dict,
            dummy_source_dict={
                0: dummy_source,
                1: dummy_source},
            minimum_travel_time_dict=minimum_travel_time_dict,
            latest_arrival=latest_arrival)
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
            tc=schedule_problem_description
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
        #  the expected costs are only the delay (which is minimum_travel_time + 1 for release time + 1 for dummy edge)
        expected_costs = minimum_travel_time + 1 + 1
        assert solution.optimization_costs == expected_costs, f"actual costs {solution.optimization_costs}, expected {expected_costs}"
        assert len(asp_solution.extract_list_of_lates()) == expected_costs
        assert len(asp_solution.extract_list_of_active_penalty()) == 0


def test_costs_forced_rerouting_two_agents():
    """Two trains were schedule to run one after the other on the same path.

    They can
    """
    topo1, edge_on_first_path1, _, dummy_source1, dummy_target1 = _make_topo2(dummy_offset=55)
    topo2, edge_on_first_path2, _, dummy_source2, dummy_target2 = _make_topo2(dummy_offset=56)

    topo_dict = {0: topo1, 1: topo2}
    minimum_travel_time = 3
    minimum_travel_time_dict = {0: minimum_travel_time, 1: minimum_travel_time}
    latest_arrival = 300

    for edge_penalty in range(1, 8):
        route_dag_constraints_dict = _get_freeze_for_scheduling(
            topo_dict=topo_dict,
            dummy_source_dict={
                0: dummy_source1,
                1: dummy_source2},
            minimum_travel_time_dict=minimum_travel_time_dict,
            latest_arrival=latest_arrival
        )
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
            tc=schedule_problem_description
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
      X
    (0,0) -> (0,1)
     |         |
    (1,0)    (1,1)
     |         |
    (2,0)    (2,1)
     |         |
    (3,0) -> (3,1)
               X

    Returns
    -------

    """
    dummy_source = Waypoint(position=(0, 0), direction=MAGIC_DIRECTION_FOR_SOURCE_TARGET)
    dummy_target = Waypoint(position=(3, 1), direction=MAGIC_DIRECTION_FOR_SOURCE_TARGET)

    topo = nx.DiGraph()

    # first path
    topo.add_edge(dummy_source,
                  Waypoint(position=(0, 0), direction=int(Grid4TransitionsEnum.SOUTH)))
    edge_on_first_path = (Waypoint(position=(0, 0), direction=int(Grid4TransitionsEnum.SOUTH)),
                          Waypoint(position=(1, 0), direction=int(Grid4TransitionsEnum.SOUTH)))
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
    return topo, edge_on_first_path, edge_on_second_path, dummy_source, dummy_target


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
        change the postion of the dummy source and sink to make them not use the same resource as

    Returns
    -------

    """
    dummy_source = Waypoint(position=(0 + dummy_offset, 0 + dummy_offset), direction=MAGIC_DIRECTION_FOR_SOURCE_TARGET)
    dummy_target = Waypoint(position=(3 + dummy_offset, 0 + dummy_offset), direction=MAGIC_DIRECTION_FOR_SOURCE_TARGET)

    topo = nx.DiGraph()

    # first path
    topo.add_edge(dummy_source,
                  Waypoint(position=(0, 0), direction=int(Grid4TransitionsEnum.SOUTH)))
    edge_on_left_path = (Waypoint(position=(0, 0), direction=int(Grid4TransitionsEnum.SOUTH)),
                         Waypoint(position=(1, 0), direction=int(Grid4TransitionsEnum.SOUTH)))
    topo.add_edge(*edge_on_left_path)
    topo.add_edge(Waypoint(position=(1, 0), direction=int(Grid4TransitionsEnum.SOUTH)),
                  Waypoint(position=(2, 0), direction=int(Grid4TransitionsEnum.SOUTH)))
    topo.add_edge(Waypoint(position=(2, 0), direction=int(Grid4TransitionsEnum.SOUTH)),
                  Waypoint(position=(3, 0), direction=int(Grid4TransitionsEnum.SOUTH)))
    topo.add_edge(Waypoint(position=(3, 0), direction=int(Grid4TransitionsEnum.SOUTH)),
                  dummy_target)
    # second path
    topo.add_edge(dummy_source,
                  Waypoint(position=(0, 1), direction=int(Grid4TransitionsEnum.SOUTH)))
    edge_on_right_path = (Waypoint(position=(0, 1), direction=int(Grid4TransitionsEnum.SOUTH)),
                          Waypoint(position=(1, 1), direction=int(Grid4TransitionsEnum.SOUTH)))
    topo.add_edge(*edge_on_right_path)
    topo.add_edge(Waypoint(position=(1, 1), direction=int(Grid4TransitionsEnum.SOUTH)),
                  Waypoint(position=(2, 1), direction=int(Grid4TransitionsEnum.SOUTH)))
    topo.add_edge(Waypoint(position=(2, 1), direction=int(Grid4TransitionsEnum.SOUTH)),
                  Waypoint(position=(3, 1), direction=int(Grid4TransitionsEnum.SOUTH)))
    topo.add_edge(Waypoint(position=(3, 1), direction=int(Grid4TransitionsEnum.SOUTH)),
                  dummy_target)
    return topo, edge_on_left_path, edge_on_right_path, dummy_source, dummy_target
