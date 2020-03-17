import os
import pprint
from typing import Dict
from typing import List
from typing import Tuple

import networkx as nx
import numpy as np
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.experiment_solvers.asp.asp_problem_description import ASPProblemDescription
from rsp.experiment_solvers.asp.asp_solve_problem import solve_problem
from rsp.experiment_solvers.data_types import SchedulingExperimentResult
from rsp.job_insertion.disjunctive_graph import draw_disjunctive_graph
from rsp.job_insertion.disjunctive_graph import force_disjunctive_edges_from_schedule
from rsp.job_insertion.disjunctive_graph import get_conjunctive_graph_by_inserting_at_end
from rsp.job_insertion.disjunctive_graph import get_trainroute_from_trainrun
from rsp.job_insertion.disjunctive_graph import left_closure
from rsp.job_insertion.disjunctive_graph import make_disjunctive_graph
from rsp.job_insertion.disjunctive_graph import make_schedule_from_conjunctive_graph
from rsp.job_insertion.disjunctive_graph import Segment
from rsp.job_insertion.disjunctive_graph import sort_vertices_by_train_start_and_earliest
from rsp.route_dag.analysis.route_dag_analysis import visualize_route_dag_constraints_simple
from rsp.route_dag.analysis.route_dag_analysis import visualize_schedule
from rsp.route_dag.generators.route_dag_generator_utils import propagate_earliest
from rsp.route_dag.generators.route_dag_generator_utils import propagate_latest
from rsp.route_dag.route_dag import get_paths_in_route_dag
from rsp.route_dag.route_dag import get_sinks_for_topo
from rsp.route_dag.route_dag import get_sources_for_topo
from rsp.route_dag.route_dag import MAGIC_DIRECTION_FOR_SOURCE_TARGET
from rsp.route_dag.route_dag import RouteDAGConstraints
from rsp.route_dag.route_dag import ScheduleProblemDescription
from rsp.utils.file_utils import check_create_folder

_pp = pprint.PrettyPrinter(indent=4)


def create_digraph_from_segments(segments: List[Segment]) -> nx.DiGraph:
    dag = nx.DiGraph()
    for segment in segments:
        for wp1, wp2 in zip(segment, segment[1:]):
            dag.add_edge(wp1, wp2)

    sources = list(get_sources_for_topo(dag))
    sinks = list(get_sinks_for_topo(dag))
    dummy_source = Waypoint(sources[0].position, direction=MAGIC_DIRECTION_FOR_SOURCE_TARGET)
    dummy_target = Waypoint(sinks[0].position, direction=MAGIC_DIRECTION_FOR_SOURCE_TARGET)

    for source in sources:
        dag.add_edge(dummy_source, source)
    for sink in sinks:
        dag.add_edge(sink, dummy_target)
    return dag


def _make_our_little_three_train_scenario() -> Dict[int, nx.DiGraph]:
    # NOTE that a synchronization step is introduced in the first and last cell
    A1 = Waypoint(position=(2, 0), direction=Grid4TransitionsEnum.SOUTH)
    B = Waypoint(position=(5, 1), direction=Grid4TransitionsEnum.EAST)
    segment_a1_b = [
        A1,
        Waypoint(position=(3, 0), direction=Grid4TransitionsEnum.SOUTH),
        Waypoint(position=(4, 0), direction=Grid4TransitionsEnum.SOUTH),
        Waypoint(position=(5, 0), direction=Grid4TransitionsEnum.SOUTH),
        B
    ]
    A2 = Waypoint(position=(8, 0), direction=Grid4TransitionsEnum.NORTH)
    segment_a2_b = [
        A2,
        Waypoint(position=(7, 0), direction=Grid4TransitionsEnum.NORTH),
        Waypoint(position=(6, 0), direction=Grid4TransitionsEnum.NORTH),
        Waypoint(position=(5, 0), direction=Grid4TransitionsEnum.NORTH),
        B
    ]
    segment_b_c = [
        B,
        Waypoint(position=(5, 2), direction=Grid4TransitionsEnum.EAST),
        Waypoint(position=(5, 3), direction=Grid4TransitionsEnum.EAST),
        Waypoint(position=(5, 4), direction=Grid4TransitionsEnum.EAST),
        Waypoint(position=(5, 5), direction=Grid4TransitionsEnum.EAST)
    ]
    D = Waypoint(position=(8, 4), direction=Grid4TransitionsEnum.EAST)
    segment_a2_d = [
        Waypoint(position=(8, 0), direction=Grid4TransitionsEnum.EAST),
        Waypoint(position=(8, 1), direction=Grid4TransitionsEnum.EAST),
        Waypoint(position=(8, 2), direction=Grid4TransitionsEnum.EAST),
        Waypoint(position=(8, 3), direction=Grid4TransitionsEnum.EAST),
        D,
    ]
    E = Waypoint(position=(7, 5), direction=Grid4TransitionsEnum.NORTH)
    segment_d_e = [
        D,
        Waypoint(position=(8, 5), direction=Grid4TransitionsEnum.NORTH),
        E
    ]
    C = Waypoint(position=(5, 5), direction=Grid4TransitionsEnum.NORTH)
    segment_e_c = [
        E,
        Waypoint(position=(6, 5), direction=Grid4TransitionsEnum.NORTH),
        Waypoint(position=(6, 4), direction=Grid4TransitionsEnum.WEST),
        Waypoint(position=(6, 5), direction=Grid4TransitionsEnum.EAST),
        C
    ]
    segment_f_e = [
        Waypoint(position=(8, 5), direction=Grid4TransitionsEnum.NORTH),
        E
    ]
    segment_dict = {
        0: [segment_f_e, segment_d_e],
        1: [segment_a2_b, segment_b_c, segment_a2_d, segment_d_e, segment_e_c],
        2: [segment_a1_b, segment_b_c]

    }
    topo_dict = {
        train: create_digraph_from_segments(segments)
        for train, segments in segment_dict.items()
    }

    for train, topo in topo_dict.items():
        assert len(list(get_sources_for_topo(topo))) == 1, f"train {train}"
        assert len(list(get_sinks_for_topo(topo))) == 1, f"train {train}"

    minimum_travel_time_dict = {
        0: 1,
        1: 1,
        2: 1
    }

    return {
        'topo_dict': topo_dict,
        'minimum_travel_time_dict': minimum_travel_time_dict
    }


def make_problem_description_for_asp_from_scenario(
        earliest_init_dict: Dict[int, int],
        topo_dict: Dict[int, nx.DiGraph],
        minimum_travel_time_dict: Dict[int, int],
        latest_arrival: int = 55,
        release_time: int = 1
):
    dummy_source_dict = {
        train: list(get_sources_for_topo(topo))[0]
        for train, topo in topo_dict.items()
    }
    dummy_target_dict = {
        train: list(get_sinks_for_topo(topo))[0]
        for train, topo in topo_dict.items()
    }

    schedule_problem_description = ScheduleProblemDescription(
        route_dag_constraints_dict={
            train: RouteDAGConstraints(
                freeze_earliest=propagate_earliest(
                    banned_set=set(),
                    earliest_dict={dummy_source_dict[train]: earliest_init_dict[train]},
                    minimum_travel_time=minimum_travel_time_dict[train],
                    force_freeze_dict={},
                    subdag_source=TrainrunWaypoint(waypoint=dummy_source_dict[train],
                                                   scheduled_at=earliest_init_dict[train]),
                    topo=topo_dict[train],
                ),
                freeze_latest=propagate_latest(
                    banned_set=set(),
                    earliest_dict={dummy_source_dict[train]: earliest_init_dict[train]},
                    latest_dict={dummy_target_dict[train]: latest_arrival},
                    latest_arrival=latest_arrival,
                    minimum_travel_time=minimum_travel_time_dict[train],
                    force_freeze_dict={},
                    topo=topo_dict[train]
                ),
                freeze_banned=[],
                freeze_visit=[]
            )
            for train in topo_dict
        },
        minimum_travel_time_dict=minimum_travel_time_dict,
        topo_dict=topo_dict,
        max_episode_steps=latest_arrival,
        route_section_penalties={train: {} for train in topo_dict},
        weight_lateness_seconds=release_time
    )

    return schedule_problem_description


def _solve_schedule_problem_and_save_route_dags(
        schedule_problem_description: ScheduleProblemDescription,
        title: str,
        output_folder: str) -> SchedulingExperimentResult:
    scheduling_problem: ASPProblemDescription = ASPProblemDescription.factory_rescheduling(
        tc=schedule_problem_description
    )
    solution, _ = solve_problem(problem=scheduling_problem)
    check_create_folder(output_folder)
    for train in list(schedule_problem_description.topo_dict.keys()):
        visualize_route_dag_constraints_simple(
            topo=schedule_problem_description.topo_dict[train],
            background_topo_dict=schedule_problem_description.topo_dict,
            f=schedule_problem_description.route_dag_constraints_dict[train],
            train_run=solution.trainruns_dict[train],
            file_name=os.path.join(output_folder, f"{title}_{train:03d}.png"),
            title=f"{title}_{train:03d}.png",
            scale=8
        )
    return solution


def dict_apply_extend(base: Dict, extension: Dict) -> Dict:
    return dict(base, **extension)


def main():
    output_folder = "job_insertion"

    # ---------------------------------------------------------------
    print("(1) our schedule scenario")
    # ---------------------------------------------------------------
    our_little_three_train_scenario = _make_our_little_three_train_scenario()
    schedule_start_time_dict = {
        0: 20,
        1: 16,
        2: 10
    }
    schedule_problem = make_problem_description_for_asp_from_scenario(
        **dict_apply_extend(
            our_little_three_train_scenario,
            {
                'earliest_init_dict': schedule_start_time_dict
            }
        )
    )

    schedule_solution = _solve_schedule_problem_and_save_route_dags(
        schedule_problem_description=schedule_problem,
        title="1_route_dag",
        output_folder=output_folder
    )
    schedule_from_asp = schedule_solution.trainruns_dict

    # sort trains and vertices for display
    sorted_trains, sorted_vertices = sort_vertices_by_train_start_and_earliest(schedule_problem, schedule_solution)

    visualize_schedule(
        trainrun_dict=schedule_from_asp,
        background_topo_dict=schedule_problem.topo_dict,
        file_name=os.path.join(output_folder, "1_schedule.png"),
        title="initial schedule"
    )

    # ---------------------------------------------------------------
    print("(2) route choice -> disjunctive graph")
    # ---------------------------------------------------------------
    trainroute_dict_schedule = {
        train: get_trainroute_from_trainrun(trainrun)
        for train, trainrun in schedule_from_asp.items()
    }
    disjunctive_graph_schedule = make_disjunctive_graph(
        minimum_travel_time_dict=schedule_problem.minimum_travel_time_dict,
        start_time_dict=schedule_start_time_dict,
        trainroute_dict=trainroute_dict_schedule)

    draw_disjunctive_graph(
        disjunctive_graph=disjunctive_graph_schedule,
        file_name=os.path.join(output_folder, f"2_disjunctive_graph_schedule.png"),
        sorted_trains=sorted_trains,
        sorted_vertices=sorted_vertices
    )

    # ---------------------------------------------------------------
    print("(3a) disjunctive_graph + full, feasible selection from schedule -> conjunctive graph")
    print("(3b) conjunctive graph -> schedule")
    # ---------------------------------------------------------------
    conjunctive_graph_schedule = force_disjunctive_edges_from_schedule(
        disjunctive_graph=disjunctive_graph_schedule,
        schedule=schedule_from_asp
    )
    draw_disjunctive_graph(
        disjunctive_graph=conjunctive_graph_schedule,
        file_name=os.path.join(output_folder, f"3a_conjunctive_graph_schedule_from_selection.png"),
        sorted_trains=sorted_trains,
        sorted_vertices=sorted_vertices
    )

    schedule_from_conjunctive_graph = make_schedule_from_conjunctive_graph(conjunctive_graph=conjunctive_graph_schedule)
    # TODO make unit tests instead...
    assert schedule_from_asp == schedule_from_conjunctive_graph

    # ---------------------------------------------------------------
    print("(4) job insertion graph")
    # ---------------------------------------------------------------
    reschedule_start_time_dict = schedule_start_time_dict.copy()
    reschedule_start_time_dict[2] = schedule_start_time_dict[2] + 6
    # train 2 start 5 later
    job_insertion_graph = make_disjunctive_graph(
        minimum_travel_time_dict=schedule_problem.minimum_travel_time_dict,
        # keep the sequences between the other two trains fixed!
        no_disjunctions={0, 1},
        start_time_dict=reschedule_start_time_dict,
        trainroute_dict=trainroute_dict_schedule
    )
    draw_disjunctive_graph(
        disjunctive_graph=job_insertion_graph,
        file_name=os.path.join(output_folder, f"4_job_insertion_graph.png"),
        sorted_trains=sorted_trains,
        sorted_vertices=sorted_vertices
    )

    # ---------------------------------------------------------------
    print("(5a) insert train at the end")
    # ---------------------------------------------------------------
    initial_conjunctive_graph, selection = get_conjunctive_graph_by_inserting_at_end(
        job_insertion_graph=job_insertion_graph,
        train_to_insert=2
    )
    np.random.seed(444)
    random_critical_arc = selection[np.random.choice(len(selection))]
    # must be incoming edge for train 2 to insert!
    assert random_critical_arc[1][0] == 2

    draw_disjunctive_graph(
        disjunctive_graph=initial_conjunctive_graph,
        file_name=os.path.join(output_folder, f"5a_initial_conjunctive_graph_with_critical_arc_to_invert.png"),
        sorted_trains=sorted_trains,
        sorted_vertices=sorted_vertices,
        highlight_edges=[random_critical_arc]
    )

    visualize_schedule(
        trainrun_dict=make_schedule_from_conjunctive_graph(initial_conjunctive_graph),
        background_topo_dict=schedule_problem.topo_dict,
        file_name=os.path.join(output_folder, "5a_schedule_insert_train_2_at_the_end.png"),
        title="initial schedule"
    )

    # ---------------------------------------------------------------
    print("(5b) left closure: swap critical arc")
    # ---------------------------------------------------------------
    conjunctive_graph_after_left_closure, cl = left_closure(
        conjunctive_graph=initial_conjunctive_graph,
        critical_arc=random_critical_arc
    )

    draw_disjunctive_graph(
        disjunctive_graph=conjunctive_graph_after_left_closure,
        file_name=os.path.join(output_folder, f"5b_conjunctive_graph_after_left_closure.png"),
        sorted_trains=sorted_trains,
        sorted_vertices=sorted_vertices,
        highlight_edges=cl
    )
    visualize_schedule(
        trainrun_dict=make_schedule_from_conjunctive_graph(conjunctive_graph_after_left_closure),
        background_topo_dict=schedule_problem.topo_dict,
        file_name=os.path.join(output_folder, "5b_reschedule_after_left_closure"),
        title="re-schedule after left closure"
    )

    # ---------------------------------------------------------------
    print("(6) naive neighborhood search -> re-schedule")
    # ---------------------------------------------------------------
    routes_dict = {
        train: get_paths_in_route_dag(topo)
        for train, topo in schedule_problem.topo_dict.items()
    }
    best_lateness, best_solutions = naive_neighborhood_search_for_3_hardcoded_trains(
        reschedule_start_time_dict=reschedule_start_time_dict,
        routes_dict=routes_dict,
        schedule=schedule_from_asp,
        topo_dict=schedule_problem.topo_dict,
        minimum_travel_time_dict=schedule_problem.minimum_travel_time_dict
    )
    print(f"best solution has lateness {best_lateness}")
    for i, sol in enumerate(best_solutions):
        print(f"solution {i + 1}/{len(best_solutions)}")
        print(_pp.pformat(sol))
        visualize_schedule(
            trainrun_dict=sol,
            background_topo_dict=schedule_problem.topo_dict,
            file_name=os.path.join(output_folder, "6a_reschedule_naive_neighborhood.png"),
            title="re-schedule ASP"
        )

    # ---------------------------------------------------------------
    print("(6b) ASP -> re-schedule for comparison")
    # ---------------------------------------------------------------
    reschedule_problem = make_problem_description_for_asp_from_scenario(
        **dict_apply_extend(
            our_little_three_train_scenario,
            {
                'earliest_init_dict': reschedule_start_time_dict
            }
        )
    )
    reschedule_solution = _solve_schedule_problem_and_save_route_dags(
        schedule_problem_description=reschedule_problem,
        title="6b_route_dag",
        output_folder=output_folder
    )
    reschedule_from_asp = reschedule_solution.trainruns_dict

    visualize_schedule(
        trainrun_dict=reschedule_from_asp,
        background_topo_dict=reschedule_problem.topo_dict,
        file_name=os.path.join(output_folder, "6b_reschedule_asp.png"),
        title="re-schedule ASP"
    )
    print(f"reschedule asp: {_pp.pformat(reschedule_from_asp)}")
    print(f"reschedule naive {_pp.pformat(best_solutions[0])}")
    assert reschedule_solution.optimization_costs == best_lateness - 6, f"{reschedule_solution.optimization_costs}, {best_lateness}"

    # ---------------------------------------------------------------
    print("(7)* local neighborhood search -> re-schedule")
    # ---------------------------------------------------------------

    print("-> done.")


def naive_neighborhood_search_for_3_hardcoded_trains(
        reschedule_start_time_dict,
        routes_dict,
        schedule,
        topo_dict,
        minimum_travel_time_dict) -> Tuple[int, TrainrunDict]:
    best_lateness = np.inf
    best_solutions = []
    for reinsert_train, topo_reinsert_train in topo_dict.items():
        # TODO generalize rather than hard code 3 trains (itertools.product?)
        for r_0 in routes_dict[0]:
            for r_1 in routes_dict[1]:
                for r_2 in routes_dict[2]:
                    chosen_routes = {
                        0: r_0,
                        1: r_1,
                        2: r_2
                    }
                    for train, route in chosen_routes.items():
                        expected_target = list(get_sinks_for_topo(topo_dict[train]))[0]
                        route_target = route[train][-1]
                        assert expected_target == route_target, \
                            f"chosen route for {train} does not go to expected target {expected_target}, but to {route_target}"
                    job_insertion_graph = make_disjunctive_graph(
                        minimum_travel_time_dict=minimum_travel_time_dict,
                        # keep the sequences between the other two trains fixed!
                        no_disjunctions={0, 1, 2}.difference({reinsert_train}),
                        start_time_dict=reschedule_start_time_dict,
                        trainroute_dict=chosen_routes
                    )

                    initial_conjunctive_graph, selection = get_conjunctive_graph_by_inserting_at_end(
                        job_insertion_graph=job_insertion_graph,
                        train_to_insert=reinsert_train
                    )
                    if len(selection) == 0:
                        print(f"no common resources between {reinsert_train} and the other trains on the chosen routes")
                        continue
                    random_critical_arc = selection[np.random.choice(len(selection))]
                    conjunctive_graph_after_left_closure, cl = left_closure(
                        conjunctive_graph=initial_conjunctive_graph,
                        critical_arc=random_critical_arc
                    )
                    re_schedule = make_schedule_from_conjunctive_graph(conjunctive_graph_after_left_closure)
                    assert len(re_schedule) == 3
                    for train, topo_reinsert_train in topo_dict.items():
                        expected_target = list(get_sinks_for_topo(topo_reinsert_train))[0]
                        actual_target = re_schedule[train][-1]
                        assert actual_target == expected_target, \
                            f"{train} did not reach target: actual {actual_target}, expected {expected_target}"

                    # sum over lateness with respect to schedule
                    lateness = sum([trainrun[-1].scheduled_at - schedule[train][-1].scheduled_at
                                    for train, trainrun in re_schedule.items()])
                    if lateness < best_lateness:
                        best_solutions = [re_schedule]
                        best_lateness = lateness
                    elif lateness == best_lateness:
                        best_solutions.append(re_schedule)
    return best_lateness, best_solutions


if __name__ == '__main__':
    main()

# TODO visualisierung: minimum running time auf edges
# TODO better scenario: multiple speed instead of individual mrt
# TODO rename freeze_***
# TODO rename ExperimentSchedulingResult....
# TODO put drawing at the end
