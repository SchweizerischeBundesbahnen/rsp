import os
import pprint
from typing import Dict
from typing import List

import networkx as nx
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.experiment_solvers.asp.asp_problem_description import ASPProblemDescription
from rsp.experiment_solvers.asp.asp_solve_problem import solve_problem
from rsp.job_insertion.disjunctive_graph import DisjunctiveGraphEdgeType
from rsp.job_insertion.disjunctive_graph import draw_disjunctive_graph
from rsp.job_insertion.disjunctive_graph import force_disjunctive_edge_in_job_insertion_graph
from rsp.job_insertion.disjunctive_graph import force_disjunctive_edges_from_schedule
from rsp.job_insertion.disjunctive_graph import get_trainroute_from_trainrun
from rsp.job_insertion.disjunctive_graph import make_disjunctive_graph
from rsp.job_insertion.disjunctive_graph import make_schedule_from_conjunctive_graph
from rsp.job_insertion.disjunctive_graph import Segment
from rsp.job_insertion.disjunctive_graph import sort_vertices_by_train_start_and_earliest
from rsp.job_insertion.disjunctive_graph import Trainroute
from rsp.route_dag.analysis.route_dag_analysis import visualize_route_dag_constraints_simple
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


def _scenerio_topo_dict() -> Dict[int, nx.DiGraph]:
    segment_a1_b = [
        Waypoint(position=(2, 0), direction=Grid4TransitionsEnum.SOUTH),
        Waypoint(position=(3, 0), direction=Grid4TransitionsEnum.SOUTH),
        Waypoint(position=(4, 0), direction=Grid4TransitionsEnum.SOUTH),
        Waypoint(position=(5, 0), direction=Grid4TransitionsEnum.SOUTH),
        Waypoint(position=(5, 1), direction=Grid4TransitionsEnum.EAST)
    ]
    segment_a2_b = [
        Waypoint(position=(8, 0), direction=Grid4TransitionsEnum.NORTH),
        Waypoint(position=(7, 0), direction=Grid4TransitionsEnum.NORTH),
        Waypoint(position=(6, 0), direction=Grid4TransitionsEnum.NORTH),
        Waypoint(position=(5, 0), direction=Grid4TransitionsEnum.NORTH),
        Waypoint(position=(5, 1), direction=Grid4TransitionsEnum.EAST)
    ]
    segment_b_c = [
        Waypoint(position=(5, 1), direction=Grid4TransitionsEnum.EAST),
        Waypoint(position=(5, 2), direction=Grid4TransitionsEnum.EAST),
        Waypoint(position=(5, 3), direction=Grid4TransitionsEnum.EAST),
        Waypoint(position=(5, 4), direction=Grid4TransitionsEnum.EAST),
        Waypoint(position=(5, 5), direction=Grid4TransitionsEnum.EAST)
    ]
    segment_a2_c = [
        Waypoint(position=(8, 0), direction=Grid4TransitionsEnum.EAST),
        Waypoint(position=(8, 1), direction=Grid4TransitionsEnum.EAST),
        Waypoint(position=(8, 2), direction=Grid4TransitionsEnum.EAST),
        Waypoint(position=(8, 3), direction=Grid4TransitionsEnum.EAST),
        Waypoint(position=(8, 4), direction=Grid4TransitionsEnum.EAST),
        Waypoint(position=(8, 5), direction=Grid4TransitionsEnum.NORTH),
        Waypoint(position=(7, 5), direction=Grid4TransitionsEnum.NORTH),
        Waypoint(position=(6, 5), direction=Grid4TransitionsEnum.NORTH),
        Waypoint(position=(6, 4), direction=Grid4TransitionsEnum.WEST),
        Waypoint(position=(6, 5), direction=Grid4TransitionsEnum.EAST),
        Waypoint(position=(5, 5), direction=Grid4TransitionsEnum.NORTH)
    ]
    train_1_segments = [segment_a1_b, segment_b_c]
    train_2_segments = [segment_a2_b, segment_b_c, segment_a2_c]

    topo_dict = {
        0: create_digraph_from_segments(train_1_segments),
        1: create_digraph_from_segments(train_2_segments)
    }
    return topo_dict


def _solve_schedule_problem_and_save_route_dags(schedule_problem_description: ScheduleProblemDescription,
                                                title: str,
                                                output_folder: str):
    reschedule_problem: ASPProblemDescription = ASPProblemDescription.factory_rescheduling(
        tc=schedule_problem_description
    )
    solution, _ = solve_problem(problem=reschedule_problem)
    check_create_folder(output_folder)
    for train in schedule_problem_description.topo_dict.keys():
        visualize_route_dag_constraints_simple(
            topo=schedule_problem_description.topo_dict[train],
            f=schedule_problem_description.route_dag_constraints_dict[train],
            train_run=solution.trainruns_dict[train],
            file_name=os.path.join(output_folder, f"{title}_{train:03d}.png"),
            title=f"{title}_{train:03d}.png",
            scale=8
        )
    return solution


def main():
    # ---------------------------------------------------------------
    print("(1) topology of hello world scenario")
    # ---------------------------------------------------------------
    output_folder = "job_insertion"
    topo_dict = _scenerio_topo_dict()

    # ---------------------------------------------------------------
    print("(2) route DAGs for schedule and re-schedule situation")
    # ---------------------------------------------------------------
    for train, topo in topo_dict.items():
        assert len(list(get_sources_for_topo(topo))) == 1, f"train {train}"
        assert len(list(get_sinks_for_topo(topo))) == 1, f"train {train}"
    dummy_source_dict = {
        train: list(get_sources_for_topo(topo))[0]
        for train, topo in topo_dict.items()
    }
    dummy_target_dict = {
        train: list(get_sinks_for_topo(topo))[0]
        for train, topo in topo_dict.items()
    }
    minimum_travel_time_dict = {
        0: 1,
        1: 1
    }

    def _make_scenario_schedule_description(earliest_init_dict: Dict[int, int]):
        schedule_problem_description = ScheduleProblemDescription(
            route_dag_constraints_dict={
                train: RouteDAGConstraints(
                    freeze_earliest=propagate_earliest(
                        banned_set=set(),
                        earliest_dict={dummy_source_dict[train]: earliest_init_dict[train]},
                        minimum_travel_time=minimum_travel_time_dict[train],
                        force_freeze_dict={},
                        subdag_source=TrainrunWaypoint(waypoint=dummy_source_dict[train], scheduled_at=10),
                        topo=topo_dict[train],
                    ),
                    freeze_latest=propagate_latest(
                        banned_set=set(),
                        earliest_dict={dummy_source_dict[train]: earliest_init_dict[train]},
                        latest_dict={dummy_target_dict[train]: 55},
                        latest_arrival=55,
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
            max_episode_steps=50,
            route_section_penalties={0: {}, 1: {}},
            weight_lateness_seconds=1
        )

        return schedule_problem_description

    schedule_earliest_init_dict = {
        0: 10,
        1: 16
    }

    schedule_problem = _make_scenario_schedule_description(earliest_init_dict=schedule_earliest_init_dict)

    # ---------------------------------------------------------------
    print("(3) route DAGS -> ASP -> schedule and re-schedule")
    # ---------------------------------------------------------------
    schedule_solution = _solve_schedule_problem_and_save_route_dags(
        schedule_problem_description=schedule_problem,
        title="schedule",
        output_folder=output_folder
    )
    schedule = schedule_solution.trainruns_dict

    # sort trains and vertices
    sorted_trains, sorted_vertices = sort_vertices_by_train_start_and_earliest(schedule_problem, schedule_solution)

    # ---------------------------------------------------------------
    print("(4) routes -> disjunctive graphs")
    # ---------------------------------------------------------------
    trainroute_dict_schedule = {
        train: get_trainroute_from_trainrun(trainrun)
        for train, trainrun in schedule.items()
    }

    disjunctive_graph_schedule = make_disjunctive_graph(minimum_travel_time_dict=minimum_travel_time_dict,
                                                        start_time_dict={
                                                            0: 10,
                                                            1: 16
                                                        },
                                                        trainroute_dict=trainroute_dict_schedule)

    draw_disjunctive_graph(
        disjunctive_graph=disjunctive_graph_schedule,
        file_name=os.path.join(output_folder, f"disjunctive_graph_schedule.png"),
        sorted_trains=sorted_trains,
        sorted_vertices=sorted_vertices
    )

    # ---------------------------------------------------------------
    print("(5a) disjunctive_graph + full, feasible selection from schedule -> conjunctive graph")
    print("(5b) conjunctive graph -> schedule")
    # ---------------------------------------------------------------
    conjunctive_graph_schedule = force_disjunctive_edges_from_schedule(
        disjunctive_graph=disjunctive_graph_schedule,
        schedule=schedule
    )
    draw_disjunctive_graph(
        disjunctive_graph=conjunctive_graph_schedule,
        file_name=os.path.join(output_folder, f"conjunctive_graph_schedule_from_selection.png"),
        sorted_trains=sorted_trains,
        sorted_vertices=sorted_vertices
    )

    schedule_from_conjunctive_graph = make_schedule_from_conjunctive_graph(conjunctive_graph=conjunctive_graph_schedule)
    # TODO make unit tests instead...
    assert schedule == schedule_from_conjunctive_graph

    # ---------------------------------------------------------------
    print("(6) job insertion graph")
    # ---------------------------------------------------------------
    job_insertion_graph = make_disjunctive_graph(
        minimum_travel_time_dict=minimum_travel_time_dict,
        start_time_dict={
            0: 15,
            1: 16
        },
        trainroute_dict=trainroute_dict_schedule
    )
    draw_disjunctive_graph(
        disjunctive_graph=job_insertion_graph,
        file_name=os.path.join(output_folder, f"job_insertion_graph.png"),
        sorted_trains=sorted_trains,
        sorted_vertices=sorted_vertices
    )

    # ---------------------------------------------------------------
    print("(7) job insertion graph + forced edge -> re-schedule")
    # ---------------------------------------------------------------
    random_forced_edge = next(((u, v) for (u, v, d) in job_insertion_graph.edges(data=True)
                               if d['type'] == DisjunctiveGraphEdgeType.DISJUNCTIVE))
    force_disjunctive_edge_in_job_insertion_graph(
        job_insertion_graph=job_insertion_graph,
        forced_edge=random_forced_edge
    )

    # ---------------------------------------------------------------
    print("(8) job insertion graph -> naive neighborhood search -> re-schedule")
    # ---------------------------------------------------------------
    trainroutes_dict: Dict[int, List[Trainroute]] = {
        train: get_paths_in_route_dag(topo)
        for train, topo in schedule_problem.topo_dict.items()
    }
    _pp.pprint(trainroutes_dict)

    # ---------------------------------------------------------------
    print("(9) generalized job insertion graph -> naive neighborhood search -> re-schedule ")
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    print("(10) generalized job insertion graph -> local neighborhood search -> re-schedule ")
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    print("(11)*** job insertion graph -> tabu neighborhood search -> schedule")
    # ---------------------------------------------------------------

    print("-> done.")


if __name__ == '__main__':
    main()

# TODO visualisierung
# ganze Szenario grau hintendran
# Verspätung gegenüber earliest
# Knoten des Fahrplans hervorheben oder Edges verstärken
# minimum running time auf edges


# TODO rename freeze_***
# TODO rename ExperimentSchedulingResult....
# TODO put drawing at the end
