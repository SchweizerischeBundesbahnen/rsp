import networkx as nx
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.scheduling.asp.asp_helper import _print_stats
from rsp.scheduling.asp_wrapper import asp_schedule_wrapper
from rsp.scheduling.propagate import propagate
from rsp.scheduling.schedule import save_schedule
from rsp.scheduling.schedule import Schedule
from rsp.scheduling.scheduling_problem import get_paths_in_route_dag
from rsp.scheduling.scheduling_problem import get_sinks_for_topo
from rsp.scheduling.scheduling_problem import get_sources_for_topo
from rsp.scheduling.scheduling_problem import RouteDAGConstraints
from rsp.scheduling.scheduling_problem import ScheduleProblemDescription
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import ScheduleParameters
from rsp.step_02_infrastructure_generation.infrastructure import Infrastructure
from rsp.step_02_infrastructure_generation.infrastructure import load_infrastructure
from rsp.utils.rsp_logger import rsp_logger


def _get_route_dag_constraints_for_scheduling(
    topo: nx.DiGraph, source_waypoint: Waypoint, minimum_travel_time: int, latest_arrival: int
) -> RouteDAGConstraints:
    earliest = {source_waypoint: 0}
    latest = {sink: latest_arrival - 1 for sink in get_sinks_for_topo(topo)}
    propagate(
        latest_dict=latest,
        earliest_dict=earliest,
        latest_arrival=latest_arrival,
        minimum_travel_time=minimum_travel_time,
        force_earliest={source_waypoint},
        force_latest=set(get_sinks_for_topo(topo)),
        must_be_visited=set(),
        topo=topo,
    )
    return RouteDAGConstraints(earliest=earliest, latest=latest,)


def create_schedule_problem_description_from_instructure(
    infrastructure: Infrastructure, number_of_shortest_paths_per_agent_schedule: int
) -> ScheduleProblemDescription:
    # deep copy dict
    topo_dict = {agent_id: topo.copy() for agent_id, topo in infrastructure.topo_dict.items()}
    # reduce topo_dict to number_of_shortest_paths_per_agent_schedule
    for _, topo in topo_dict.items():
        paths = get_paths_in_route_dag(topo)
        paths = paths[:number_of_shortest_paths_per_agent_schedule]
        remaining_vertices = {vertex for path in paths for vertex in path}
        topo.remove_nodes_from(set(topo.nodes).difference(remaining_vertices))

    schedule_problem_description = ScheduleProblemDescription(
        route_dag_constraints_dict={
            agent_id: _get_route_dag_constraints_for_scheduling(
                minimum_travel_time=infrastructure.minimum_travel_time_dict[agent_id],
                topo=topo_dict[agent_id],
                source_waypoint=next(get_sources_for_topo(topo_dict[agent_id])),
                latest_arrival=infrastructure.max_episode_steps,
            )
            for agent_id, topo in topo_dict.items()
        },
        minimum_travel_time_dict=infrastructure.minimum_travel_time_dict,
        topo_dict=topo_dict,
        max_episode_steps=infrastructure.max_episode_steps,
        route_section_penalties={agent_id: {} for agent_id in topo_dict.keys()},
        weight_lateness_seconds=1,
    )
    return schedule_problem_description


def gen_schedule(schedule_parameters: ScheduleParameters, infrastructure: Infrastructure, debug: bool = False) -> Schedule:
    """A.1.2 Create schedule from parameter ranges.

    Parameters
    ----------
    infrastructure
    schedule_parameters
    debug

    Returns
    -------
    """
    rsp_logger.info(f"gen_schedule {schedule_parameters}")
    schedule_problem = create_schedule_problem_description_from_instructure(
        infrastructure=infrastructure, number_of_shortest_paths_per_agent_schedule=schedule_parameters.number_of_shortest_paths_per_agent_schedule
    )
    if debug:
        for agent_id, topo in schedule_problem.topo_dict.items():
            rsp_logger.info(f"    {agent_id} has {len(get_paths_in_route_dag(topo))} paths in scheduling")
            rsp_logger.info(f"    {agent_id} has {len(get_paths_in_route_dag(infrastructure.topo_dict[agent_id]))} paths in infrastructure")

    schedule_result = asp_schedule_wrapper(schedule_problem_description=schedule_problem, asp_seed_value=schedule_parameters.asp_seed_value, debug=debug)
    rsp_logger.info(f"done gen_schedule {schedule_parameters}")
    return Schedule(schedule_problem_description=schedule_problem, schedule_experiment_result=schedule_result)


def gen_and_save_schedule(schedule_parameters: ScheduleParameters, base_directory: str):
    infra_id = schedule_parameters.infra_id
    infra, infra_parameters = load_infrastructure(base_directory=base_directory, infra_id=infra_id)
    rsp_logger.info(f"gen schedule for [{infra_id}/{schedule_parameters.schedule_id}] {infra_parameters} {schedule_parameters}")
    schedule = gen_schedule(infrastructure=infra, schedule_parameters=schedule_parameters)
    save_schedule(schedule=schedule, schedule_parameters=schedule_parameters, base_directory=base_directory)
    _print_stats(schedule.schedule_experiment_result.solver_statistics)
    return schedule_parameters
