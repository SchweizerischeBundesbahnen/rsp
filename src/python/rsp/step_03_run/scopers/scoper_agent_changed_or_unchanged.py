import networkx as nx
from flatland.envs.rail_trainrun_data_structures import Trainrun
from rsp.scheduling.scheduling_problem import ScheduleProblemDescription
from rsp.step_02_setup.data_types import ExperimentMalfunction
from rsp.step_03_run.scopers.scoper_online_unrestricted import scoper_online_unrestricted


def scoper_changed_or_unchanged(
    agent_id: int,
    # pytorch convention for in-place operations: postfixed with underscore.
    topo_: nx.DiGraph,
    schedule_trainrun: Trainrun,
    online_unrestricted_problem: ScheduleProblemDescription,
    malfunction: ExperimentMalfunction,
    minimum_travel_time: int,
    latest_arrival: int,
    changed: bool,
    max_window_size_from_earliest: int,
    time_flexibility: bool,
):
    """"scoper changed or unchanged":

    - if no change for train between schedule and re-schedule,
      - keep the exact train run if `exact`
      - keep the same route with flexibility
    - if any change for train between schedule and re-schedule, open up everything as in full re-scheduling
    """

    if changed:
        route_dag_constraints = online_unrestricted_problem.route_dag_constraints_dict[agent_id]
        return route_dag_constraints.earliest.copy(), route_dag_constraints.latest.copy(), online_unrestricted_problem.topo_dict[agent_id].copy()
    elif not time_flexibility:
        schedule = {trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at for trainrun_waypoint in set(schedule_trainrun)}
        nodes_to_keep = {trainrun_waypoint.waypoint for trainrun_waypoint in schedule_trainrun}
        nodes_to_remove = {node for node in topo_.nodes if node not in nodes_to_keep}
        topo_.remove_nodes_from(nodes_to_remove)
        return schedule, schedule, topo_
    else:
        schedule_waypoints = {trainrun_waypoint.waypoint for trainrun_waypoint in schedule_trainrun}
        to_remove = {node for node in topo_.nodes if node not in schedule_waypoints}
        topo_.remove_nodes_from(to_remove)
        earliest, latest = scoper_online_unrestricted(
            agent_id=agent_id,
            topo_=topo_,
            schedule_trainrun=schedule_trainrun,
            minimum_travel_time=minimum_travel_time,
            malfunction=malfunction,
            latest_arrival=latest_arrival,
            max_window_size_from_earliest=max_window_size_from_earliest,
        )
        return earliest, latest, topo_
