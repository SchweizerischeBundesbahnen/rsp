import networkx as nx
from flatland.envs.rail_trainrun_data_structures import Trainrun

from rsp.schedule_problem_description.data_types_and_utils import ScheduleProblemDescription


def scoper_changed_or_unchanged(
        agent_id: int,
        # pytorch convention for in-place operations: postfixed with underscore.
        topo_: nx.DiGraph,
        full_reschedule_trainrun: Trainrun,
        full_reschedule_problem: ScheduleProblemDescription,
        unchanged: bool
):
    """"scoper changed or unchanged":

    - if no change for train between schedule and re-schedule, keep the exact train run
    - if any change for train between schedule and re-schedule, open up everything as in full re-scheduling
    """

    if unchanged:
        route_dag_constraints = full_reschedule_problem.route_dag_constraints_dict[agent_id]
        return route_dag_constraints.earliest.copy(), route_dag_constraints.latest.copy(), full_reschedule_problem.topo_dict[agent_id].copy()
    else:
        schedule = {trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at for trainrun_waypoint in set(full_reschedule_trainrun)}
        nodes_to_keep = {trainrun_waypoint.waypoint for trainrun_waypoint in full_reschedule_trainrun}
        nodes_to_remove = {node for node in topo_.nodes if node not in nodes_to_keep}
        topo_.remove_nodes_from(nodes_to_remove)
        return schedule, schedule, topo_
