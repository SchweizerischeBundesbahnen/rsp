import pprint
from typing import Dict

import numpy as np
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from rsp.experiment_solvers.data_types import ExperimentMalfunction
from rsp.schedule_problem_description.data_types_and_utils import RouteDAGConstraints
from rsp.schedule_problem_description.data_types_and_utils import ScheduleProblemDescription
from rsp.schedule_problem_description.data_types_and_utils import TopoDict
from rsp.schedule_problem_description.route_dag_constraints.propagate import verify_consistency_of_route_dag_constraints_for_agent
from rsp.schedule_problem_description.route_dag_constraints.scoper_zero import _extract_route_section_penalties
from rsp.utils.data_types import RouteDAGConstraintsDict

_pp = pprint.PrettyPrinter(indent=4)


def scoper_trivially_perfect_for_all_agents(
        full_reschedule_trainrun_dict: TrainrunDict,
        malfunction: ExperimentMalfunction,
        minimum_travel_time_dict: Dict[int, int],
        max_episode_steps: int,
        # pytorch convention for in-place operations: postfixed with underscore.
        delta_trivially_perfect_topo_dict_: TopoDict,
        schedule_trainrun_dict: TrainrunDict,
        weight_route_change: int,
        weight_lateness_seconds: int,
        max_window_size_from_earliest: int = np.inf) -> ScheduleProblemDescription:
    """The scoper trivially_perfect only opens up the malfunction agent and the
    same amount of agents as were changed in the full re-reschedule, but chosen
    trivially_perfectly.

    Parameters
    ----------

    full_reschedule_trainrun_dict: TrainrunDict
        the magic information of the full re-reschedule
    malfunction: ExperimentMalfunction
        the malfunction; used to determine the waypoint after the malfunction
    minimum_travel_time_dict: Dict[int,int]
        the minimumum travel times for the agents
    max_episode_steps:
        latest arrival
    delta_trivially_perfect_topo_dict_:
        the topologies used for scheduling
    schedule_trainrun_dict: TrainrunDict
        the reschedule S0
    max_window_size_from_earliest: int
        maximum window size as offset from earliest. => "Cuts off latest at earliest + earliest_time_windows when doing
        back propagation of latest"
    weight_lateness_seconds
        how much
    weight_route_change
    Returns
    -------
    ScheduleProblemDesccription
    """
    freeze_dict: RouteDAGConstraintsDict = {}

    for agent_id, full_reschedule_trainrun in full_reschedule_trainrun_dict.items():
        topo_ = delta_trivially_perfect_topo_dict_[agent_id]
        reschedule = {trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at for trainrun_waypoint in set(full_reschedule_trainrun)}
        nodes_to_keep = {trainrun_waypoint.waypoint for trainrun_waypoint in full_reschedule_trainrun}
        nodes_to_remove = {node for node in topo_.nodes if node not in nodes_to_keep}
        topo_.remove_nodes_from(nodes_to_remove)
        freeze_dict[agent_id] = RouteDAGConstraints(
            earliest=reschedule,
            latest=reschedule
        )

    # TODO SIM-324 pull out verification
    for agent_id, _ in freeze_dict.items():
        verify_consistency_of_route_dag_constraints_for_agent(
            agent_id=agent_id,
            route_dag_constraints=freeze_dict[agent_id],
            topo=delta_trivially_perfect_topo_dict_[agent_id],
            malfunction=malfunction,
            max_window_size_from_earliest=max_window_size_from_earliest,
        )
        # N.B. re-reschedule train run must not necessarily be open in route dag constraints!

    return ScheduleProblemDescription(
        route_dag_constraints_dict=freeze_dict,
        minimum_travel_time_dict=minimum_travel_time_dict,
        topo_dict=delta_trivially_perfect_topo_dict_,
        max_episode_steps=max_episode_steps,
        route_section_penalties=_extract_route_section_penalties(
            schedule_trainruns=schedule_trainrun_dict,
            topo_dict=delta_trivially_perfect_topo_dict_,
            weight_route_change=weight_route_change
        ),
        weight_lateness_seconds=weight_lateness_seconds
    )
