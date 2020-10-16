import pprint
from typing import Dict

import numpy as np
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from rsp.scheduling.propagate import verify_consistency_of_route_dag_constraints_for_agent
from rsp.scheduling.scheduling_problem import ScheduleProblemDescription
from rsp.scheduling.scheduling_problem import TopoDict
from rsp.step_02_setup.data_types import ExperimentMalfunction
from rsp.step_03_run.route_dag_constraints.scoper_zero import _extract_route_section_penalties
from rsp.step_03_run.route_dag_constraints.scoper_zero import scoper_zero
from rsp.utils.data_types import RouteDAGConstraintsDict

_pp = pprint.PrettyPrinter(indent=4)


def scoper_no_rerouting_for_all_agents(
    full_reschedule_trainrun_dict: TrainrunDict,
    full_reschedule_problem: ScheduleProblemDescription,
    malfunction: ExperimentMalfunction,
    minimum_travel_time_dict: Dict[int, int],
    max_episode_steps: int,
    # pytorch convention for in-place operations: postfixed with underscore.
    topo_dict_: TopoDict,
    schedule_trainrun_dict: TrainrunDict,
    weight_route_change: int,
    weight_lateness_seconds: int,
    max_window_size_from_earliest: int = np.inf,
) -> ScheduleProblemDescription:
    """The scoper naive only opens up the differences between the schedule and
    the imaginary re-schedule. It gives no additional routing flexibility!

    Parameters
    ----------

    full_reschedule_problem
    full_reschedule_trainrun_dict: TrainrunDict
        the magic information of the full re-schedule
    malfunction: ExperimentMalfunction
        the malfunction; used to determine the waypoint after the malfunction
    minimum_travel_time_dict: Dict[int,int]
        the minimumum travel times for the agents
    max_episode_steps:
        latest arrival
    topo_dict_:
        the topologies used for scheduling
    schedule_trainrun_dict: TrainrunDict
        the schedule S0
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
    for agent_id, schedule_trainrun in schedule_trainrun_dict.items():
        topo_ = topo_dict_[agent_id]
        schedule_waypoints = {trainrun_waypoint.waypoint for trainrun_waypoint in schedule_trainrun}
        to_remove = {node for node in topo_.nodes if node not in schedule_waypoints}
        topo_.remove_nodes_from(to_remove)
        freeze_dict[agent_id] = scoper_zero(
            agent_id=agent_id,
            topo_=topo_,
            schedule_trainrun=schedule_trainrun_dict[agent_id],
            minimum_travel_time=minimum_travel_time_dict[agent_id],
            malfunction=malfunction,
            latest_arrival=max_episode_steps,
            max_window_size_from_earliest=max_window_size_from_earliest,
        )

    # TODO SIM-324 pull out verification
    for agent_id, _ in freeze_dict.items():
        verify_consistency_of_route_dag_constraints_for_agent(
            agent_id=agent_id,
            route_dag_constraints=freeze_dict[agent_id],
            topo=topo_dict_[agent_id],
            malfunction=malfunction,
            max_window_size_from_earliest=max_window_size_from_earliest,
        )
        # N.B. re-schedule train run must not necessarily be be open in route dag constraints

    return ScheduleProblemDescription(
        route_dag_constraints_dict=freeze_dict,
        minimum_travel_time_dict=minimum_travel_time_dict,
        topo_dict=topo_dict_,
        max_episode_steps=max_episode_steps,
        route_section_penalties=_extract_route_section_penalties(
            schedule_trainruns=schedule_trainrun_dict, topo_dict=topo_dict_, weight_route_change=weight_route_change
        ),
        weight_lateness_seconds=weight_lateness_seconds,
    )
