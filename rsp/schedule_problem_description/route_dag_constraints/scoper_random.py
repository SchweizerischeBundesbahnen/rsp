import pprint
from typing import Dict

import numpy as np
from flatland.envs.rail_trainrun_data_structures import TrainrunDict

from rsp.schedule_problem_description.data_types_and_utils import RouteDAGConstraints
from rsp.schedule_problem_description.data_types_and_utils import ScheduleProblemDescription
from rsp.schedule_problem_description.data_types_and_utils import TopoDict
from rsp.schedule_problem_description.route_dag_constraints.propagate import verify_consistency_of_route_dag_constraints_for_agent
from rsp.schedule_problem_description.route_dag_constraints.propagate import verify_trainrun_satisfies_route_dag_constraints
from rsp.schedule_problem_description.route_dag_constraints.scoper_online import scoper_changed_or_unchanged
from rsp.schedule_problem_description.route_dag_constraints.scoper_zero import _extract_route_section_penalties
from rsp.utils.data_types import ExperimentMalfunction
from rsp.utils.data_types import RouteDAGConstraintsDict

_pp = pprint.PrettyPrinter(indent=4)


def scoper_random_for_all_agents(
        full_reschedule_trainrun_dict: TrainrunDict,
        full_reschedule_problem: ScheduleProblemDescription,
        malfunction: ExperimentMalfunction,
        minimum_travel_time_dict: Dict[int, int],
        max_episode_steps: int,
        # pytorch convention for in-place operations: postfixed with underscore.
        delta_random_topo_dict_to_: TopoDict,
        schedule_trainrun_dict: TrainrunDict,
        weight_route_change: int,
        weight_lateness_seconds: int,
        max_window_size_from_earliest: int = np.inf) -> ScheduleProblemDescription:
    """The scoper random only opens up the differences between the schedule and
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
    delta_random_topo_dict_to_:
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
    topo_dict: TopoDict = {}

    unchanged = [
        agent_id
        for agent_id, full_reschedule_trainrun in full_reschedule_trainrun_dict.items()

        if set(full_reschedule_trainrun) == set(schedule_trainrun_dict[agent_id])

    ]
    nb_agents = len(full_reschedule_trainrun_dict)
    p_unchanged = len(unchanged) / nb_agents
    randomized_unchanged = np.random.choice(a=[True, False], size=nb_agents, p=[p_unchanged, 1 - p_unchanged])

    for agent_id in schedule_trainrun_dict.keys():
        earliest_dict, latest_dict, topo = scoper_changed_or_unchanged(
            agent_id=agent_id,
            topo_=delta_random_topo_dict_to_[agent_id],
            full_reschedule_trainrun=full_reschedule_trainrun_dict[agent_id],
            full_reschedule_problem=full_reschedule_problem,
            unchanged=(agent_id != malfunction.agent_id or randomized_unchanged[agent_id])
        )
        freeze_dict[agent_id] = RouteDAGConstraints(
            earliest=earliest_dict,
            latest=latest_dict
        )
        topo_dict[agent_id] = topo

    # TODO SIM-324 pull out verification
    for agent_id, _ in freeze_dict.items():
        verify_consistency_of_route_dag_constraints_for_agent(
            agent_id=agent_id,
            route_dag_constraints=freeze_dict[agent_id],
            topo=topo_dict[agent_id],
            malfunction=malfunction,
            max_window_size_from_earliest=max_window_size_from_earliest,
        )
        # re-schedule train run must be open in route dag constraints
        verify_trainrun_satisfies_route_dag_constraints(
            agent_id=agent_id,
            route_dag_constraints=freeze_dict[agent_id],
            scheduled_trainrun=full_reschedule_trainrun_dict[agent_id]
        )

    return ScheduleProblemDescription(
        route_dag_constraints_dict=freeze_dict,
        minimum_travel_time_dict=minimum_travel_time_dict,
        topo_dict=topo_dict,
        max_episode_steps=max_episode_steps,
        route_section_penalties=_extract_route_section_penalties(
            schedule_trainruns=schedule_trainrun_dict,
            topo_dict=topo_dict,
            weight_route_change=weight_route_change
        ),
        weight_lateness_seconds=weight_lateness_seconds
    )
