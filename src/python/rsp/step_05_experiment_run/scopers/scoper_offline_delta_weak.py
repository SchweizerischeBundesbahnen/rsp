import pprint
from typing import Dict

import numpy as np
from flatland.envs.rail_trainrun_data_structures import TrainrunDict

from rsp.scheduling.propagate import verify_consistency_of_route_dag_constraints_for_agent
from rsp.scheduling.propagate import verify_trainrun_satisfies_route_dag_constraints
from rsp.scheduling.scheduling_problem import RouteDAGConstraints
from rsp.scheduling.scheduling_problem import RouteDAGConstraintsDict
from rsp.scheduling.scheduling_problem import ScheduleProblemDescription
from rsp.scheduling.scheduling_problem import TopoDict
from rsp.step_05_experiment_run.experiment_malfunction import ExperimentMalfunction
from rsp.step_05_experiment_run.scopers.scoper_agent_wise import AgentWiseChange
from rsp.step_05_experiment_run.scopers.scoper_agent_wise import scoper_agent_wise
from rsp.step_05_experiment_run.scopers.scoper_online_unrestricted import _extract_route_section_penalties

_pp = pprint.PrettyPrinter(indent=4)


def scoper_offline_delta_weak_for_all_agents(
    online_unrestricted_trainrun_dict: TrainrunDict,
    online_unrestricted_problem: ScheduleProblemDescription,
    malfunction: ExperimentMalfunction,
    minimum_travel_time_dict: Dict[int, int],
    latest_arrival: int,
    # pytorch convention for in-place operations: postfixed with underscore.
    topo_dict_: TopoDict,
    schedule_trainrun_dict: TrainrunDict,
    weight_route_change: int,
    weight_lateness_seconds: int,
    max_window_size_from_earliest: int = np.inf,
) -> ScheduleProblemDescription:
    """The scoper offline delta weak only opens up all changed agents and
    freezes all unchanged agents to the intitial schedule.

    Parameters
    ----------

    online_unrestricted_problem
    online_unrestricted_trainrun_dict: TrainrunDict
        the magic information of the full re-schedule
    malfunction: ExperimentMalfunction
        the malfunction; used to determine the waypoint after the malfunction
    minimum_travel_time_dict: Dict[int,int]
        the minimumum travel times for the agents
    latest_arrival:
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
    topo_dict: TopoDict = {}
    changed_dict = {agent_id: online_unrestricted_trainrun_dict[agent_id] != schedule_trainrun_dict[agent_id] for agent_id in schedule_trainrun_dict}

    # TODO SIM-324 pull out verification
    assert malfunction.agent_id in changed_dict
    for agent_id in schedule_trainrun_dict.keys():
        earliest_dict, latest_dict, topo = scoper_agent_wise(
            agent_id=agent_id,
            topo_=topo_dict_[agent_id],
            schedule_trainrun=schedule_trainrun_dict[agent_id],
            online_unrestricted_problem=online_unrestricted_problem,
            malfunction=malfunction,
            latest_arrival=latest_arrival,
            max_window_size_from_earliest=max_window_size_from_earliest,
            minimum_travel_time=minimum_travel_time_dict[agent_id],
            # freeze all unchanged agents - we know this is feasible!
            agent_wise_change=AgentWiseChange.unrestricted if changed_dict[agent_id] else AgentWiseChange.fully_restricted,
        )
        freeze_dict[agent_id] = RouteDAGConstraints(earliest=earliest_dict, latest=latest_dict)
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
            agent_id=agent_id, route_dag_constraints=freeze_dict[agent_id], scheduled_trainrun=online_unrestricted_trainrun_dict[agent_id]
        )

    return ScheduleProblemDescription(
        route_dag_constraints_dict=freeze_dict,
        minimum_travel_time_dict=minimum_travel_time_dict,
        topo_dict=topo_dict,
        max_episode_steps=latest_arrival,
        route_section_penalties=_extract_route_section_penalties(
            schedule_trainruns=schedule_trainrun_dict, topo_dict=topo_dict, weight_route_change=weight_route_change
        ),
        weight_lateness_seconds=weight_lateness_seconds,
    )
