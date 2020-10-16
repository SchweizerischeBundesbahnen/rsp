import pprint
from typing import Dict

import numpy as np
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from rsp.scheduling.propagate import verify_consistency_of_route_dag_constraints_for_agent
from rsp.scheduling.scheduling_problem import RouteDAGConstraints
from rsp.scheduling.scheduling_problem import ScheduleProblemDescription
from rsp.scheduling.scheduling_problem import TopoDict
from rsp.step_02_setup.data_types import ExperimentMalfunction
from rsp.step_03_run.route_dag_constraints.scoper_agent_changed_or_unchanged import scoper_changed_or_unchanged
from rsp.step_03_run.route_dag_constraints.scoper_zero import _extract_route_section_penalties
from rsp.utils.data_types import RouteDAGConstraintsDict

_pp = pprint.PrettyPrinter(indent=4)


def scoper_random_for_all_agents(
    full_reschedule_trainrun_dict: TrainrunDict,
    full_reschedule_problem: ScheduleProblemDescription,
    malfunction: ExperimentMalfunction,
    minimum_travel_time_dict: Dict[int, int],
    latest_arrival: int,
    # pytorch convention for in-place operations: postfixed with underscore.
    delta_random_topo_dict_to_: TopoDict,
    schedule_trainrun_dict: TrainrunDict,
    weight_route_change: int,
    weight_lateness_seconds: int,
    max_window_size_from_earliest: int,
    changed_running_agents_online: int,
) -> ScheduleProblemDescription:
    """The scoper random only opens up the malfunction agent and the same
    amount of agents as were changed in the full re-schedule, but chosen
    randomly.

    Parameters
    ----------

    full_reschedule_problem
    full_reschedule_trainrun_dict: TrainrunDict
        the magic information of the full re-schedule
    malfunction: ExperimentMalfunction
        the malfunction; used to determine the waypoint after the malfunction
    minimum_travel_time_dict: Dict[int,int]
        the minimumum travel times for the agents
    latest_arrival:
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

    agents_running_after_malfunction = [
        agent_id for agent_id, schedule_trainrun in schedule_trainrun_dict.items() if schedule_trainrun[-1].scheduled_at >= malfunction.time_step
    ]
    assert malfunction.agent_id in agents_running_after_malfunction

    changed_agents = np.random.choice(agents_running_after_malfunction, changed_running_agents_online)

    for agent_id in schedule_trainrun_dict.keys():
        earliest_dict, latest_dict, topo = scoper_changed_or_unchanged(
            agent_id=agent_id,
            topo_=delta_random_topo_dict_to_[agent_id],
            schedule_trainrun=schedule_trainrun_dict[agent_id],
            full_reschedule_problem=full_reschedule_problem,
            changed=(agent_id in changed_agents),
            # N.B. we do not require malfunction agent to have re-routing flexibility!
            time_flexibility=True,
            malfunction=malfunction,
            latest_arrival=latest_arrival,
            max_window_size_from_earliest=max_window_size_from_earliest,
            minimum_travel_time=minimum_travel_time_dict[agent_id],
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
        # N.B. re-schedule train run must not necessarily be open in route dag constraints!

    return (
        ScheduleProblemDescription(
            route_dag_constraints_dict=freeze_dict,
            minimum_travel_time_dict=minimum_travel_time_dict,
            topo_dict=topo_dict,
            max_episode_steps=latest_arrival,
            route_section_penalties=_extract_route_section_penalties(
                schedule_trainruns=schedule_trainrun_dict, topo_dict=topo_dict, weight_route_change=weight_route_change
            ),
            weight_lateness_seconds=weight_lateness_seconds,
        ),
        set(changed_agents),
    )
