import pprint
from typing import Dict
from typing import Set
from typing import Tuple

import numpy as np
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from rsp.scheduling.propagate import verify_consistency_of_route_dag_constraints_for_agent
from rsp.scheduling.scheduling_problem import RouteDAGConstraints
from rsp.scheduling.scheduling_problem import RouteDAGConstraintsDict
from rsp.scheduling.scheduling_problem import ScheduleProblemDescription
from rsp.scheduling.scheduling_problem import TopoDict
from rsp.step_02_setup.data_types import ExperimentMalfunction
from rsp.step_03_run.scopers.scoper_agent_changed_or_unchanged import scoper_changed_or_unchanged
from rsp.step_03_run.scopers.scoper_online_unrestricted import _extract_route_section_penalties
from rsp.transmission_chains.transmission_chains import extract_transmission_chains_from_schedule
from rsp.transmission_chains.transmission_chains import validate_transmission_chains
from rsp.utils.resource_occupation import extract_resource_occupations

_pp = pprint.PrettyPrinter(indent=4)


def scoper_online_transmission_chains_for_all_agents(
    online_unrestricted_trainrun_dict: TrainrunDict,
    online_unrestricted_problem: ScheduleProblemDescription,
    malfunction: ExperimentMalfunction,
    minimum_travel_time_dict: Dict[int, int],
    latest_arrival: int,
    # pytorch convention for in-place operations: postfixed with underscore.
    delta_online_topo_dict_to_: TopoDict,
    schedule_trainrun_dict: TrainrunDict,
    weight_route_change: int,
    weight_lateness_seconds: int,
    time_flexibility: bool,
    max_window_size_from_earliest: int = np.inf,
) -> Tuple[ScheduleProblemDescription, Set[int]]:
    """The scoper online only opens up the differences between the schedule and
    the imaginary re-schedule. It gives no additional routing flexibility!

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
    delta_online_topo_dict_to_:
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
    # 1. compute the forward-only wave of the malfunction
    schedule_occupations = extract_resource_occupations(schedule=schedule_trainrun_dict)
    transmission_chains = extract_transmission_chains_from_schedule(malfunction=malfunction, occupations=schedule_occupations)
    validate_transmission_chains(transmission_chains=transmission_chains)

    # 2. compute reached agents
    online_reached_agents = {transmission_chain[-1].hop_off.agent_id for transmission_chain in transmission_chains}

    freeze_dict: RouteDAGConstraintsDict = {}
    topo_dict: TopoDict = {}
    # TODO SIM-324 pull out verification
    assert malfunction.agent_id in online_reached_agents
    for agent_id in schedule_trainrun_dict.keys():
        earliest_dict, latest_dict, topo = scoper_changed_or_unchanged(
            agent_id=agent_id,
            topo_=delta_online_topo_dict_to_[agent_id],
            schedule_trainrun=schedule_trainrun_dict[agent_id],
            online_unrestricted_problem=online_unrestricted_problem,
            malfunction=malfunction,
            latest_arrival=latest_arrival,
            max_window_size_from_earliest=max_window_size_from_earliest,
            minimum_travel_time=minimum_travel_time_dict[agent_id],
            changed=(agent_id in online_reached_agents),
            time_flexibility=time_flexibility,
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
        online_reached_agents,
    )
