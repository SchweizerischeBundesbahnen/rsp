from typing import Dict

import networkx as nx
import numpy as np
from flatland.envs.rail_trainrun_data_structures import TrainrunDict

from rsp.schedule_problem_description.data_types_and_utils import ScheduleProblemDescription
from rsp.schedule_problem_description.route_dag_constraints.route_dag_generator_reschedule_generic import _extract_route_section_penalties
from rsp.schedule_problem_description.route_dag_constraints.route_dag_generator_reschedule_generic import delta_zero
from rsp.schedule_problem_description.route_dag_constraints.route_dag_generator_utils import verify_consistency_of_route_dag_constraints_for_agent
from rsp.schedule_problem_description.route_dag_constraints.route_dag_generator_utils import verify_trainrun_satisfies_route_dag_constraints
from rsp.utils.data_types import ExperimentMalfunction


# TODO SIM-613 do we need this wrapper?
def get_schedule_problem_for_full_rescheduling(malfunction: ExperimentMalfunction,
                                               schedule_trainruns: TrainrunDict,
                                               minimum_travel_time_dict: Dict[int, int],
                                               topo_dict: Dict[int, nx.DiGraph],
                                               latest_arrival: int,
                                               max_window_size_from_earliest: int = np.inf
                                               ) -> ScheduleProblemDescription:
    """Returns the experiment freeze for the full re-scheduling problem. Wraps
    the generic freeze by freezing everything up to and including the
    malfunction.

    See param description there.
    """
    spd = ScheduleProblemDescription(
        route_dag_constraints_dict={
            agent_id: delta_zero(
                schedule_trainrun=schedule_trainruns[agent_id],
                minimum_travel_time=minimum_travel_time_dict[agent_id],
                topo=topo_dict[agent_id],
                malfunction=malfunction,
                agent_id=agent_id,
                latest_arrival=latest_arrival,
                max_window_size_from_earliest=max_window_size_from_earliest
            )
            for agent_id in schedule_trainruns},
        topo_dict=topo_dict,
        minimum_travel_time_dict=minimum_travel_time_dict,
        max_episode_steps=latest_arrival,
        route_section_penalties=_extract_route_section_penalties(schedule_trainruns, topo_dict),
        weight_lateness_seconds=1
    )
    # TODO SIM-324 pull out verification
    for agent_id in spd.route_dag_constraints_dict:
        verify_consistency_of_route_dag_constraints_for_agent(
            agent_id=agent_id,
            topo=topo_dict[agent_id],
            route_dag_constraints=spd.route_dag_constraints_dict[agent_id],
            # TODO SIM-613
            # force_freeze=force_freeze[agent_id], # noqa E800
            malfunction=malfunction if malfunction.agent_id == agent_id else None,
            max_window_size_from_earliest=max_window_size_from_earliest
        )
        verify_trainrun_satisfies_route_dag_constraints(
            agent_id=agent_id,
            route_dag_constraints=spd.route_dag_constraints_dict[agent_id],
            scheduled_trainrun=list(
                filter(lambda trainrun_waypoint: trainrun_waypoint.scheduled_at <= malfunction.time_step,
                       schedule_trainruns[agent_id]))
        )
    return spd
