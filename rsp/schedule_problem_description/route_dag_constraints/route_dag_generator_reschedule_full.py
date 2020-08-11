from typing import Dict

import networkx as nx
import numpy as np
from flatland.envs.rail_trainrun_data_structures import TrainrunDict

from rsp.schedule_problem_description.data_types_and_utils import ScheduleProblemDescription
from rsp.schedule_problem_description.route_dag_constraints.route_dag_generator_reschedule_generic import \
    generic_schedule_problem_description_for_rescheduling
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
    return generic_schedule_problem_description_for_rescheduling(
        malfunction=malfunction,
        schedule_trainruns=schedule_trainruns,
        minimum_travel_time_dict=minimum_travel_time_dict,
        force_freeze={agent_id: [trainrun_waypoint
                                 for trainrun_waypoint in schedule_trainrun
                                 if trainrun_waypoint.scheduled_at <= malfunction.time_step
                                 ]
                      for agent_id, schedule_trainrun in schedule_trainruns.items()
                      },
        topo_dict=topo_dict,
        latest_arrival=latest_arrival,
        max_window_size_from_earliest=max_window_size_from_earliest
    )
