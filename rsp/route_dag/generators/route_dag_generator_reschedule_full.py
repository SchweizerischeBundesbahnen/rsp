from typing import Dict

import networkx as nx
from flatland.envs.rail_trainrun_data_structures import TrainrunDict

from rsp.route_dag.generators.route_dag_generator_reschedule_generic import generic_experiment_freeze_for_rescheduling
from rsp.route_dag.route_dag import RouteDAG
from rsp.utils.data_types import ExperimentMalfunction


def get_freeze_for_full_rescheduling(malfunction: ExperimentMalfunction,
                                     schedule_trainruns: TrainrunDict,
                                     minimum_travel_time_dict: Dict[int, int],
                                     topo_dict: Dict[int, nx.DiGraph],
                                     latest_arrival: int
                                     ) -> RouteDAG:
    """Returns the experiment freeze for the full re-scheduling problem. Wraps
    the generic freeze by freezing everything up to and including the
    malfunction.

    See param description there.
    """
    return generic_experiment_freeze_for_rescheduling(
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
        latest_arrival=latest_arrival
    )
