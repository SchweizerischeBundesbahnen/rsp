from typing import Dict
from typing import NamedTuple

import networkx as nx

ExperimentMalfunction = NamedTuple("ExperimentMalfunction", [("time_step", int), ("agent_id", int), ("malfunction_duration", int)])

# TODO SIM-661 we should separate grid generation from agent placement, speed generation and topo_dict extraction (shortest paths);
#  however, we do not pass the city information out of FLATland to place agents.
Infrastructure = NamedTuple("Infrastructure", [("topo_dict", Dict[int, nx.DiGraph]), ("minimum_travel_time_dict", Dict[int, int]), ("max_episode_steps", int)])
