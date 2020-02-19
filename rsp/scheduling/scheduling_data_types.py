"""Represents a (re-)scheduling problem in a solver-independent way."""
from typing import Dict
from typing import NamedTuple

import networkx as nx

from rsp.utils.data_types import ExperimentFreezeDict

ScheduleProblemDescription = NamedTuple('ScheduleProblemDescription', [
    ('experiment_freeze_dict', ExperimentFreezeDict),
    ('minimum_travel_time_dict', Dict[int, int]),
    ('topo_dict', Dict[int, nx.DiGraph]),
    ('max_episode_steps', int)
])
