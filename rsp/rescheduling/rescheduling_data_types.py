# TODO SIM-239 rename or remove; can we use generic container for RouteDAGs?
from typing import Dict
from typing import NamedTuple

from flatland.envs.rail_trainrun_data_structures import TrainrunDict

from rsp.utils.data_types import AgentsPathsDict
from rsp.utils.data_types import ExperimentFreezeDict

ReScheduleProblemDescription = NamedTuple('ReScheduleProblemDescription', [
    ('experiment_freeze_dict', ExperimentFreezeDict),
    ('schedule_trainruns_to_minimize_for', TrainrunDict),
    ('minimum_travel_time_dict', Dict[int, int]),
    # TODO SIM-239 use graph instead
    ('agents_paths_dict', AgentsPathsDict),
    ('max_episode_steps', int)
])
