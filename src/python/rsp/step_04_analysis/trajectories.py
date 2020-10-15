from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple

from flatland.envs.rail_trainrun_data_structures import Waypoint
from rsp.schedule_problem_description.data_types_and_utils import ScheduleProblemDescription
from rsp.utils.data_types import LeftClosedInterval
from rsp.utils.data_types import ResourceOccupation
from rsp.utils.data_types import SortedResourceOccupationsPerAgent
from rsp.utils.global_constants import RELEASE_TIME

Trajectory = List[Tuple[Optional[int], Optional[int]]]  # Time and sorted resource, optional
Trajectories = Dict[int, Trajectory]  # Int in the dict is the agent handle
SpaceTimeDifference = NamedTuple('Space_Time_Difference', [('changed_agents', Trajectories),
                                                           ('additional_information', Dict)])


# Information used for plotting time-resource-graphs: Sorting is dict mapping ressource to int value used to sort
# resources for nice visualization

def time_windows_as_resource_occupations_per_agent(problem: ScheduleProblemDescription) -> SortedResourceOccupationsPerAgent:
    time_windows_per_agent = {}

    for agent_id, route_dag_constraints in problem.route_dag_constraints_dict.items():
        time_windows_per_agent[agent_id] = []
        for waypoint, earliest in route_dag_constraints.earliest.items():
            waypoint: Waypoint = waypoint
            resource = waypoint.position
            latest = route_dag_constraints.latest[waypoint]
            time_windows_per_agent[agent_id].append(ResourceOccupation(
                interval=LeftClosedInterval(earliest, latest + RELEASE_TIME),
                resource=resource,
                agent_id=agent_id,
                direction=waypoint.direction
            ))
    return time_windows_per_agent
