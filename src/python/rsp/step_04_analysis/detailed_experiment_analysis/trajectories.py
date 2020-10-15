from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Set
from typing import Tuple

from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.rail_trainrun_data_structures import Waypoint
from rsp.schedule_problem_description.data_types_and_utils import ScheduleProblemDescription
from rsp.step_04_analysis.detailed_experiment_analysis.schedule_plotting import PlottingInformation
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


def explode_trajectories(trajectories: Trajectories) -> Dict[int, Set[Tuple[int, int]]]:
    """Return for each agent the pairs of `(resource,time)` corresponding to
    the trajectories.

    Parameters
    ----------
    trajectories

    Returns
    -------
    Dict indexed by `agent_id`, containing `(resource,time_step)` pairs.
    """
    exploded = {agent_id: set() for agent_id in trajectories.keys()}
    for agent_id, trajectory in trajectories.items():
        # ensure we have triplets (resource,from_time), (resource,to_time), (None,None)
        assert len(trajectory) % 3 == 0
        while len(trajectory) > 0:
            (resource, from_time), (resource, to_time), (_, _) = trajectory[:3]
            for time in range(from_time, to_time + 1):
                exploded[agent_id].add((resource, time))
            trajectory = trajectory[3:]
    return exploded


def get_difference_in_time_space_trajectories(base_trajectories: Trajectories, target_trajectories: Trajectories) -> SpaceTimeDifference:
    """
    Compute the difference between schedules and return in plot ready format (in base but not in target)
    Parameters
    ----------
    base_trajectories
    target_trajectories

    Returns
    -------

    """
    # Detect changes to original schedule
    traces_influenced_agents: Trajectories = {}
    additional_information = dict()
    # explode trajectories in order to be able to do point-wise diff!
    base_trajectories_exploded = explode_trajectories(base_trajectories)
    target_trajectories_exploded = explode_trajectories(target_trajectories)
    for agent_id in base_trajectories.keys():
        difference_exploded = base_trajectories_exploded[agent_id] - target_trajectories_exploded[agent_id]

        if len(difference_exploded) > 0:
            trace = []
            for (resource, time_step) in difference_exploded:
                # TODO we draw one-dot strokes, should we collapse to longer strokes?
                #  We want to keep the triplet structure in the trajectories in order not to have to distinguish between cases!
                trace.append((resource, time_step))
                trace.append((resource, time_step))
                trace.append((None, None))
            traces_influenced_agents[agent_id] = trace
            additional_information.update({agent_id: True})
        else:
            traces_influenced_agents[agent_id] = [(None, None)]
            additional_information.update({agent_id: False})
    space_time_difference = SpaceTimeDifference(changed_agents=traces_influenced_agents,
                                                additional_information=additional_information)
    return space_time_difference


def trajectories_from_resource_occupations_per_agent(
        resource_occupations_schedule: SortedResourceOccupationsPerAgent,
        plotting_information: PlottingInformation
) -> Trajectories:
    """
    Build trajectories for time-resource graph
    Parameters
    ----------
    resource_occupations_schedule

    Returns
    -------

    """
    resource_sorting = plotting_information.sorting
    width = plotting_information.grid_width
    schedule_trajectories: Trajectories = {}
    for agent_handle, resource_ocupations in resource_occupations_schedule.items():
        train_time_path = []
        for resource_ocupation in resource_ocupations:
            position = coordinate_to_position(width, [resource_ocupation.resource])[0]
            # TODO dirty hack: add positions from re-scheduling to resource_sorting in the first place instead of workaround here!
            if position not in resource_sorting:
                resource_sorting[position] = len(resource_sorting)
            train_time_path.append((resource_sorting[position], resource_ocupation.interval.from_incl))
            train_time_path.append((resource_sorting[position], resource_ocupation.interval.to_excl))
            train_time_path.append((None, None))
        schedule_trajectories[agent_handle] = train_time_path
    return schedule_trajectories
