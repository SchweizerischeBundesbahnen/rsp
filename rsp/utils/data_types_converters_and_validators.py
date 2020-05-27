from itertools import chain
from typing import Tuple

from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.rail_trainrun_data_structures import TrainrunDict

from rsp.utils.data_types import LeftClosedInterval
from rsp.utils.data_types import PlottingInformation
from rsp.utils.data_types import ResourceOccupation
from rsp.utils.data_types import ResourceSorting
from rsp.utils.data_types import SortedResourceOccupationsPerAgentDict
from rsp.utils.data_types import SortedResourceOccupationsPerResourceDict
from rsp.utils.data_types import TrainScheduleDict
from rsp.utils.data_types import Trajectories


def extract_resource_occupations(
        schedule: TrainrunDict,
        release_time: int
) -> Tuple[SortedResourceOccupationsPerResourceDict, SortedResourceOccupationsPerAgentDict]:
    """Extract the resource occuptaions from the (unexpanded) `TrainrunDict`.

    Parameters
    ----------
    schedule
    release_time

    Returns
    -------
    resource_occupations_per_resource, resource_occupations_per_agent
    """
    resource_occupations_per_resource: SortedResourceOccupationsPerResourceDict = {}
    resource_occupations_per_agent: SortedResourceOccupationsPerAgentDict = {}
    for agent_id, trainrun in schedule.items():
        resource_occupations_per_agent[agent_id] = []
        for entry_event, exit_event in zip(trainrun, trainrun[1:]):
            resource = entry_event.waypoint.position
            from_incl = entry_event.scheduled_at
            to_excl = exit_event.scheduled_at + release_time
            ro = ResourceOccupation(interval=LeftClosedInterval(from_incl, to_excl), agent_id=agent_id, resource=resource)
            resource_occupations_per_resource.setdefault(resource, []).append(ro)
            resource_occupations_per_agent[agent_id].append(ro)
    # sort occupations by interval's lower bound
    for _, occupations in resource_occupations_per_resource.items():
        occupations.sort(key=lambda _ro: _ro.interval.from_incl)
    return resource_occupations_per_resource, resource_occupations_per_agent


def verify_extracted_resource_occupations(  # noqa: C901
        resource_occupations_per_resource: SortedResourceOccupationsPerResourceDict,
        resource_occupations_per_agent: SortedResourceOccupationsPerAgentDict,
        release_time: int
):
    """Check consistency of the two dicts.

    Parameters
    ----------
    resource_occupations_per_resource
    resource_occupations_per_agent
    release_time
    """
    # 1. resource occupations per resource must be be for the relevant resource
    for resource, occupations in resource_occupations_per_resource.items():
        for ro in occupations:
            assert ro.resource == resource

    # 2. resource occupations must be mutually exclusive
    for occupations in resource_occupations_per_resource.values():
        for ro_1, ro_2 in zip(occupations, occupations[1:]):
            # TODO SIM-517 cleanup dummy synchronization step at the beginning makes two intervals for the same agent at the same resource, which is OK.
            assert ro_2.interval.from_incl >= ro_1.interval.to_excl or ro_1.agent_id == ro_2.agent_id, f"{ro_1} {ro_2}"

    # 3. resource occupations per agent must be for the relevant agent
    for agent_id, occupations in resource_occupations_per_agent.items():
        for ro in occupations:
            assert ro.agent_id == agent_id

    # 4. resource occupations per agent must be sorted (but may overlap by release time)
    for occupations in resource_occupations_per_agent.values():
        for ro_1, ro_2 in zip(occupations, occupations[1:]):
            assert ro_2.interval.from_incl == ro_1.interval.to_excl - release_time, f"{ro_1} {ro_2}"

    # 5. intervals must be non-empty
    for occupations in chain(resource_occupations_per_resource.values(), resource_occupations_per_agent.values()):
        for ro in occupations:
            assert ro.interval.to_excl > ro.interval.from_incl


# TODO used in plotting only, not here
def trajectories_from_resource_occupations_per_agent(
        resource_occupations_schedule: SortedResourceOccupationsPerAgentDict,
        resource_sorting: ResourceSorting,
        width: int) -> Trajectories:
    """

    Parameters
    ----------
    resource_occupations_schedule
    resource_sorting
    width

    Returns
    -------

    """
    schedule_trajectories: Trajectories = []
    for _, resource_ocupations in resource_occupations_schedule.items():
        train_time_path = []
        for resource_ocupation in resource_ocupations:
            position = coordinate_to_position(width, [resource_ocupation.resource])[0]
            # TODO dirty hack: add positions from re-scheduling to resource_sorting in the first place instead of workaround here!
            if position not in resource_sorting:
                resource_sorting[position] = len(resource_sorting)
            train_time_path.append((resource_sorting[position], resource_ocupation.interval.from_incl))
            train_time_path.append((resource_sorting[position], resource_ocupation.interval.to_excl))
            train_time_path.append((None, None))
        schedule_trajectories.append(train_time_path)
    return schedule_trajectories


# TODO remove, use only trajectories_from_resource_occupations_per_agent
def trajectories_from_train_schedule_dict(
        train_schedule_dict: TrainScheduleDict,
        plotting_information: PlottingInformation,
) -> Trajectories:
    """

    Parameters
    ----------
    plotting_information
    train_schedule_dict

    Returns
    -------

    """
    resource_sorting = plotting_information.sorting
    schedule_trajectories: Trajectories = []
    width = plotting_information.grid_width
    for _, train_schedule in train_schedule_dict.items():
        train_time_path = []
        previous_waypoint = None
        entry_time = -1
        for time_step, waypoint in train_schedule.items():

            if previous_waypoint is None and waypoint is not None:
                entry_time = time_step
                previous_waypoint = waypoint
            elif previous_waypoint != waypoint:
                position = coordinate_to_position(width, [previous_waypoint.position])[0]
                # TODO dirty hack: add positions from re-scheduling to resource_sorting in the first place instead of workaround here!
                if position not in resource_sorting:
                    resource_sorting[position] = len(resource_sorting)

                train_time_path.append((resource_sorting[position], entry_time))
                train_time_path.append((resource_sorting[position], time_step))
                train_time_path.append((None, None))

                previous_waypoint = waypoint
                entry_time = time_step

        schedule_trajectories.append(train_time_path)
    return schedule_trajectories
