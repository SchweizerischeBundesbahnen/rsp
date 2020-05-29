from itertools import chain

from flatland.envs.rail_trainrun_data_structures import TrainrunDict

from rsp.utils.data_types import LeftClosedInterval
from rsp.utils.data_types import Resource
from rsp.utils.data_types import ResourceOccupation
from rsp.utils.data_types import ResourceOccupationPerAgentAndTimeStep
from rsp.utils.data_types import ScheduleAsResourceOccupations
from rsp.utils.data_types import SortedResourceOccupationsPerAgent
from rsp.utils.data_types import SortedResourceOccupationsPerResource


def extract_resource_occupations(
        schedule: TrainrunDict,
        release_time: int
) -> ScheduleAsResourceOccupations:
    """Extract the resource occuptaions from the (unexpanded) `TrainrunDict`.

    Parameters
    ----------
    schedule
    release_time

    Returns
    -------
    resource_occupations_per_resource, resource_occupations_per_agent
    """
    resource_occupations_per_resource: SortedResourceOccupationsPerResource = {}
    resource_occupations_per_agent: SortedResourceOccupationsPerAgent = {}
    resource_occupations_per_agent_and_timestep: ResourceOccupationPerAgentAndTimeStep = {}
    for agent_id, trainrun in schedule.items():
        resource_occupations_per_agent[agent_id] = []
        for entry_event, exit_event in zip(trainrun, trainrun[1:]):
            resource = Resource(*entry_event.waypoint.position)
            from_incl = entry_event.scheduled_at
            to_excl = exit_event.scheduled_at + release_time
            ro = ResourceOccupation(
                interval=LeftClosedInterval(from_incl, to_excl),
                resource=resource,
                direction=entry_event.waypoint.direction,
                agent_id=agent_id
            )
            resource_occupations_per_resource.setdefault(resource, []).append(ro)
            resource_occupations_per_agent[agent_id].append(ro)
            for time_step in range(from_incl, to_excl):
                resource_occupations_per_agent_and_timestep.setdefault((agent_id, time_step), []).append(ro)

    # sort occupations by interval's lower bound
    for _, occupations in resource_occupations_per_resource.items():
        occupations.sort(key=lambda _ro: _ro.interval.from_incl)
    return ScheduleAsResourceOccupations(resource_occupations_per_resource, resource_occupations_per_agent, resource_occupations_per_agent_and_timestep)


def verify_schedule_as_resource_occupations(  # noqa: C901
        schedule_as_resource_occupations: ScheduleAsResourceOccupations,
        release_time: int
):
    """Check consistency of the two dicts.

    Parameters
    ----------
    schedule_as_resource_occupations: ScheduleAsResourceOccupations
    release_time: int
    """
    # 1. resource occupations per resource must be be for the relevant resource
    for resource, occupations in schedule_as_resource_occupations.sorted_resource_occupations_per_resource.items():
        for ro in occupations:
            assert ro.resource == resource

    # 2. resource occupations must be mutually exclusive
    for occupations in schedule_as_resource_occupations.sorted_resource_occupations_per_resource.values():
        for ro_1, ro_2 in zip(occupations, occupations[1:]):
            # TODO SIM-517 cleanup dummy synchronization step at the beginning makes two intervals for the same agent at the same resource, which is OK.
            assert ro_2.interval.from_incl >= ro_1.interval.to_excl or ro_1.agent_id == ro_2.agent_id, f"{ro_1} {ro_2}"

    # 3. resource occupations per agent must be for the relevant agent
    for agent_id, occupations in schedule_as_resource_occupations.sorted_resource_occupations_per_agent.items():
        for ro in occupations:
            assert ro.agent_id == agent_id

    # 4. resource occupations per agent must be sorted (but may overlap by release time)
    for occupations in schedule_as_resource_occupations.sorted_resource_occupations_per_agent.values():
        for ro_1, ro_2 in zip(occupations, occupations[1:]):
            assert ro_2.interval.from_incl == ro_1.interval.to_excl - release_time, f"{ro_1} {ro_2}"

    # 5. intervals must be non-empty
    for occupations in chain(
            schedule_as_resource_occupations.sorted_resource_occupations_per_resource.values(),
            schedule_as_resource_occupations.sorted_resource_occupations_per_agent.values()):
        for ro in occupations:
            assert ro.interval.to_excl > ro.interval.from_incl
