"""Data types used in the experiment for the real time rescheduling research
project."""
from itertools import chain
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Tuple

from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from rsp.scheduling.scheduling_problem import RouteDAGConstraintsDict

LeftClosedInterval = NamedTuple("LeftClosedInterval", [("from_incl", int), ("to_excl", int)])
Resource = NamedTuple("Resource", [("row", int), ("column", int)])
ResourceOccupation = NamedTuple("ResourceOccupation", [("interval", LeftClosedInterval), ("resource", Resource), ("direction", int), ("agent_id", int)])

# sorted list of non-overlapping resource occupations per resource
SortedResourceOccupationsPerResource = Dict[Resource, List[ResourceOccupation]]

# sorted list of resource occupations per agent; resource occupations overlap by release time!
SortedResourceOccupationsPerAgent = Dict[int, List[ResourceOccupation]]

# list of resource occupations per agent and time-step (there are multiple resource occupations if the previous resource is not released yet)
ResourceOccupationPerAgentAndTimeStep = Dict[Tuple[int, int], List[ResourceOccupation]]

ScheduleAsResourceOccupations = NamedTuple(
    "ScheduleAsResourceOccupations",
    [
        ("sorted_resource_occupations_per_resource", SortedResourceOccupationsPerResource),
        ("sorted_resource_occupations_per_agent", SortedResourceOccupationsPerAgent),
        ("resource_occupations_per_agent_and_time_step", ResourceOccupationPerAgentAndTimeStep),
    ],
)

TimeWindow = ResourceOccupation
# list of time windows per resource sorted by lower bound; time windows may overlap!
TimeWindowsPerResourceAndTimeStep = Dict[Tuple[Resource, int], List[TimeWindow]]

# sorted list of time windows per agent
TimeWindowsPerAgentSortedByLowerBound = Dict[int, List[TimeWindow]]

SchedulingProblemInTimeWindows = NamedTuple(
    "SchedulingProblemInTimeWindows",
    [
        ("time_windows_per_resource_and_time_step", TimeWindowsPerResourceAndTimeStep),
        ("time_windows_per_agent_sorted_by_lower_bound", TimeWindowsPerAgentSortedByLowerBound),
    ],
)


def extract_time_windows(
    route_dag_constraints_dict: RouteDAGConstraintsDict, minimum_travel_time_dict: Dict[int, int], release_time: int
) -> SchedulingProblemInTimeWindows:
    """Derive time windows from constraints.

    Parameters
    ----------
    route_dag_constraints_dict
    minimum_travel_time_dict
    release_time

    Returns
    -------
    """
    time_windows_per_resource_and_time_step: TimeWindowsPerResourceAndTimeStep = {}
    time_windows_per_agent_sorted_by_lower_bound: TimeWindowsPerAgentSortedByLowerBound = {}

    for agent_id, route_dag_constraints in route_dag_constraints_dict.items():
        for waypoint, earliest in route_dag_constraints.earliest.items():
            # TODO actually, we should take latest leaving event for resource!
            latest = route_dag_constraints.latest[waypoint] + minimum_travel_time_dict[agent_id] + release_time
            resource = Resource(*waypoint.position)
            time_window = TimeWindow(interval=LeftClosedInterval(earliest, latest), resource=resource, direction=waypoint.direction, agent_id=agent_id)
            for time_step in range(earliest, latest):
                time_windows_per_resource_and_time_step.setdefault((resource, time_step), [])
                # TODO duplicates because of different directions?
                if time_window not in time_windows_per_resource_and_time_step:
                    time_windows_per_resource_and_time_step[(resource, time_step)].append(time_window)
            time_windows_per_agent_sorted_by_lower_bound.setdefault(agent_id, []).append(time_window)
    for _, time_windows in time_windows_per_agent_sorted_by_lower_bound.items():
        time_windows.sort(key=lambda t_w: t_w.interval.from_incl,)

    return SchedulingProblemInTimeWindows(
        time_windows_per_agent_sorted_by_lower_bound=time_windows_per_agent_sorted_by_lower_bound,
        time_windows_per_resource_and_time_step=time_windows_per_resource_and_time_step,
    )


def extract_resource_occupations(schedule: TrainrunDict, release_time: int) -> ScheduleAsResourceOccupations:
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
                interval=LeftClosedInterval(from_incl, to_excl), resource=resource, direction=entry_event.waypoint.direction, agent_id=agent_id
            )
            resource_occupations_per_resource.setdefault(resource, []).append(ro)
            resource_occupations_per_agent[agent_id].append(ro)
            for time_step in range(from_incl, to_excl):
                resource_occupations_per_agent_and_timestep.setdefault((agent_id, time_step), []).append(ro)

    # sort occupations by interval's lower bound
    for _, occupations in resource_occupations_per_resource.items():
        occupations.sort(key=lambda _ro: _ro.interval.from_incl)

    return ScheduleAsResourceOccupations(resource_occupations_per_resource, resource_occupations_per_agent, resource_occupations_per_agent_and_timestep)


def verify_schedule_as_resource_occupations(schedule_as_resource_occupations: ScheduleAsResourceOccupations, release_time: int):  # noqa: C901
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
            assert ro_2.interval.from_incl >= ro_1.interval.to_excl, f"{ro_1} {ro_2}"

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
        schedule_as_resource_occupations.sorted_resource_occupations_per_agent.values(),
    ):
        for ro in occupations:
            assert ro.interval.to_excl > ro.interval.from_incl
