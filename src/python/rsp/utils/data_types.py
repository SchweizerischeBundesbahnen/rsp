"""Data types used in the experiment for the real time rescheduling research
project."""
import pprint
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Tuple

from rsp.scheduling.scheduling_problem import RouteDAGConstraints
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

_pp = pprint.PrettyPrinter(indent=4)


def experiment_freeze_dict_pretty_print(d: RouteDAGConstraintsDict):
    for agent_id, route_dag_constraints in d.items():
        prefix = f"agent {agent_id} "
        experiment_freeze_pretty_print(route_dag_constraints, prefix)


def experiment_freeze_pretty_print(route_dag_constraints: RouteDAGConstraints, prefix: str = ""):
    print(f"{prefix}earliest={_pp.pformat(route_dag_constraints.earliest)}")
    print(f"{prefix}latest={_pp.pformat(route_dag_constraints.latest)}")
