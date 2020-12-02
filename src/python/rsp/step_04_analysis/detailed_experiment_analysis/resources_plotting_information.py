# Plotting Data Structures
from typing import Dict
from typing import NamedTuple
from typing import Optional
from typing import Tuple

from flatland.core.grid.grid_utils import coordinate_to_position

from rsp.utils.resource_occupation import ResourceOccupation
from rsp.utils.resource_occupation import ScheduleAsResourceOccupations

Resource = NamedTuple("Resource", [("row", int), ("column", int)])

ResourceSorting = Dict[Resource, int]

PlottingInformation = NamedTuple("PlottingInformation", [("sorting", ResourceSorting), ("dimensions", Tuple[int, int]), ("grid_width", int)])


def extract_plotting_information(
    schedule_as_resource_occupations: ScheduleAsResourceOccupations, grid_depth: int, sorting_agent_id: Optional[int] = None
) -> PlottingInformation:
    """Extract plotting information.

    Parameters
    ----------
    schedule_as_resource_occupations:
    grid_depth
        Ranges of the window to be shown, used for consistent plotting
    sorting_agent_id
        agent id to be used for sorting the resources
    Returns
    -------
    PlottingInformation
        The extracted plotting information.
    """
    sorted_index = 0
    max_time = 0
    sorting = {}
    # If specified, sort according to path of agent with sorting_agent_id
    if sorting_agent_id is not None and sorting_agent_id in schedule_as_resource_occupations.sorted_resource_occupations_per_agent:
        for resource_occupation in sorted(schedule_as_resource_occupations.sorted_resource_occupations_per_agent[sorting_agent_id]):
            position = coordinate_to_position(grid_depth, [resource_occupation.resource])[0]
            time = resource_occupation.interval.to_excl
            if time > max_time:
                max_time = time
            if position not in sorting:
                sorting[position] = sorted_index
                sorted_index += 1

    # Sort the rest of the resources according to agent handle sorting
    for _, sorted_resource_occupations in sorted(schedule_as_resource_occupations.sorted_resource_occupations_per_agent.items()):
        for resource_occupation in sorted_resource_occupations:
            resource_occupation: ResourceOccupation = resource_occupation
            time = resource_occupation.interval.to_excl
            if time > max_time:
                max_time = time
            position = coordinate_to_position(grid_depth, [resource_occupation.resource])[0]
            if position not in sorting:
                sorting[position] = sorted_index
                sorted_index += 1
    max_ressource = max(list(sorting.values()))
    plotting_information = PlottingInformation(sorting=sorting, dimensions=(max_ressource, max_time), grid_width=grid_depth)
    return plotting_information
