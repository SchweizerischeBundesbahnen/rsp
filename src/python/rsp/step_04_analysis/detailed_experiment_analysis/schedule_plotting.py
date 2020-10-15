# Plotting Data Structures
from typing import Dict
from typing import NamedTuple
from typing import Optional
from typing import Tuple

from flatland.core.grid.grid_utils import coordinate_to_position
from rsp.experiment_solvers.data_types import ExperimentMalfunction
from rsp.step_03_run.experiment_results_analysis import ExperimentResultsAnalysis
from rsp.utils.data_types import ResourceOccupation
from rsp.utils.data_types import ScheduleAsResourceOccupations
from rsp.utils.data_types_converters_and_validators import extract_resource_occupations
from rsp.utils.data_types_converters_and_validators import verify_schedule_as_resource_occupations
from rsp.utils.global_constants import RELEASE_TIME

Resource = NamedTuple('Resource', [
    ('row', int),
    ('column', int)])

ResourceSorting = Dict[Resource, int]

PlottingInformation = NamedTuple('PlottingInformation', [
    ('sorting', ResourceSorting),
    ('dimensions', Tuple[int, int]),
    ('grid_width', int)])

SchedulePlotting = NamedTuple('SchedulePlotting', [
    ('schedule_as_resource_occupations', ScheduleAsResourceOccupations),
    ('reschedule_full_as_resource_occupations', ScheduleAsResourceOccupations),
    ('reschedule_delta_perfect_as_resource_occupations', ScheduleAsResourceOccupations),
    ('malfunction', ExperimentMalfunction),
    ('plotting_information', PlottingInformation)
])


def extract_schedule_plotting(
        experiment_result: ExperimentResultsAnalysis,
        sorting_agent_id: Optional[int] = None) -> SchedulePlotting:
    """Extract the scheduling information from a experiment data for plotting.

    Parameters
    ----------
    experiment_result
        Experiment results for plotting
    sorting_agent_id
        Agent according to which trainrun the resources will be sorted
    Returns
    -------
    """
    schedule = experiment_result.solution_full
    reschedule_full = experiment_result.solution_full_after_malfunction
    reschedule_delta_perfect = experiment_result.solution_delta_perfect_after_malfunction
    schedule_as_resource_occupations: ScheduleAsResourceOccupations = extract_resource_occupations(
        schedule=schedule,
        release_time=RELEASE_TIME)
    verify_schedule_as_resource_occupations(schedule_as_resource_occupations=schedule_as_resource_occupations,
                                            release_time=RELEASE_TIME)
    reschedule_full_as_resource_occupations = extract_resource_occupations(
        schedule=reschedule_full,
        release_time=RELEASE_TIME)
    verify_schedule_as_resource_occupations(schedule_as_resource_occupations=reschedule_full_as_resource_occupations,
                                            release_time=RELEASE_TIME)
    reschedule_delta_perfect_as_resource_occupations = extract_resource_occupations(
        schedule=reschedule_delta_perfect,
        release_time=RELEASE_TIME)
    verify_schedule_as_resource_occupations(schedule_as_resource_occupations=reschedule_delta_perfect_as_resource_occupations,
                                            release_time=RELEASE_TIME)
    plotting_information: PlottingInformation = extract_plotting_information(
        schedule_as_resource_occupations=schedule_as_resource_occupations,
        grid_depth=experiment_result.experiment_parameters.infra_parameters.width,
        sorting_agent_id=sorting_agent_id)
    return SchedulePlotting(
        schedule_as_resource_occupations=schedule_as_resource_occupations,
        reschedule_full_as_resource_occupations=reschedule_full_as_resource_occupations,
        reschedule_delta_perfect_as_resource_occupations=reschedule_delta_perfect_as_resource_occupations,
        plotting_information=plotting_information,
        malfunction=experiment_result.malfunction
    )


def extract_plotting_information(
        schedule_as_resource_occupations: ScheduleAsResourceOccupations,
        grid_depth: int,
        sorting_agent_id: Optional[int] = None) -> PlottingInformation:
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
