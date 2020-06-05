
# Plotting Data Structures
from typing import NamedTuple, Dict, Tuple

from rsp.experiment_solvers.data_types import ExperimentMalfunction
from rsp.utils.data_types import ScheduleAsResourceOccupations

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
    ('reschedule_delta_as_resource_occupations', ScheduleAsResourceOccupations),
    ('malfunction', ExperimentMalfunction),
    ('plotting_information', PlottingInformation)
])