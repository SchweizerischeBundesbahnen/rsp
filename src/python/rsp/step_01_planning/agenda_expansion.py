from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
from deprecated import deprecated
from rsp.scheduling.schedule import Schedule
from rsp.step_01_planning.experiment_parameters_and_ranges import ExperimentAgenda
from rsp.step_01_planning.experiment_parameters_and_ranges import ExperimentParameters
from rsp.step_01_planning.experiment_parameters_and_ranges import InfrastructureParameters
from rsp.step_01_planning.experiment_parameters_and_ranges import InfrastructureParametersRange
from rsp.step_01_planning.experiment_parameters_and_ranges import ParameterRangesAndSpeedData
from rsp.step_01_planning.experiment_parameters_and_ranges import ReScheduleParameters
from rsp.step_01_planning.experiment_parameters_and_ranges import ReScheduleParametersRange
from rsp.step_01_planning.experiment_parameters_and_ranges import ScheduleParameters
from rsp.step_01_planning.experiment_parameters_and_ranges import ScheduleParametersRange
from rsp.step_01_planning.experiment_parameters_and_ranges import SpeedData
from rsp.utils.global_constants import get_defaults
from rsp.utils.global_constants import GlobalConstants
from rsp.utils.rsp_logger import rsp_logger


def span_n_grid(collected_parameters: List, open_dimensions: List) -> list:
    """Recursive function to generate all combinations of parameters given the
    open_dimensions.
    Parameters
    ----------
    collected_parameters: list
        The parameter sets filled so far in the recurions, starts out empty
    open_dimensions: list
        Parameter dimensions we have not yet included in the set
    Returns
    -------
    list of parameter sets for ExperimentAgenda
    """
    full_params = []
    if len(open_dimensions) == 0:
        return [collected_parameters]

    for parameter in open_dimensions[0]:
        full_params.extend(span_n_grid(collected_parameters + [parameter], open_dimensions[1:]))

    return full_params


def expand_range_to_parameter_set(parameter_ranges: List[Tuple[int, int, int]], debug: bool = False) -> List[List[int]]:
    """Expand parameter ranges.

    Parameters
    ----------


    Returns
    -------
    ExperimentAgenda built from the ParameterRanges
    """
    number_of_dimensions = len(parameter_ranges)
    parameter_values = [[] for _ in range(number_of_dimensions)]

    # Setup experiment parameters
    for dim_idx, dimensions in enumerate(parameter_ranges):
        if dimensions[-1] > 1:
            step = np.abs(dimensions[1] - dimensions[0]) / dimensions[-1]
            assert step > 0, "You should defined a number of items in the interval that makes the step < 1.0, check your parameters"
            parameter_values[dim_idx] = np.arange(dimensions[0], dimensions[1], step, dtype=int)
        else:
            parameter_values[dim_idx] = [dimensions[0]]
    full_param_set = span_n_grid([], parameter_values)
    return full_param_set


def expand_infrastructure_parameter_range(
    infrastructure_parameter_range: InfrastructureParametersRange, speed_data: SpeedData, grid_mode: bool = True
) -> List[InfrastructureParameters]:
    expanded = expand_range_to_parameter_set(infrastructure_parameter_range)
    return [
        InfrastructureParameters(*([infra_id] + params[:4] + [grid_mode] + params[4:7] + [speed_data, params[7]])) for infra_id, params in enumerate(expanded)
    ]


def expand_schedule_parameter_range(schedule_parameter_range: ScheduleParametersRange, infra_id: int) -> List[ScheduleParameters]:
    expanded = expand_range_to_parameter_set(schedule_parameter_range)
    return [ScheduleParameters(*([infra_id, schedule_id] + params)) for schedule_id, params in enumerate(expanded)]


def filter_experiment_agenda(current_experiment_parameters, experiment_ids) -> bool:
    return current_experiment_parameters.experiment_id in experiment_ids


def create_experiment_agenda_from_infrastructure_and_schedule_ranges(
    experiment_name: str,
    reschedule_parameters_range: ReScheduleParametersRange,
    infra_parameters_list: List[InfrastructureParameters],
    infra_schedule_dict: Dict[InfrastructureParameters, List[Tuple[ScheduleParameters, Schedule]]],
    experiments_per_grid_element: int = 1,
    global_constants: GlobalConstants = None,
):
    list_of_re_schedule_parameters = [ReScheduleParameters(*expanded) for expanded in expand_range_to_parameter_set(reschedule_parameters_range)]
    infra_parameters_dict = {infra_parameters.infra_id: infra_parameters for infra_parameters in infra_parameters_list}
    experiments = []

    # we want to be able to have different number of schedules for two infrastructures, therefore, we increment counters
    experiment_id = 0
    grid_id = 0
    infra_id_schedule_id = 0
    for infra_id, list_of_schedule_parameters in infra_schedule_dict.items():
        infra_parameters: InfrastructureParameters = infra_parameters_dict[infra_id]
        for schedule_parameters, _ in list_of_schedule_parameters:
            for re_schedule_parameters in list_of_re_schedule_parameters:
                # allow for malfunction agent range to be greater than number of agents of this infrastructure
                if re_schedule_parameters.malfunction_agent_id >= infra_parameters.number_of_agents:
                    continue
                for _ in range(experiments_per_grid_element):
                    experiments.append(
                        ExperimentParameters(
                            experiment_id=experiment_id,
                            schedule_parameters=schedule_parameters,
                            infra_parameters=infra_parameters,
                            grid_id=grid_id,
                            infra_id_schedule_id=infra_id_schedule_id,
                            re_schedule_parameters=re_schedule_parameters,
                        )
                    )
                    experiment_id += 1
                grid_id += 1
            infra_id_schedule_id += 1
    return ExperimentAgenda(
        experiment_name=experiment_name, global_constants=get_defaults() if global_constants is None else global_constants, experiments=experiments
    )


# TODO remove deprecated ParameterRangesAndSpeedData
@deprecated(reason="You should use hiearchy level ranges.")
def create_experiment_agenda_from_parameter_ranges_and_speed_data(
    experiment_name: str,
    parameter_ranges_and_speed_data: ParameterRangesAndSpeedData,
    flatland_seed: int = 12,
    experiments_per_grid_element: int = 1,
    debug: bool = False,
) -> ExperimentAgenda:
    """Create an experiment agenda given a range of parameters defined as
    ParameterRanges.
    Parameters
    ----------

    parameter_ranges_and_speed_data
    flatland_seed
    experiment_name: str
        Name of the experiment
    experiments_per_grid_element: int
        Number of runs with different seed per parameter set we want to run
    debug

    Returns
    -------
    ExperimentAgenda built from the ParameterRanges
    """
    parameter_ranges = parameter_ranges_and_speed_data.parameter_ranges
    number_of_dimensions = len(parameter_ranges)
    parameter_values = [[] for _ in range(number_of_dimensions)]

    # Setup experiment parameters
    for dim_idx, dimensions in enumerate(parameter_ranges):
        if dimensions[-1] > 1:
            parameter_values[dim_idx] = np.arange(dimensions[0], dimensions[1], np.abs(dimensions[1] - dimensions[0]) / dimensions[-1], dtype=int)
        else:
            parameter_values[dim_idx] = [dimensions[0]]
    full_param_set = span_n_grid([], parameter_values)
    experiment_list = []
    for grid_id, parameter_set in enumerate(full_param_set):
        for run_of_this_grid_element in range(experiments_per_grid_element):
            experiment_id = grid_id * experiments_per_grid_element + run_of_this_grid_element
            # 0: size_range
            # 1: agent_range
            # 2: in_city_rail_range
            # 3: out_city_rail_range
            # 4: city_range
            # 5: earliest_malfunction
            # 6: malfunction_duration
            # 7: number_of_shortest_paths_per_agent
            # 8: max_window_size_from_earliest
            # 9: asp_seed_value
            # 10: weight_route_change
            # 11: weight_lateness_seconds
            current_experiment = ExperimentParameters(
                experiment_id=experiment_id,
                grid_id=grid_id,
                infra_id_schedule_id=grid_id,
                infra_parameters=InfrastructureParameters(
                    infra_id=grid_id,
                    speed_data=parameter_ranges_and_speed_data.speed_data,
                    width=parameter_set[0],
                    height=parameter_set[0],
                    flatland_seed_value=flatland_seed + run_of_this_grid_element,
                    max_num_cities=parameter_set[4],
                    # Do we need to have this true?
                    grid_mode=False,
                    max_rail_between_cities=parameter_set[3],
                    max_rail_in_city=parameter_set[2],
                    number_of_agents=parameter_set[1],
                    number_of_shortest_paths_per_agent=parameter_set[7],
                ),
                schedule_parameters=ScheduleParameters(
                    infra_id=grid_id, schedule_id=grid_id, asp_seed_value=parameter_set[9], number_of_shortest_paths_per_agent_schedule=1
                ),
                re_schedule_parameters=ReScheduleParameters(
                    earliest_malfunction=parameter_set[5],
                    malfunction_duration=parameter_set[6],
                    malfunction_agent_id=0,
                    weight_route_change=parameter_set[10],
                    weight_lateness_seconds=parameter_set[11],
                    max_window_size_from_earliest=parameter_set[8],
                    number_of_shortest_paths_per_agent=10,
                    asp_seed_value=94,
                ),
            )

            experiment_list.append(current_experiment)
    experiment_agenda = ExperimentAgenda(experiment_name=experiment_name, global_constants=get_defaults(), experiments=experiment_list)
    rsp_logger.info("Generated an agenda with {} experiments".format(len(experiment_list)))
    return experiment_agenda
