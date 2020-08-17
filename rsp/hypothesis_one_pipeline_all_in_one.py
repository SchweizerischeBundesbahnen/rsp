from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

from rsp.hypothesis_one_data_analysis import hypothesis_one_data_analysis
from rsp.utils.data_types import ExperimentAgenda
from rsp.utils.data_types import ParameterRanges
from rsp.utils.data_types import ParameterRangesAndSpeedData
from rsp.utils.experiments import AVAILABLE_CPUS
from rsp.utils.experiments import create_experiment_agenda
from rsp.utils.experiments import create_experiment_folder_name
from rsp.utils.experiments import EXPERIMENT_AGENDA_SUBDIRECTORY_NAME
from rsp.utils.experiments import hypothesis_one_gen_schedule
from rsp.utils.experiments import run_experiment_agenda
from rsp.utils.file_utils import check_create_folder


def get_agenda_pipeline_params_001_simple_setting() -> ParameterRangesAndSpeedData:
    parameter_ranges = ParameterRanges(agent_range=[2, 2, 1],
                                       size_range=[18, 18, 1],
                                       in_city_rail_range=[2, 2, 1],
                                       out_city_rail_range=[1, 1, 1],
                                       city_range=[2, 2, 1],
                                       earliest_malfunction=[5, 5, 1],
                                       malfunction_duration=[20, 20, 1],
                                       number_of_shortest_paths_per_agent=[10, 10, 1],
                                       max_window_size_from_earliest=[np.inf, np.inf, 1],
                                       asp_seed_value=[94, 94, 1],
                                       # route change is penalized the same as 60 seconds delay
                                       weight_route_change=[60, 60, 1],
                                       weight_lateness_seconds=[1, 1, 1],
                                       )
    # Define the desired speed profiles
    speed_data = {1.: 0.25,  # Fast passenger train
                  1. / 2.: 0.25,  # Fast freight train
                  1. / 3.: 0.25,  # Slow commuter train
                  1. / 4.: 0.25}  # Slow freight train
    return ParameterRangesAndSpeedData(parameter_ranges=parameter_ranges, speed_data=speed_data)


def get_agenda_pipeline_params_002_a_bit_more_advanced() -> ParameterRangesAndSpeedData:
    parameter_ranges = ParameterRanges(agent_range=[50, 50, 1],
                                       size_range=[40, 40, 1],
                                       in_city_rail_range=[3, 3, 1],
                                       out_city_rail_range=[2, 2, 1],
                                       city_range=[5, 5, 1],
                                       earliest_malfunction=[10, 10, 1],
                                       malfunction_duration=[20, 20, 1],
                                       number_of_shortest_paths_per_agent=[10, 10, 1],
                                       max_window_size_from_earliest=[30, 30, 1],
                                       asp_seed_value=[94, 94, 1],
                                       # route change is penalized the same as 60 seconds delay
                                       weight_route_change=[60, 60, 1],
                                       weight_lateness_seconds=[1, 1, 1]
                                       )
    # Define the desired speed profiles
    speed_data = {1.: 0.25,  # Fast passenger train
                  1. / 2.: 0.25,  # Fast freight train
                  1. / 3.: 0.25,  # Slow commuter train
                  1. / 4.: 0.25}  # Slow freight train
    return ParameterRangesAndSpeedData(parameter_ranges=parameter_ranges, speed_data=speed_data)


def get_agenda_pipeline_params_003_a_bit_more_advanced() -> ParameterRangesAndSpeedData:
    parameter_ranges = ParameterRanges(
        agent_range=[50, 150, 100],
        size_range=[50, 50, 1],
        in_city_rail_range=[3, 3, 1],
        out_city_rail_range=[2, 2, 1],
        city_range=[10, 10, 1],
        earliest_malfunction=[1, 1, 1],
        malfunction_duration=[50, 50, 1],
        number_of_shortest_paths_per_agent=[10, 10, 1],
        max_window_size_from_earliest=[60, 60, 1],
        asp_seed_value=[94, 94, 1],
        # route change is penalized the same as 30 seconds delay
        weight_route_change=[30, 30, 1],
        weight_lateness_seconds=[1, 1, 1]
    )
    # Define the desired speed profiles
    speed_data = {1.: 0.25,  # Fast passenger train
                  1. / 2.: 0.25,  # Fast freight train
                  1. / 3.: 0.25,  # Slow commuter train
                  1. / 4.: 0.25}  # Slow freight train
    return ParameterRangesAndSpeedData(parameter_ranges=parameter_ranges, speed_data=speed_data)


def hypothesis_one_pipeline_all_in_one(
        parameter_ranges_and_speed_data: ParameterRangesAndSpeedData,
        experiment_base_directory: Optional[str] = '.',
        experiment_ids: Optional[List[int]] = None,
        qualitative_analysis_experiment_ids: Optional[List[int]] = None,
        asp_export_experiment_ids: Optional[List[int]] = None,
        experiment_name: str = "exp_hypothesis_one",
        run_analysis: bool = True,
        parallel_compute: int = AVAILABLE_CPUS // 2,
        flatland_seed: int = 12,
        # take only half of avilable cpus so the machine stays responsive
        experiments_per_grid_element: int = 1
) -> Tuple[str, ExperimentAgenda]:
    """
    Run full pipeline A.1 -> A.2 - B - C

    Parameters
    ----------
    experiment_name
    experiment_ids
        filter for experiment ids (data generation)
    qualitative_analysis_experiment_ids
        filter for data analysis on the generated data
    asp_export_experiment_ids
        filter for data analysis on the generated data
    experiment_base_directory
        base directory from the same agenda with serialized schedule and malfunction.
        - if given, the schedule is not re-generated
        - if not given, a schedule is generate in a non-deterministc fashion
    parallel_compute
        degree of parallelization; must not be larger than available cores.
    run_analysis
    parameter_ranges_and_speed_data
    parallel_compute

    Returns
    -------
    str
        experiment_base_folder_name
    """

    # A.1 Experiment Planning: Create an experiment agenda out of the parameter ranges
    experiment_data_directory_name = create_experiment_folder_name(experiment_name)
    experiment_data_directory = f'{experiment_base_directory}/{experiment_data_directory_name}'
    experiment_agenda_directory = f'{experiment_base_directory}/{EXPERIMENT_AGENDA_SUBDIRECTORY_NAME}'

    check_create_folder(experiment_base_directory)
    check_create_folder(experiment_data_directory)
    check_create_folder(experiment_agenda_directory)

    # Create schedule and malfunction
    hypothesis_one_gen_schedule(
        parameter_ranges_and_speed_data=parameter_ranges_and_speed_data,
        experiment_agenda_directory=experiment_agenda_directory
    )

    experiment_agenda = create_experiment_agenda(experiment_name=experiment_name,
                                                 parameter_ranges_and_speed_data=parameter_ranges_and_speed_data,
                                                 flatland_seed=flatland_seed,
                                                 experiments_per_grid_element=experiments_per_grid_element)

    experiment_output_directory = run_experiment_agenda(
        # TODO should only be folder and optional filter of experiment_ids?
        experiment_agenda=experiment_agenda,
        run_experiments_parallel=parallel_compute,
        show_results_without_details=True,
        verbose=False,
        experiment_ids=experiment_ids,
        experiment_base_directory=experiment_base_directory
    )

    # C. Experiment Analysis
    if run_analysis:
        hypothesis_one_data_analysis(
            # TODO SIM-650 should be run_folder_name?
            experiment_base_directory=experiment_output_directory,
            analysis_2d=True,
            qualitative_analysis_experiment_ids=qualitative_analysis_experiment_ids,
            asp_export_experiment_ids=asp_export_experiment_ids
        )

    return experiment_output_directory, experiment_agenda
