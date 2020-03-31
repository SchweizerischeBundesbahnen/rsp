from typing import List
from typing import Optional

import numpy as np

from rsp.hypothesis_one_data_analysis import hypothesis_one_data_analysis
from rsp.utils.data_types import ParameterRanges
from rsp.utils.data_types import ParameterRangesAndSpeedData
from rsp.utils.experiments import create_experiment_agenda
from rsp.utils.experiments import run_experiment_agenda


def get_first_agenda_pipeline_params() -> ParameterRangesAndSpeedData:
    parameter_ranges = ParameterRanges(agent_range=[2, 50, 30],
                                       size_range=[30, 50, 10],
                                       in_city_rail_range=[6, 6, 1],
                                       out_city_rail_range=[2, 2, 1],
                                       city_range=[20, 20, 1],
                                       earliest_malfunction=[20, 20, 1],
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


def hypothesis_one_pipeline(parameter_ranges_and_speed_data: ParameterRangesAndSpeedData,
                            experiment_ids: Optional[List[int]] = None,
                            copy_agenda_from_base_directory: Optional[str] = None,
                            experiment_name: str = "exp_hypothesis_one",
                            run_anaylsis: bool = True,
                            parallel_compute: bool = True) -> str:
    """
    Run full pipeline A - B - C

    Parameters
    ----------
    parallel_compute
    run_anaylsis
    parameter_ranges
    speed_data
    experiment_ids
        filter for experiment ids
    copy_agenda_from_base_directory
        base directory from the same agenda with serialized schedule and malfunction.
        - if given, the schedule is not re-generated
        - if not given, a schedule is generate in a non-deterministc fashion

    Returns
    -------
    str
        experiment_base_folder_name
    """

    # A. Experiment Planning: Create an experiment agenda out of the parameter ranges
    experiment_agenda = create_experiment_agenda(
        experiment_name=experiment_name,
        parameter_ranges_and_speed_data=parameter_ranges_and_speed_data,
        experiments_per_grid_element=1
    )
    # B. Experiments: setup, then run
    experiment_base_folder_name, _ = run_experiment_agenda(
        experiment_agenda=experiment_agenda,
        run_experiments_parallel=parallel_compute,
        show_results_without_details=True,
        verbose=False,
        experiment_ids=experiment_ids,
        copy_agenda_from_base_directory=copy_agenda_from_base_directory
    )
    # C. Experiment Analysis
    if run_anaylsis:
        hypothesis_one_data_analysis(
            experiment_base_directory=experiment_base_folder_name,
            analysis_2d=True,
            analysis_3d=False,
            qualitative_analysis_experiment_ids=[]
        )
    return experiment_base_folder_name


def hypothesis_one_main():
    parameter_ranges_and_speed_data = get_first_agenda_pipeline_params()
    hypothesis_one_pipeline(parameter_ranges_and_speed_data=parameter_ranges_and_speed_data,
                            experiment_ids=None,
                            copy_agenda_from_base_directory=None,
                            )


if __name__ == '__main__':
    hypothesis_one_main()
