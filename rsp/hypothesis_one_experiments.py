from typing import List
from typing import Optional

import numpy as np

from rsp.hypothesis_one_data_analysis import hypothesis_one_data_analysis
from rsp.utils.data_types import ExperimentAgenda
from rsp.utils.data_types import ParameterRanges
from rsp.utils.data_types import ParameterRangesAndSpeedData
from rsp.utils.experiments import AVAILABLE_CPUS
from rsp.utils.experiments import create_experiment_agenda
from rsp.utils.experiments import run_experiment_agenda


def get_first_agenda_pipeline_params() -> ParameterRangesAndSpeedData:
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


def hypothesis_one_pipeline(parameter_ranges_and_speed_data: ParameterRangesAndSpeedData,
                            experiment_ids: Optional[List[int]] = None,
                            qualitative_analysis_experiment_ids: Optional[List[int]] = None,
                            asp_export_experiment_ids: Optional[List[int]] = None,
                            copy_agenda_from_base_directory: Optional[str] = None,
                            experiment_name: str = "exp_hypothesis_one",
                            run_analysis: bool = True,
                            parallel_compute: int = AVAILABLE_CPUS) -> str:
    """
    Run full pipeline A.1 -> A.2 - B - C

    Parameters
    ----------
    experiment_name
    parameter_ranges_and_speed_data
    parallel_compute
    run_anaylsis
    parameter_ranges
    speed_data
    experiment_ids
        filter for experiment ids (data generation)
    qualitative_analysis_experiment_ids
        filter for data analysis on the generated data
    asp_export_experiment_ids
        filter for data analysis on the generated data
    copy_agenda_from_base_directory
        base directory from the same agenda with serialized schedule and malfunction.
        - if given, the schedule is not re-generated
        - if not given, a schedule is generate in a non-deterministc fashion
    parallel_compute
        degree of parallelization; must not be larger than available cores.
    run_analysis

    Returns
    -------
    str
        experiment_base_folder_name
    """

    # A.1 Experiment Planning: Create an experiment agenda out of the parameter ranges
    experiment_agenda = create_experiment_agenda(
        experiment_name=experiment_name,
        parameter_ranges_and_speed_data=parameter_ranges_and_speed_data,
        experiments_per_grid_element=10
    )
    # [ A.2 -> B ]* -> C
    experiment_base_folder_name = hypothesis_one_pipeline_without_setup(
        copy_agenda_from_base_directory=copy_agenda_from_base_directory,
        experiment_agenda=experiment_agenda,
        experiment_ids=experiment_ids,
        parallel_compute=parallel_compute,
        qualitative_analysis_experiment_ids=qualitative_analysis_experiment_ids,
        asp_export_experiment_ids=asp_export_experiment_ids,
        run_analysis=run_analysis)
    return experiment_base_folder_name


def hypothesis_one_pipeline_without_setup(experiment_agenda: ExperimentAgenda,
                                          experiment_ids: Optional[List[int]] = None,
                                          qualitative_analysis_experiment_ids: Optional[List[int]] = None,
                                          asp_export_experiment_ids: Optional[List[int]] = None,
                                          copy_agenda_from_base_directory: Optional[str] = None,
                                          run_analysis: bool = True,
                                          parallel_compute: bool = True):
    """Run pipeline from A.2 -> C."""
    # [A.2 -> B]* Experiments: setup, then run
    experiment_base_folder_name, _ = run_experiment_agenda(
        experiment_agenda=experiment_agenda,
        run_experiments_parallel=parallel_compute,
        show_results_without_details=True,
        verbose=False,
        experiment_ids=experiment_ids,
        copy_agenda_from_base_directory=copy_agenda_from_base_directory
    )
    # C. Experiment Analysis
    if run_analysis:
        hypothesis_one_data_analysis(
            experiment_base_directory=experiment_base_folder_name,
            analysis_2d=True,
            analysis_3d=False,
            qualitative_analysis_experiment_ids=qualitative_analysis_experiment_ids,
            asp_export_experiment_ids=asp_export_experiment_ids
        )
    return experiment_base_folder_name


def hypothesis_one_main():
    parameter_ranges_and_speed_data = get_first_agenda_pipeline_params()
    hypothesis_one_pipeline(
        parameter_ranges_and_speed_data=parameter_ranges_and_speed_data,
        experiment_ids=None,
        parallel_compute=1
    )


if __name__ == '__main__':
    hypothesis_one_main()
