from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from rsp.step_04_analysis.data_analysis_all_in_one import hypothesis_one_data_analysis
from rsp.utils.data_types import ExperimentAgenda
from rsp.utils.data_types import ExperimentParameters
from rsp.utils.data_types import parameter_ranges_and_speed_data_to_hiearchical
from rsp.utils.data_types import ParameterRanges
from rsp.utils.data_types import ParameterRangesAndSpeedData
from rsp.utils.data_types import ReScheduleParametersRange
from rsp.utils.experiments import AVAILABLE_CPUS
from rsp.utils.experiments import create_experiment_agenda_from_infrastructure_and_schedule_ranges
from rsp.utils.experiments import create_experiment_folder_name
from rsp.utils.experiments import create_infrastructure_and_schedule_from_ranges
from rsp.utils.experiments import EXPERIMENT_DATA_SUBDIRECTORY_NAME
from rsp.utils.experiments import list_infrastructure_and_schedule_params_from_base_directory
from rsp.utils.experiments import load_experiment_agenda_from_file
from rsp.utils.experiments import load_experiments_results
from rsp.utils.experiments import run_experiment_agenda
from rsp.utils.experiments import save_experiment_agenda_and_hash_to_file
from rsp.utils.file_utils import check_create_folder
from rsp.utils.rsp_logger import rsp_logger


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


def hypothesis_one_pipeline_all_in_one(
        parameter_ranges_and_speed_data: ParameterRangesAndSpeedData,
        experiment_base_directory: str,
        qualitative_analysis_experiment_ids: Optional[List[int]] = None,
        experiment_name: str = "exp_hypothesis_one",
        run_analysis: bool = True,
        parallel_compute: int = AVAILABLE_CPUS // 2
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

    check_create_folder(experiment_base_directory)
    check_create_folder(experiment_data_directory)

    infra_parameters_range, speed_data, schedule_parameters_range, reschedule_parameters_range = parameter_ranges_and_speed_data_to_hiearchical(
        parameter_ranges_and_speed_data=parameter_ranges_and_speed_data
    )

    create_infrastructure_and_schedule_from_ranges(
        base_directory=experiment_base_directory,
        infrastructure_parameters_range=infra_parameters_range,
        schedule_parameters_range=schedule_parameters_range,
        speed_data=parameter_ranges_and_speed_data.speed_data
    )

    experiment_agenda, experiment_output_directory = list_from_base_directory_and_run_experiment_agenda(
        experiment_base_directory=experiment_base_directory,
        reschedule_parameters_range=reschedule_parameters_range,
        experiment_name=experiment_name,
        parallel_compute=parallel_compute,
    )

    # C. Experiment Analysis
    if run_analysis:
        hypothesis_one_data_analysis(
            experiment_output_directory=experiment_output_directory,
            analysis_2d=True,
            qualitative_analysis_experiment_ids=qualitative_analysis_experiment_ids,
        )

    return experiment_output_directory, experiment_agenda


def list_from_base_directory_and_run_experiment_agenda(
        reschedule_parameters_range: ReScheduleParametersRange,
        experiment_base_directory: str,
        experiment_name: str,
        experiment_output_directory: Optional[str] = None,
        filter_experiment_agenda: Callable[[ExperimentParameters], bool] = None,
        parallel_compute: int = AVAILABLE_CPUS // 2,
        experiments_per_grid_element: int = 1,
        experiment_filter=None):
    """

    Parameters
    ----------
    reschedule_parameters_range
    experiment_base_directory
    experiment_name
    experiment_output_directory
        if given, incremental re-run of this agenda; filter is applied to this
    filter_experiment_agenda
    parallel_compute
    experiments_per_grid_element
    experiment_filter

    Returns
    -------

    """
    if experiment_output_directory is not None:
        rsp_logger.info(f"============================================================================================================")
        rsp_logger.info(f"loading agenda <- {experiment_output_directory}")
        rsp_logger.info(f"============================================================================================================")

        experiment_agenda = load_experiment_agenda_from_file(experiment_folder_name=experiment_output_directory)
        len_before_filtering = len(experiment_agenda.experiments)
        if experiment_filter is not None:
            experiment_agenda = ExperimentAgenda(
                experiment_name=experiment_name,
                experiments=[experiment for experiment in experiment_agenda.experiments if experiment_filter(experiment)]
            )
        rsp_logger.info(
            f"after applying filter, there are {len(experiment_agenda.experiments)} experiments out of {len_before_filtering}: \n" + str(experiment_agenda))
        experiment_data_directory = f'{experiment_output_directory}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}'
        experiment_agenda = ExperimentAgenda(
            experiment_name=experiment_name,
            experiments=[
                experiment for experiment in experiment_agenda.experiments
                if load_experiments_results(experiment_data_folder_name=experiment_data_directory, experiment_id=experiment.experiment_id) is None
            ]
        )
        rsp_logger.info(
            f"after filtering out experiments already run from {experiment_output_directory}, "
            f"there are {len(experiment_agenda.experiments)} experiments: \n" + str(experiment_agenda))
    else:

        infra_parameters_list, infra_schedule_dict = list_infrastructure_and_schedule_params_from_base_directory(
            base_directory=experiment_base_directory
        )

        experiment_agenda = create_experiment_agenda_from_infrastructure_and_schedule_ranges(
            experiment_name=experiment_name,
            reschedule_parameters_range=reschedule_parameters_range,
            infra_parameters_list=infra_parameters_list,
            infra_schedule_dict=infra_schedule_dict,
            experiments_per_grid_element=experiments_per_grid_element
        )
        experiment_output_directory = f"{experiment_base_directory}/" + create_experiment_folder_name(experiment_agenda.experiment_name)
        len_before_filtering = len(experiment_agenda.experiments)
        rsp_logger.info(f"============================================================================================================")
        rsp_logger.info(f"saving agenda -> {experiment_output_directory} with {len_before_filtering} experiments")
        rsp_logger.info(f"============================================================================================================")

        # N.B. save unfiltered agenda! This allows to reload agenda and run a subset of experiments
        save_experiment_agenda_and_hash_to_file(
            output_base_folder=experiment_output_directory,
            experiment_agenda=experiment_agenda
        )

        if experiment_filter is not None:
            experiment_agenda = ExperimentAgenda(
                experiment_name=experiment_name,
                experiments=[experiment for experiment in experiment_agenda.experiments if experiment_filter(experiment)]
            )
        rsp_logger.info(
            f"after applying filter, there are {len(experiment_agenda.experiments)} experiments out of {len_before_filtering}: \n" + str(experiment_agenda))
    experiment_output_directory = run_experiment_agenda(
        experiment_agenda=experiment_agenda,
        run_experiments_parallel=parallel_compute,
        filter_experiment_agenda=filter_experiment_agenda,
        verbose=False,
        experiment_output_directory=experiment_output_directory,
        experiment_base_directory=experiment_base_directory
    )
    return experiment_agenda, experiment_output_directory
