import shutil
from typing import List
from typing import Optional

import numpy as np

from rsp.experiment_solvers.data_types import ScheduleAndMalfunction
from rsp.hypothesis_one_data_analysis import hypothesis_one_data_analysis
from rsp.logger import rsp_logger
from rsp.utils.data_types import ExperimentAgenda
from rsp.utils.data_types import ParameterRanges
from rsp.utils.data_types import ParameterRangesAndSpeedData
from rsp.utils.experiments import AVAILABLE_CPUS
from rsp.utils.experiments import create_experiment_agenda
from rsp.utils.experiments import exists_schedule_and_malfunction
from rsp.utils.experiments import EXPERIMENT_AGENDA_SUBDIRECTORY_NAME
from rsp.utils.experiments import gen_malfunction
from rsp.utils.experiments import load_experiment_agenda_from_file
from rsp.utils.experiments import load_schedule_and_malfunction
from rsp.utils.experiments import remove_dummy_stuff_from_schedule_and_malfunction_pickle
from rsp.utils.experiments import run_experiment_agenda
from rsp.utils.experiments import save_experiment_agenda_and_hash_to_file
from rsp.utils.experiments import save_schedule_and_malfunction


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


def get_agenda_pipeline_malfunction_variation(schedule_gen) -> ParameterRangesAndSpeedData:
    if schedule_gen:
        parameter_ranges = ParameterRanges(agent_range=[100, 100, 1],
                                           size_range=[50, 50, 1],
                                           in_city_rail_range=[3, 3, 1],
                                           out_city_rail_range=[2, 2, 1],
                                           city_range=[10, 10, 1],
                                           earliest_malfunction=[1, 1, 1],
                                           malfunction_duration=[50, 50, 1],
                                           number_of_shortest_paths_per_agent=[10, 10, 1],
                                           max_window_size_from_earliest=[60, 60, 1],
                                           asp_seed_value=[94, 94, 1],
                                           # route change is penalized the same as 1 second delay
                                           weight_route_change=[30, 30, 1],
                                           weight_lateness_seconds=[1, 1, 1]
                                           )
    else:
        parameter_ranges = ParameterRanges(agent_range=[100, 100, 1],
                                           size_range=[50, 50, 1],
                                           in_city_rail_range=[3, 3, 1],
                                           out_city_rail_range=[2, 2, 1],
                                           city_range=[10, 10, 1],
                                           earliest_malfunction=[169, 250, 10],
                                           malfunction_duration=[50, 50, 1],
                                           number_of_shortest_paths_per_agent=[10, 10, 1],
                                           max_window_size_from_earliest=[60, 60, 1],
                                           asp_seed_value=[94, 94, 1],
                                           # route change is penalized the same as 1 second delay
                                           weight_route_change=[30, 30, 1],
                                           weight_lateness_seconds=[1, 1, 1]
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
                            parallel_compute: int = AVAILABLE_CPUS // 2,
                            # take only half of avilable cpus so the machine stays responsive
                            gen_only: bool = False,
                            experiments_per_grid_element: int = 1
                            ) -> str:
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
    copy_agenda_from_base_directory
        base directory from the same agenda with serialized schedule and malfunction.
        - if given, the schedule is not re-generated
        - if not given, a schedule is generate in a non-deterministc fashion
    parallel_compute
        degree of parallelization; must not be larger than available cores.
    run_analysis
    parameter_ranges_and_speed_data
    parallel_compute
    gen_only

    Returns
    -------
    str
        experiment_base_folder_name
    """

    # A.1 Experiment Planning: Create an experiment agenda out of the parameter ranges
    experiment_agenda = create_experiment_agenda(
        experiment_name=experiment_name,
        parameter_ranges_and_speed_data=parameter_ranges_and_speed_data,
        experiments_per_grid_element=experiments_per_grid_element,
    )
    # [ A.2 -> B ]* -> C
    experiment_base_folder_name = hypothesis_one_pipeline_without_setup(
        copy_agenda_from_base_directory=copy_agenda_from_base_directory,
        experiment_agenda=experiment_agenda,
        experiment_ids=experiment_ids,
        parallel_compute=parallel_compute,
        qualitative_analysis_experiment_ids=qualitative_analysis_experiment_ids,
        asp_export_experiment_ids=asp_export_experiment_ids,
        run_analysis=run_analysis,
        gen_only=gen_only
    )
    return experiment_base_folder_name


def hypothesis_one_pipeline_without_setup(experiment_agenda: ExperimentAgenda,
                                          experiment_ids: Optional[List[int]] = None,
                                          qualitative_analysis_experiment_ids: Optional[List[int]] = None,
                                          asp_export_experiment_ids: Optional[List[int]] = None,
                                          copy_agenda_from_base_directory: Optional[str] = None,
                                          run_analysis: bool = True,
                                          parallel_compute: int = AVAILABLE_CPUS // 2,
                                          # take only half of avilable cpus so the machine stays responsive
                                          gen_only: bool = False
                                          ):
    """Run pipeline from A.2 -> C.

    [A.2 -> B]* Experiments: setup, then run
    """
    if experiment_ids is None:
        experiment_ids = [exp.experiment_id for exp in experiment_agenda.experiments]
    # TODO remove again

    if copy_agenda_from_base_directory is not None:
        for experiment_id in experiment_ids:
            remove_dummy_stuff_from_schedule_and_malfunction_pickle(
                experiment_agenda_directory=f"{copy_agenda_from_base_directory}/{EXPERIMENT_AGENDA_SUBDIRECTORY_NAME}",
                experiment_id=experiment_id
            )

    experiment_base_folder_name, _ = run_experiment_agenda(
        experiment_agenda=experiment_agenda,
        run_experiments_parallel=parallel_compute,
        show_results_without_details=True,
        verbose=False,
        experiment_ids=experiment_ids,
        copy_agenda_from_base_directory=copy_agenda_from_base_directory,
        gen_only=gen_only
    )
    if gen_only:
        return experiment_base_folder_name

    # C. Experiment Analysis
    if run_analysis:
        hypothesis_one_data_analysis(
            experiment_base_directory=experiment_base_folder_name,
            analysis_2d=True,
            qualitative_analysis_experiment_ids=qualitative_analysis_experiment_ids,
            asp_export_experiment_ids=asp_export_experiment_ids
        )
    return experiment_base_folder_name


def hypothesis_one_main():
    rsp_logger.info(f"RUN FULL (WITH SCHEDULE GENERATION)")
    parameter_ranges_and_speed_data = get_agenda_pipeline_malfunction_variation()
    hypothesis_one_pipeline(
        parameter_ranges_and_speed_data=parameter_ranges_and_speed_data,
        qualitative_analysis_experiment_ids=[],
        asp_export_experiment_ids=[],
        parallel_compute=AVAILABLE_CPUS // 2  # take only half of avilable cpus so the machine stays responsive
    )


def hypothesis_one_rerun_without_regen_schedule(copy_agenda_from_base_directory: str, nb_runs: int = 1):
    rsp_logger.info(f"RERUN from {copy_agenda_from_base_directory} WITHOUT REGEN SCHEDULE")
    experiment_agenda_directory = f'{copy_agenda_from_base_directory}/{EXPERIMENT_AGENDA_SUBDIRECTORY_NAME}'
    experiment_agenda = load_experiment_agenda_from_file(experiment_agenda_directory)

    experiment_agenda = ExperimentAgenda(experiment_name=experiment_agenda.experiment_name,
                                         experiments=experiment_agenda.experiments * nb_runs)

    experiment_ids = [
        experiment.experiment_id
        for experiment in experiment_agenda.experiments
        if exists_schedule_and_malfunction(
            experiment_agenda_directory=experiment_agenda_directory,
            experiment_id=experiment.experiment_id)
    ]

    hypothesis_one_pipeline_without_setup(
        experiment_agenda=experiment_agenda,
        qualitative_analysis_experiment_ids=[],
        asp_export_experiment_ids=[],
        copy_agenda_from_base_directory=copy_agenda_from_base_directory,
        parallel_compute=AVAILABLE_CPUS // 2,  # take only half of avilable cpus so the machine stays responsive
        experiment_ids=experiment_ids
    )


def hypothesis_one_rerun_one_experiment_with_new_params_same_schedule(
        copy_agenda_from_base_directory: str,
        experiment_parameters: ParameterRangesAndSpeedData = None,
        base_experiment_id: int = 0,
        experiment_agenda_name: Optional[str] = None,
        parallel_compute: int = AVAILABLE_CPUS // 2
):
    """Simple method to run experiments with new parameters without the need to
    generate the schedule.
    Takes the schedule of the `base_experiment_id` and prepares a new agenda with malfunction according from the parameters.
    The new agenda is prepared in temporary folder which is then passed to the pipeline `hypothesis_one_pipeline_without_setup`
    using `copy_agenda_from_base_directory` from the temporary folder.
    .

    Parameters
    ----------
    copy_agenda_from_base_directory
        Agenda containing schedule.
    experiment_parameters
        New set of parameters that will be used for the experiments
    base_experiment_id
        The experiment to choose from the original agenda
    experiment_agenda_name
    parallel_compute

    Returns
    -------
    """
    rsp_logger.info(f"RERUN from {copy_agenda_from_base_directory} WITHOUT REGEN SCHEDULE")

    # Load the previous agenda
    rsp_logger.info("Loading Agenda and Schedule")
    experiment_agenda_directory = f'{copy_agenda_from_base_directory}/{EXPERIMENT_AGENDA_SUBDIRECTORY_NAME}'
    loaded_experiment_agenda = load_experiment_agenda_from_file(experiment_agenda_directory)

    if experiment_agenda_name is not None:
        loaded_experiment_agenda = ExperimentAgenda(
            experiment_name=experiment_agenda_name,
            experiments=loaded_experiment_agenda.experiments
        )

    loaded_schedule_and_malfunction = load_schedule_and_malfunction(
        experiment_agenda_directory=experiment_agenda_directory, experiment_id=base_experiment_id)

    # Create new experiment agenda
    rsp_logger.info("Creating New Agenda")
    experiment_agenda = create_experiment_agenda(
        experiment_name=loaded_experiment_agenda.experiment_name,
        parameter_ranges_and_speed_data=experiment_parameters,
        experiments_per_grid_element=1,
    )

    # Save the new agenda into a tmp folder
    rsp_logger.info("Saving New Agenda")
    tmp_experiment_folder = './tmp_experiment_folder'
    shutil.rmtree(tmp_experiment_folder, ignore_errors=True)
    tmp_experiment_agenda_directory = f'{tmp_experiment_folder}/{EXPERIMENT_AGENDA_SUBDIRECTORY_NAME}'

    save_experiment_agenda_and_hash_to_file(experiment_agenda_folder_name=tmp_experiment_agenda_directory,
                                            experiment_agenda=experiment_agenda)

    # Generate the malfunction experiments
    rsp_logger.info("Generating Malfunctions")
    for experiment in experiment_agenda.experiments:
        rsp_logger.info("Generating malfunction for experiment {}".format(experiment.experiment_id))

        malfunction = gen_malfunction(
            malfunction_duration=experiment.malfunction_duration,
            earliest_malfunction=experiment.earliest_malfunction,
            schedule_trainruns=loaded_schedule_and_malfunction.schedule_experiment_result.trainruns_dict)
        rsp_logger.info("Generated malfunction for agent {} at time {} for {} steps".format(malfunction.agent_id,
                                                                                            malfunction.time_step,
                                                                                            malfunction.malfunction_duration))
        # Use the same schedule and only vary the malfunction
        schedule_and_malfunction = ScheduleAndMalfunction(
            schedule_problem_description=loaded_schedule_and_malfunction.schedule_problem_description,
            schedule_experiment_result=loaded_schedule_and_malfunction.schedule_experiment_result,
            experiment_malfunction=malfunction)
        # Save the newly generated schedule malfunction pairs
        save_schedule_and_malfunction(schedule_and_malfunction=schedule_and_malfunction,
                                      experiment_agenda_directory=tmp_experiment_agenda_directory,
                                      experiment_id=experiment.experiment_id)

    # Run Pipeline
    rsp_logger.info("Running Pipeline with new Parameters and Malfunctions")

    hypothesis_one_pipeline_without_setup(
        experiment_agenda=experiment_agenda,
        qualitative_analysis_experiment_ids=[],
        asp_export_experiment_ids=[],
        copy_agenda_from_base_directory=tmp_experiment_folder,
        parallel_compute=parallel_compute
    )
    # Cleanup files
    shutil.rmtree(tmp_experiment_folder, ignore_errors=True)


def hypothesis_one_rerun_with_regen_schedule(copy_agenda_from_base_directory: str):
    rsp_logger.info(f"RERUN from {copy_agenda_from_base_directory} WITH REGEN SCHEDULE")
    experiment_agenda = load_experiment_agenda_from_file(copy_agenda_from_base_directory + "/agenda")
    hypothesis_one_pipeline_without_setup(
        experiment_agenda=experiment_agenda,
        qualitative_analysis_experiment_ids=[],
        asp_export_experiment_ids=[],
        parallel_compute=1,
        experiment_ids=list(range(10))
    )


def hypothesis_one_gen_schedule(parameter_ranges_and_speed_data: ParameterRangesAndSpeedData = None):
    rsp_logger.info("GEN SCHEDULE ONLY")

    experiment_base_folder_name = hypothesis_one_pipeline(
        parameter_ranges_and_speed_data=parameter_ranges_and_speed_data,
        gen_only=True,
        experiment_ids=None,
        parallel_compute=1,
    )
    return experiment_base_folder_name


def hypothesis_one_malfunction_analysis(
        agenda_folder: str = None,
        base_experiment_id: int = 0,
        parallel_compute: int = AVAILABLE_CPUS // 2, ):
    rsp_logger.info(f"MALFUNCTION INVESTIGATION")
    # Generate Schedule
    if agenda_folder is None:
        parameter_ranges_and_speed_data = get_agenda_pipeline_malfunction_variation(schedule_gen=True)
        experiment_base_folder_name = hypothesis_one_gen_schedule(parameter_ranges_and_speed_data)
    else:
        experiment_base_folder_name = agenda_folder
    # Generate examples with different malfunctions
    parameter_ranges_and_speed_data = get_agenda_pipeline_malfunction_variation(schedule_gen=False)

    hypothesis_one_rerun_one_experiment_with_new_params_same_schedule(
        experiment_base_folder_name,
        experiment_parameters=parameter_ranges_and_speed_data,
        base_experiment_id=base_experiment_id,
        experiment_agenda_name=f"agent_{base_experiment_id}_malfunction",
        parallel_compute=parallel_compute
    )


if __name__ == '__main__':
    # do not commit your own calls !
    pass
    hypothesis_one_malfunction_analysis(
        agenda_folder='../rsp-data/agent_0_malfunction_2020_05_27T19_45_49/',
        parallel_compute=1
    )
