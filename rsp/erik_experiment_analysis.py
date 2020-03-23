"""Analysis of the experiment data for hypothesis one.

Hypothesis 1:
    We can compute good recourse actions, i.e., an adapted plan within the time budget,
    if all the variables are fixed, except those related to services that are affected by the
    disruptions implicitly or explicitly.

Hypothesis 2:
    Machine learning can predict services that are affected by disruptions implicitly or
    explicitly. Hypothesis 3: If hypothesis 2 is true, in addition, machine
    learning can predict the state of the system in the next time period
    after re-scheduling.
"""
from typing import List

from pandas import DataFrame

from rsp.utils.data_types import ExperimentAgenda
from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.data_types import convert_list_of_experiment_results_analysis_to_data_frame
from rsp.utils.data_types import convert_pandas_series_experiment_results_analysis
from rsp.utils.experiment_render_utils import visualize_experiment
from rsp.utils.experiments import EXPERIMENT_AGENDA_SUBDIRECTORY_NAME
from rsp.utils.experiments import EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME
from rsp.utils.experiments import EXPERIMENT_DATA_SUBDIRECTORY_NAME
from rsp.utils.experiments import load_and_expand_experiment_results_from_data_folder
from rsp.utils.experiments import load_experiment_agenda_from_file
from rsp.utils.file_utils import check_create_folder


def hypothesis_one_qualitative_analysis(experiment_base_directory: str,
                                        analysis_2d: bool = False,
                                        analysis_3d: bool = False,
                                        route_dag: bool = True,
                                        qualitative_analysis_experiment_ids: List[int] = None,
                                        flatland_rendering: bool = False):
    """

    Parameters
    ----------
    route_dag
    experiment_base_directory
    analysis_2d
    analysis_3d
    qualitative_analysis_experiment_ids
    flatland_rendering
    debug
    """
    # Import the desired experiment results
    experiment_analysis_directory = f'{experiment_base_directory}/{EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME}/'
    experiment_data_directory = f'{experiment_base_directory}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}'
    experiment_agenda_directory = f'{experiment_base_directory}/{EXPERIMENT_AGENDA_SUBDIRECTORY_NAME}'

    # Create output directoreis
    check_create_folder(experiment_analysis_directory)

    experiment_results_list: List[ExperimentResultsAnalysis] = load_and_expand_experiment_results_from_data_folder(
        experiment_data_directory)
    experiment_agenda: ExperimentAgenda = load_experiment_agenda_from_file(experiment_agenda_directory)

    print(experiment_data_directory)
    print(experiment_agenda)

    # convert to data frame for statistical analysis
    experiment_data: DataFrame = convert_list_of_experiment_results_analysis_to_data_frame(experiment_results_list)

    # qualitative explorative analysis
    if qualitative_analysis_experiment_ids:
        filtered_experiments = list(filter(
            lambda experiment: experiment.experiment_id in qualitative_analysis_experiment_ids,
            experiment_agenda.experiments))
        for experiment in filtered_experiments:
            row = experiment_data[experiment_data['experiment_id'] == experiment.experiment_id].iloc[0]
            experiment_results_analysis: ExperimentResultsAnalysis = convert_pandas_series_experiment_results_analysis(
                row)

            visualize_experiment(experiment_parameters=experiment,
                                 experiment_results_analysis=experiment_results_analysis,
                                 experiment_analysis_directory=experiment_analysis_directory,
                                 analysis_2d=analysis_2d,
                                 analysis_3d=analysis_3d,
                                 route_dag=route_dag,
                                 flatland_rendering=flatland_rendering)


if __name__ == '__main__':
    hypothesis_one_qualitative_analysis(experiment_base_directory='./exp_hypothesis_one_2020_03_17T07_01_49',
                                        analysis_2d=True,
                                        analysis_3d=False,
                                        route_dag=False,
                                        qualitative_analysis_experiment_ids=[12])
