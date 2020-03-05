from typing import List

from pandas import DataFrame

from rsp.utils.analysis_tools import average_over_trials
from rsp.utils.analysis_tools import two_dimensional_scatter_plot
from rsp.utils.data_types import convert_list_of_experiment_results_analysis_to_data_frame
from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.experiments import load_and_expand_experiment_results_from_folder


def _load_and_average(data_folder):
    experiment_results_list: List[ExperimentResultsAnalysis] = load_and_expand_experiment_results_from_folder(data_folder)
    experiment_data: DataFrame = convert_list_of_experiment_results_analysis_to_data_frame(experiment_results_list)
    averaged_data, std_data = average_over_trials(experiment_data)
    return averaged_data


def _extract_times_for_experiment_id(experiment_data, experiment_id):
    row = experiment_data[experiment_data['experiment_id'] == experiment_id].iloc[0]
    time_full_after_malfunction = row['time_full_after_malfunction']
    time_delta_after_malfunction = row['time_delta_after_malfunction']
    return time_delta_after_malfunction, time_full_after_malfunction


def compare_runtimes(data_folder1: str,
                     data_folder2: str,
                     experiment_ids: List[int]):
    experiment_data2: DataFrame = _load_and_average(data_folder2)
    experiment_data1: DataFrame = _load_and_average(data_folder1)

    for experiment_id in experiment_ids:
        time_delta_after_malfunction1, time_full_after_malfunction1 = _extract_times_for_experiment_id(experiment_data1,
                                                                                                       experiment_id)
        time_delta_after_malfunction2, time_full_after_malfunction2 = _extract_times_for_experiment_id(experiment_data2,
                                                                                                       experiment_id)
        print(f"time_delta_after_malfunction: {time_delta_after_malfunction1} --> {time_delta_after_malfunction2}")

    _scatter(experiment_data1=experiment_data1,
             experiment_data2=experiment_data2,
             data_folder=data_folder1,
             column='time_full_after_malfunction')
    _scatter(experiment_data1=experiment_data1,
             experiment_data2=experiment_data2,
             data_folder=data_folder1,
             column='time_delta_after_malfunction')


def _scatter(experiment_data1: DataFrame,
             experiment_data2: DataFrame,
             data_folder: str,
             column: str):
    two_dimensional_scatter_plot(data=experiment_data2,
                                 baseline=experiment_data1[column].values,
                                 columns=['experiment_id', column],
                                 title=f'difference {column}',
                                 output_folder=data_folder,
                                 )


if __name__ == '__main__':
    compare_runtimes(
        data_folder1='./exp_hypothesis_one_2020_03_02T16_57_41',
        data_folder2='./exp_hypothesis_one_2020_03_02T17_19_54',
        experiment_ids=[]
    )
