from typing import List

from pandas import DataFrame

from rsp.utils.analysis_tools import average_over_grid_id
from rsp.utils.analysis_tools import two_dimensional_scatter_plot
from rsp.utils.data_types import convert_list_of_experiment_results_analysis_to_data_frame
from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.experiments import load_and_expand_experiment_results_from_folder


def _load_and_average(data_folder):
    experiment_results_list: List[ExperimentResultsAnalysis] = load_and_expand_experiment_results_from_folder(
        data_folder)
    experiment_data: DataFrame = convert_list_of_experiment_results_analysis_to_data_frame(experiment_results_list)
    averaged_data, std_data = average_over_grid_id(experiment_data)
    return averaged_data


def _load_without_average(data_folder):
    experiment_results_list: List[ExperimentResultsAnalysis] = load_and_expand_experiment_results_from_folder(
        data_folder)
    experiment_data: DataFrame = convert_list_of_experiment_results_analysis_to_data_frame(experiment_results_list)
    return experiment_data


def _extract_times_for_experiment_id(experiment_data, experiment_id):
    row = experiment_data[experiment_data['experiment_id'] == experiment_id].iloc[0]
    time_full_after_malfunction = row['time_full_after_malfunction']
    time_delta_after_malfunction = row['time_delta_after_malfunction']
    return time_delta_after_malfunction, time_full_after_malfunction


def compare_runtimes(data_folder1: str,
                     data_folder2: str,
                     experiment_ids: List[int]):
    experiment_data2: DataFrame = _load_without_average(data_folder2)
    experiment_data1: DataFrame = _load_without_average(data_folder1)

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
    _scatter(experiment_data1=experiment_data1,
             experiment_data2=experiment_data2,
             data_folder=data_folder1,
             column='costs_full_after_malfunction')
    _scatter(experiment_data1=experiment_data1,
             experiment_data2=experiment_data2,
             data_folder=data_folder1,
             column='costs_delta_after_malfunction')


def _scatter(experiment_data1: DataFrame,
             experiment_data2: DataFrame,
             data_folder: str,
             column: str):
    min_len = min(len(experiment_data1), len(experiment_data2))
    for i in range(min_len):
        assert experiment_data1['experiment_id'].values[i] == experiment_data2['experiment_id'].values[i]

    two_dimensional_scatter_plot(data=experiment_data2,
                                 baseline_data=experiment_data1,
                                 columns=['experiment_id', column],
                                 title=f'difference {column}',
                                 output_folder=data_folder,
                                 link_column=None
                                 )


if __name__ == '__main__':
    compare_runtimes(
        data_folder1='./exp_hypothesis_one_2020_03_04T19_19_00',
        data_folder2='./exp_hypothesis_one_2020_03_10T22_10_19',
        experiment_ids=[]
    )
