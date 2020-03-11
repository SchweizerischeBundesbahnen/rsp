import os
from typing import List

from pandas import DataFrame

from rsp.utils.analysis_tools import two_dimensional_scatter_plot
from rsp.utils.experiments import load_without_average
from rsp.utils.file_utils import check_create_folder


def _extract_times_for_experiment_id(experiment_data, experiment_id):
    """Extract experiment data for experiment_id and return time for full and
    delta re-scheduling.

    Parameters
    ----------
    experiment_data
    experiment_id

    Returns
    -------
    """
    row = experiment_data[experiment_data['experiment_id'] == experiment_id].iloc[0]
    time_full_after_malfunction = row['time_full_after_malfunction']
    time_delta_after_malfunction = row['time_delta_after_malfunction']
    return time_delta_after_malfunction, time_full_after_malfunction


def compare_runtimes(data_folder1: str,
                     data_folder2: str,
                     output_folder: str,
                     experiment_ids: List[int]):
    """Compare run times and solution costs of two pipeline runs.

    Parameters
    ----------
    data_folder1
        folder with baseline data
    data_folder2
        folder with new data
    experiment_ids
        filter for experiment ids
    """

    experiment_data2: DataFrame = load_without_average(data_folder2)
    experiment_data1: DataFrame = load_without_average(data_folder1)

    output_folder = os.path.join(output_folder, os.path.split(data_folder1)[-1] + '_' + os.path.split(data_folder2)[-1])

    for experiment_id in experiment_ids:
        time_delta_after_malfunction1, time_full_after_malfunction1 = _extract_times_for_experiment_id(experiment_data1,
                                                                                                       experiment_id)
        time_delta_after_malfunction2, time_full_after_malfunction2 = _extract_times_for_experiment_id(experiment_data2,
                                                                                                       experiment_id)
        print(f"time_delta_after_malfunction: {time_delta_after_malfunction1} --> {time_delta_after_malfunction2}")
    min_len = min(len(experiment_data1), len(experiment_data2))

    # verify that the experiment ids match for the first min_len experiments
    for i in range(min_len):
        assert experiment_data1['experiment_id'].values[i] == experiment_data2['experiment_id'].values[i], \
            f"at {i} {experiment_data1['experiment_id'].values[i]} - {experiment_data2['experiment_id'].values[i]}\n" \
            f"{experiment_data1['experiment_id'].values} - {experiment_data2['experiment_id'].values}"

    _scatter_for_two_runs(experiment_data1=experiment_data1,
                          experiment_data2=experiment_data2,
                          data_folder=data_folder1,
                          column='time_full_after_malfunction')
    _scatter_for_two_runs(experiment_data1=experiment_data1,
                          experiment_data2=experiment_data2,
                          data_folder=data_folder1,
                          column='time_delta_after_malfunction')
    _scatter_for_two_runs(experiment_data1=experiment_data1,
                          experiment_data2=experiment_data2,
                          data_folder=data_folder1,
                          column='costs_full_after_malfunction')
    _scatter_for_two_runs(experiment_data1=experiment_data1,
                          experiment_data2=experiment_data2,
                          data_folder=data_folder1,
                          column='costs_delta_after_malfunction')


def _scatter_for_two_runs(experiment_data1: DataFrame,
                          experiment_data2: DataFrame,
                          column: str,
                          output_folder: str):
    """
    Compare two pipeline runs by plotting the values of a column in both frames per `experiment_id`.
    The first frame is considered the baseline and the scatter point labels are suffixed with "_b".
    Parameters
    ----------
    experiment_data1:
        baseline data
    experiment_data2:
        new data
    data_folder
    column
    output_folder

    Returns
    -------

    """
    # verify that experiment_ids match (in case failed experiments, we do not want to combine non-matching data!)
    min_len = min(len(experiment_data1), len(experiment_data2))
    for i in range(min_len):
        assert experiment_data1['experiment_id'].values[i] == experiment_data2['experiment_id'].values[i]

    check_create_folder(output_folder)

    two_dimensional_scatter_plot(data=experiment_data2,
                                 baseline_data=experiment_data1,
                                 columns=['experiment_id', column],
                                 title=f'difference {column}',
                                 output_folder=output_folder,
                                 link_column=None
                                 )


if __name__ == '__main__':
    # allow for non-matching pkl files
    # this should be safe here since we only consider solve times and solution costs
    COMPATIBILITY_MODE = True
    compare_runtimes(
        data_folder1='./exp_hypothesis_one_2020_03_04T19_19_00',
        data_folder2='./exp_hypothesis_one_2020_03_10T22_10_19',
        output_folder='.',
        experiment_ids=[]
    )
