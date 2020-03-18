import os
import warnings
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


def compare_runtimes(
        data_folder1: str,
        data_folder2: str,
        experiment_ids: List[int],
        output_enclosing_folder: str = '.',
        fail_on_missing_experiment_ids: bool = True
) -> str:
    """Compare run times and solution costs of two pipeline runs.

    Parameters
    ----------

    data_folder1
        folder with baseline data
    data_folder2
        folder with new data
    experiment_ids
        filter for experiment ids
    output_enclosing_folder
        the output is put in a subfolder of this folder taking the two experiment names
    """

    output_folder = os.path.join(output_enclosing_folder,
                                 _extract_experiment_name_from_data_folder_path(
                                     data_folder1) + '_' + _extract_experiment_name_from_data_folder_path(data_folder2))
    print(f"compare runtimes {data_folder1} - {data_folder2} -> {output_folder}")

    experiment_data2: DataFrame = load_without_average(data_folder2)
    experiment_data1: DataFrame = load_without_average(data_folder1)

    for experiment_id in experiment_ids:
        time_delta_after_malfunction1, time_full_after_malfunction1 = _extract_times_for_experiment_id(experiment_data1,
                                                                                                       experiment_id)
        time_delta_after_malfunction2, time_full_after_malfunction2 = _extract_times_for_experiment_id(experiment_data2,
                                                                                                       experiment_id)
        print(f"time_delta_after_malfunction: {time_delta_after_malfunction1} --> {time_delta_after_malfunction2}")

    # ensure that experiment_ids are the same (ignored failures)
    experiment_data1_experiment_ids = set(experiment_data1['experiment_id'].values)
    experiment_data2_experiment_ids = set(experiment_data2['experiment_id'].values)
    experiment_ids_common = experiment_data1_experiment_ids.intersection(experiment_data2_experiment_ids)
    only_experiment_data1_experiment_ids = experiment_data1_experiment_ids.difference(experiment_data2_experiment_ids)
    only_experiment_data2_experiment_ids = experiment_data2_experiment_ids.difference(experiment_data1_experiment_ids)
    if len(only_experiment_data1_experiment_ids) > 0:
        warnings.warn(
            f"experiment_ids only in {data_folder1} but not in {data_folder2}:" + "\n - ".join([str(id) for id in
                                                                                                only_experiment_data1_experiment_ids]))
    if len(only_experiment_data2_experiment_ids) > 0:
        warnings.warn(
            f"experiment_ids only in {data_folder2} but not in {data_folder1}:" + "\n - ".join([str(id) for id in
                                                                                                only_experiment_data2_experiment_ids]))
    if len(only_experiment_data1_experiment_ids) + len(only_experiment_data2_experiment_ids) > 0 \
            and fail_on_missing_experiment_ids:
        raise AssertionError(f"Not same experiment_ids in the two runs to compare ({data_folder1}, {data_folder2})")

    experiment_data1 = experiment_data1.loc[experiment_data1['experiment_id'].isin(experiment_ids_common)]
    experiment_data2 = experiment_data2.loc[experiment_data2['experiment_id'].isin(experiment_ids_common)]

    _scatter_for_two_runs(experiment_data1=experiment_data1,
                          experiment_data2=experiment_data2,
                          output_folder=output_folder,
                          column='time_full_after_malfunction')
    _scatter_for_two_runs(experiment_data1=experiment_data1,
                          experiment_data2=experiment_data2,
                          output_folder=output_folder,
                          column='time_delta_after_malfunction')
    _scatter_for_two_runs(experiment_data1=experiment_data1,
                          experiment_data2=experiment_data2,
                          output_folder=output_folder,
                          column='costs_full_after_malfunction')
    _scatter_for_two_runs(experiment_data1=experiment_data1,
                          experiment_data2=experiment_data2,
                          output_folder=output_folder,
                          column='costs_delta_after_malfunction')
    return output_folder


def _extract_experiment_name_from_data_folder_path(data_folder1):
    # sanitize / paths under windows, and split ((head-*tail*)-tail) to extract to first to last path element
    return os.path.split(os.path.split(data_folder1.replace('/', os.sep))[0])[1]


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
    assert set(experiment_data1['experiment_id'].values) == set(experiment_data2['experiment_id'].values)

    check_create_folder(output_folder)

    two_dimensional_scatter_plot(data=experiment_data2,
                                 baseline_data=experiment_data1,
                                 columns=['experiment_id', column],
                                 title=f'difference {column}',
                                 output_folder=output_folder
                                 )
