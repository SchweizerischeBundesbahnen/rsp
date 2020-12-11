import os
from typing import Callable
from typing import List
from typing import Tuple
from typing import Union

import pandas as pd
import tqdm
from pandas import DataFrame

from rsp.global_data_configuration import BASELINE_DATA_FOLDER
from rsp.global_data_configuration import EXPERIMENT_DATA_SUBDIRECTORY_NAME
from rsp.step_05_experiment_run.experiment_results import ExperimentResults
from rsp.step_05_experiment_run.experiment_results_analysis import convert_list_of_experiment_results_analysis_to_data_frame
from rsp.step_05_experiment_run.experiment_results_analysis import expand_experiment_results_for_analysis
from rsp.step_05_experiment_run.experiment_results_analysis import ExperimentResultsAnalysis
from rsp.step_05_experiment_run.experiment_results_analysis import temporary_backwards_compatibility_scope
from rsp.step_05_experiment_run.experiment_results_analysis_online_unrestricted import (
    convert_list_of_experiment_results_analysis_online_unrestricted_to_data_frame,
)
from rsp.step_05_experiment_run.experiment_results_analysis_online_unrestricted import expand_experiment_results_online_unrestricted
from rsp.step_05_experiment_run.experiment_results_analysis_online_unrestricted import ExperimentResultsAnalysisOnlineUnrestricted
from rsp.utils.file_utils import get_experiment_id_from_filename
from rsp.utils.file_utils import newline_and_flush_stdout_and_stderr
from rsp.utils.pickle_helper import _pickle_dump
from rsp.utils.pickle_helper import _pickle_load
from rsp.utils.rsp_logger import rsp_logger


def save_experiment_results_to_file(experiment_results: ExperimentResults, file_name: str, csv_only: bool = False, online_unrestricted_only: bool = False):
    """Save the data frame with all the result from an experiment into a given
    file.
    Parameters
    ----------
    experiment_results: List of experiment results
       List containing all the experiment results
    file_name: str
        File name containing path and name of file we want to store the experiment results
    csv_only:bool
        write only csv or also pkl?
    Returns
    -------
    """
    if not csv_only:
        _pickle_dump(obj=experiment_results, file_name=file_name)
    if online_unrestricted_only:
        experiment_data = convert_list_of_experiment_results_analysis_online_unrestricted_to_data_frame(
            [expand_experiment_results_online_unrestricted(experiment_results)]
        )
    else:
        experiment_data: pd.DataFrame = convert_list_of_experiment_results_analysis_to_data_frame([expand_experiment_results_for_analysis(experiment_results)])
    experiment_data.to_csv(file_name.replace(".pkl", ".csv"))


def load_and_expand_experiment_results_from_data_folder(
    experiment_data_folder_name: str, experiment_ids: List[int] = None, re_save_csv_after_expansion: bool = False, online_unrestricted_only: bool = False,
) -> Tuple[List[ExperimentResults], List[Union[ExperimentResultsAnalysis, ExperimentResultsAnalysisOnlineUnrestricted]]]:
    """Load results as DataFrame to do further analysis.
    Parameters
    ----------
    experiment_data_folder_name: str
        Folder name of experiment where all experiment files are stored
    experiment_ids
        List of experiment ids which should be loaded, if None all experiments in experiment_folder are loaded
    Returns
    -------
    DataFrame containing the loaded experiment results
    """

    experiment_results_list_analysis = []
    experiment_results_list = []

    files = os.listdir(experiment_data_folder_name)
    rsp_logger.info(f"loading and expanding experiment results from {experiment_data_folder_name}")
    # nicer printing when tdqm print to stderr and we have logging to stdout shown in to the same console (IDE, separated in files)
    newline_and_flush_stdout_and_stderr()
    for file in tqdm.tqdm([file for file in files if "agenda" not in file]):
        file_name = os.path.join(experiment_data_folder_name, file)
        if not file_name.endswith(".pkl"):
            continue

        # filter experiments according to defined experiment_ids
        exp_id = get_experiment_id_from_filename(file_name)
        if experiment_ids is not None and exp_id not in experiment_ids:
            continue
        try:
            file_data: ExperimentResults = _pickle_load(file_name=file_name)
            experiment_results_list.append(file_data)
            if online_unrestricted_only:
                results_for_analysis = expand_experiment_results_online_unrestricted(file_data)
                experiment_results_list_analysis.append(results_for_analysis)
                if re_save_csv_after_expansion:
                    experiment_data: pd.DataFrame = convert_list_of_experiment_results_analysis_online_unrestricted_to_data_frame([results_for_analysis])
                    experiment_data.to_csv(file_name.replace(".pkl", ".csv"))
            else:
                results_for_analysis = expand_experiment_results_for_analysis(file_data)
                experiment_results_list_analysis.append(results_for_analysis)
                if re_save_csv_after_expansion:
                    # ensure it is nonified
                    experiment_data: pd.DataFrame = convert_list_of_experiment_results_analysis_to_data_frame([results_for_analysis])
                    experiment_data.to_csv(file_name.replace(".pkl", ".csv"))
        except Exception as e:
            rsp_logger.warn(f"skipping {file} because of {e}")
            rsp_logger.warn(e, exc_info=True)

    # nicer printing when tdqm print to stderr and we have logging to stdout shown in to the same console (IDE, separated in files)
    newline_and_flush_stdout_and_stderr()
    rsp_logger.info(f" -> done loading and expanding experiment results from {experiment_data_folder_name} done")

    return experiment_results_list, experiment_results_list_analysis


def load_data_from_individual_csv_in_data_folder(
    experiment_data_folder_name: str, experiment_ids: List[int] = None, online_unrestricted_only: bool = False
) -> DataFrame:
    """Load results as DataFrame to do further analysis.
    Parameters
    ----------
    experiment_data_folder_name: str
        Folder name of experiment where all experiment files are stored
    experiment_ids
        List of experiment ids which should be loaded, if None all experiments in experiment_folder are loaded
    Returns
    -------
    DataFrame containing the loaded experiment results
    """

    list_of_frames = []
    files = os.listdir(experiment_data_folder_name)
    rsp_logger.info(f"loading individual csv experiment results from {experiment_data_folder_name}")
    # nicer printing when tdqm print to stderr and we have logging to stdout shown in to the same console (IDE, separated in files)
    newline_and_flush_stdout_and_stderr()

    for file in tqdm.tqdm([file for file in files if "agenda" not in file]):
        file_name = os.path.join(experiment_data_folder_name, file)
        if not file_name.endswith(".csv"):
            continue

        # filter experiments according to defined experiment_ids
        exp_id = get_experiment_id_from_filename(file_name)
        if experiment_ids is not None and exp_id not in experiment_ids:
            continue
        try:
            list_of_frames.append(pd.read_csv(file_name))
        except Exception as e:
            rsp_logger.warn(f"skipping {file} because of {e}")

    # nicer printing when tdqm print to stderr and we have logging to stdout shown in to the same console (IDE, separated in files)
    newline_and_flush_stdout_and_stderr()
    rsp_logger.info(f" -> done loading individual csv results from {experiment_data_folder_name} done")

    experiment_data = pd.concat(list_of_frames)

    if not online_unrestricted_only:
        temporary_backwards_compatibility_scope(experiment_data)
    return experiment_data


def load_and_filter_experiment_results_analysis_online_unrestricted(
    experiment_base_directory: str = BASELINE_DATA_FOLDER,
    experiments_of_interest: List[int] = None,
    from_cache: bool = False,
    from_individual_csv: bool = True,
    local_filter_experiment_results_analysis_data_frame: Callable[[DataFrame], DataFrame] = None,
) -> DataFrame:
    if from_cache:
        experiment_data_filtered = pd.read_csv(f"{experiment_base_directory}.csv")
    else:
        if from_individual_csv:
            experiment_data: pd.DataFrame = load_data_from_individual_csv_in_data_folder(
                experiment_data_folder_name=f"{experiment_base_directory}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}",
                experiment_ids=experiments_of_interest,
                online_unrestricted_only=True,
            )
        else:
            _, experiment_results_analysis_list = load_and_expand_experiment_results_from_data_folder(
                experiment_data_folder_name=f"{experiment_base_directory}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}",
                experiment_ids=experiments_of_interest,
                online_unrestricted_only=True,
            )
            experiment_data: pd.DataFrame = convert_list_of_experiment_results_analysis_online_unrestricted_to_data_frame(experiment_results_analysis_list)

        if local_filter_experiment_results_analysis_data_frame is not None:
            experiment_data_filtered = local_filter_experiment_results_analysis_data_frame(experiment_data)
            print(f"removed {len(experiment_data) - len(experiment_data_filtered)}/{len(experiment_data)} rows")
        else:
            experiment_data_filtered = experiment_data
        experiment_data_filtered.to_csv(f"{experiment_base_directory}.csv")
    return experiment_data_filtered
