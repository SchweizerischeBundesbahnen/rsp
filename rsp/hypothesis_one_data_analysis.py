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
from typing import Dict
from typing import List

import tqdm
from pandas import DataFrame

from rsp.asp_plausibility.asp_plausi import asp_plausi_analysis
from rsp.asp_plausibility.potassco_export import potassco_export
from rsp.route_dag.analysis.rescheduling_analysis_utils import analyze_experiment
from rsp.route_dag.analysis.rescheduling_verification_utils import plausibility_check_experiment_results
from rsp.utils.analysis_tools import two_dimensional_scatter_plot
from rsp.utils.data_types import convert_list_of_experiment_results_analysis_to_data_frame
from rsp.utils.data_types import convert_pandas_series_experiment_results_analysis
from rsp.utils.data_types import ExperimentAgenda
from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.experiment_render_utils import visualize_experiment
from rsp.utils.experiments import EXPERIMENT_AGENDA_SUBDIRECTORY_NAME
from rsp.utils.experiments import EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME
from rsp.utils.experiments import EXPERIMENT_DATA_SUBDIRECTORY_NAME
from rsp.utils.experiments import EXPERIMENT_POTASSCO_SUBDIRECTORY_NAME
from rsp.utils.experiments import load_and_expand_experiment_results_from_data_folder
from rsp.utils.experiments import load_experiment_agenda_from_file
from rsp.utils.file_utils import check_create_folder
from rsp.utils.file_utils import newline_and_flush_stdout_and_stderr


def _2d_analysis(data: DataFrame,
                 output_folder: str = None):
    explanations = {
        'speed_up': 'time_full_after_malfunction / time_delta_after_malfunction',
        'time_full_after_malfunction': 'solver time for re-scheduling without Oracle information',
        'time_delta_after_malfunction': 'solver time for re-scheduling with Oracle information'
    }
    for y_axis in ['speed_up', 'time_full_after_malfunction', 'time_delta_after_malfunction']:
        for x_axis in ['experiment_id', 'n_agents', 'size', 'size_used']:
            explanation = (f"({explanations[y_axis]})" if y_axis in explanations else "")
            two_dimensional_scatter_plot(
                columns=[x_axis, y_axis],
                data=data,
                show_global_mean=True if x_axis == 'experiment_id' else False,
                title=f'{y_axis} per {x_axis} {explanation}',
                output_folder=output_folder
            )
    for x_axis_prefix in ['nb_resource_conflicts_']:
        for y_axis_prefix in ['time_']:
            for suffix in ['full_after_malfunction', 'delta_after_malfunction']:
                x_axis = x_axis_prefix + suffix
                y_axis = y_axis_prefix + suffix
                explanation = (f"({explanations[y_axis]})" if y_axis in explanations else "")
                two_dimensional_scatter_plot(
                    data=data,
                    columns=[x_axis, y_axis],
                    title=f'{y_axis} per {x_axis} {explanation})',
                    output_folder=output_folder
                )


def hypothesis_one_data_analysis(experiment_base_directory: str,
                                 analysis_2d: bool = False,
                                 analysis_3d: bool = False,
                                 qualitative_analysis_experiment_ids: List[int] = None,
                                 asp_export_experiment_ids: List[int] = None,
                                 flatland_rendering: bool = True
                                 ):
    """

    Parameters
    ----------
    analysis_2d
    analysis_3d
    qualitative_analysis_experiment_ids
    asp_export_experiment_ids
    experiment_base_directory
    flatland_rendering
    """
    # Import the desired experiment results
    experiment_analysis_directory = f'{experiment_base_directory}/{EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME}/'
    experiment_data_directory = f'{experiment_base_directory}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}'
    experiment_agenda_directory = f'{experiment_base_directory}/{EXPERIMENT_AGENDA_SUBDIRECTORY_NAME}'
    experiment_potassco_directory = f'{experiment_base_directory}/{EXPERIMENT_POTASSCO_SUBDIRECTORY_NAME}'

    # Create output directoreis
    check_create_folder(experiment_analysis_directory)

    experiment_results_list: List[ExperimentResultsAnalysis] = load_and_expand_experiment_results_from_data_folder(
        experiment_data_directory)
    experiment_agenda: ExperimentAgenda = load_experiment_agenda_from_file(experiment_agenda_directory)

    print(experiment_agenda)

    # Plausibility tests on experiment data
    _run_plausibility_tests_on_experiment_data(experiment_results_list)

    # convert to data frame for statistical analysis
    experiment_data: DataFrame = convert_list_of_experiment_results_analysis_to_data_frame(experiment_results_list)

    # save experiment data to .tsv for Excel inspection
    experiment_data.to_csv(f"{experiment_data_directory}/data.tsv", sep="\t")

    # quantitative analysis
    if analysis_2d:
        _2d_analysis(
            data=experiment_data,
            output_folder=f'{experiment_analysis_directory}/main_results'
        )
        asp_plausi_analysis(
            experiment_results_list=experiment_results_list,
            output_folder=f'{experiment_analysis_directory}/asp_plausi'
        )
    if analysis_3d:
        raise NotImplementedError()

    # qualitative explorative analysis
    if qualitative_analysis_experiment_ids:
        filtered_experiments = list(filter(
            lambda experiment: experiment.experiment_id in qualitative_analysis_experiment_ids,
            experiment_agenda.experiments))
        for experiment in filtered_experiments:
            row = experiment_data[experiment_data['experiment_id'] == experiment.experiment_id].iloc[0]
            experiment_results_analysis: ExperimentResultsAnalysis = convert_pandas_series_experiment_results_analysis(
                row)

            analyze_experiment(experiment_results_analysis=experiment_results_analysis)
            visualize_experiment(experiment_parameters=experiment,
                                 experiment_results_analysis=experiment_results_analysis,
                                 experiment_analysis_directory=experiment_analysis_directory,
                                 analysis_2d=analysis_2d,
                                 analysis_3d=analysis_3d,
                                 flatland_rendering=flatland_rendering)
    if asp_export_experiment_ids:
        potassco_export(experiment_potassco_directory=experiment_potassco_directory,
                        experiment_results_list=experiment_results_list,
                        asp_export_experiment_ids=asp_export_experiment_ids)


def lateness_to_cost(weight_lateness_seconds: int, lateness_dict: Dict[int, int]) -> Dict[int, int]:
    """Map lateness per agent to costs for lateness.

    Parameters
    ----------
    weight_lateness_seconds
    lateness_dict

    Returns
    -------
    """
    # TODO hard-coded constants for delay model, same as in delay_linear_within_one_minute.lp
    PENALTY_LEAP_AT = 60
    PENALTY_LEAP = 5000000 + PENALTY_LEAP_AT * weight_lateness_seconds
    return sum([(PENALTY_LEAP
                 if lateness > PENALTY_LEAP_AT
                 else lateness * weight_lateness_seconds)
                for agent_id, lateness in lateness_dict.items()])


def _run_plausibility_tests_on_experiment_data(l: List[ExperimentResultsAnalysis]):
    print("Running plausibility tests on experiment data...")
    # nicer printing when tdqm print to stderr and we have logging to stdout shown in to the same console (IDE, separated in files)
    newline_and_flush_stdout_and_stderr()
    for experiment_results_analysis in tqdm.tqdm(l):
        experiment_id = experiment_results_analysis.experiment_id
        plausibility_check_experiment_results(experiment_results=experiment_results_analysis)
        costs_full_after_malfunction: int = experiment_results_analysis.costs_full_after_malfunction
        lateness_full_after_malfunction: Dict[int, int] = experiment_results_analysis.lateness_full_after_malfunction
        sum_route_section_penalties_full_after_malfunction: Dict[
            int, int] = experiment_results_analysis.sum_route_section_penalties_full_after_malfunction
        costs_delta_after_malfunction: int = experiment_results_analysis.costs_delta_after_malfunction
        lateness_delta_after_malfunction: Dict[int, int] = experiment_results_analysis.lateness_delta_after_malfunction
        sum_route_section_penalties_delta_after_malfunction: Dict[
            int, int] = experiment_results_analysis.sum_route_section_penalties_delta_after_malfunction
        costs_lateness_full_after_malfunction: int = lateness_to_cost(
            weight_lateness_seconds=experiment_results_analysis.experiment_parameters.weight_lateness_seconds,
            lateness_dict=lateness_full_after_malfunction)
        sum_all_route_section_penalties_full_after_malfunction: int = sum(
            sum_route_section_penalties_full_after_malfunction.values())
        costs_lateness_delta_after_malfunction: int = lateness_to_cost(
            weight_lateness_seconds=experiment_results_analysis.experiment_parameters.weight_lateness_seconds,
            lateness_dict=lateness_delta_after_malfunction)
        sum_all_route_section_penalties_delta_after_malfunction: int = sum(
            sum_route_section_penalties_delta_after_malfunction.values())

        assert costs_full_after_malfunction == costs_lateness_full_after_malfunction + sum_all_route_section_penalties_full_after_malfunction, \
            f"experiment {experiment_id}: " \
            f"costs_full_after_malfunction={costs_full_after_malfunction}, " \
            f"costs_lateness_full_after_malfunction={costs_lateness_full_after_malfunction}, " \
            f"sum_all_route_section_penalties_full_after_malfunction={sum_all_route_section_penalties_full_after_malfunction}, "
        assert costs_delta_after_malfunction == costs_lateness_delta_after_malfunction + sum_all_route_section_penalties_delta_after_malfunction, \
            f"experiment {experiment_id}: " \
            f"costs_delta_after_malfunction={costs_delta_after_malfunction}, " \
            f"costs_lateness_delta_after_malfunction={costs_lateness_delta_after_malfunction}, " \
            f"sum_all_route_section_penalties_delta_after_malfunction={sum_all_route_section_penalties_delta_after_malfunction}, "
    # nicer printing when tdqm print to stderr and we have logging to stdout shown in to the same console (IDE, separated in files)
    newline_and_flush_stdout_and_stderr()
    print("  -> Done plausibility tests on experiment data.")


if __name__ == '__main__':
    hypothesis_one_data_analysis(
        experiment_base_directory='./exp_hypothesis_one_2020_03_21T00_15_08',
        analysis_2d=True,
        analysis_3d=False,
        qualitative_analysis_experiment_ids=range(270, 300),
        asp_export_experiment_ids=range(270, 300)
    )
