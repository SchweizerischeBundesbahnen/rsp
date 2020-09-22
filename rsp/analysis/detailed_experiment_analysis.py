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
from typing import Optional

import numpy as np
import pandas as pd
import tqdm
from pandas import DataFrame

from rsp.analysis.compute_time_analysis import extract_schedule_plotting
from rsp.analysis.compute_time_analysis import plot_agent_specific_delay
from rsp.analysis.compute_time_analysis import plot_box_plot
from rsp.analysis.compute_time_analysis import plot_box_plot_from_traces
from rsp.analysis.compute_time_analysis import plot_changed_agents
from rsp.analysis.compute_time_analysis import plot_histogram_from_delay_data
from rsp.analysis.compute_time_analysis import plot_resource_occupation_heat_map
from rsp.analysis.compute_time_analysis import plot_resource_time_diagrams
from rsp.analysis.compute_time_analysis import plot_route_dag
from rsp.analysis.compute_time_analysis import plot_shared_heatmap
from rsp.analysis.compute_time_analysis import plot_speed_up
from rsp.analysis.compute_time_analysis import plot_time_density
from rsp.analysis.compute_time_analysis import plot_time_window_resource_trajectories
from rsp.analysis.compute_time_analysis import plot_total_lateness
from rsp.analysis.compute_time_analysis import plot_train_paths
from rsp.asp_plausibility.asp_plausi import visualize_hypotheses_asp
from rsp.asp_plausibility.potassco_export import potassco_export
from rsp.experiment_solvers.data_types import SchedulingExperimentResult
from rsp.schedule_problem_description.analysis.rescheduling_verification_utils import plausibility_check_experiment_results
from rsp.schedule_problem_description.data_types_and_utils import get_paths_in_route_dag
from rsp.schedule_problem_description.data_types_and_utils import ScheduleProblemDescription
from rsp.schedule_problem_description.data_types_and_utils import ScheduleProblemEnum
from rsp.utils.data_types import after_malfunction_scopes
from rsp.utils.data_types import all_scopes
from rsp.utils.data_types import convert_list_of_experiment_results_analysis_to_data_frame
from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.data_types import speed_up_scopes
from rsp.utils.experiment_render_utils import visualize_experiment
from rsp.utils.experiments import EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME
from rsp.utils.experiments import EXPERIMENT_DATA_SUBDIRECTORY_NAME
from rsp.utils.experiments import EXPERIMENT_POTASSCO_SUBDIRECTORY_NAME
from rsp.utils.experiments import load_and_expand_experiment_results_from_data_folder
from rsp.utils.file_utils import check_create_folder
from rsp.utils.file_utils import newline_and_flush_stdout_and_stderr
from rsp.utils.general_helpers import catch_zero_division_error_as_minus_one
from rsp.utils.global_constants import DELAY_MODEL_RESOLUTION
from rsp.utils.global_constants import DELAY_MODEL_UPPER_BOUND_LINEAR_PENALTY
from rsp.utils.rsp_logger import rsp_logger


def _derive_numbers_for_correlation_analysis(
        r: SchedulingExperimentResult,
        p: ScheduleProblemDescription,
        suffix: str):
    nb_topo_edges_per_agent = [
        len(topo.edges)
        for agent, topo in p.topo_dict.items()
    ]
    nb_topo_paths_per_agent = [
        len(get_paths_in_route_dag(topo))
        for agent, topo in p.topo_dict.items()
    ]
    d = {
        f'nb_resource_conflicts_{suffix}': r.nb_conflicts,
        f'sum_nb_topo_edges_all_agents': sum(nb_topo_edges_per_agent),
    }
    for q in [0.9, 0.7, 0.5, 0.3]:
        d[f'quantile_nb_topo_edges_{q}'] = np.quantile(a=nb_topo_edges_per_agent, q=q)
        d[f'quantile_nb_topo_paths_per_agent_{q}'] = np.quantile(a=nb_topo_paths_per_agent, q=q)
    return d


def visualize_hypothesis_009_rescheduling_times_grow_exponentially_in_the_number_of_time_window_overlaps(
        experiment_results_list: List[ExperimentResultsAnalysis],
        output_folder: Optional[str] = None):
    data_frame = pd.DataFrame(data=[
        {
            **r._asdict(),
            'ratio_nb_resource_conflicts': catch_zero_division_error_as_minus_one(
                lambda:
                r.nb_resource_conflicts_full_after_malfunction / r.nb_resource_conflicts_delta_perfect_after_malfunction),
            **_derive_numbers_for_correlation_analysis(r=r.results_full, p=r.problem_full, suffix="full"),
            **_derive_numbers_for_correlation_analysis(r=r.results_full_after_malfunction,
                                                       p=r.problem_full_after_malfunction,
                                                       suffix="full_after_malfunction"),
            **_derive_numbers_for_correlation_analysis(r=r.results_delta_perfect_after_malfunction,
                                                       p=r.problem_delta_perfect_after_malfunction,
                                                       suffix="delta_perfect_after_malfunction"),
        }
        for r in experiment_results_list])

    columns_of_interest = [
        f'nb_resource_conflicts_{item}'
        for item in ['full', 'full_after_malfunction', 'delta_perfect_after_malfunction']
    ]
    for axis_of_interest in ['experiment_id', 'n_agents', 'size', 'size_used_full_after_malfunction',
                             'changed_agents_percentage_delta_perfect_after_malfunction']:
        plot_box_plot(
            experiment_data=data_frame,
            axis_of_interest=axis_of_interest,
            columns_of_interest=columns_of_interest,
            title=f"009_rescheduling_times_grow_exponentially_in_the_number_of_time_window_overlaps: "
                  f"correlation of {axis_of_interest} with resource conflicts",
            output_folder=output_folder,
            file_name_prefix="009"
        )

    plot_box_plot_from_traces(
        experiment_data=data_frame,
        output_folder=output_folder,
        pdf_file="009_nb_resource_conflict__time.pdf",
        title="009_rescheduling_times_grow_exponentially_in_the_number_of_time_window_overlaps:\n"
              "Correlation of resource conflicts and runtime",
        traces=[(f'nb_resource_conflicts_{item}', f'solver_statistics_times_total_{item}') for item in
                ['full', 'full_after_malfunction', 'delta_perfect_after_malfunction']],
        x_axis_title='nb_resource_conflict',
    )

    plot_box_plot_from_traces(
        experiment_data=data_frame,
        output_folder=output_folder,
        pdf_file="009_nb_resource_conflict__time.pdf",
        title="009_rescheduling_times_grow_exponentially_in_the_number_of_time_window_overlaps:\n"
              'Correlation of ratio of nb_resource_conflicts and speed_up_delta_perfect_after_malfunction?',
        traces=[('ratio_nb_resource_conflicts', 'speed_up_delta_perfect_after_malfunction')],
        x_axis_title='ratio_nb_resource_conflicts'
    )


HYPOTHESIS_ONE_COLUMNS_OF_INTEREST = [f'solver_statistics_times_total_{scope}' for scope in all_scopes]


# TODO SIM-672 should we remove analysis stuff from pipeline, only have it in notebooks and tests (from dummydata maybe?)
def hypothesis_one_analysis_visualize_computational_time_comparison(
        experiment_data: DataFrame,
        output_folder: str = None):
    for axis_of_interest in ['experiment_id', 'n_agents', 'size', 'size_used_full_after_malfunction', 'solver_statistics_times_total_full_after_malfunction']:
        plot_box_plot(
            experiment_data=experiment_data,
            axis_of_interest=axis_of_interest,
            columns_of_interest=HYPOTHESIS_ONE_COLUMNS_OF_INTEREST,
            output_folder=output_folder
        )


def hypothesis_one_analysis_visualize_lateness(
        experiment_data: DataFrame,
        output_folder: str = None):
    plot_box_plot(
        experiment_data=experiment_data,
        axis_of_interest='experiment_id',
        columns_of_interest=[f'total_lateness_{scope}' for scope in after_malfunction_scopes],
        output_folder=output_folder,
        title='Total delay',
        color_offset=1
    )


def hypothesis_one_analysis_visualize_speed_up(experiment_data: DataFrame,
                                               output_folder: str = None):
    for scope in speed_up_scopes:
        experiment_data[f'speed_up_{scope}_solve_time'] = \
            experiment_data['solver_statistics_times_solve_full_after_malfunction'] / \
            experiment_data[f'solver_statistics_times_solve_{scope}']
        experiment_data[f'speed_up_{scope}_non_solve_time'] = (
                (experiment_data['solver_statistics_times_total_full_after_malfunction'] -
                 experiment_data['solver_statistics_times_solve_full_after_malfunction']) /
                (experiment_data[f'solver_statistics_times_total_{scope}'] - experiment_data[f'solver_statistics_times_solve_{scope}']))

    for axis_of_interest, axis_of_interest_suffix in {
        'experiment_id': '',
        'n_agents': '',
        'size': '',
        'size_used_full_after_malfunction': '',
        'solver_statistics_times_total_full_after_malfunction': '[s]',
        'changed_agents_percentage_delta_perfect_after_malfunction': '[-]'
    }.items():
        for speed_up_col_pattern, y_axis_title in [
            ('speed_up_{}', 'Speed-up full solver time [-]'),
            ('speed_up_{}_solve_time', 'Speed-up solver time solving only [-]'),
            ('speed_up_{}_non_solve_time', 'Speed-up solver time non-processing (grounding etc.) [-]'),
            ('changed_agents_percentage_{}', 'Percentage of changed agents [-]'),
            ('effective_costs_ratio_{}', 'Costs ratio [-]'),
        ]:
            plot_speed_up(
                experiment_data=experiment_data,
                axis_of_interest=axis_of_interest,
                axis_of_interest_suffix=axis_of_interest_suffix,
                output_folder=output_folder,
                cols=[speed_up_col_pattern.format(speed_up_series) for speed_up_series in speed_up_scopes],
                y_axis_title=y_axis_title
            )


def hypothesis_one_data_analysis(experiment_output_directory: str,
                                 analysis_2d: bool = False,
                                 asp_export_experiment_ids: List[int] = None,
                                 qualitative_analysis_experiment_ids: List[int] = None,
                                 save_as_tsv: bool = False
                                 ):
    """

    Parameters
    ----------
    analysis_2d
    asp_export_experiment_ids
    experiment_output_directory
    save_as_tsv
    qualitative_analysis_experiment_ids
    """

    # Import the desired experiment results
    experiment_analysis_directory = f'{experiment_output_directory}/{EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME}/'
    experiment_data_directory = f'{experiment_output_directory}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}'
    experiment_potassco_directory = f'{experiment_output_directory}/{EXPERIMENT_POTASSCO_SUBDIRECTORY_NAME}'

    # Create output directoreis
    check_create_folder(experiment_analysis_directory)

    experiment_results_list: List[ExperimentResultsAnalysis] = load_and_expand_experiment_results_from_data_folder(
        experiment_data_directory)

    # Plausibility tests on experiment data
    _run_plausibility_tests_on_experiment_data(experiment_results_list)

    # convert to data frame for statistical analysis
    experiment_data: DataFrame = convert_list_of_experiment_results_analysis_to_data_frame(experiment_results_list)

    if save_as_tsv:
        # save experiment data to .tsv for Excel inspection
        experiment_data.to_csv(f"{experiment_data_directory}/data.tsv", sep="\t")

    # quantitative analysis
    # TODO SIM-672 should we remove analysis_2d in favor of notebooks which are tested in ci?
    if analysis_2d:
        # main results
        hypothesis_one_analysis_visualize_computational_time_comparison(
            experiment_data=experiment_data,
            output_folder=f'{experiment_analysis_directory}/main_results'
        )
        hypothesis_one_analysis_visualize_speed_up(
            experiment_data=experiment_data,
            output_folder=f'{experiment_analysis_directory}/main_results'
        )

        # TODO SIM-674 SIM-672 group by notebook?
        experiment_result_of_interest = experiment_results_list[0]
        agent_of_interest = 0
        schedule_plotting = extract_schedule_plotting(experiment_result=experiment_result_of_interest)
        plot_time_window_resource_trajectories(experiment_result=experiment_result_of_interest, schedule_plotting=schedule_plotting)
        plot_shared_heatmap(schedule_plotting=schedule_plotting, experiment_result=experiment_result_of_interest)
        plot_resource_time_diagrams(schedule_plotting=schedule_plotting, with_diff=True)
        plot_histogram_from_delay_data(experiment_results=experiment_result_of_interest)
        plot_total_lateness(experiment_results=experiment_result_of_interest)
        plot_agent_specific_delay(experiment_results=experiment_result_of_interest)
        plot_changed_agents(experiment_results=experiment_result_of_interest)
        plot_route_dag(experiment_results_analysis=experiment_result_of_interest,
                       agent_id=agent_of_interest,
                       suffix_of_constraints_to_visualize=ScheduleProblemEnum.PROBLEM_SCHEDULE)
        plot_resource_occupation_heat_map(
            schedule_plotting=schedule_plotting,
            plotting_information=schedule_plotting.plotting_information,
            title_suffix='Schedule'
        )
        plot_train_paths(
            plotting_data=schedule_plotting,
            agent_ids=[agent_of_interest])
        plot_time_density(schedule_plotting.schedule_as_resource_occupations)

    hypothesis_one_analysis_visualize_lateness(experiment_data=experiment_data, output_folder=f'{experiment_analysis_directory}/main_results')
    visualize_hypothesis_009_rescheduling_times_grow_exponentially_in_the_number_of_time_window_overlaps(
        experiment_results_list=experiment_results_list,
        output_folder=f'{experiment_analysis_directory}/plausi'
    )
    visualize_hypotheses_asp(
        experiment_data=experiment_data,
        output_folder=f'{experiment_analysis_directory}/plausi'
    )

    # TODO should we remove qualitative_analysis_experiment_ids in favor of notebooks which are tested in ci?
    if qualitative_analysis_experiment_ids:
        for experiment_result in experiment_results_list:
            if experiment_result.experiment_id not in qualitative_analysis_experiment_ids:
                continue
            visualize_experiment(
                experiment_parameters=experiment_result.experiment_parameters,
                experiment_results_analysis=experiment_result,
                experiment_analysis_directory=experiment_analysis_directory,
                # TODO SIM-443
                flatland_rendering=False
            )

    if asp_export_experiment_ids:
        potassco_export(experiment_potassco_directory=experiment_potassco_directory,
                        experiment_results_list=experiment_results_list,
                        asp_export_experiment_ids=asp_export_experiment_ids)


def lateness_to_effective_cost(weight_lateness_seconds: int, lateness_dict: Dict[int, int]) -> Dict[int, int]:
    """Map lateness per agent to costs for lateness.

    Parameters
    ----------
    weight_lateness_seconds
    lateness_dict

    Returns
    -------
    """
    penalty_leap_at = DELAY_MODEL_UPPER_BOUND_LINEAR_PENALTY
    penalty_leap = 5000000 + penalty_leap_at * weight_lateness_seconds
    return sum([(penalty_leap
                 if lateness > penalty_leap_at
                 else (lateness // DELAY_MODEL_RESOLUTION) * DELAY_MODEL_RESOLUTION * weight_lateness_seconds)
                for agent_id, lateness in lateness_dict.items()])


def _run_plausibility_tests_on_experiment_data(l: List[ExperimentResultsAnalysis]):
    rsp_logger.info("Running plausibility tests on experiment data...")
    # nicer printing when tdqm print to stderr and we have logging to stdout shown in to the same console (IDE, separated in files)
    newline_and_flush_stdout_and_stderr()
    for experiment_results_analysis in tqdm.tqdm(l):
        experiment_id = experiment_results_analysis.experiment_id
        plausibility_check_experiment_results(experiment_results=experiment_results_analysis)
        effective_costs_full_after_malfunction: int = experiment_results_analysis.effective_costs_full_after_malfunction
        lateness_full_after_malfunction: Dict[int, int] = experiment_results_analysis.lateness_full_after_malfunction
        effective_costs_from_route_section_penalties_full_after_malfunction: Dict[
            int, int] = experiment_results_analysis.effective_costs_from_route_section_penalties_full_after_malfunction
        effective_costs_delta_perfect_after_malfunction: int = experiment_results_analysis.effective_costs_delta_perfect_after_malfunction
        lateness_delta_perfect_after_malfunction: Dict[int, int] = experiment_results_analysis.lateness_delta_perfect_after_malfunction
        effective_costs_from_route_section_penalties_delta_perfect_after_malfunction: Dict[
            int, int] = experiment_results_analysis.effective_costs_from_route_section_penalties_delta_perfect_after_malfunction
        effective_costs_lateness_full_after_malfunction: int = lateness_to_effective_cost(
            weight_lateness_seconds=experiment_results_analysis.experiment_parameters.weight_lateness_seconds,
            lateness_dict=lateness_full_after_malfunction)
        sum_all_route_section_penalties_full_after_malfunction: int = sum(
            effective_costs_from_route_section_penalties_full_after_malfunction.values())
        effective_costs_lateness_delta_perfect_after_malfunction: int = lateness_to_effective_cost(
            weight_lateness_seconds=experiment_results_analysis.experiment_parameters.weight_lateness_seconds,
            lateness_dict=lateness_delta_perfect_after_malfunction)
        sum_all_route_section_penalties_delta_perfect_after_malfunction: int = sum(
            effective_costs_from_route_section_penalties_delta_perfect_after_malfunction.values())

        assert effective_costs_full_after_malfunction == (
                effective_costs_lateness_full_after_malfunction + sum_all_route_section_penalties_full_after_malfunction), \
            f"experiment {experiment_id}: " \
            f"effective_costs_full_after_malfunction={effective_costs_full_after_malfunction}, " \
            f"costs_lateness_full_after_malfunction={effective_costs_lateness_full_after_malfunction}, " \
            f"sum_all_route_section_penalties_full_after_malfunction={sum_all_route_section_penalties_full_after_malfunction}, "
        assert (effective_costs_delta_perfect_after_malfunction ==
                effective_costs_lateness_delta_perfect_after_malfunction + sum_all_route_section_penalties_delta_perfect_after_malfunction), \
            f"experiment {experiment_id}: " \
            f"effective_costs_delta_perfect_after_malfunction={effective_costs_delta_perfect_after_malfunction}, " \
            f"costs_lateness_delta_perfect_after_malfunction={effective_costs_lateness_delta_perfect_after_malfunction}, " \
            f"sum_all_route_section_penalties_delta_perfect_after_malfunction={sum_all_route_section_penalties_delta_perfect_after_malfunction}, "
    # nicer printing when tdqm print to stderr and we have logging to stdout shown in to the same console (IDE, separated in files)
    newline_and_flush_stdout_and_stderr()
    rsp_logger.info("  -> done plausibility tests on experiment data.")


if __name__ == '__main__':
    # TODO SIM-672 make unit test instead? do we need an offline version? extract main from here in the latter case.
    hypothesis_one_data_analysis(
        experiment_output_directory='./rsp/exp_hypothesis_one_2020_03_21T12_57_55',
        analysis_2d=True,
        asp_export_experiment_ids=[270, 275, 280, 285, 290, 295]
    )
