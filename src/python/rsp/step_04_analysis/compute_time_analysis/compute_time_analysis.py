"""Rendering methods to use with jupyter notebooks."""
from typing import List

from pandas import DataFrame
from rsp.step_04_analysis.plot_utils import plot_binned_box_plot
from rsp.step_04_analysis.plot_utils import plot_box_plot
from rsp.utils.data_types import after_malfunction_scopes_visualization
from rsp.utils.data_types import all_scopes_visualization
from rsp.utils.data_types import prediction_scopes_visualization
from rsp.utils.data_types import speed_up_scopes_visualization

HYPOTHESIS_ONE_COLUMNS_OF_INTEREST = [f'solver_statistics_times_total_{scope}' for scope in all_scopes_visualization]


def hypothesis_one_analysis_visualize_computational_time_comparison(
        experiment_data: DataFrame,
        output_folder: str = None,
        columns_of_interest: List[str] = HYPOTHESIS_ONE_COLUMNS_OF_INTEREST
):
    for axis_of_interest in ['experiment_id', 'n_agents', 'size', 'size_used_full_after_malfunction', 'solver_statistics_times_total_full_after_malfunction']:
        plot_box_plot(
            experiment_data=experiment_data,
            axis_of_interest=axis_of_interest,
            columns_of_interest=columns_of_interest,
            output_folder=output_folder
        )


def hypothesis_one_analysis_visualize_lateness(
        experiment_data: DataFrame,
        output_folder: str = None):
    for axis_of_interest, axis_of_interest_suffix in {
        'infra_id_schedule_id': '',
    }.items():
        for speed_up_col_pattern, y_axis_title in [
            ('costs_{}', 'Costs [-]'),
            ('costs_ratio_{}', 'Costs ratio [-]'),
            ('lateness_{}', 'Lateness (unweighted) [-]'),
            ('costs_from_route_section_penalties_{}', 'Weighted costs from route section penalties [-]'),
        ]:
            plot_binned_box_plot(
                experiment_data=experiment_data,
                axis_of_interest=axis_of_interest,
                axis_of_interest_suffix=axis_of_interest_suffix,
                output_folder=output_folder,
                cols=[speed_up_col_pattern.format(scope) for scope in speed_up_scopes_visualization],
                y_axis_title=y_axis_title,
            )


def hypothesis_one_analysis_prediction_quality(
        experiment_data: DataFrame,
        output_folder: str = None):
    for axis_of_interest, axis_of_interest_suffix in {
        'experiment_id': '',
        'infra_id_schedule_id': '',
    }.items():
        plot_binned_box_plot(
            experiment_data=experiment_data,
            axis_of_interest=axis_of_interest,
            axis_of_interest_suffix=axis_of_interest_suffix,
            output_folder=output_folder,
            cols=['n_agents', 'changed_agents_full_after_malfunction'] +
                 [prediction_col + '_' + scope
                  for scope in prediction_scopes_visualization
                  for prediction_col in [
                      'changed_agents',
                      'predicted_changed_agents_number',
                      'predicted_changed_agents_false_positives',
                      'predicted_changed_agents_false_negatives'
                  ]
                  ],
            y_axis_title='Prediction Quality Counts'
        )
        plot_binned_box_plot(
            experiment_data=experiment_data,
            axis_of_interest=axis_of_interest,
            axis_of_interest_suffix=axis_of_interest_suffix,
            output_folder=output_folder,
            cols=['changed_agents_percentage_full_after_malfunction'] +
                 [prediction_col + '_' + scope
                  for scope in prediction_scopes_visualization
                  for prediction_col in [
                      'changed_agents_percentage',
                      'predicted_changed_agents_percentage',
                      'predicted_changed_agents_false_positives_percentage',
                      'predicted_changed_agents_false_negatives_percentage'
                  ]
                  ],
            y_axis_title='Prediction Quality Percentage'
        )


def hypothesis_one_analysis_visualize_speed_up(experiment_data: DataFrame,
                                               output_folder: str = None):
    for axis_of_interest, axis_of_interest_suffix in {
        'experiment_id': '',
        'solver_statistics_times_total_full_after_malfunction': '[s]',
    }.items():
        for speed_up_col_pattern, y_axis_title in [
            ('speed_up_{}', 'Speed-up full solver time [-]'),
            ('speed_up_solve_time_{}', 'Speed-up solver time solving only [-]'),
            ('speed_up_non_solve_time_{}', 'Speed-up solver time non-processing (grounding etc.) [-]'),
        ]:
            plot_binned_box_plot(
                experiment_data=experiment_data,
                axis_of_interest=axis_of_interest,
                axis_of_interest_suffix=axis_of_interest_suffix,
                output_folder=output_folder,
                cols=[speed_up_col_pattern.format(speed_up_series) for speed_up_series in speed_up_scopes_visualization],
                y_axis_title=y_axis_title
            )


def hypothesis_one_analysis_visualize_changed_agents(experiment_data: DataFrame,
                                                     output_folder: str = None):
    for axis_of_interest, axis_of_interest_suffix in {
        'infra_id_schedule_id': '',
        'solver_statistics_times_total_full_after_malfunction': '[s]',
    }.items():
        for speed_up_col_pattern, y_axis_title in [
            ('changed_agents_{}', 'Number of changed agents [-]'),
            ('changed_agents_percentage_{}', 'Percentage of changed agents [-]'),
        ]:
            plot_binned_box_plot(
                experiment_data=experiment_data,
                axis_of_interest=axis_of_interest,
                axis_of_interest_suffix=axis_of_interest_suffix,
                output_folder=output_folder,
                cols=[speed_up_col_pattern.format(speed_up_series) for speed_up_series in after_malfunction_scopes_visualization],
                y_axis_title=y_axis_title
            )