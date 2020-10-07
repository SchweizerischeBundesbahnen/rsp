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
from pandas import DataFrame

from rsp.analysis.compute_time_analysis import plot_box_plot
from rsp.analysis.compute_time_analysis import plot_speed_up
from rsp.utils.data_types import after_malfunction_scopes
from rsp.utils.data_types import all_scopes
from rsp.utils.data_types import prediction_scopes
from rsp.utils.data_types import speed_up_scopes

HYPOTHESIS_ONE_COLUMNS_OF_INTEREST = [f'solver_statistics_times_total_{scope}' for scope in all_scopes]


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
    # TODO SIM-672 should not be necessary  - why is costs_ratio not computed correctly
    for scope in after_malfunction_scopes:
        experiment_data[f'costs_ratio_{scope}'] = \
            experiment_data['costs_full_after_malfunction'] / \
            experiment_data[f'costs_{scope}']

    for axis_of_interest, axis_of_interest_suffix in {
        'grid_id': '',
    }.items():
        for speed_up_col_pattern, y_axis_title in [
            ('costs_{}', 'Costs [-]'),
            ('costs_ratio_{}', 'Costs ratio [-]'),
            ('lateness_{}', 'Lateness (unweighted) [-]'),
            ('costs_from_route_section_penalties_{}', 'Weighted costs from route section penalties [-]'),
        ]:
            plot_speed_up(
                experiment_data=experiment_data,
                axis_of_interest=axis_of_interest,
                axis_of_interest_suffix=axis_of_interest_suffix,
                output_folder=output_folder,
                cols=[speed_up_col_pattern.format(scope) for scope in after_malfunction_scopes],
                y_axis_title=y_axis_title,
            )


def hypothesis_one_analysis_prediction_quality(
        experiment_data: DataFrame,
        output_folder: str = None):
    for axis_of_interest, axis_of_interest_suffix in {
        'experiment_id': '',
        'grid_id': '',
    }.items():
        plot_speed_up(
            experiment_data=experiment_data,
            axis_of_interest=axis_of_interest,
            axis_of_interest_suffix=axis_of_interest_suffix,
            output_folder=output_folder,
            cols=['n_agents', 'changed_agents_full_after_malfunction'] +
                 [prediction_col + '_' + scope
                  for scope in prediction_scopes
                  for prediction_col in [
                      'changed_agents',
                      'predicted_changed_agents_number',
                      'predicted_changed_agents_false_positives',
                      'predicted_changed_agents_false_negatives'
                  ]
                  ],
            y_axis_title='Prediction Quality Counts'
        )
        plot_speed_up(
            experiment_data=experiment_data,
            axis_of_interest=axis_of_interest,
            axis_of_interest_suffix=axis_of_interest_suffix,
            output_folder=output_folder,
            cols=['changed_agents_percentage_full_after_malfunction'] +
                 [prediction_col + '_' + scope
                  for scope in prediction_scopes
                  for prediction_col in [
                      'predicted_changed_agents_percentage',
                      'predicted_changed_agents_false_positives_percentage',
                      'predicted_changed_agents_false_negatives_percentage'
                  ]
                  ],
            y_axis_title='Prediction Quality Percentage'
        )


def hypothesis_one_analysis_visualize_speed_up(experiment_data: DataFrame,
                                               output_folder: str = None):
    # TODO SIM-672  do we still need this? should be in expansion?
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
        'solver_statistics_times_total_full_after_malfunction': '[s]',
    }.items():
        for speed_up_col_pattern, y_axis_title in [
            ('speed_up_{}', 'Speed-up full solver time [-]'),
            ('speed_up_{}_solve_time', 'Speed-up solver time solving only [-]'),
            ('speed_up_{}_non_solve_time', 'Speed-up solver time non-processing (grounding etc.) [-]'),
        ]:
            plot_speed_up(
                experiment_data=experiment_data,
                axis_of_interest=axis_of_interest,
                axis_of_interest_suffix=axis_of_interest_suffix,
                output_folder=output_folder,
                cols=[speed_up_col_pattern.format(speed_up_series) for speed_up_series in speed_up_scopes],
                y_axis_title=y_axis_title
            )


def hypothesis_one_analysis_visualize_changed_agents(experiment_data: DataFrame,
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
        'grid_id': '',
        'solver_statistics_times_total_full_after_malfunction': '[s]',
    }.items():
        for speed_up_col_pattern, y_axis_title in [
            ('changed_agents_{}', 'Number of changed agents [-]'),
            ('changed_agents_percentage_{}', 'Percentage of changed agents [-]'),
        ]:
            plot_speed_up(
                experiment_data=experiment_data,
                axis_of_interest=axis_of_interest,
                axis_of_interest_suffix=axis_of_interest_suffix,
                output_folder=output_folder,
                cols=[speed_up_col_pattern.format(speed_up_series) for speed_up_series in after_malfunction_scopes],
                y_axis_title=y_axis_title
            )
