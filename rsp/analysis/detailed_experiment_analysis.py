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
    plot_box_plot(
        experiment_data=experiment_data,
        axis_of_interest='experiment_id',
        columns_of_interest=[f'lateness_{scope}' for scope in after_malfunction_scopes],
        output_folder=output_folder,
        title='Total delay',
        color_offset=1
    )
    plot_box_plot(
        experiment_data=experiment_data,
        axis_of_interest='experiment_id',
        columns_of_interest=[f'costs_{scope}' for scope in after_malfunction_scopes],
        output_folder=output_folder,
        title='Effective costs',
        color_offset=1
    )

    for axis_of_interest, axis_of_interest_suffix in {
        'experiment_id': '',
        'solver_statistics_times_total_full_after_malfunction': '[s]'
    }.items():
        for speed_up_col_pattern, y_axis_title in [
            ('costs_ratio_{}', 'Effective costs ratio [-]'),
        ]:
            plot_speed_up(
                experiment_data=experiment_data,
                axis_of_interest=axis_of_interest,
                axis_of_interest_suffix=axis_of_interest_suffix,
                output_folder=output_folder,
                cols=[speed_up_col_pattern.format(speed_up_series) for speed_up_series in speed_up_scopes],
                y_axis_title=y_axis_title,

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
        'solver_statistics_times_total_full_after_malfunction': '[s]',
    }.items():
        for speed_up_col_pattern, y_axis_title in [
            ('speed_up_{}', 'Speed-up full solver time [-]'),
            ('speed_up_{}_solve_time', 'Speed-up solver time solving only [-]'),
            ('speed_up_{}_non_solve_time', 'Speed-up solver time non-processing (grounding etc.) [-]'),
            ('changed_agents_percentage_{}', 'Percentage of changed agents [-]'),
        ]:
            plot_speed_up(
                experiment_data=experiment_data,
                axis_of_interest=axis_of_interest,
                axis_of_interest_suffix=axis_of_interest_suffix,
                output_folder=output_folder,
                cols=[speed_up_col_pattern.format(speed_up_series) for speed_up_series in speed_up_scopes],
                y_axis_title=y_axis_title
            )
