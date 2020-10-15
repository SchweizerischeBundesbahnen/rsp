from typing import Optional

import pandas as pd
from rsp.analysis.compute_time_analysis import plot_box_plot
from rsp.analysis.compute_time_analysis import plot_box_plot_from_traces
from rsp.utils.data_types import all_scopes_visualization


def visualize_hypotheses_asp(
        experiment_data: pd.DataFrame,
        output_folder: Optional[str] = None):
    suffixes = all_scopes_visualization

    # problem reduction in terms of shared, conflicts, choices
    for column_prefix in ['nb_resource_conflicts', 'solver_statistics_conflicts', 'solver_statistics_choices']:
        experiment_data[f'{column_prefix}_ratio'] = \
            experiment_data[f'{column_prefix}_full_after_malfunction'] / \
            experiment_data[f'{column_prefix}_delta_perfect_after_malfunction']

    plot_box_plot(
        experiment_data=experiment_data,
        axis_of_interest='experiment_id',
        columns_of_interest=[f'nb_resource_conflicts_' + item for item in suffixes],
        title='problem reduction in terms of resource conflicts (shared)',
        y_axis_title="shared[-]",
        file_name_prefix="shared",
        output_folder=output_folder
    )
    plot_box_plot(
        experiment_data=experiment_data,
        axis_of_interest='experiment_id',
        columns_of_interest=[f'nb_resource_conflicts_ratio'],
        title='problem reduction in terms of resource conflicts (shared)',
        y_axis_title="shared ratio[-]",
        file_name_prefix="shared_ratio",
        output_folder=output_folder
    )
    plot_box_plot(
        experiment_data=experiment_data,
        axis_of_interest='experiment_id',
        columns_of_interest=[f'solver_statistics_conflicts_' + item for item in suffixes],
        title='problem reduction ratio in terms of conflicts in the model',
        y_axis_title="conflicts[-]",
        file_name_prefix="nb_resource_conflicts_ratio",
        output_folder=output_folder
    )
    plot_box_plot(
        experiment_data=experiment_data,
        axis_of_interest='experiment_id',
        columns_of_interest=[f'solver_statistics_conflicts_ratio'],
        title='problem reduction in terms of conflicts in the model',
        y_axis_title="conflicts ratio[-]",
        file_name_prefix="conflicts",
        output_folder=output_folder
    )
    plot_box_plot(
        experiment_data=experiment_data,
        axis_of_interest='experiment_id',
        columns_of_interest=[f'solver_statistics_choices_' + item for item in suffixes],
        title='problem reduction in terms of choices during solving',
        y_axis_title="choices[-]",
        file_name_prefix="choices",
        output_folder=output_folder
    )
    plot_box_plot(
        experiment_data=experiment_data,
        axis_of_interest='experiment_id',
        columns_of_interest=[f'solver_statistics_choices_ratio'],
        title='problem reduction in terms of choices during solving',
        y_axis_title="choices ratio[-]",
        file_name_prefix="choices ratio",
        output_folder=output_folder
    )

    # 003_ratio_asp_grounding_solving: solver should spend most of the time solving: compare solve and total times
    plot_box_plot(
        experiment_data=experiment_data,
        axis_of_interest='experiment_id',
        columns_of_interest=[f'solve_total_ratio_' + item for item in suffixes],
        title='003_ratio_asp_grounding_solving:\n'
              'solver should spend most of the time solving: compare solve and total solver time',
        y_axis_title="Ratio[-]",
        file_name_prefix="003",
        output_folder=output_folder
    )
    # 002_asp_absolute_total_solver_times
    plot_box_plot(
        experiment_data=experiment_data,
        axis_of_interest='experiment_id',
        columns_of_interest=[f'solver_statistics_times_total_' + item for item in suffixes] +
                            [f'solver_statistics_times_total_without_solve_' + item for item in suffixes] +
                            [f'solver_statistics_times_solve_' + item for item in suffixes] +
                            [f'solver_statistics_times_unsat_' + item for item in suffixes] +
                            [f'solver_statistics_times_sat_' + item for item in suffixes],
        title=f'002_asp_absolute_total_solver_times:\n'
              f' solver should spend most of the time solving: comparison total_time time and solve_time ',
        output_folder=output_folder,
        file_name_prefix="002"
    )
    # 004_ratio_asp_solve_propagation: propagation times should be low in comparison to solve times
    plot_box_plot(
        experiment_data=experiment_data,
        axis_of_interest='experiment_id',
        columns_of_interest=([f'summed_user_accu_propagations_' + item for item in suffixes] +
                             [f'solver_statistics_times_solve_' + item for item in suffixes]),
        title=f'004_ratio_asp_solve_propagation:\n'
              'propagation times should be low in comparison to solve times: '
              f'compare solve_time (b) against summed propagation times of user accu',
        y_axis_title="Nb summed_user_accu_propagations[-]",
        output_folder=output_folder,
        file_name_prefix="004"
    )
    plot_box_plot(
        experiment_data=experiment_data,
        axis_of_interest='experiment_id',
        columns_of_interest=([f'summed_user_step_propagations_' + item for item in suffixes] +
                             [f'solver_statistics_times_solve_' + item for item in suffixes]),
        title='004_ratio_asp_solve_propagation:\n'
              'propagation times should be low in comparison to solve times: '
              f'comparison of solve_time (b) against summed propagation times of user step',
        output_folder=output_folder,
        file_name_prefix="004",
        y_axis_title="Nb summed_user_step_propagations[-]"
    )

    # 005_ratio_asp_conflict_choice: choice conflict ratio should be close to 1;
    # if the ratio is high, the problem might be large, but not difficult
    plot_box_plot(
        experiment_data=experiment_data,
        axis_of_interest='experiment_id',
        columns_of_interest=[f'choice_conflict_ratio_' + item for item in suffixes],
        title=f'005_ratio_asp_conflict_choice: choice conflict ratio should be small; '
              f'if the ratio is high, the problem might be large, but not difficult; '
              f'choice conflict ratio',
        y_axis_title="Ratio[-]",
        output_folder=output_folder,
        file_name_prefix="005"
    )

    # Choices as predictor?
    plot_box_plot_from_traces(
        experiment_data=experiment_data,
        traces=[('solver_statistics_times_total_' + item, 'solver_statistics_choices_' + item) for item in suffixes],
        title=f'XXX_choices_are_good_predictor_of_solution_time: '
              'Choices represent the size of the solution space (how many routing alternatives, '
              f'how many potential resource conflicts).\n'
              f'How much are choices and solution times correlated?',
        output_folder=output_folder,
        x_axis_title="choices",
        pdf_file="XXX_choices_as_predictor.pdf"
    )
    # Conflicts as predictor?
    plot_box_plot_from_traces(
        experiment_data=experiment_data,
        traces=[('solver_statistics_times_total_' + item, 'solver_statistics_conflicts_' + item) for item in suffixes],
        title=f'XXX_conflicts_are_good_predictor_of_longer_solution_time_than_expected: '
              'Conflicts represent the number of routing alternatives and .\n'
              f'How much are conflicts and solution times correlated?',
        output_folder=output_folder,
        pdf_file="XXX_conflicts_as_predictor.pdf",
        x_axis_title="conflicts"
    )
    # Shared as predictor?
    plot_box_plot_from_traces(
        experiment_data=experiment_data,
        traces=[('solver_statistics_times_total_' + item, 'nb_resource_conflicts_' + item) for item in suffixes],
        title=f'XXX_shared_are_good_predictor_of_longer_solution_time_than_expected: '
              'Shared are resource conflicts of time windows.\n'
              f'How much does number of resource conflicts predict solution times?',
        output_folder=output_folder,
        pdf_file="XXX_shared_as_predictor.pdf",
        x_axis_title="shared"
    )

    # Optimization progress
    # TODO SIM-376 for the time being only plot final costs
    plot_box_plot(
        experiment_data=experiment_data,
        axis_of_interest='experiment_id',
        columns_of_interest=[f'solver_statistics_costs_' + item for item in suffixes],
        title=f'XXX_low_cost_optimal_solutions_may_be_harder_to_find:\n'
              f'final optimization costs per experiment_id',
        y_axis_title="Costs[??]",
        output_folder=output_folder,
        file_name_prefix="XXX_low_cost_optimal_solutions_may_be_harder_to_find"
    )
    plot_box_plot_from_traces(
        experiment_data=experiment_data,
        traces=[('solver_statistics_times_total_' + item, 'solver_statistics_costs_' + item) for item in suffixes],
        title=f'XXX_low_cost_optimal_solutions_may_be_harder_to_find\n'
              f'Are solver times and optimization costs correlated? '
              f'Final optimization costs per total_time',
        output_folder=output_folder,
        x_axis_title="costs",
        pdf_file="XXX_low_cost_optimal_solutions_may_be_harder_to_find.pdf",
    )
