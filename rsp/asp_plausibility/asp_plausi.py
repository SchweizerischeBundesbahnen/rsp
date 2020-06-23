from typing import Optional

import pandas as pd

from rsp.compute_time_analysis.compute_time_analysis import plot_computational_times
from rsp.compute_time_analysis.compute_time_analysis import plot_computional_times_from_traces


def visualize_hypotheses_asp(
        experiment_data: pd.DataFrame,
        output_folder: Optional[str] = None):
    suffixes = ['full', 'full_after_malfunction', 'delta_after_malfunction']
    # 003_ratio_asp_grounding_solving: solver should spend most of the time solving: compare solve and total times
    plot_computational_times(
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
    plot_computational_times(
        experiment_data=experiment_data,
        axis_of_interest='experiment_id',
        columns_of_interest=[f'solve_time_' + item for item in suffixes] + [f'total_time_' + item for item in suffixes],
        title=f'002_asp_absolute_total_solver_times:\n'
              f' solver should spend most of the time solving: comparison total_time time and solve_time ',
        output_folder=output_folder,
        file_name_prefix="002"
    )
    # 004_ratio_asp_solve_propagation: propagation times should be low in comparison to solve times
    plot_computational_times(
        experiment_data=experiment_data,
        axis_of_interest='experiment_id',
        columns_of_interest=[f'user_accu_propagations_' + item for item in suffixes] + [f'solve_time_' + item for item
                                                                                        in suffixes],
        title=f'004_ratio_asp_solve_propagation:\n'
              'propagation times should be low in comparison to solve times: '
              f'compare solve_time (b) against summed propagation times of user accu',
        y_axis_title="Nb user_accu_propagations[-]",
        output_folder=output_folder,
        file_name_prefix="004"
    )
    plot_computational_times(
        experiment_data=experiment_data,
        axis_of_interest='experiment_id',
        columns_of_interest=[f'user_step_propagations_' + item for item in suffixes] + [f'solve_time_' + item for item
                                                                                        in suffixes],
        title='004_ratio_asp_solve_propagation:\n'
              'propagation times should be low in comparison to solve times: '
              f'comparison of solve_time (b) against summed propagation times of user step',
        output_folder=output_folder,
        file_name_prefix="004",
        y_axis_title="Nb user_step_propagations[-]"
    )

    # 005_ratio_asp_conflict_choice: choice conflict ratio should be close to 1;
    # if the ratio is high, the problem might be large, but not difficult
    plot_computational_times(
        experiment_data=experiment_data,
        axis_of_interest='experiment_id',
        columns_of_interest=[f'choice_conflict_ratio_' + item for item in suffixes],
        title=f'005_ratio_asp_conflict_choice: choice conflict ratio should be close to 1; '
              f'if the ratio is high, the problem might be large, but not difficult; '
              f'choice conflict ratio',
        y_axis_title="Ratio[-]",
        output_folder=output_folder,
        file_name_prefix="005"
    )

    # Choices
    plot_computional_times_from_traces(
        experiment_data=experiment_data,
        traces=[('total_time_' + item, 'choices_' + item) for item in suffixes],
        title=f'XXX_choices_are_good_predictor_of_solution_time: '
              'Choices represent the size of the solution space (how many routing alternatives, '
              f'how many potential resource conflicts).\n'
              f'How much are choices and solution times correlated?',
        output_folder=output_folder,
        x_axis_title="choices",
        pdf_file="XXX_choices.pdf"
    )
    # Conflicts
    plot_computional_times_from_traces(
        experiment_data=experiment_data,
        traces=[('total_time_' + item, 'conflicts_' + item) for item in suffixes],
        title=f'XXX_conflicts_are_good_predictor_of_longer_solution_time_than_expected: '
              'Conflicts represent the number of routing alternatives and .\n'
              f'How much are conflicts and solution times correlated?',
        output_folder=output_folder,
        pdf_file="XXX_conflicts.pdf",
        x_axis_title="conflicts"
    )
    # Shared
    plot_computional_times_from_traces(
        experiment_data=experiment_data,
        traces=[('total_time_' + item, 'nb_resource_conflicts_' + item) for item in suffixes],
        title=f'XXX_shared_are_good_predictor_of_longer_solution_time_than_expected: '
              'Shared are resource conflicts of time windows.\n'
              f'How much does number of resource conflicts predict solution times?',
        output_folder=output_folder,
        pdf_file="XXX_shared.pdf",
        x_axis_title="shared"
    )

    # Optimization progress
    # TODO SIM-376 for the time being only plot final costs
    plot_computational_times(
        experiment_data=experiment_data,
        axis_of_interest='experiment_id',
        columns_of_interest=[f'costs_' + item for item in suffixes],
        title=f'XXX_low_cost_optimal_solutions_may_be_harder_to_find:\n'
              f'final optimization costs per experiment_id',
        y_axis_title="Costs[??]",
        output_folder=output_folder,
        file_name_prefix="XXX"
    )
    plot_computional_times_from_traces(
        experiment_data=experiment_data,
        traces=[('total_time_' + item, 'costs_' + item) for item in suffixes],
        title=f'XXX_low_cost_optimal_solutions_may_be_harder_to_find\n'
              f'Are solver times and optimization costs correlated? '
              f'Final optimization costs per total_time',
        output_folder=output_folder,
        x_axis_title="costs",
        pdf_file="XXX_costs.pdf",
    )
