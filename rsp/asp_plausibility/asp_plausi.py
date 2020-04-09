from typing import List
from typing import Optional

import pandas as pd

from rsp.compute_time_analysis.compute_time_analysis import plot_computational_times
from rsp.compute_time_analysis.compute_time_analysis import plot_computional_times_from_traces
from rsp.experiment_solvers.data_types import SchedulingExperimentResult
from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.general_helpers import catch_zero_division_error_as_minus_one


def _expand_asp_solver_statistics_for_asp_plausi(r: SchedulingExperimentResult, suffix: str):
    return {
        f'solve_total_ratio_{suffix}':
            catch_zero_division_error_as_minus_one(
                lambda:
                r.solver_statistics["summary"]["times"]["solve"] /
                r.solver_statistics["summary"]["times"]["total"]
            ),
        f'solve_time_{suffix}':
            r.solver_statistics["summary"]["times"]["solve"],
        f'total_time_{suffix}':
            r.solver_statistics["summary"]["times"]["total"],
        f'choice_conflict_ratio_{suffix}':
            catch_zero_division_error_as_minus_one(
                lambda:
                r.solver_statistics["solving"]["solvers"]["choices"] /
                r.solver_statistics["solving"]["solvers"]["conflicts"]
            ),
        f'choices_{suffix}':
            r.solver_statistics["solving"]["solvers"]["choices"],
        f'conflicts_{suffix}':
            r.solver_statistics["solving"]["solvers"]["conflicts"],
        f'costs_{suffix}': r.solver_statistics["summary"]["costs"][0],
        f'user_accu_propagations_{suffix}':
            sum(map(lambda d: d["Propagation(s)"],
                    r.solver_statistics["user_accu"]["DifferenceLogic"]["Thread"])),
        f'user_step_propagations_{suffix}':
            sum(map(lambda d: d["Propagation(s)"],
                    r.solver_statistics["user_step"]["DifferenceLogic"]["Thread"])),
    }


def visualize_hypotheses_asp(
        experiment_results_list: List[ExperimentResultsAnalysis],
        output_folder: Optional[str] = None):
    data_frame = pd.DataFrame(data=[
        {
            **r._asdict(),
            **_expand_asp_solver_statistics_for_asp_plausi(r=r.results_full, suffix="full"),
            **_expand_asp_solver_statistics_for_asp_plausi(r=r.results_full_after_malfunction,
                                                           suffix="full_after_malfunction"),
            **_expand_asp_solver_statistics_for_asp_plausi(r=r.results_delta_after_malfunction,
                                                           suffix="delta_after_malfunction"),

        }
        for r in experiment_results_list])
    suffixes = ['full', 'full_after_malfunction', 'delta_after_malfunction']
    # 003_ratio_asp_grounding_solving: solver should spend most of the time solving: compare solve and total times
    plot_computational_times(
        experiment_data=data_frame,
        axis_of_interest='experiment_id',
        columns_of_interest=[f'solve_total_ratio_' + item for item in suffixes],
        title='003_ratio_asp_grounding_solving:\n'
              'solver should spend most of the time solving: compare solve and total solver time',
        y_axis_title="Ratio[-]",
        output_folder=output_folder)
    # 002_asp_absolute_total_solver_times
    plot_computational_times(
        experiment_data=data_frame,
        axis_of_interest='experiment_id',
        columns_of_interest=[f'solve_time_' + item for item in suffixes] + [f'total_time_' + item for item in suffixes],
        title=f'002_asp_absolute_total_solver_times:\n'
              f' solver should spend most of the time solving: comparison total_time time and solve_time ',
        output_folder=output_folder
    )
    # 004_ratio_asp_solve_propagation: propagation times should be low in comparison to solve times
    plot_computational_times(
        experiment_data=data_frame,
        axis_of_interest='experiment_id',
        columns_of_interest=[f'user_accu_propagations_' + item for item in suffixes] + [f'solve_time_' + item for item
                                                                                        in suffixes],
        title=f'004_ratio_asp_solve_propagation:\n'
              'propagation times should be low in comparison to solve times: '
              f'compare solve_time (b) against summed propagation times of user accu',
        y_axis_title="Nb user_accu_propagations[-]",
        output_folder=output_folder
    )
    plot_computational_times(
        experiment_data=data_frame,
        axis_of_interest='experiment_id',
        columns_of_interest=[f'user_step_propagations_' + item for item in suffixes] + [f'solve_time_' + item for item
                                                                                        in suffixes],
        title='004_ratio_asp_solve_propagation:\n'
              'propagation times should be low in comparison to solve times: '
              f'comparison of solve_time (b) against summed propagation times of user step',
        y_axis_title="Nb user_step_propagations[-]",
        output_folder=output_folder
    )

    # 005_ratio_asp_conflict_choice: choice conflict ratio should be close to 1;
    # if the ratio is high, the problem might be large, but not difficult
    plot_computational_times(
        experiment_data=data_frame,
        axis_of_interest='experiment_id',
        columns_of_interest=[f'choice_conflict_ratio_' + item for item in suffixes],
        title=f'005_ratio_asp_conflict_choice: choice conflict ratio should be close to 1; '
              f'if the ratio is high, the problem might be large, but not difficult; '
              f'choice conflict ratio',
        y_axis_title="Ratio[-]",
        output_folder=output_folder
    )

    # Choices
    plot_computional_times_from_traces(
        experiment_data=data_frame,
        traces=[('total_time_' + item, 'choices_' + item) for item in suffixes],
        title=f'XXX_choices_are_good_predictor_of_solution_time: '
              'Choices represent the size of the solution space (how many routing alternatives, '
              f'how many potential resource conflicts).\n'
              f'How much are choices and solution times correlated?',
        output_folder=output_folder,
        pdf_file="choices.pdf",
        x_axis_title="choices"
    )
    # Conflicts
    plot_computional_times_from_traces(
        experiment_data=data_frame,
        traces=[('total_time_' + item, 'conflicts_' + item) for item in suffixes],
        title=f'XXX_choices_are_good_predictor_of_longer_solution_time_than_expected: '
              'Conflicts represent the number of routing alternatives and .\n'
              f'How much are conflicts and solution times correlated?',
        output_folder=output_folder,
        pdf_file="conflicts.pdf",
        x_axis_title="conflicts"
    )

    # Optimization progress
    # TODO SIM-376 for the time being only plot final costs
    plot_computational_times(
        experiment_data=data_frame,
        axis_of_interest='experiment_id',
        columns_of_interest=[f'costs_' + item for item in suffixes],
        title=f'XXX_low_cost_optimal_solutions_may_be_harder_to_find:\n'
              f'final optimization costs per experiment_id',
        output_folder=output_folder,
        y_axis_title="Costs[??]",
    )
    plot_computional_times_from_traces(
        experiment_data=data_frame,
        traces=[('total_time_' + item, 'costs_' + item) for item in suffixes],
        title=f'XXX_low_cost_optimal_solutions_may_be_harder_to_find\n'
              f'Are solver times and optimization costs correlated? '
              f'Final optimization costs per total_time',
        output_folder=output_folder,
        x_axis_title="costs"
    )
