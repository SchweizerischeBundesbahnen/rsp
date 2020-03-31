from typing import List

import pandas as pd

from rsp.experiment_solvers.data_types import SchedulingExperimentResult
from rsp.utils.analysis_tools import two_dimensional_scatter_plot
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


def asp_plausi_analysis(
        experiment_results_list: List[ExperimentResultsAnalysis],
        output_folder: str):
    data_frame = pd.DataFrame(data=[
        {
            'experiment_id': r.experiment_id,
            **_expand_asp_solver_statistics_for_asp_plausi(r=r.results_full, suffix="full"),
            **_expand_asp_solver_statistics_for_asp_plausi(r=r.results_full_after_malfunction,
                                                           suffix="full_after_malfunction"),
            **_expand_asp_solver_statistics_for_asp_plausi(r=r.results_delta_after_malfunction,
                                                           suffix="delta_after_malfunction"),

        }
        for r in experiment_results_list])
    for item in ['full', 'full_after_malfunction', 'delta_after_malfunction']:
        # 003_ratio_asp_grounding_solving: solver should spend most of the time solving: compare solve and total times
        two_dimensional_scatter_plot(
            data=data_frame,
            columns=['experiment_id', 'solve_total_ratio_' + item],
            title='003_ratio_asp_grounding_solving:\n'
                  'solver should spend most of the time solving: compare solve and total solver time for ' + item,
            output_folder=output_folder,
            show_global_mean=True
        )
        # 002_asp_absolute_total_solver_times
        two_dimensional_scatter_plot(
            data=data_frame,
            columns=['experiment_id', 'solve_time_' + item],
            baseline_column='total_time_' + item,
            title=f'002_asp_absolute_total_solver_times:\n'
                  f' solver should spend most of the time solving: comparison total solver time (b) and solve_time for {item}',
            output_folder=output_folder
        )
        # 004_ratio_asp_solve_propagation: propagation times should be low in comparison to solve times
        two_dimensional_scatter_plot(
            data=data_frame,
            columns=['experiment_id', 'user_accu_propagations_' + item],
            baseline_column='solve_time_' + item,
            title=f'004_ratio_asp_solve_propagation:\n'
                  'propagation times should be low in comparison to solve times: '
                  f'compare solve_time (b) against summed propagation times of user accu {item}',
            output_folder=output_folder
        )
        two_dimensional_scatter_plot(
            data=data_frame,
            columns=['experiment_id', 'user_step_propagations_' + item],
            baseline_column='solve_time_' + item,
            title='004_ratio_asp_solve_propagation:\n'
                  'propagation times should be low in comparison to solve times: '
                  f'comparison of solve_time (b) against summed propagation times of user step {item}',
            output_folder=output_folder
        )

        # 005_ratio_asp_conflict_choice: choice conflict ratio should be close to 1; if the ratio is high, the problem might be large, but not difficult
        two_dimensional_scatter_plot(
            data=data_frame,
            columns=['experiment_id', 'choice_conflict_ratio_' + item],
            title=f'005_ratio_asp_conflict_choice: choice conflict ratio should be close to 1; '
                  f'if the ratio is high, the problem might be large, but not difficult; '
                  f'choice conflict ratio {item}',
            output_folder=output_folder,
            show_global_mean=True
        )

        # Choices
        two_dimensional_scatter_plot(
            data=data_frame,
            columns=['total_time_' + item, 'choices_' + item],
            title=f'XXX_choices_are_good_predictor_of_solution_time: Choices represent the size of the solution space (how many routing alternatives, '
                  f'how many potential resource conflicts).\n'
                  f'How much are choices and solution times correlated? time_{item} choices_{item}',
            output_folder=output_folder
        )
        # Conflicts
        two_dimensional_scatter_plot(
            data=data_frame,
            columns=['total_time_' + item, 'conflicts_' + item],
            title=f'XXX_choices_are_good_predictor_of_longer_solution_time_than_expected: Conflicts represent the number of routing alternatives and .\n'
                  f'How much are conflicts and solution times correlated? time_{item} conflicts_{item}',
            output_folder=output_folder
        )

        # Optimization progress
        # TODO SIM-376 for the time being only plot final costs
        two_dimensional_scatter_plot(
            data=data_frame,
            columns=['experiment_id', 'costs_' + item],
            title=f'XXX_low_cost_optimal_solutions_may_be_harder_to_find:\n'
                  f'final optimization costs_{item} per experiment_id',
            output_folder=output_folder
        )
        two_dimensional_scatter_plot(
            data=data_frame,
            columns=['total_time_' + item, 'costs_' + item],
            title=f'XXX_low_cost_optimal_solutions_may_be_harder_to_find\n'
                  f'Are solver times and optimization costs correlated? '
                  f'Final optimization costs_{item} per costs_{item}',
            output_folder=output_folder
        )
