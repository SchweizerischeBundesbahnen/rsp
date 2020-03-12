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

import pandas as pd
from networkx.drawing.tests.test_pylab import plt
from pandas import DataFrame

from rsp.route_dag.analysis.rescheduling_analysis_utils import analyze_experiment
from rsp.route_dag.analysis.rescheduling_verification_utils import plausibility_check_experiment_results
from rsp.utils.analysis_tools import average_over_grid_id
from rsp.utils.analysis_tools import three_dimensional_scatter_plot
from rsp.utils.analysis_tools import two_dimensional_scatter_plot
from rsp.utils.data_types import convert_list_of_experiment_results_analysis_to_data_frame
from rsp.utils.data_types import convert_pandas_series_experiment_results_analysis
from rsp.utils.data_types import ExperimentAgenda
from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.experiment_render_utils import visualize_experiment
from rsp.utils.experiments import EXPERIMENT_ANALYSIS_DIRECTORY_NAME
from rsp.utils.experiments import EXPERIMENT_DATA_DIRECTORY_NAME
from rsp.utils.experiments import load_and_expand_experiment_results_from_folder
from rsp.utils.experiments import load_experiment_agenda_from_file
from rsp.utils.file_utils import check_create_folder


def _2d_analysis(averaged_data: DataFrame, std_data: DataFrame, output_folder: str = None):
    for column in ['n_agents', 'size', 'size_used']:
        two_dimensional_scatter_plot(data=averaged_data,
                                     error=std_data,
                                     columns=[column, 'speed_up'],
                                     colors=['black' if inv_speed_up < 1 else 'red' for inv_speed_up in
                                             averaged_data['time_delta_after_malfunction'] / averaged_data[
                                                 'time_full_after_malfunction']],
                                     title='speed_up delta-rescheduling against re-scheduling',
                                     output_folder=output_folder
                                     )
        two_dimensional_scatter_plot(data=averaged_data,
                                     error=std_data,
                                     columns=[column, 'time_full'],
                                     title='scheduling for comparison',
                                     output_folder=output_folder
                                     )
        two_dimensional_scatter_plot(data=averaged_data, error=std_data,
                                     columns=[column, 'time_full_after_malfunction'],
                                     title='re-scheduling',
                                     output_folder=output_folder
                                     )
        two_dimensional_scatter_plot(data=averaged_data, error=std_data,
                                     columns=[column, 'time_delta_after_malfunction'],
                                     title='delta re-scheduling',
                                     output_folder=output_folder
                                     )

    # resource conflicts
    two_dimensional_scatter_plot(data=averaged_data,
                                 error=std_data,
                                 columns=['nb_resource_conflicts_full',
                                          'time_full'],
                                 title='effect of resource conflicts',
                                 output_folder=output_folder,
                                 )
    two_dimensional_scatter_plot(data=averaged_data,
                                 error=std_data,
                                 columns=['nb_resource_conflicts_full_after_malfunction',
                                          'time_full_after_malfunction'],
                                 title='effect of resource conflicts',
                                 output_folder=output_folder,
                                 )
    two_dimensional_scatter_plot(data=averaged_data,
                                 error=std_data,
                                 columns=['nb_resource_conflicts_delta_after_malfunction',
                                          'time_delta_after_malfunction'],
                                 title='effect of resource conflicts',
                                 output_folder=output_folder,
                                 )

    # nb paths
    two_dimensional_scatter_plot(data=averaged_data,
                                 error=std_data,
                                 columns=['path_search_space_rsp_full',
                                          'time_full_after_malfunction'],
                                 title='impact of number of considered paths over all agents',
                                 output_folder=output_folder,
                                 xscale='log'
                                 )
    two_dimensional_scatter_plot(data=averaged_data,
                                 error=std_data,
                                 columns=['path_search_space_rsp_delta',
                                          'time_delta_after_malfunction'],
                                 title='impact of number of considered paths over all agents',
                                 output_folder=output_folder,
                                 xscale='log'
                                 )
    two_dimensional_scatter_plot(data=averaged_data,
                                 error=std_data,
                                 columns=['path_search_space_rsp_delta',
                                          'n_agents'],
                                 title='impact of number of considered paths over all agents',
                                 output_folder=output_folder,
                                 xscale='log'
                                 )


def _3d_analysis(averaged_data: DataFrame, std_data: DataFrame):
    fig = plt.figure()
    three_dimensional_scatter_plot(data=averaged_data,
                                   error=std_data,
                                   columns=['n_agents', 'size', 'speed_up'],
                                   fig=fig,
                                   subplot_pos='111',
                                   colors=['black' if z_value < 1 else 'red' for z_value in averaged_data['speed_up']])
    three_dimensional_scatter_plot(data=averaged_data, error=std_data,
                                   columns=['n_agents', 'size', 'time_full'],
                                   fig=fig,
                                   subplot_pos='121')
    three_dimensional_scatter_plot(data=averaged_data, error=std_data,
                                   columns=['n_agents', 'size', 'time_full_after_malfunction'],
                                   fig=fig,
                                   subplot_pos='211')
    three_dimensional_scatter_plot(data=averaged_data, error=std_data,
                                   columns=['n_agents', 'size', 'time_delta_after_malfunction'],
                                   fig=fig,
                                   subplot_pos='221', )
    fig.set_size_inches(15, 15)
    plt.show()


def _asp_plausi_analysis(experiment_results_list: List[ExperimentResultsAnalysis], output_folder=str):
    def _catch_zero_division_error_as_minus_one(l):
        try:
            return l()
        except ZeroDivisionError:
            return -1

    data_frame = pd.DataFrame(data=[
        {
            'experiment_id': r.experiment_id,

            # scheduling = full # noqa E800
            'solve_total_ratio_full':
                _catch_zero_division_error_as_minus_one(
                    lambda:
                    r.results_full.solver_statistics["summary"]["times"]["solve"] /
                    r.results_full.solver_statistics["summary"]["times"]["total"]
                ),
            'solve_time_full':
                r.results_full.solver_statistics["summary"]["times"]["solve"],
            'total_time_full':
                r.results_full.solver_statistics["summary"]["times"]["total"],
            'choice_conflict_ratio_full':
                _catch_zero_division_error_as_minus_one(
                    lambda:
                    r.results_full.solver_statistics["solving"]["solvers"]["choices"] /
                    r.results_full.solver_statistics["solving"]["solvers"]["conflicts"]
                ),
            'user_accu_propagations_full':
                sum(map(lambda d: d["Propagation(s)"],
                        r.results_full.solver_statistics["user_accu"]["DifferenceLogic"]["Thread"])),
            'user_step_propagations_full':
                sum(map(lambda d: d["Propagation(s)"],
                        r.results_full.solver_statistics["user_step"]["DifferenceLogic"]["Thread"])),

            # re-scheduling without delta = full_after_malfunction
            'solve_total_ratio_full_after_malfunction':
                _catch_zero_division_error_as_minus_one(
                    lambda:
                    r.results_full_after_malfunction.solver_statistics["summary"]["times"]["solve"] /
                    r.results_full_after_malfunction.solver_statistics["summary"]["times"]["total"]
                ),
            'solve_time_full_after_malfunction':
                r.results_full_after_malfunction.solver_statistics["summary"]["times"]["solve"],
            'total_time_full_after_malfunction':
                r.results_full_after_malfunction.solver_statistics["summary"]["times"]["total"],
            'choice_conflict_ratio_full_after_malfunction':
                _catch_zero_division_error_as_minus_one(
                    lambda:
                    r.results_full_after_malfunction.solver_statistics["solving"]["solvers"]["choices"] /
                    r.results_full_after_malfunction.solver_statistics["solving"]["solvers"]["conflicts"]
                ),
            'user_accu_propagations_full_after_malfunction':
                sum(map(lambda d: d["Propagation(s)"],
                        r.results_full_after_malfunction.solver_statistics["user_accu"]["DifferenceLogic"]["Thread"])),
            'user_step_propagations_full_after_malfunction':
                sum(map(lambda d: d["Propagation(s)"],
                        r.results_full_after_malfunction.solver_statistics["user_step"]["DifferenceLogic"]["Thread"])),

            # re-scheduling with delta = delta_after_malfunction
            'solve_total_ratio_delta_after_malfunction':
                _catch_zero_division_error_as_minus_one(
                    lambda:
                    r.results_delta_after_malfunction.solver_statistics["summary"]["times"]["solve"] /
                    r.results_delta_after_malfunction.solver_statistics["summary"]["times"]["total"]
                ),
            'solve_time_delta_after_malfunction':
                r.results_delta_after_malfunction.solver_statistics["summary"]["times"]["solve"],
            'total_time_delta_after_malfunction':
                r.results_delta_after_malfunction.solver_statistics["summary"]["times"]["total"],
            'choice_conflict_ratio_delta_after_malfunction':
                _catch_zero_division_error_as_minus_one(
                    lambda:
                    r.results_delta_after_malfunction.solver_statistics["solving"]["solvers"]["choices"] /
                    r.results_delta_after_malfunction.solver_statistics["solving"]["solvers"]["conflicts"]
                ),
            'user_accu_propagations_delta_after_malfunction':
                sum(map(lambda d: d["Propagation(s)"],
                        r.results_delta_after_malfunction.solver_statistics["user_accu"]["DifferenceLogic"]["Thread"])),
            'user_step_propagations_delta_after_malfunction':
                sum(map(lambda d: d["Propagation(s)"],
                        r.results_delta_after_malfunction.solver_statistics["user_step"]["DifferenceLogic"]["Thread"])),
        }
        for r in experiment_results_list])
    for item in ['full', 'full_after_malfunction', 'delta_after_malfunction']:
        # 1. solver should spend most of the time solving: compare solve and total times
        two_dimensional_scatter_plot(data=data_frame,
                                     columns=['experiment_id', 'solve_total_ratio_' + item],
                                     title='relative comparison of solve and total solver time for ' + item,
                                     output_folder=output_folder,
                                     link_column=None
                                     )
        two_dimensional_scatter_plot(data=data_frame,
                                     columns=['experiment_id', 'solve_total_ratio_' + item],
                                     baseline_column='solve_time_' + item,
                                     title='absolute comparison of total solver time and solve_time (b) for ' + item,
                                     output_folder=output_folder,
                                     link_column=None
                                     )
        # 2. propagation times should be low in comparison to solve times
        two_dimensional_scatter_plot(data=data_frame,
                                     columns=['experiment_id', 'solve_time_' + item],
                                     baseline_column='user_accu_propagations_' + item,
                                     title='comparison of absolute values of solve_time against summed propagation times of user accu (b) ' + item,
                                     output_folder=output_folder,
                                     link_column=None
                                     )
        two_dimensional_scatter_plot(data=data_frame,
                                     columns=['experiment_id', 'solve_time_' + item],
                                     baseline_column='user_step_propagations_' + item,
                                     title='comparison of absolute values of solve_time against summed propagation times of user step (b) ' + item,
                                     output_folder=output_folder,
                                     link_column=None
                                     )

        # 3. choice conflict ratio should be close to 1; if the ratio is high, the problem might be large, but not difficult
        two_dimensional_scatter_plot(data=data_frame,
                                     columns=['experiment_id', 'choice_conflict_ratio_' + item],
                                     title='choice conflict ratio ' + item,
                                     output_folder=output_folder,
                                     link_column=None
                                     )


def hypothesis_one_data_analysis(experiment_base_directory: str,
                                 analysis_2d: bool = False,
                                 analysis_3d: bool = False,
                                 qualitative_analysis_experiment_ids: List[int] = None,
                                 flatland_rendering: bool = True
                                 ):
    """

    Parameters
    ----------
    experiment_base_directory
    analysis_2d
    analysis_3d
    qualitative_analysis_experiment_ids
    flatland_rendering
    debug
    """
    # Import the desired experiment results
    experiment_analysis_directory = f'{experiment_base_directory}/{EXPERIMENT_ANALYSIS_DIRECTORY_NAME}/'
    experiment_data_directory = f'{experiment_base_directory}/{EXPERIMENT_DATA_DIRECTORY_NAME}'

    # Create output directoreis
    check_create_folder(experiment_analysis_directory)

    experiment_results_list: List[ExperimentResultsAnalysis] = load_and_expand_experiment_results_from_folder(
        experiment_data_directory)
    experiment_agenda: ExperimentAgenda = load_experiment_agenda_from_file(experiment_data_directory)

    print(experiment_data_directory)
    print(experiment_agenda)
    # Plausibility tests on experiment data
    _run_plausibility_tests_on_experiment_data(experiment_results_list)

    # convert to data frame for statistical analysis
    experiment_data: DataFrame = convert_list_of_experiment_results_analysis_to_data_frame(experiment_results_list)

    # previews
    preview_cols = ['speed_up', 'time_delta_after_malfunction', 'experiment_id',
                    'nb_resource_conflicts_delta_after_malfunction', 'path_search_space_rsp_full']
    for preview_col in preview_cols:
        print(preview_col)
        print(experiment_data[preview_col])
        print(experiment_data[preview_col])
    print(experiment_data.dtypes)

    print("Averaging...")
    averaged_data, std_data = average_over_grid_id(experiment_data)
    print("  -> Done averaging.")

    # quantitative analysis
    if analysis_2d:
        _2d_analysis(averaged_data, std_data, output_folder=experiment_analysis_directory)
        _asp_plausi_analysis(experiment_results_list, output_folder=experiment_analysis_directory)
    if analysis_3d:
        _3d_analysis(averaged_data, std_data)

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
                                 data_frame=experiment_data,
                                 experiment_results_analysis=experiment_results_analysis,
                                 experiment_analysis_directory=experiment_analysis_directory,
                                 analysis_2d=analysis_2d,
                                 analysis_3d=analysis_3d,
                                 flatland_rendering=flatland_rendering)


def _run_plausibility_tests_on_experiment_data(l: List[ExperimentResultsAnalysis]):
    print("Running plausibility tests on experiment data...")
    for experiment_results_analysis in l:
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

        sum_lateness_full_after_malfunction: int = sum(lateness_full_after_malfunction.values())
        sum_all_route_section_penalties_full_after_malfunction: int = sum(
            sum_route_section_penalties_full_after_malfunction.values())
        sum_lateness_delta_after_malfunction: int = sum(lateness_delta_after_malfunction.values())
        sum_all_route_section_penalties_delta_after_malfunction: int = sum(
            sum_route_section_penalties_delta_after_malfunction.values())

        assert costs_full_after_malfunction == sum_lateness_full_after_malfunction + sum_all_route_section_penalties_full_after_malfunction, \
            f"experiment {experiment_id}: " \
            f"costs_full_after_malfunction={costs_full_after_malfunction}, " \
            f"sum_lateness_full_after_malfunction={sum_lateness_full_after_malfunction}, " \
            f"sum_all_route_section_penalties_full_after_malfunction={sum_all_route_section_penalties_full_after_malfunction}, "
        assert costs_delta_after_malfunction == sum_lateness_delta_after_malfunction + sum_all_route_section_penalties_delta_after_malfunction, \
            f"experiment {experiment_id}: " \
            f"costs_delta_after_malfunction={costs_delta_after_malfunction}, " \
            f"sum_lateness_delta_after_malfunction={sum_lateness_delta_after_malfunction}, " \
            f"sum_all_route_section_penalties_delta_after_malfunction={sum_all_route_section_penalties_delta_after_malfunction}, "
    print("  -> Done plausibility tests on experiment data.")


if __name__ == '__main__':
    hypothesis_one_data_analysis(experiment_base_directory='./exp_hypothesis_one_2020_03_04T19_19_00',
                                 analysis_2d=True,
                                 analysis_3d=False,
                                 qualitative_analysis_experiment_ids=[12]
                                 )
