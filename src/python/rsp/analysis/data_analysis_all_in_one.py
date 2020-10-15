from typing import List

from pandas import DataFrame
from rsp.analysis.compute_time_analysis import extract_schedule_plotting
from rsp.analysis.compute_time_analysis import plot_agent_specific_delay
from rsp.analysis.compute_time_analysis import plot_changed_agents
from rsp.analysis.compute_time_analysis import plot_histogram_from_delay_data
from rsp.analysis.compute_time_analysis import plot_lateness
from rsp.analysis.compute_time_analysis import plot_resource_occupation_heat_map
from rsp.analysis.compute_time_analysis import plot_resource_time_diagrams
from rsp.analysis.compute_time_analysis import plot_route_dag
from rsp.analysis.compute_time_analysis import plot_shared_heatmap
from rsp.analysis.compute_time_analysis import plot_time_density
from rsp.analysis.compute_time_analysis import plot_time_window_resource_trajectories
from rsp.analysis.compute_time_analysis import plot_train_paths
from rsp.analysis.detailed_experiment_analysis import hypothesis_one_analysis_prediction_quality
from rsp.analysis.detailed_experiment_analysis import hypothesis_one_analysis_visualize_changed_agents
from rsp.analysis.detailed_experiment_analysis import hypothesis_one_analysis_visualize_computational_time_comparison
from rsp.analysis.detailed_experiment_analysis import hypothesis_one_analysis_visualize_lateness
from rsp.analysis.detailed_experiment_analysis import hypothesis_one_analysis_visualize_speed_up
from rsp.analysis.detailed_experiment_analysis import plot_agent_speeds
from rsp.analysis.detailed_experiment_analysis import plot_nb_route_alternatives
from rsp.analysis.detailed_experiment_analysis import plot_time_window_sizes
from rsp.asp_plausibility.asp_plausi import visualize_hypotheses_asp
from rsp.schedule_problem_description.analysis.route_dag_analysis import visualize_route_dag_constraints_simple_wrapper
from rsp.schedule_problem_description.data_types_and_utils import ScheduleProblemEnum
from rsp.utils.data_types import convert_list_of_experiment_results_analysis_to_data_frame
from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.data_types import filter_experiment_results_analysis_data_frame
from rsp.utils.experiment_render_utils import visualize_experiment
from rsp.utils.experiments import EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME
from rsp.utils.experiments import EXPERIMENT_DATA_SUBDIRECTORY_NAME
from rsp.utils.experiments import load_and_expand_experiment_results_from_data_folder
from rsp.utils.file_utils import check_create_folder
from rsp.utils.global_data_configuration import BASELINE_DATA_FOLDER


def hypothesis_one_data_analysis(
        experiment_output_directory: str,
        analysis_2d: bool = False,
        qualitative_analysis_experiment_ids: List[int] = None,
        save_as_tsv: bool = False
):
    """

    Parameters
    ----------
    analysis_2d
    experiment_output_directory
    save_as_tsv
    qualitative_analysis_experiment_ids
    """

    # Import the desired experiment results
    experiment_analysis_directory = f'{experiment_output_directory}/{EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME}/'
    experiment_data_directory = f'{experiment_output_directory}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}'

    # Create output directoreis
    check_create_folder(experiment_analysis_directory)

    experiment_results_list_nonified: List[ExperimentResultsAnalysis] = load_and_expand_experiment_results_from_data_folder(
        experiment_data_folder_name=experiment_data_directory,
        nonify_all_structured_fields=True
    )

    # convert to data frame for statistical analysis
    experiment_data: DataFrame = convert_list_of_experiment_results_analysis_to_data_frame(experiment_results_list_nonified)
    experiment_data = filter_experiment_results_analysis_data_frame(experiment_data)

    if save_as_tsv:
        # save experiment data to .tsv for Excel inspection
        experiment_data.to_csv(f"{experiment_data_directory}/data.tsv", sep="\t")

    # quantitative analysis
    results_folder = f'{experiment_analysis_directory}/main_results'
    if analysis_2d:
        # main results
        hypothesis_one_analysis_visualize_computational_time_comparison(
            experiment_data=experiment_data,
            output_folder=results_folder
        )
        hypothesis_one_analysis_visualize_speed_up(
            experiment_data=experiment_data,
            output_folder=results_folder
        )
        hypothesis_one_analysis_visualize_lateness(
            experiment_data=experiment_data,
            output_folder=results_folder
        )
        visualize_hypotheses_asp(
            experiment_data=experiment_data,
            output_folder=results_folder
        )
        hypothesis_one_analysis_visualize_changed_agents(
            experiment_data=experiment_data,
            output_folder=results_folder
        )
        hypothesis_one_analysis_prediction_quality(
            experiment_data=experiment_data,
            output_folder=results_folder
        )

    if qualitative_analysis_experiment_ids:
        experiment_results_list: List[ExperimentResultsAnalysis] = load_and_expand_experiment_results_from_data_folder(
            experiment_data_folder_name=experiment_data_directory,
            experiment_ids=qualitative_analysis_experiment_ids
        )
        for experiment_result in experiment_results_list:
            visualize_experiment(
                experiment_parameters=experiment_result.experiment_parameters,
                experiment_results_analysis=experiment_result,
                experiment_analysis_directory=experiment_analysis_directory,
                flatland_rendering=False
            )

            agent_of_interest = experiment_result.malfunction.agent_id
            output_folder_of_interest = f'{results_folder}/experiment_{experiment_result.experiment_id:04d}_agent_{agent_of_interest:04d}/'
            schedule_plotting = extract_schedule_plotting(experiment_result=experiment_result)
            plot_time_window_resource_trajectories(
                experiment_result=experiment_result,
                schedule_plotting=schedule_plotting,
                output_folder=output_folder_of_interest
            )
            plot_shared_heatmap(
                schedule_plotting=schedule_plotting,
                experiment_result=experiment_result,
                output_folder=output_folder_of_interest
            )
            plot_resource_time_diagrams(
                schedule_plotting=schedule_plotting,
                with_diff=True,
                output_folder=output_folder_of_interest
            )
            plot_histogram_from_delay_data(
                experiment_results=experiment_result,
                output_folder=output_folder_of_interest
            )
            plot_lateness(
                experiment_results=experiment_result,
                output_folder=output_folder_of_interest
            )
            plot_agent_specific_delay(
                experiment_results=experiment_result,
                output_folder=output_folder_of_interest
            )
            plot_changed_agents(
                experiment_results=experiment_result,
                output_folder=output_folder_of_interest
            )
            plot_route_dag(
                experiment_results_analysis=experiment_result,
                agent_id=agent_of_interest,
                suffix_of_constraints_to_visualize=ScheduleProblemEnum.PROBLEM_SCHEDULE,
                output_folder=output_folder_of_interest
            )
            plot_nb_route_alternatives(
                experiment_results=experiment_result,
                output_folder=output_folder_of_interest
            )
            plot_agent_speeds(
                experiment_results=experiment_result,
                output_folder=output_folder_of_interest
            )
            plot_time_window_sizes(
                experiment_results=experiment_result,
                output_folder=output_folder_of_interest
            )
            plot_resource_occupation_heat_map(
                schedule_plotting=schedule_plotting,
                plotting_information=schedule_plotting.plotting_information,
                title_suffix='Schedule',
                output_folder=output_folder_of_interest
            )
            plot_train_paths(
                plotting_data=schedule_plotting,
                agent_ids=[agent_of_interest],
                file_name=f'{output_folder_of_interest}/train_paths.pdf' if output_folder_of_interest is not None else None
            )
            plot_time_density(
                schedule_as_resource_occupations=schedule_plotting.schedule_as_resource_occupations,
                output_folder=output_folder_of_interest
            )

            visualize_route_dag_constraints_simple_wrapper(
                schedule_problem_description=experiment_result.problem_full,
                trainrun_dict=None,
                experiment_malfunction=experiment_result.malfunction,
                agent_id=agent_of_interest,
                file_name=f"{output_folder_of_interest}/schedule_route_dag.pdf"
            )


if __name__ == '__main__':
    hypothesis_one_data_analysis(
        experiment_output_directory=BASELINE_DATA_FOLDER,
        analysis_2d=True,
    )
