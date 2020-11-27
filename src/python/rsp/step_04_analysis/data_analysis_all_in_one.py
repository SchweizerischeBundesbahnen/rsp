import os
from typing import Dict
from typing import List
from typing import Optional

import networkx as nx
from flatland.envs.rail_trainrun_data_structures import Trainrun
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from pandas import DataFrame

from rsp.scheduling.scheduling_problem import get_paths_in_route_dag
from rsp.scheduling.scheduling_problem import ScheduleProblemDescription
from rsp.scheduling.scheduling_problem import ScheduleProblemEnum
from rsp.step_02_setup.data_types import ExperimentMalfunction
from rsp.step_03_run.experiment_results import ExperimentResults
from rsp.step_03_run.experiment_results_analysis import convert_list_of_experiment_results_analysis_to_data_frame
from rsp.step_03_run.experiment_results_analysis import ExperimentResultsAnalysis
from rsp.step_03_run.experiment_results_analysis import filter_experiment_results_analysis_data_frame
from rsp.step_03_run.experiments import EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME
from rsp.step_03_run.experiments import EXPERIMENT_DATA_SUBDIRECTORY_NAME
from rsp.step_03_run.experiments import load_and_expand_experiment_results_from_data_folder
from rsp.step_04_analysis.compute_time_analysis.asp_plausi import visualize_asp_problem_reduction
from rsp.step_04_analysis.compute_time_analysis.asp_plausi import visualize_asp_solver_stats
from rsp.step_04_analysis.compute_time_analysis.compute_time_analysis import hypothesis_one_analysis_prediction_quality
from rsp.step_04_analysis.compute_time_analysis.compute_time_analysis import hypothesis_one_analysis_visualize_changed_agents
from rsp.step_04_analysis.compute_time_analysis.compute_time_analysis import hypothesis_one_analysis_visualize_computational_time_comparison
from rsp.step_04_analysis.compute_time_analysis.compute_time_analysis import hypothesis_one_analysis_visualize_lateness
from rsp.step_04_analysis.compute_time_analysis.compute_time_analysis import hypothesis_one_analysis_visualize_speed_up
from rsp.step_04_analysis.detailed_experiment_analysis.detailed_experiment_analysis import plot_agent_specific_delay
from rsp.step_04_analysis.detailed_experiment_analysis.detailed_experiment_analysis import plot_agent_speeds
from rsp.step_04_analysis.detailed_experiment_analysis.detailed_experiment_analysis import plot_changed_agents
from rsp.step_04_analysis.detailed_experiment_analysis.detailed_experiment_analysis import plot_nb_route_alternatives
from rsp.step_04_analysis.detailed_experiment_analysis.detailed_experiment_analysis import plot_resource_occupation_heat_map
from rsp.step_04_analysis.detailed_experiment_analysis.detailed_experiment_analysis import plot_route_dag
from rsp.step_04_analysis.detailed_experiment_analysis.detailed_experiment_analysis import plot_shared_heatmap
from rsp.step_04_analysis.detailed_experiment_analysis.detailed_experiment_analysis import plot_time_density
from rsp.step_04_analysis.detailed_experiment_analysis.detailed_experiment_analysis import plot_time_resource_trajectories
from rsp.step_04_analysis.detailed_experiment_analysis.detailed_experiment_analysis import plot_time_resource_trajectories_all_scopes
from rsp.step_04_analysis.detailed_experiment_analysis.detailed_experiment_analysis import plot_time_window_sizes
from rsp.step_04_analysis.detailed_experiment_analysis.detailed_experiment_analysis import plot_time_windows_all_scopes
from rsp.step_04_analysis.detailed_experiment_analysis.detailed_experiment_analysis import plot_train_paths
from rsp.step_04_analysis.detailed_experiment_analysis.detailed_experiment_analysis import print_path_stats
from rsp.step_04_analysis.detailed_experiment_analysis.resources_plotting_information import extract_plotting_information
from rsp.step_04_analysis.detailed_experiment_analysis.resources_plotting_information import PlottingInformation
from rsp.step_04_analysis.detailed_experiment_analysis.route_dag_analysis import visualize_route_dag_constraints
from rsp.step_04_analysis.detailed_experiment_analysis.route_dag_analysis import visualize_route_dag_constraints_simple_wrapper
from rsp.step_04_analysis.detailed_experiment_analysis.trajectories import extract_trajectories_for_all_scopes
from rsp.step_04_analysis.detailed_experiment_analysis.trajectories import get_difference_in_time_space_trajectories
from rsp.step_04_analysis.detailed_experiment_analysis.trajectories import trajectories_from_resource_occupations_per_agent
from rsp.step_04_analysis.malfunction_analysis.disturbance_propagation import compute_disturbance_propagation_graph
from rsp.step_04_analysis.malfunction_analysis.disturbance_propagation import plot_delay_propagation_graph
from rsp.step_04_analysis.malfunction_analysis.disturbance_propagation import resource_occpuation_from_transmission_chains
from rsp.step_04_analysis.malfunction_analysis.malfunction_analysis import plot_delay_propagation_2d
from rsp.step_04_analysis.malfunction_analysis.malfunction_analysis import plot_histogram_from_delay_data
from rsp.step_04_analysis.malfunction_analysis.malfunction_analysis import plot_lateness
from rsp.step_04_analysis.malfunction_analysis.malfunction_analysis import print_situation_overview
from rsp.transmission_chains.transmission_chains import extract_transmission_chains_from_schedule
from rsp.utils.file_utils import check_create_folder
from rsp.utils.global_data_configuration import BASELINE_DATA_FOLDER
from rsp.utils.resource_occupation import extract_resource_occupations_for_all_scopes


def hypothesis_one_data_analysis(
    experiment_output_directory: str, analysis_2d: bool = False, qualitative_analysis_experiment_ids: List[int] = None, save_as_tsv: bool = False
):
    """

    Parameters
    ----------
    analysis_2d
        run compute time analysis?
    experiment_output_directory
    save_as_tsv
    qualitative_analysis_experiment_ids
        run detailed analysis and malfunction analysis on these experiments
    """

    # Import the desired experiment results
    experiment_analysis_directory = f"{experiment_output_directory}/{EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME}/"
    experiment_data_directory = f"{experiment_output_directory}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}"

    # Create output directoreis
    check_create_folder(experiment_analysis_directory)

    _, experiment_results_analysis_list = load_and_expand_experiment_results_from_data_folder(experiment_data_folder_name=experiment_data_directory)

    # convert to data frame for statistical analysis
    experiment_data: DataFrame = convert_list_of_experiment_results_analysis_to_data_frame(experiment_results_analysis_list)
    experiment_data = filter_experiment_results_analysis_data_frame(experiment_data)

    if save_as_tsv:
        # save experiment data to .tsv for Excel inspection
        experiment_data.to_csv(f"{experiment_data_directory}/data.tsv", sep="\t")

    # quantitative analysis
    results_folder = f"{experiment_analysis_directory}/main_results"
    if analysis_2d:
        # main results
        _compute_time_analysis(experiment_data, results_folder)

    if qualitative_analysis_experiment_ids:
        experiment_results_list, experiment_results_analysis_list = load_and_expand_experiment_results_from_data_folder(
            experiment_data_folder_name=experiment_data_directory, experiment_ids=qualitative_analysis_experiment_ids
        )
        for experiment_results, experiment_results_analysis in zip(experiment_results_list, experiment_results_analysis_list):
            _route_dag_constraints_analysis(
                experiment_results=experiment_results,
                experiment_analysis_directory=experiment_analysis_directory,
                experiment_results_analysis=experiment_results_analysis,
            )

            agent_of_interest = experiment_results.malfunction.agent_id
            output_folder_of_interest = (
                f"{results_folder}/experiment_{experiment_results.experiment_parameters.experiment_id:04d}_agent_{agent_of_interest:04d}/"
            )
            _detailed_experiment_results(
                experiment_results=experiment_results,
                experiment_results_analysis=experiment_results_analysis,
                output_folder_of_interest=output_folder_of_interest,
            )
            _malfunction_analysis(
                experiment_results=experiment_results,
                experiment_results_analysis=experiment_results_analysis,
                output_folder_of_interest=output_folder_of_interest,
            )


def _compute_time_analysis(experiment_data: DataFrame, results_folder: str):
    hypothesis_one_analysis_visualize_computational_time_comparison(experiment_data=experiment_data, output_folder=results_folder)
    hypothesis_one_analysis_visualize_speed_up(experiment_data=experiment_data, output_folder=results_folder)
    hypothesis_one_analysis_visualize_lateness(experiment_data=experiment_data, output_folder=results_folder)
    visualize_asp_problem_reduction(experiment_data=experiment_data, output_folder=results_folder)
    visualize_asp_solver_stats(experiment_data=experiment_data, output_folder=results_folder)
    hypothesis_one_analysis_visualize_changed_agents(experiment_data=experiment_data, output_folder=results_folder)
    hypothesis_one_analysis_prediction_quality(experiment_data=experiment_data, output_folder=results_folder)


def _detailed_experiment_results(experiment_results: ExperimentResults, experiment_results_analysis: ExperimentResultsAnalysis, output_folder_of_interest: str):
    agent_of_interest = experiment_results.malfunction.agent_id
    resource_occupations_for_all_scopes = extract_resource_occupations_for_all_scopes(experiment_result=experiment_results)
    plotting_information: PlottingInformation = extract_plotting_information(
        schedule_as_resource_occupations=resource_occupations_for_all_scopes.schedule,
        grid_depth=experiment_results.experiment_parameters.infra_parameters.width,
        sorting_agent_id=agent_of_interest,
    )
    trajectories_for_all_scopes = extract_trajectories_for_all_scopes(
        schedule_as_resource_occupations_all_scopes=resource_occupations_for_all_scopes, plotting_information=plotting_information
    )
    plot_time_windows_all_scopes(experiment_results=experiment_results, plotting_information=plotting_information, output_folder=output_folder_of_interest)
    plot_time_resource_trajectories_all_scopes(
        experiment_results=experiment_results, plotting_information=plotting_information, output_folder=output_folder_of_interest
    )
    plot_shared_heatmap(plotting_information=plotting_information, experiment_result=experiment_results, output_folder=output_folder_of_interest)
    plot_changed_agents(experiment_results=experiment_results, output_folder=output_folder_of_interest)
    plot_route_dag(
        experiment_results=experiment_results,
        agent_id=agent_of_interest,
        suffix_of_constraints_to_visualize=ScheduleProblemEnum.PROBLEM_SCHEDULE,
        output_folder=output_folder_of_interest,
    )
    plot_nb_route_alternatives(experiment_results=experiment_results, output_folder=output_folder_of_interest)
    print_path_stats(experiment_results=experiment_results)
    plot_agent_speeds(experiment_results=experiment_results, output_folder=output_folder_of_interest)
    plot_time_window_sizes(experiment_results=experiment_results, output_folder=output_folder_of_interest)
    plot_resource_occupation_heat_map(
        schedule_as_resource_occupations=resource_occupations_for_all_scopes.schedule,
        reschedule_as_resource_occupations=resource_occupations_for_all_scopes.offline_delta,
        plotting_information=plotting_information,
        title_suffix="Schedule",
        output_folder=output_folder_of_interest,
    )
    plot_train_paths(
        schedule_as_resource_occupations=resource_occupations_for_all_scopes.schedule,
        plotting_information=plotting_information,
        agent_ids=[agent_of_interest],
        pdf_file=f"{output_folder_of_interest}/train_paths.pdf" if output_folder_of_interest is not None else None,
    )
    plot_time_density(schedule_as_resource_occupations=resource_occupations_for_all_scopes.schedule, output_folder=output_folder_of_interest)
    return plotting_information, resource_occupations_for_all_scopes, trajectories_for_all_scopes


def _malfunction_analysis(experiment_results: ExperimentResults, experiment_results_analysis: ExperimentResultsAnalysis, output_folder_of_interest: str):
    agent_of_interest = experiment_results.malfunction.agent_id
    resource_occupations_for_all_scopes = extract_resource_occupations_for_all_scopes(experiment_result=experiment_results)
    plotting_information: PlottingInformation = extract_plotting_information(
        schedule_as_resource_occupations=resource_occupations_for_all_scopes.schedule,
        grid_depth=experiment_results.experiment_parameters.infra_parameters.width,
        sorting_agent_id=agent_of_interest,
    )
    trajectories_for_all_scopes = extract_trajectories_for_all_scopes(
        schedule_as_resource_occupations_all_scopes=resource_occupations_for_all_scopes, plotting_information=plotting_information
    )
    # malfunction analysis
    transmission_chains = extract_transmission_chains_from_schedule(
        malfunction=experiment_results.malfunction, occupations=resource_occupations_for_all_scopes.schedule
    )
    distance_matrix, minimal_depth = compute_disturbance_propagation_graph(
        transmission_chains=transmission_chains, number_of_trains=experiment_results.experiment_parameters.infra_parameters.number_of_agents
    )
    _, changed_agents_dict = get_difference_in_time_space_trajectories(
        base_trajectories=trajectories_for_all_scopes.schedule, target_trajectories=trajectories_for_all_scopes.offline_delta
    )
    true_positives = resource_occpuation_from_transmission_chains(transmission_chains, changed_agents_dict)
    unchanged_agents = {}
    for agent in changed_agents_dict:
        unchanged_agents[agent] = not changed_agents_dict[agent]
    false_positives = resource_occpuation_from_transmission_chains(transmission_chains, unchanged_agents)
    true_positives_trajectories = trajectories_from_resource_occupations_per_agent({0: true_positives}, plotting_information)
    false_positives_trajectories = trajectories_from_resource_occupations_per_agent({0: false_positives}, plotting_information)
    print_situation_overview(resource_occupations_for_all_scopes=resource_occupations_for_all_scopes, malfunction=experiment_results.malfunction)
    plot_time_resource_trajectories(
        title="Malfunction Propagation in Schedule",
        trajectories=trajectories_for_all_scopes.schedule,
        plotting_information=plotting_information,
        malfunction=experiment_results.malfunction,
        true_positives=true_positives_trajectories,
        false_positives=false_positives_trajectories,
        output_folder=output_folder_of_interest,
    )
    plot_delay_propagation_2d(
        plotting_information=plotting_information,
        malfunction=experiment_results.malfunction,
        schedule_as_resource_occupations=resource_occupations_for_all_scopes.offline_delta,
        delay_information=experiment_results_analysis.lateness_per_agent_offline_delta,
        depth_dict=minimal_depth,
        pdf_file=f"{output_folder_of_interest}/delay_propagation_2d.pdf",
    )
    plot_delay_propagation_graph(
        minimal_depth=minimal_depth,
        distance_matrix=distance_matrix,
        changed_agents=changed_agents_dict,
        pdf_file=f"{output_folder_of_interest}/delay_propagation_graph.pdf",
    )
    plot_histogram_from_delay_data(experiment_results_analysis=experiment_results_analysis, output_folder=output_folder_of_interest)
    plot_lateness(experiment_results_analysis=experiment_results_analysis, output_folder=output_folder_of_interest)
    plot_agent_specific_delay(experiment_results_analysis=experiment_results_analysis, output_folder=output_folder_of_interest)


def _route_dag_constraints_analysis(
    experiment_results: ExperimentResults, experiment_results_analysis: ExperimentResultsAnalysis, experiment_analysis_directory: str = None,
):
    """Render the experiment the DAGs.

    Parameters
    ----------
    experiment_analysis_directory
        Folder to store FLATland pngs and mpeg to
    """
    experiment_parameters = experiment_results.experiment_parameters

    train_runs_schedule: TrainrunDict = experiment_results.results_schedule.trainruns_dict
    train_runs_online_unrestricted: TrainrunDict = experiment_results.results_online_unrestricted.trainruns_dict
    train_runs_offline_delta: TrainrunDict = experiment_results.results_offline_delta.trainruns_dict

    problem_online_unrestricted: ScheduleProblemDescription = experiment_results.problem_online_unrestricted
    costs_online_unrestricted: ScheduleProblemDescription = experiment_results_analysis.costs_online_unrestricted
    problem_rsp_reduced_scope_perfect: ScheduleProblemDescription = experiment_results.problem_offline_delta
    costs_offline_delta: ScheduleProblemDescription = experiment_results_analysis.costs_offline_delta
    problem_schedule: ScheduleProblemDescription = experiment_results.problem_schedule
    malfunction: ExperimentMalfunction = experiment_results.malfunction
    n_agents: int = experiment_results_analysis.n_agents
    lateness_online_unrestricted: Dict[int, int] = experiment_results_analysis.lateness_per_agent_online_unrestricted
    costs_from_route_section_penalties_per_agent_online_unrestricted: Dict[
        int, int
    ] = experiment_results_analysis.costs_from_route_section_penalties_per_agent_online_unrestricted
    lateness_offline_delta: Dict[int, int] = experiment_results_analysis.lateness_per_agent_offline_delta
    costs_from_route_section_penalties_per_agent_offline_delta: Dict[
        int, int
    ] = experiment_results_analysis.costs_from_route_section_penalties_per_agent_offline_delta

    experiment_output_folder = f"{experiment_analysis_directory}/experiment_{experiment_parameters.experiment_id:04d}_analysis"
    route_dag_folder = f"{experiment_output_folder}/route_graphs"

    # Check and create the folders
    check_create_folder(experiment_output_folder)
    check_create_folder(route_dag_folder)

    visualize_route_dag_constraints_simple_wrapper(
        schedule_problem_description=experiment_results.problem_schedule,
        trainrun_dict=None,
        experiment_malfunction=experiment_results.malfunction,
        agent_id=experiment_results.malfunction.agent_id,
        file_name=f"{route_dag_folder}/schedule_route_dag.pdf",
    )

    for agent_id in problem_rsp_reduced_scope_perfect.route_dag_constraints_dict.keys():
        # TODO SIM-650 since the scheduling topo might now only contain one path per agent,
        #  we should visualize with respect to the full route DAG as in infrastructure and visualize removed edges
        topo = problem_schedule.topo_dict[agent_id]
        train_run_online_unrestricted = train_runs_online_unrestricted[agent_id]
        train_run_offline_delta = train_runs_offline_delta[agent_id]
        train_run_schedule: Trainrun = train_runs_schedule[agent_id]

        # schedule input
        visualize_route_dag_constraints(
            constraints_to_visualize=problem_schedule.route_dag_constraints_dict[agent_id],
            trainrun_to_visualize=train_run_schedule,
            vertex_lateness={},
            costs_from_route_section_penalties_per_agent_and_edge={},
            route_section_penalties=problem_schedule.route_section_penalties[agent_id],
            title=_make_title(
                agent_id, experiment_parameters, malfunction, n_agents, topo, k=experiment_parameters.infra_parameters.number_of_shortest_paths_per_agent
            ),
            file_name=(
                os.path.join(route_dag_folder, f"experiment_{experiment_parameters.experiment_id:04d}_agent_{agent_id}_route_graph_schedule.pdf")
                if experiment_analysis_directory is not None
                else None
            ),
            topo=topo,
            train_run_schedule=train_run_schedule,
            train_run_online_unrestricted=train_run_online_unrestricted,
            train_run_offline_delta=train_run_offline_delta,
        )
        # delta perfect after malfunction
        visualize_route_dag_constraints(
            constraints_to_visualize=problem_rsp_reduced_scope_perfect.route_dag_constraints_dict[agent_id],
            trainrun_to_visualize=train_run_offline_delta,
            vertex_lateness=experiment_results_analysis.vertex_lateness_offline_delta[agent_id],
            costs_from_route_section_penalties_per_agent_and_edge=(
                experiment_results_analysis.costs_from_route_section_penalties_per_agent_and_edge_offline_delta[agent_id]
            ),
            route_section_penalties=problem_rsp_reduced_scope_perfect.route_section_penalties[agent_id],
            title=_make_title(
                agent_id,
                experiment_parameters,
                malfunction,
                n_agents,
                topo,
                k=experiment_parameters.infra_parameters.number_of_shortest_paths_per_agent,
                costs=costs_offline_delta,
                eff_lateness_agent=lateness_offline_delta[agent_id],
                eff_costs_from_route_section_penalties_per_agent_agent=costs_from_route_section_penalties_per_agent_offline_delta[agent_id],
            ),
            file_name=(
                os.path.join(route_dag_folder, f"experiment_{experiment_parameters.experiment_id:04d}_agent_{agent_id}_route_graph_rsp_offline_delta.pdf")
                if experiment_analysis_directory is not None
                else None
            ),
            topo=topo,
            train_run_schedule=train_runs_schedule[agent_id],
            train_run_online_unrestricted=train_run_online_unrestricted,
            train_run_offline_delta=train_run_offline_delta,
        )
        # full rescheduling
        visualize_route_dag_constraints(
            constraints_to_visualize=problem_online_unrestricted.route_dag_constraints_dict[agent_id],
            trainrun_to_visualize=train_run_online_unrestricted,
            vertex_lateness=experiment_results_analysis.vertex_lateness_online_unrestricted[agent_id],
            costs_from_route_section_penalties_per_agent_and_edge=(
                experiment_results_analysis.costs_from_route_section_penalties_per_agent_and_edge_online_unrestricted[agent_id]
            ),
            route_section_penalties=problem_online_unrestricted.route_section_penalties[agent_id],
            title=_make_title(
                agent_id,
                experiment_parameters,
                malfunction,
                n_agents,
                topo,
                k=experiment_parameters.infra_parameters.number_of_shortest_paths_per_agent,
                costs=costs_online_unrestricted,
                eff_lateness_agent=lateness_online_unrestricted[agent_id],
                eff_costs_from_route_section_penalties_per_agent_agent=costs_from_route_section_penalties_per_agent_online_unrestricted[agent_id],
            ),
            file_name=(
                os.path.join(route_dag_folder, f"experiment_{experiment_parameters.experiment_id:04d}_agent_{agent_id}_route_graph_rsp_schedule.pdf")
                if experiment_analysis_directory is not None
                else None
            ),
            topo=topo,
            train_run_schedule=train_runs_schedule[agent_id],
            train_run_online_unrestricted=train_run_online_unrestricted,
            train_run_offline_delta=train_run_offline_delta,
        )


def _make_title(
    agent_id: str,
    experiment,
    malfunction: ExperimentMalfunction,
    n_agents: int,
    topo: nx.DiGraph,
    k: int,
    costs: Optional[int] = None,
    eff_lateness_agent: Optional[int] = None,
    eff_costs_from_route_section_penalties_per_agent_agent: Optional[int] = None,
):
    title = (
        f"experiment {experiment.experiment_id}\n"
        f"agent {agent_id}/{n_agents}\n"
        f"{malfunction}\n"
        f"k={k}\n"
        f"all paths in topo {len(get_paths_in_route_dag(topo))}\n"
        f"open paths in topo {len(get_paths_in_route_dag(topo))}\n"
    )
    if costs is not None:
        title += f"costs (all)={costs}\n"
    if eff_lateness_agent is not None:
        title += f"lateness (agent)={eff_lateness_agent}\n"
    if eff_costs_from_route_section_penalties_per_agent_agent is not None:
        title += f"costs_from_route_section_penalties_per_agent (agent)={eff_costs_from_route_section_penalties_per_agent_agent}\n"
    return title


if __name__ == "__main__":
    hypothesis_one_data_analysis(
        experiment_output_directory=BASELINE_DATA_FOLDER, analysis_2d=True,
    )
