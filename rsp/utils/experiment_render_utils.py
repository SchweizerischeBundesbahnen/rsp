import os
from typing import Dict
from typing import Optional

import networkx as nx
from flatland.envs.rail_trainrun_data_structures import Trainrun
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from pandas import DataFrame

from rsp.route_dag.analysis.route_dag_analysis import visualize_route_dag_constraints
from rsp.route_dag.route_dag import get_paths_for_route_dag_constraints
from rsp.route_dag.route_dag import get_paths_in_route_dag
from rsp.route_dag.route_dag import RouteDAGConstraints
from rsp.route_dag.route_dag import ScheduleProblemDescription
from rsp.utils.analysis_tools import plot_weg_zeit_diagramm_3d
from rsp.utils.analysis_tools import save_weg_zeit_diagramm_2d
from rsp.utils.analysis_tools import visualize_agent_density
from rsp.utils.data_types import ExperimentMalfunction
from rsp.utils.data_types import ExperimentParameters
from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.experiments import create_env_pair_for_experiment
from rsp.utils.file_utils import check_create_folder
from rsp.utils.flatland_replay_utils import replay_and_verify_trainruns


def visualize_experiment(
        experiment_parameters: ExperimentParameters,
        experiment_results_analysis: ExperimentResultsAnalysis,
        data_frame: DataFrame,
        data_folder: str = None,
        output_folder: str = None,
        flatland_rendering: bool = False,
        convert_to_mpeg: bool = True):
    """Render the experiment the DAGs and the FLATland png/mpeg in the
    analysis.

    Parameters
    ----------
    experiment_parameters: ExperimentParameters
        experiment parameters
    data_frame: DataFrame
        Pandas data frame with one experiment.
    data_folder
        Folder to store FLATland pngs and mpeg to
    flatland_rendering
        Flatland rendering?
    convert_to_mpeg
        Converts the rendering to mpeg
    """

    rows = data_frame.loc[data_frame['experiment_id'] == experiment_parameters.experiment_id]

    static_rail_env, malfunction_rail_env = create_env_pair_for_experiment(experiment_parameters)
    train_runs_input: TrainrunDict = rows['solution_full'].iloc[0]
    train_runs_full_after_malfunction: TrainrunDict = rows['solution_full_after_malfunction'].iloc[0]
    train_runs_delta_after_malfunction: TrainrunDict = rows['solution_delta_after_malfunction'].iloc[0]

    problem_rsp_full: ScheduleProblemDescription = rows['problem_full_after_malfunction'].iloc[0]
    costs_full_after_malfunction: ScheduleProblemDescription = rows['costs_full_after_malfunction'].iloc[0]
    problem_rsp_delta: ScheduleProblemDescription = rows['problem_delta_after_malfunction'].iloc[0]
    costs_delta_after_malfunction: ScheduleProblemDescription = rows['costs_delta_after_malfunction'].iloc[0]
    problem_schedule: ScheduleProblemDescription = rows['problem_full'].iloc[0]
    malfunction: ExperimentMalfunction = rows['malfunction'].iloc[0]
    n_agents: int = rows['n_agents'].iloc[0]
    lateness_full_after_malfunction: Dict[int, int] = rows['lateness_full_after_malfunction'].iloc[0]
    sum_route_section_penalties_full_after_malfunction: Dict[int, int] = \
        rows['sum_route_section_penalties_full_after_malfunction'].iloc[0]
    lateness_delta_after_malfunction: Dict[int, int] = rows['lateness_delta_after_malfunction'].iloc[0]
    sum_route_section_penalties_delta_after_malfunction: Dict[int, int] = \
        rows['sum_route_section_penalties_delta_after_malfunction'].iloc[0]

    experiment_output_folder = f"{output_folder}/experiment_{experiment_parameters.experiment_id:04d}_analysis"
    route_dag_folder = f"{experiment_output_folder}/Route_Graphs"
    metric_folder = f"{experiment_output_folder}/Metrics"
    rendering_folder = f"{experiment_output_folder}/Rendering"

    # Check and create the folders
    check_create_folder(experiment_output_folder)
    check_create_folder(route_dag_folder)
    check_create_folder(metric_folder)
    check_create_folder(rendering_folder)

    for agent_id in problem_rsp_delta.route_dag_constraints_dict.keys():
        # IMPORTANT: we visualize with respect to the full schedule DAG,
        #            but the banned elements are not passed to solver any more!
        # TODO SIM-190 documentation about this
        topo = problem_schedule.topo_dict[agent_id]
        train_run_full_after_malfunction = train_runs_full_after_malfunction[agent_id]
        train_run_delta_after_malfunction = train_runs_delta_after_malfunction[agent_id]
        train_run_input: Trainrun = train_runs_input[agent_id]

        # schedule input
        visualize_route_dag_constraints(
            topo=topo,
            train_run_input=train_run_input,
            train_run_full_after_malfunction=train_run_full_after_malfunction,
            train_run_delta_after_malfunction=train_run_delta_after_malfunction,
            f=problem_schedule.route_dag_constraints_dict[agent_id],
            vertex_eff_lateness={},
            edge_eff_route_penalties={},
            route_section_penalties=problem_schedule.route_section_penalties[agent_id],
            title=_make_title(agent_id, experiment_parameters, malfunction, n_agents, topo,
                              problem_schedule.route_dag_constraints_dict[agent_id],
                              k=experiment_parameters.number_of_shortest_paths_per_agent),
            file_name=(os.path.join(route_dag_folder,
                                    f"experiment_{experiment_parameters.experiment_id:04d}_agent_{agent_id}_route_graph_schedule.png")
                       if output_folder is not None else None)
        )
        # delta after malfunction
        visualize_route_dag_constraints(
            topo=topo,
            train_run_input=train_runs_input[agent_id],
            train_run_full_after_malfunction=train_run_full_after_malfunction,
            train_run_delta_after_malfunction=train_run_delta_after_malfunction,
            f=problem_rsp_delta.route_dag_constraints_dict[agent_id],
            vertex_eff_lateness=experiment_results_analysis.vertex_eff_lateness_delta_after_malfunction[agent_id],
            edge_eff_route_penalties=experiment_results_analysis.edge_eff_route_penalties_delta_after_malfunction[
                agent_id],
            route_section_penalties=problem_rsp_delta.route_section_penalties[agent_id],
            title=_make_title(
                agent_id, experiment_parameters, malfunction, n_agents, topo,
                problem_rsp_delta.route_dag_constraints_dict[agent_id],
                k=experiment_parameters.number_of_shortest_paths_per_agent,
                costs=costs_delta_after_malfunction,
                eff_lateness_agent=lateness_delta_after_malfunction[agent_id],
                eff_sum_route_section_penalties_agent=sum_route_section_penalties_delta_after_malfunction[agent_id]),
            file_name=(os.path.join(route_dag_folder,
                                    f"experiment_{experiment_parameters.experiment_id:04d}_agent_{agent_id}_route_graph_rsp_delta.png")
                       if output_folder is not None else None)
        )
        # full rescheduling
        visualize_route_dag_constraints(
            topo=topo,
            train_run_input=train_runs_input[agent_id],
            train_run_full_after_malfunction=train_run_full_after_malfunction,
            train_run_delta_after_malfunction=train_run_delta_after_malfunction,
            f=problem_rsp_full.route_dag_constraints_dict[agent_id],
            vertex_eff_lateness=experiment_results_analysis.vertex_eff_lateness_full_after_malfunction[agent_id],
            edge_eff_route_penalties=experiment_results_analysis.edge_eff_route_penalties_full_after_malfunction[
                agent_id],
            route_section_penalties=problem_rsp_full.route_section_penalties[agent_id],
            title=_make_title(
                agent_id, experiment_parameters, malfunction, n_agents, topo,
                problem_rsp_full.route_dag_constraints_dict[agent_id],
                k=experiment_parameters.number_of_shortest_paths_per_agent,
                costs=costs_full_after_malfunction,
                eff_lateness_agent=lateness_full_after_malfunction[agent_id],
                eff_sum_route_section_penalties_agent=sum_route_section_penalties_full_after_malfunction[agent_id]),
            file_name=(os.path.join(route_dag_folder,
                                    f"experiment_{experiment_parameters.experiment_id:04d}_agent_{agent_id}_route_graph_rsp_full.png")
                       if output_folder is not None else None)
        )

    # Generate aggregated visualization
    save_weg_zeit_diagramm_2d(experiment_data=experiment_results_analysis, output_folder=metric_folder)

    visualize_agent_density(experiment_results_analysis, output_folder=metric_folder)

    replay_and_verify_trainruns(data_folder=data_folder,
                                experiment_id=experiment_results_analysis.experiment_id,
                                expected_malfunction=experiment_results_analysis.malfunction,
                                rendering=flatland_rendering,
                                rail_env=malfunction_rail_env,
                                trainruns=train_runs_full_after_malfunction,
                                convert_to_mpeg=convert_to_mpeg)

    plot_weg_zeit_diagramm_3d(experiment_results_analysis)


def _make_title(agent_id: str,
                experiment,
                malfunction: ExperimentMalfunction,
                n_agents: int,
                topo: nx.DiGraph,
                current_constraints: RouteDAGConstraints,
                k: int,
                costs: Optional[int] = None,
                eff_lateness_agent: Optional[int] = None,
                eff_sum_route_section_penalties_agent: Optional[int] = None,
                ):
    title = f"experiment {experiment.experiment_id}\n" \
            f"agent {agent_id}/{n_agents}\n" \
            f"{malfunction}\n" \
            f"k={k}\n" \
            f"all paths in topo {len(get_paths_in_route_dag(topo))}\n" \
            f"open paths in topo {len(get_paths_for_route_dag_constraints(topo, current_constraints))}\n"
    if costs is not None:
        title += f"costs (all)={costs}\n"
    if eff_lateness_agent is not None:
        title += f"lateness (agent)={eff_lateness_agent}\n"
    if eff_sum_route_section_penalties_agent is not None:
        title += f"sum_route_section_penalties (agent)={eff_sum_route_section_penalties_agent}\n"
    return title
