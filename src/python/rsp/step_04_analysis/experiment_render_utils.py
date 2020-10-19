import os
import pprint
import warnings
from typing import Dict
from typing import Optional

import networkx as nx
import numpy as np
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_trainrun_data_structures import Trainrun
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import Waypoint
from rsp.scheduling.scheduling_problem import get_paths_in_route_dag
from rsp.scheduling.scheduling_problem import ScheduleProblemDescription
from rsp.step_01_planning.experiment_parameters_and_ranges import ExperimentParameters
from rsp.step_02_setup.data_types import ExperimentMalfunction
from rsp.step_03_run.experiment_results_analysis import ExperimentResultsAnalysis
from rsp.step_03_run.experiments import create_env_from_experiment_parameters
from rsp.step_04_analysis.detailed_experiment_analysis.route_dag_analysis import visualize_route_dag_constraints
from rsp.utils.file_utils import check_create_folder

_pp = pprint.PrettyPrinter(indent=4)

# For each time step, agent's location: int key is the time step at which the train occupies the resource
# (agents occupy only one resource at a time, no release time)
TrainSchedule = Dict[int, Waypoint]
# TrainSchedule for all  trains: int key is the agent handle for which the schedule is returned
TrainScheduleDict = Dict[int, TrainSchedule]


def convert_trainrun_dict_to_train_schedule_dict(trainrun_dict: TrainrunDict) -> TrainScheduleDict:
    """
    Converts a `TrainrunDict` (only entry times into a new position) into a dict with the waypoint for each agent and time step.
    Parameters
    ----------
    trainrun_dict: TrainrunDict
        for each agent, a list of time steps with new position

    Returns
    -------
    TrainScheduleDict
        for each agent and time step, the current position (not considering release times)

    """
    train_schedule_dict: TrainScheduleDict = {}
    for agent_id, trainrun in trainrun_dict.items():
        train_schedule: TrainSchedule = {}
        train_schedule_dict[agent_id] = train_schedule
        time_step = 0
        current_position = None
        end_time_step = trainrun[-1].scheduled_at
        for next_trainrun_waypoint in trainrun[1:]:
            while time_step + 1 < next_trainrun_waypoint.scheduled_at:
                train_schedule[time_step] = current_position
                time_step += 1
            assert time_step + 1 == next_trainrun_waypoint.scheduled_at
            if time_step + 1 == end_time_step:
                train_schedule[time_step] = None
                break
            current_position = next_trainrun_waypoint.waypoint
            train_schedule[time_step] = current_position
            time_step += 1
    return train_schedule_dict


def render_trainruns(  # noqa:C901
    rail_env: RailEnv,
    trainruns: TrainrunDict,
    title: str = None,
    malfunction: Optional[ExperimentMalfunction] = None,
    experiment_id: int = 0,
    data_folder: Optional[str] = None,
    convert_to_mpeg: bool = False,
    mpeg_resolution: str = "640x360",
    show: bool = False,
):
    """

    Parameters
    ----------
    rail_env:
        FLATland `RailEnv`. Be aware it is `reset` correctly.
    trainruns: TrainrunDict
        a train run per agent
    title: str
        Title to set for folder
    malfunction: Optional[ExperimentMalfunction]
        if a malfunction is supposed to happen, pass it here.
    data_folder: Optional[str]
        if `rendering=True`, save the visualization to a png per time step
    experiment_id:
        if `rendering=True` and `data_folder` is given, use the pngs/mpegs to files with this identification
    convert_to_mpeg: bool
        if `rendering=True` and `data_folder` is defined, convert the generated pngs to an mpeg
    mpeg_resolution:
        resolution to use if mpeg is generated
    show: bool
        show panel while rendering?

    Returns
    -------

    """
    image_output_directory = None

    if data_folder:
        foldername = f"experiment_{experiment_id}_rendering_output"

        if title is not None:
            foldername = f"experiment_{experiment_id}_rendering_output_{title}"
        image_output_directory = os.path.join(data_folder, f"experiment_{experiment_id:04d}_analysis", foldername)

        check_create_folder(image_output_directory)
    train_schedule_dict: TrainScheduleDict = convert_trainrun_dict_to_train_schedule_dict(trainrun_dict=trainruns)
    max_episode_steps = np.max([time_step for agent_id, train_schedule in train_schedule_dict.items() for time_step in train_schedule.keys()])
    renderer = init_renderer_for_env(rail_env)

    for time_step in range(max_episode_steps):
        for agent_id, train_schedule in train_schedule_dict.items():
            waypoint = train_schedule.get(time_step, None)
            if waypoint is None:
                rail_env.agents[agent_id].position = None
            else:
                rail_env.agents[agent_id].position = waypoint.position
                rail_env.agents[agent_id].direction = waypoint.direction
        if malfunction is not None:
            malfunction_start = malfunction.time_step
            malfunction_end = malfunction_start + malfunction.malfunction_duration
            if malfunction_start <= time_step < malfunction_end:
                rail_env.agents[malfunction.agent_id].malfunction_data["malfunction"] = malfunction_end - time_step
            else:
                rail_env.agents[malfunction.agent_id].malfunction_data["malfunction"] = 0
        # TODO SIM-516  simplify: test_id, solver_name???
        render_env(renderer, test_id=0, solver_name="data_analysis", i_step=time_step, show=show, image_output_directory=image_output_directory)

    try:
        cleanup_renderer_for_env(renderer)
    except AttributeError as e:
        # TODO why does this happen?
        warnings.warn(str(e))
    if convert_to_mpeg:
        import ffmpeg

        (
            ffmpeg.input(f"{image_output_directory}/flatland_frame_0000_%04d_data_analysis.png", r="5", s=mpeg_resolution)
            .output(f"{image_output_directory}/experiment_{experiment_id}_flatland_data_analysis.mp4", crf=15, pix_fmt="yuv420p", vcodec="libx264")
            .overwrite_output()
            .run()
        )


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def init_renderer_for_env(env: RailEnv):
    from flatland.utils.rendertools import AgentRenderVariant
    from flatland.utils.rendertools import RenderTool

    return RenderTool(
        env,
        gl="PILSVG",
        agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
        show_debug=True,
        clear_debug_text=True,
        screen_height=1000,
        screen_width=1000,
    )


def render_env(renderer, test_id: int, solver_name, i_step: int, show=True, image_output_directory: Optional[str] = "./rendering_output"):
    """

    Parameters
    ----------
    renderer: Optional[RenderTool]
    test_id: int
        id in file name
    solver_name:
        solver name for file name
    i_step: int
        used in file name
    show:
        render without showing?
    image_output_directory: Optional[str]
        store files to this directory if given
    """
    from flatland.utils.rendertools import RenderTool

    renderer: RenderTool = renderer
    renderer.render_env(show=show, show_observations=False, show_predictions=False)
    if image_output_directory is not None:
        if not os.path.exists(image_output_directory):
            os.makedirs(image_output_directory)
        renderer.gl.save_image(os.path.join(image_output_directory, "flatland_frame_{:04d}_{:04d}_{}.png".format(test_id, i_step, solver_name)))


def cleanup_renderer_for_env(renderer):
    from flatland.utils.rendertools import RenderTool

    renderer: RenderTool = renderer
    # close renderer window
    try:
        renderer.close_window()
    except AttributeError as e:
        # TODO why does this happen?
        warnings.warn(str(e))


def visualize_experiment(
    experiment_parameters: ExperimentParameters,
    experiment_results_analysis: ExperimentResultsAnalysis,
    experiment_analysis_directory: str = None,
    route_dag: bool = True,
    flatland_rendering: bool = False,
    convert_to_mpeg: bool = True,
):
    """Render the experiment the DAGs and the FLATland png/mpeg in the
    analysis.

    Parameters
    ----------
    route_dag
        toggle route-dag rendering
    experiment_parameters: ExperimentParameters
        experiment parameters
    experiment_analysis_directory
        Folder to store FLATland pngs and mpeg to
    flatland_rendering
        Flatland rendering?
    convert_to_mpeg
        Converts the rendering to mpeg
    """

    rail_env = create_env_from_experiment_parameters(experiment_parameters.infra_parameters)
    train_runs_schedule: TrainrunDict = experiment_results_analysis.solution_schedule
    train_runs_online_unrestricted: TrainrunDict = experiment_results_analysis.solution_online_unrestricted
    train_runs_offline_delta: TrainrunDict = experiment_results_analysis.solution_offline_delta

    problem_online_unrestricted: ScheduleProblemDescription = experiment_results_analysis.problem_online_unrestricted
    costs_online_unrestricted: ScheduleProblemDescription = experiment_results_analysis.costs_online_unrestricted
    problem_rsp_reduced_scope_perfect: ScheduleProblemDescription = experiment_results_analysis.problem_offline_delta
    costs_offline_delta: ScheduleProblemDescription = experiment_results_analysis.costs_offline_delta
    problem_schedule: ScheduleProblemDescription = experiment_results_analysis.problem_schedule
    malfunction: ExperimentMalfunction = experiment_results_analysis.malfunction
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
    metric_folder = f"{experiment_output_folder}/metrics"
    rendering_folder = f"{experiment_output_folder}/rendering"

    # Check and create the folders
    check_create_folder(experiment_output_folder)
    check_create_folder(route_dag_folder)
    check_create_folder(metric_folder)
    check_create_folder(rendering_folder)

    if route_dag:
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

    # Generate aggregated visualization
    if flatland_rendering:
        render_trainruns(
            data_folder=rendering_folder,
            experiment_id=experiment_results_analysis.experiment_id,
            malfunction=experiment_results_analysis.malfunction,
            rail_env=rail_env,
            trainruns=train_runs_online_unrestricted,
            convert_to_mpeg=convert_to_mpeg,
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
