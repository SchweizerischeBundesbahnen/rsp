import os
import time
from typing import Dict
from typing import Optional

import networkx as nx
from flatland.action_plan.action_plan import ControllerFromTrainruns
from flatland.action_plan.action_plan_player import ControllerFromTrainrunsReplayerRenderCallback
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_trainrun_data_structures import Trainrun
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from pandas import DataFrame

from rsp.experiment_solvers.experiment_solver_utils import replay
from rsp.route_dag.analysis.route_dag_analysis import visualize_route_dag_constraints
from rsp.route_dag.route_dag import get_paths_for_route_dag_constraints
from rsp.route_dag.route_dag import get_paths_in_route_dag
from rsp.route_dag.route_dag import RouteDAGConstraints
from rsp.route_dag.route_dag import ScheduleProblemDescription
from rsp.utils.data_types import ExperimentMalfunction
from rsp.utils.data_types import ExperimentParameters
from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.data_types import TrainSchedule
from rsp.utils.data_types import TrainScheduleDict
from rsp.utils.experiments import create_env_pair_for_experiment
from rsp.utils.file_utils import check_create_folder


def convert_trainrundict_to_entering_positions_for_all_timesteps(trainrun_dict: TrainrunDict) -> TrainScheduleDict:
    """
    Converts a `TrainrunDict` (only entry times into a new position) into a dict with the waypoint for each agent and agent time step.
    The positions are the new positions the agents ent
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
        for trainrun_waypoint in trainrun:
            while time_step < trainrun_waypoint.scheduled_at:
                train_schedule[time_step] = current_position
                time_step += 1
            current_position = trainrun_waypoint.waypoint
            train_schedule[time_step] = current_position
    return train_schedule_dict


def convert_trainrundict_to_positions_after_flatland_timestep(trainrun_dict: TrainrunDict) -> TrainScheduleDict:
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


def visualize_experiment(
        experiment_parameters: ExperimentParameters,
        experiment_results_analysis: ExperimentResultsAnalysis,
        data_frame: DataFrame,
        data_folder: str = None,
        flatland_rendering: bool = False,
        convert_to_mpeg: bool = True):
    """Render the experiment the DAGs and the FLATland png/mpeg in the
    analysis.

    Parameters
    ----------
    experiment_parameters: ExperimentParameters
        experiment parameters for all trials
    data_frame: DataFrame
        Pandas data frame with one ore more trials of this experiment.
    data_folder
        Folder to store FLATland pngs and mpeg to
    flatland_rendering
        Flatland rendering?
    convert_to_mpeg
        Converts the rendering to mpeg
    """
    # TODO remove iloc[0] stuff

    # find first row for this experiment (iloc[0]
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

    experiment_output_folder = f"{data_folder}/experiment_{experiment_parameters.experiment_id:04d}_analysis"
    check_create_folder(experiment_output_folder)

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
            file_name=(os.path.join(experiment_output_folder,
                                    f"experiment_{experiment_parameters.experiment_id:04d}_agent_{agent_id}_route_graph_schedule.png")
                       if data_folder is not None else None)
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
            file_name=(os.path.join(experiment_output_folder,
                                    f"experiment_{experiment_parameters.experiment_id:04d}_agent_{agent_id}_route_graph_rsp_delta.png")
                       if data_folder is not None else None)
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
            file_name=(os.path.join(experiment_output_folder,
                                    f"experiment_{experiment_parameters.experiment_id:04d}_agent_{agent_id}_route_graph_rsp_full.png")
                       if data_folder is not None else None)
        )

    _replay_flatland(data_folder=data_folder,
                     experiment_results_analysis=experiment_results_analysis,
                     flatland_rendering=flatland_rendering,
                     rail_env=malfunction_rail_env,
                     trainruns=train_runs_full_after_malfunction,
                     convert_to_mpeg=convert_to_mpeg)


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


def _replay_flatland(data_folder: str,
                     experiment_results_analysis: ExperimentResultsAnalysis,
                     rail_env: RailEnv,
                     trainruns: TrainrunDict,
                     convert_to_mpeg: bool = False,
                     flatland_rendering: bool = False):
    controller_from_train_runs = ControllerFromTrainruns(rail_env,
                                                         trainruns)
    image_output_directory = None
    if flatland_rendering:
        image_output_directory = os.path.join(data_folder,
                                              f"experiment_{experiment_results_analysis.experiment_id:04d}_analysis",
                                              f"experiment_{experiment_results_analysis.experiment_id}_rendering_output")
        check_create_folder(image_output_directory)
    replay(
        controller_from_train_runs=controller_from_train_runs,
        env=rail_env,
        stop_on_malfunction=False,
        solver_name='data_analysis',
        disable_verification_in_replay=False,
        rendering=flatland_rendering,
        image_output_directory=image_output_directory,
        debug=True,
        expected_malfunction=experiment_results_analysis.malfunction,
        expected_positions=convert_trainrundict_to_positions_after_flatland_timestep(trainruns)
    )
    if flatland_rendering and convert_to_mpeg:
        import ffmpeg
        (ffmpeg
         .input(f'{image_output_directory}/flatland_frame_0000_%04d_data_analysis.png', r='5', s='1920x1080')
         .output(
            f'{image_output_directory}/experiment_{experiment_results_analysis.experiment_id}_flatland_data_analysis.mp4',
            crf=15,
            pix_fmt='yuv420p', vcodec='libx264')
         .overwrite_output()
         .run()
         )


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def make_render_call_back_for_replay(env: RailEnv,
                                     rendering: bool = False) -> ControllerFromTrainrunsReplayerRenderCallback:
    if rendering:
        from flatland.utils.rendertools import AgentRenderVariant
        from flatland.utils.rendertools import RenderTool
        renderer = RenderTool(env, gl="PILSVG",
                              agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                              show_debug=True,
                              clear_debug_text=True,
                              screen_height=1000,
                              screen_width=1000)

    def render(*argv):
        if rendering:
            renderer.render_env(show=True, show_observations=False, show_predictions=False)
            time.sleep(2)

    return render


def init_renderer_for_env(env: RailEnv, rendering: bool = False):
    if rendering:
        from flatland.utils.rendertools import AgentRenderVariant
        from flatland.utils.rendertools import RenderTool
        return RenderTool(env, gl="PILSVG",
                          agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                          show_debug=True,
                          clear_debug_text=True,
                          screen_height=1000,
                          screen_width=1000)


def render_env(renderer, test_id: int, solver_name, i_step: int,
               image_output_directory: Optional[str] = './rendering_output'):
    if renderer is not None:
        from flatland.utils.rendertools import RenderTool
        renderer: RenderTool = renderer
        renderer.render_env(show=True, show_observations=False, show_predictions=False)
        if image_output_directory is not None:
            if not os.path.exists(image_output_directory):
                os.makedirs(image_output_directory)
            renderer.gl.save_image(os.path.join(image_output_directory,
                                                "flatland_frame_{:04d}_{:04d}_{}.png".format(test_id,
                                                                                             i_step,
                                                                                             solver_name)))


def cleanup_renderer_for_env(renderer):
    if renderer:
        from flatland.utils.rendertools import RenderTool
        renderer: RenderTool = renderer
        # close renderer window
        renderer.close_window()
