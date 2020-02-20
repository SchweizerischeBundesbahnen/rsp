import os
import time
from typing import Optional

from flatland.action_plan.action_plan import ControllerFromTrainruns
from flatland.action_plan.action_plan_player import ControllerFromTrainrunsReplayerRenderCallback
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from pandas import DataFrame

from rsp.experiment_solvers.experiment_solver_utils import replay
from rsp.route_dag.analysis.route_dag_analysis import visualize_route_dag_constraints
from rsp.route_dag.route_dag import RouteDAGConstraintsDict
from rsp.utils.data_types import ExperimentMalfunction
from rsp.utils.data_types import ExperimentParameters
from rsp.utils.data_types import ExperimentResults
from rsp.utils.experiments import create_env_pair_for_experiment
from rsp.utils.file_utils import check_create_folder


def visualize_experiment(
        experiment: ExperimentParameters,
        data_frame: DataFrame,
        data_folder: str = None,
        flatland_rendering: bool = False,
        convert_to_mpeg: bool = True):
    """Render the experiment the DAGs and the FLATland png/mpeg in the
    analysis.

    Parameters
    ----------
    experiment: ExperimentParameters
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

    # find first row for this experiment (iloc[0]
    rows = data_frame.loc[data_frame['experiment_id'] == experiment.experiment_id]

    static_rail_env, malfunction_rail_env = create_env_pair_for_experiment(experiment)
    train_runs_input: TrainrunDict = rows['solution_full'].iloc[0]
    train_runs_full_after_malfunction: TrainrunDict = rows['solution_full_after_malfunction'].iloc[0]
    train_runs_delta_after_malfunction: TrainrunDict = rows['solution_delta_after_malfunction'].iloc[0]

    route_dag_constraints_full: RouteDAGConstraintsDict = rows['route_dag_constraints_full'].iloc[0]
    route_dag_constraints_rsp_full: RouteDAGConstraintsDict = rows['route_dag_constraints_full_after_malfunction'].iloc[
        0]
    route_dag_constraints_rsp_delta: RouteDAGConstraintsDict = \
        rows['route_dag_constraints_delta_after_malfunction'].iloc[0]
    malfunction: ExperimentMalfunction = rows['malfunction'].iloc[0]
    n_agents: int = rows['n_agents'].iloc[0]

    topo_dict = rows['topo_dict'].iloc[0]
    experiment_output_folder = f"{data_folder}/experiment_{experiment.experiment_id:04d}_analysis"
    check_create_folder(experiment_output_folder)

    for agent_id in route_dag_constraints_rsp_delta:
        visualize_route_dag_constraints(
            topo=topo_dict[agent_id],
            train_run_input=train_runs_input[agent_id],
            train_run_full_after_malfunction=train_runs_full_after_malfunction[agent_id],
            train_run_delta_after_malfunction=train_runs_delta_after_malfunction[agent_id],
            f=route_dag_constraints_full[agent_id],
            title=f"experiment {experiment.experiment_id}\nagent {agent_id}/{n_agents}\n{malfunction}",
            file_name=(os.path.join(experiment_output_folder,
                                    f"experiment_{experiment.experiment_id:04d}_agent_{agent_id}_route_graph_schedule.png")
                       if data_folder is not None else None)
        )
        visualize_route_dag_constraints(
            topo=topo_dict[agent_id],
            train_run_input=train_runs_input[agent_id],
            train_run_full_after_malfunction=train_runs_full_after_malfunction[agent_id],
            train_run_delta_after_malfunction=train_runs_delta_after_malfunction[agent_id],
            f=route_dag_constraints_rsp_delta[agent_id],
            title=f"experiment {experiment.experiment_id}\nagent {agent_id}/{n_agents}\n{malfunction}",
            file_name=(os.path.join(experiment_output_folder,
                                    f"experiment_{experiment.experiment_id:04d}_agent_{agent_id}_route_graph_rsp_delta.png")
                       if data_folder is not None else None)
        )
        visualize_route_dag_constraints(
            topo=topo_dict[agent_id],
            train_run_input=train_runs_input[agent_id],
            train_run_full_after_malfunction=train_runs_full_after_malfunction[agent_id],
            train_run_delta_after_malfunction=train_runs_delta_after_malfunction[agent_id],
            f=route_dag_constraints_rsp_full[agent_id],
            title=f"experiment {experiment.experiment_id}\nagent {agent_id}/{n_agents}\n{malfunction}",
            file_name=(os.path.join(experiment_output_folder,
                                    f"experiment_{experiment.experiment_id:04d}_agent_{agent_id}_route_graph_rsp_full.png")
                       if data_folder is not None else None)
        )

    _replay_flatland(data_folder=data_folder,
                     experiment=experiment,
                     flatland_rendering=flatland_rendering,
                     rail_env=malfunction_rail_env,
                     trainruns=train_runs_full_after_malfunction,
                     convert_to_mpeg=convert_to_mpeg)


def _replay_flatland(data_folder: str,
                     experiment: ExperimentResults,
                     rail_env: RailEnv,
                     trainruns: TrainrunDict,
                     convert_to_mpeg: bool = False,
                     flatland_rendering: bool = False):
    controller_from_train_runs = ControllerFromTrainruns(rail_env,
                                                         trainruns)
    image_output_directory = None
    if flatland_rendering:
        image_output_directory = os.path.join(data_folder, f"experiment_{experiment.experiment_id:04d}_analysis",
                                              f"experiment_{experiment.experiment_id}_rendering_output")
        check_create_folder(image_output_directory)
    replay(
        controller_from_train_runs=controller_from_train_runs,
        env=rail_env,
        stop_on_malfunction=False,
        solver_name='data_analysis',
        disable_verification_in_replay=False,
        rendering=flatland_rendering,
        image_output_directory=image_output_directory
    )
    if flatland_rendering and convert_to_mpeg:
        import ffmpeg
        (ffmpeg
         .input(f'{image_output_directory}/flatland_frame_0000_%04d_data_analysis.png', r='5', s='1920x1080')
         .output(f'{image_output_directory}/experiment_{experiment.experiment_id}_flatland_data_analysis.mp4', crf=15,
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
