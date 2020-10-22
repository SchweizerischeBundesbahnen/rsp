import os
import pprint
import warnings
from pathlib import Path
from typing import Dict
from typing import Optional

import numpy as np
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import Waypoint
from rsp.step_02_setup.data_types import ExperimentMalfunction
from rsp.step_03_run.experiment_results_analysis import ExperimentResultsAnalysis
from rsp.step_03_run.experiments import create_env_from_experiment_parameters
from rsp.step_03_run.experiments import EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME
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
    print(image_output_directory)
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

        print(image_output_directory)
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


def render_flatland_env(  # noqa
    data_folder: str, experiment_data: ExperimentResultsAnalysis, experiment_id: int, render_schedule: bool = True, render_reschedule: bool = True
):
    """
    Method to render the environment for visual inspection
    Parameters
    ----------
    data_folder: str
        Folder name to store and load images from
    experiment_data: ExperimentResultsAnalysis
        experiment data used for visualization
    experiment_id: int
        ID of experiment we like to visualize
    render_reschedule
    render_schedule

    Returns
    -------
    File paths to generated videos to render in the notebook
    """

    # Generate environment for rendering
    rail_env = create_env_from_experiment_parameters(experiment_data.experiment_parameters.infra_parameters)

    # Generate aggregated visualization
    output_folder = f"{data_folder}/{EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME}/"
    check_create_folder(output_folder)
    video_src_schedule = None
    video_src_reschedule = None

    # Generate the Schedule video
    if render_schedule:
        # Import the generated video
        title = "Schedule"
        video_src_schedule = os.path.join(
            output_folder,
            f"experiment_{experiment_data.experiment_id:04d}_analysis",
            f"experiment_{experiment_data.experiment_id}_rendering_output_{title}/",
            f"experiment_{experiment_id}_flatland_data_analysis.mp4",
        )

        # Only render if file is not yet created
        if not os.path.exists(video_src_schedule):
            render_trainruns(
                data_folder=output_folder,
                experiment_id=experiment_data.experiment_id,
                title=title,
                rail_env=rail_env,
                trainruns=experiment_data.solution_schedule,
                convert_to_mpeg=True,
            )

    # Generate the Reschedule video
    if render_reschedule:
        # Import the generated video
        title = "Reschedule"
        video_src_reschedule = os.path.join(
            output_folder,
            f"experiment_{experiment_data.experiment_id:04d}_analysis",
            f"experiment_{experiment_data.experiment_id}_rendering_output_{title}/",
            f"experiment_{experiment_id}_flatland_data_analysis.mp4",
        )
        # Only render if file is not yet created
        if not os.path.exists(video_src_reschedule):
            render_trainruns(
                data_folder=output_folder,
                experiment_id=experiment_data.experiment_id,
                malfunction=experiment_data.malfunction,
                title=title,
                rail_env=rail_env,
                trainruns=experiment_data.solution_online_unrestricted,
                convert_to_mpeg=True,
            )

    return Path(video_src_schedule) if render_schedule else None, Path(video_src_reschedule) if render_reschedule else None
