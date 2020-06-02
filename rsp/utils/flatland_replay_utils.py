import os
import pprint
import warnings
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.route_dag.route_dag import MAGIC_DIRECTION_FOR_SOURCE_TARGET
from rsp.utils.data_types import ExperimentMalfunction
from rsp.utils.file_utils import check_create_folder

_pp = pprint.PrettyPrinter(indent=4)

# For each time step, agent's location: int key is the time step at which the train occupies the resource
# (agents occupy only one resource at a time, no release time)
TrainSchedule = Dict[int, Waypoint]
# TrainSchedule for all  trains: int key is the agent handle for which the schedule is returned
TrainScheduleDict = Dict[int, TrainSchedule]

# key: agent.handle, value: Waypoint (position and direction)
CurentFLATlandPositions = Dict[int, Waypoint]

# key: time_step, value: Waypoint (position and direction)
# N.B. position and direction are taken at before this step() is executed in FLATland!
AgentFLATlandPositions = Dict[int, Waypoint]

# key: time_step, value: dict[agent.handle]->Waypoint (position and direction)
FLATlandPositionsPerTimeStep = Dict[int, CurentFLATlandPositions]


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


def extract_trainrun_dict_from_flatland_positions(
        initial_directions: Dict[int, int],
        initial_positions: Dict[int, Tuple[int, int]],
        schedule: FLATlandPositionsPerTimeStep,
        targets: Dict[int, Tuple[int, int]]) -> TrainrunDict:
    """Convert FLATland positions to a TrainrunDict: for each agent, the cell
    entry events.
    Parameters
    ----------
    initial_directions
    initial_positions
    schedule
    targets
    Returns
    -------
    """
    trainrun_dict = {agent_id: [] for agent_id in initial_directions.keys()}
    for agent_id in trainrun_dict:
        curr_pos = None
        curr_dir = None
        for time_step in schedule:
            next_waypoint = schedule[time_step][agent_id]

            # are we running?
            if next_waypoint.position is not None:
                if next_waypoint.position != curr_pos:
                    # are we starting?
                    if curr_pos is None:
                        # sanity checks
                        assert time_step >= 1
                        assert next_waypoint.position == initial_positions[agent_id]
                        assert next_waypoint.direction == initial_directions[agent_id]

                        # when entering the grid in time_step t, the agent has a position only before t+1 -> entry event at t!
                        trainrun_dict[agent_id].append(
                            TrainrunWaypoint(
                                waypoint=Waypoint(
                                    position=next_waypoint.position,
                                    direction=MAGIC_DIRECTION_FOR_SOURCE_TARGET),
                                scheduled_at=time_step - 1))

                    # when the agent has a new position before time_step t, this corresponds to an entry at t!
                    trainrun_dict[agent_id].append(
                        TrainrunWaypoint(
                            waypoint=next_waypoint,
                            scheduled_at=time_step))

            # are we done?
            if next_waypoint.position is None and curr_pos is not None:
                # when the agent enters the target cell, it vanishes immediately in FLATland.
                # TODO in the rsp model, we will add a transition to the dummy time of time 1 + release time 1 -> is there a problem?
                #  (We might lose capacity in the rsp formulation)
                trainrun_dict[agent_id].append(
                    TrainrunWaypoint(
                        waypoint=Waypoint(
                            position=targets[agent_id],
                            direction=curr_dir),
                        scheduled_at=time_step))

                # sanity check: no jumping in the grid, no full check that the we respect the infrastructure layout!
                assert abs(curr_pos[0] - targets[agent_id][0]) + abs(curr_pos[1] - targets[agent_id][1]) == 1, \
                    f"agent {agent_id}: curr_pos={curr_pos} - target={targets[agent_id]}"
            curr_pos = next_waypoint.position
            curr_dir = next_waypoint.direction
    return trainrun_dict


def render_trainruns(rail_env: RailEnv,  # noqa:C901
                     trainruns: TrainrunDict,
                     title: str = None,
                     malfunction: Optional[ExperimentMalfunction] = None,
                     experiment_id: int = 0,
                     data_folder: Optional[str] = None,
                     convert_to_mpeg: bool = False,
                     mpeg_resolution: str = '640x360',
                     show: bool = False):
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
        image_output_directory = os.path.join(data_folder,
                                              f"experiment_{experiment_id:04d}_analysis",
                                              foldername)

        check_create_folder(image_output_directory)
    train_schedule_dict: TrainScheduleDict = convert_trainrun_dict_to_train_schedule_dict(trainrun_dict=trainruns)
    max_episode_steps = np.max([time_step for agent_id, train_schedule in train_schedule_dict.items() for time_step in train_schedule.keys()])
    renderer = init_renderer_for_env(rail_env)

    for time_step in range(max_episode_steps):
        # TODO malfunction
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
        # TODO SIM-516  simplify: test_id, solver_name???
        render_env(renderer,
                   test_id=0,
                   solver_name='data_analysis',
                   i_step=time_step,
                   show=show,
                   image_output_directory=image_output_directory)

    try:
        cleanup_renderer_for_env(renderer)
    except AttributeError as e:
        # TODO why does this happen?
        warnings.warn(str(e))
    if convert_to_mpeg:
        import ffmpeg
        (ffmpeg
         .input(f'{image_output_directory}/flatland_frame_0000_%04d_data_analysis.png', r='5', s=mpeg_resolution)
         .output(
            f'{image_output_directory}/experiment_{experiment_id}_flatland_data_analysis.mp4',
            crf=15,
            pix_fmt='yuv420p', vcodec='libx264')
         .overwrite_output()
         .run()
         )


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def init_renderer_for_env(env: RailEnv):
    from flatland.utils.rendertools import AgentRenderVariant
    from flatland.utils.rendertools import RenderTool
    return RenderTool(env, gl="PILSVG",
                      agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                      show_debug=True,
                      clear_debug_text=True,
                      screen_height=1000,
                      screen_width=1000)


def render_env(renderer,
               test_id: int,
               solver_name, i_step: int,
               show=True,
               image_output_directory: Optional[str] = './rendering_output'):
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
        renderer.gl.save_image(os.path.join(image_output_directory,
                                            "flatland_frame_{:04d}_{:04d}_{}.png".format(test_id,
                                                                                         i_step,
                                                                                         solver_name)))


def cleanup_renderer_for_env(renderer):
    from flatland.utils.rendertools import RenderTool
    renderer: RenderTool = renderer
    # close renderer window
    try:
        renderer.close_window()
    except AttributeError as e:
        # TODO why does this happen?
        warnings.warn(str(e))
