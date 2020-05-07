import os
import pprint
import time
import warnings
from typing import Optional

import numpy as np
from flatland.action_plan.action_plan import ActionPlanElement
from flatland.action_plan.action_plan import ControllerFromTrainruns
from flatland.action_plan.action_plan_player import ControllerFromTrainrunsReplayerRenderCallback
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env import RailEnvActions
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint

from rsp.utils.data_types import ExperimentMalfunction
from rsp.utils.data_types import TrainSchedule
from rsp.utils.data_types import TrainScheduleDict
from rsp.utils.file_utils import check_create_folder

_pp = pprint.PrettyPrinter(indent=4)


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


def replay_and_verify_trainruns(rail_env: RailEnv,
                                trainruns: TrainrunDict,
                                title: str = None,
                                expected_malfunction: Optional[ExperimentMalfunction] = None,
                                rendering: bool = False,
                                experiment_id: str = '0',
                                data_folder: Optional[str] = None,
                                convert_to_mpeg: bool = False,
                                mpeg_resolution: str = '640x360',
                                debug: bool = False) -> int:
    """

    Parameters
    ----------
    rail_env:
        FLATland `RailEnv`. Be aware it is `reset` correctly.
    trainruns: TrainrunDict
        a train run per agent
    title: str
        Title to set for folder
    expected_malfunction: Optional[ExperimentMalfunction]
        if a malfunction is supposed to happen, pass it here.
    rendering: bool
        render the replay?
    data_folder: Optional[str]
        if `rendering=True`, save the visualization to a png per time step
    experiment_id:
        if `rendering=True` and `data_folder` is given, use the pngs/mpegs to files with this identification
    convert_to_mpeg: bool
        if `rendering=True` and `data_folder` is defined, convert the generated pngs to an mpeg
    mpeg_resolution:
        resolution to use if mpeg is generated
    debug: bool
        show debug output

    Returns
    -------

    """
    controller_from_train_runs = create_controller_from_trainruns_and_malfunction(
        env=rail_env,
        trainrun_dict=trainruns,
        expected_malfunction=expected_malfunction,
        debug=debug)
    image_output_directory = None

    if data_folder:
        foldername = f"experiment_{experiment_id}_rendering_output"

        if title is not None:
            foldername = f"experiment_{experiment_id}_rendering_output_{title}"
        image_output_directory = os.path.join(data_folder,
                                              f"experiment_{experiment_id:04d}_analysis",
                                              foldername)

        check_create_folder(image_output_directory)
    total_reward = replay(
        controller_from_train_runs=controller_from_train_runs,
        env=rail_env,
        stop_on_malfunction=False,
        solver_name='data_analysis',
        rendering=rendering,
        image_output_directory=image_output_directory,
        debug=debug,
        expected_flatland_positions=convert_trainrundict_to_positions_after_flatland_timestep(trainruns)
    )
    if rendering and convert_to_mpeg:
        print("curdir=" + str(os.curdir))
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
    return total_reward


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
    if renderer is not None:
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
    if renderer:
        from flatland.utils.rendertools import RenderTool
        renderer: RenderTool = renderer
        # close renderer window
        try:
            renderer.close_window()
        except AttributeError as e:
            # TODO why does this happen?
            warnings.warn(str(e))


def replay(env: RailEnv,  # noqa: C901
           solver_name: str,
           controller_from_train_runs: ControllerFromTrainruns,
           expected_flatland_positions: Optional[TrainScheduleDict] = None,
           rendering: bool = False,
           show: bool = False,
           debug: bool = False,
           loop_index: int = 0,
           stop_on_malfunction: bool = False,
           image_output_directory: Optional[str] = None) -> Optional[ExperimentMalfunction]:
    """Replay the solution an check whether the actions againts FLATland env
    can be performed as against. Verifies that the solution is indeed a
    solution in the FLATland sense.

    Parameters
    ----------
    solver_name: bool
        The name of the solver for debugging purposes.
    env
        The env to run the verification with
    rendering_call_back
        Called every step in replay
    debug
        Display debugging information
    loop_index
        Used for display, should identify the problem instance
    expected_malfunction
        If provided and disable_verification_in_replay == False, it is checked that the malfunction happens as expected.
    stop_on_malfunction
        If true, stops and returns upon entering into malfunction; in this case returns the malfunction
    controller_from_train_runs: ActionPlanDict
        The action plan to replay
    rendering: bool
        render in replay?
    show: bool
        if rendering=True, show or only save to files?
    image_output_directory: str
        write png files to this directory
    expected_flatland_positions:
        if given, verify that the agent's position match the expectations. At index t, we expect the position after step(t,t+1).

    Returns
    -------
    Optional[Malfunction]
        The malfunction in `stop_on_malfunction` mode, the total reward else.
    """
    total_reward = 0
    time_step = 0
    if rendering:
        from rsp.utils.flatland_replay_utils import init_renderer_for_env
        renderer = init_renderer_for_env(env, rendering)
    while not env.dones['__all__'] and time_step <= env._max_episode_steps:
        fail = False
        if fail:
            raise Exception("Unexpected state. See above for !!=unexpected position, MM=unexpected malfuntion")

        actions = controller_from_train_runs.act(time_step)

        if debug:
            print(f"env._elapsed_steps={env._elapsed_steps}")
            print("actions [{}]->[{}] actions={}".format(time_step, time_step + 1, actions))

        obs, all_rewards, done, _ = env.step(actions)
        total_reward += sum(np.array(list(all_rewards.values())))
        if debug:
            for agent in env.agents:
                print(f"agent {agent.handle} "
                      f"speed_data={agent.speed_data} "
                      f"position={agent.position} "
                      f"direction={agent.direction} "
                      f"malfunction_data={agent.malfunction_data} "
                      f"target={agent.target}")
        if expected_flatland_positions is not None:
            for agent_id, train_schedule in expected_flatland_positions.items():
                actual_position = env.agents[agent_id].position
                if train_schedule.get(time_step, None) is None:
                    assert actual_position is None, \
                        f"[{time_step}] agent {agent_id} expected to have left already/note have departed, actual position={actual_position}"
                else:
                    waypoint = train_schedule[time_step]
                    expected_position = waypoint.position
                    if actual_position != expected_position:
                        print(f"expected trainrun for {agent_id}")
                        print(_pp.pformat(controller_from_train_runs.trainrun_dict[agent_id]))
                        print(f"expected positions for agent {agent_id}")
                        print(_pp.pformat(expected_flatland_positions[agent_id]))
                        print(f"action plan for agent {agent_id}")
                        print(_pp.pformat(controller_from_train_runs.action_plan[agent_id]))
                    assert actual_position == expected_position, \
                        f"[{time_step}] agent {agent_id} expected position={expected_position}, actual position={actual_position}"

        if stop_on_malfunction:
            for agent in env.agents:
                if agent.malfunction_data['malfunction'] > 0:
                    # malfunction duration is already decreased by one in this step(), therefore add +1!
                    return ExperimentMalfunction(time_step, agent.handle, agent.malfunction_data['malfunction'] + 1)

        if rendering:
            from rsp.utils.flatland_replay_utils import render_env
            render_env(renderer,
                       test_id=loop_index,
                       solver_name=solver_name,
                       i_step=time_step,
                       show=show,
                       image_output_directory=image_output_directory)

        # if all agents have reached their goals, break
        if done['__all__']:
            break
        time_step += 1
    if rendering:
        from rsp.utils.flatland_replay_utils import cleanup_renderer_for_env
        try:
            cleanup_renderer_for_env(renderer)
        except AttributeError as e:
            # TODO why does this happen?
            warnings.warn(str(e))
    if stop_on_malfunction:
        return None
    else:
        return total_reward


def create_controller_from_trainruns_and_malfunction(trainrun_dict: TrainrunDict,
                                                     env: RailEnv,
                                                     expected_malfunction: Optional[ExperimentMalfunction] = None,
                                                     debug: bool = False) -> ControllerFromTrainruns:
    """Creates an action plan (fix schedule, when an action has which positions
    and when it has to take which actions).

    Parameters
    ----------
    trainrun_dict
    env
    expected_malfunction
    debug

    Returns
    -------
    ControllerFromTrainruns
    """

    # tweak 1: replace dummy source by real source
    # N.B: the dummy target is already removed coming from the ASP wrappers!
    solution_trainruns_tweaked = {}
    for agent_id, trainrun in trainrun_dict.items():
        solution_trainruns_tweaked[agent_id] = \
            [TrainrunWaypoint(waypoint=trainrun[1].waypoint, scheduled_at=trainrun[0].scheduled_at)] + trainrun[2:]

    controller_from_train_runs = ControllerFromTrainruns(env, solution_trainruns_tweaked)

    # tweak 2: introduce malfunction
    # ControllerFromTrainruns does not know about malfunctions!
    # with malfunction, instead of travel_time before the first point after the malfunction ends,
    # take that action at that time - malfunction_duration - travel_time!
    # TODO should we move this to FLATland?
    if expected_malfunction is not None:
        malfunction_agend_id = expected_malfunction.agent_id
        expected_malfunction_end = expected_malfunction.time_step + expected_malfunction.malfunction_duration
        minimum_travel_time = int(np.ceil(1 / env.agents[malfunction_agend_id].speed_data['speed']))

        # determine the next section entry after the malfunction
        trainrun_waypoint_after_malfunction: TrainrunWaypoint = next(
            trainrun_waypoint
            for trainrun_waypoint in controller_from_train_runs.trainrun_dict[malfunction_agend_id]
            if (trainrun_waypoint.scheduled_at >= (expected_malfunction_end))
        )

        # in the action plan created by FLATland from the re-scheduling trainruns,
        # we may have a STOP before the next movement after the malfunction
        time_step_of_action_that_could_be_interrupted_by_malfunction = trainrun_waypoint_after_malfunction.scheduled_at - minimum_travel_time
        time_step_to_take_action_instead = (trainrun_waypoint_after_malfunction.scheduled_at
                                            - expected_malfunction.malfunction_duration
                                            - minimum_travel_time)

        agent_action_dict_to_tweak = {
            action_plan_element.scheduled_at: action_plan_element.action
            for action_plan_element in controller_from_train_runs.action_plan[malfunction_agend_id]
        }

        # a STOP action at the beginning of the malfunction is ignored by FLATland (actions during malfunction, even at start, are ignore)
        if (agent_action_dict_to_tweak.get(expected_malfunction.time_step, None) == RailEnvActions.STOP_MOVING and
                expected_malfunction_end not in agent_action_dict_to_tweak):
            if debug:
                print(f"tweaking agent {malfunction_agend_id} for {expected_malfunction} "
                      "which stops at malfunction beginning")
                print(f" -> {agent_action_dict_to_tweak[expected_malfunction_end]} instead of "
                      f"{RailEnvActions.STOP_MOVING} at {expected_malfunction_end}"
                      )
            agent_action_dict_to_tweak[expected_malfunction_end] = RailEnvActions.STOP_MOVING
        else:
            agent_action_dict_to_tweak[time_step_to_take_action_instead] = agent_action_dict_to_tweak[
                time_step_of_action_that_could_be_interrupted_by_malfunction]
            if debug:
                print(
                    f"tweaking agent {malfunction_agend_id} for {expected_malfunction}: "
                    f"action at {time_step_to_take_action_instead} "
                    f" ->  {agent_action_dict_to_tweak[time_step_of_action_that_could_be_interrupted_by_malfunction]}")
        controller_from_train_runs.action_plan[malfunction_agend_id] = [
            ActionPlanElement(scheduled_at=scheduled_at, action=action)
            for scheduled_at, action in sorted(agent_action_dict_to_tweak.items(), key=lambda item: item[0])
        ]

    return controller_from_train_runs
