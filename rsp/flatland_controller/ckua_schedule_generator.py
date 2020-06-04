import pprint
from time import perf_counter
from typing import List

import numpy as np
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint
from libs.cell_graph_agent import AgentWayStep

from rsp.experiment_solvers.trainrun_utils import verify_trainrun_dict_simple
from rsp.flatland_controller.ckua_flatland_controller import CkUaController
from rsp.logger import rsp_logger
from rsp.route_dag.route_dag import MAGIC_DIRECTION_FOR_SOURCE_TARGET
from rsp.utils.flatland_replay_utils import extract_trainrun_dict_from_flatland_positions
from rsp.utils.flatland_replay_utils import FLATlandPositionsPerTimeStep
from rsp.utils.flatland_replay_utils import render_trainruns

_pp = pprint.PrettyPrinter(indent=4)


def ckua_generate_schedule(  # noqa:C901
        env: RailEnv,
        random_seed: int,
        rendering: bool = False,
        show: bool = False,
        max_steps: int = np.inf) -> [TrainrunDict, int]:
    # setup the env
    observation, info = env.reset(False, False, False, random_seed=random_seed)

    # Setup the controller (solver)
    flatland_controller = CkUaController()
    flatland_controller.setup(env)

    if rendering:
        from flatland.utils.rendertools import AgentRenderVariant
        from flatland.utils.rendertools import RenderTool
        env_renderer = RenderTool(env=env,
                                  gl="PILSVG",
                                  agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS,
                                  show_debug=True,
                                  screen_height=1600,
                                  screen_width=1600)

        env_renderer.render_env(show=show, show_observations=False, show_predictions=False)

    if max_steps == np.inf:
        max_steps = env._max_episode_steps

    flatland_positions: FLATlandPositionsPerTimeStep = {}

    start_time = perf_counter()
    env.reset(False, False, False, random_seed=random_seed)
    steps = 0
    flatland_controller = CkUaController()
    flatland_controller.setup(env)
    actions_per_step = {}

    while steps < max_steps:
        if steps == 0:
            flatland_positions[0] = {}
            for agent in env.agents:
                flatland_positions[steps][agent.handle] = Waypoint(position=agent.position,
                                                                   direction=agent.direction)

        action_dict = flatland_controller.controller(env, observation, info, env.get_num_agents())
        actions_per_step[steps] = action_dict

        _, _, done, _ = env.step(action_dict)

        if rendering:
            # Environment step which returns the observations for all agents, their corresponding
            # reward and whether their are done
            env_renderer.render_env(show=show, show_observations=False, show_predictions=False)
            print(steps)

        steps += 1
        flatland_positions[steps] = {}
        for agent in env.agents:
            flatland_positions[steps][agent.handle] = Waypoint(position=agent.position,
                                                               direction=agent.direction)

        if done['__all__']:

            print(f"_max_episode_steps={env._max_episode_steps}, _elapsed_steps={env._elapsed_steps}" + str(
                (env._max_episode_steps is not None) and (env._elapsed_steps >= env._max_episode_steps)))

            if sum([1 for agent in env.agents if (agent.status == RailAgentStatus.READY_TO_DEPART or agent.status == RailAgentStatus.ACTIVE)]) > 0:
                not_done_agents = [agent for agent in env.agents if (agent.status == RailAgentStatus.READY_TO_DEPART or agent.status == RailAgentStatus.ACTIVE)]
                rsp_logger.warn(
                    f"not all agents done: {not_done_agents}")
            print(f"done after {steps}")
            if not ((env._max_episode_steps is not None) and (
                    env._elapsed_steps >= env._max_episode_steps)):
                break
    print(f"took {perf_counter() - start_time:5.2f}")
    for _, actions in actions_per_step.items():
        for agent_id, action_dict in actions.items():
            actions[agent_id] = int(action_dict)

    has_selected_way = [len(flatland_controller.dispatcher.controllers[agent.handle].selected_way) > 0 for agent in env.agents]
    assert np.alltrue(has_selected_way)
    if rendering:
        # Environment step which returns the observations for all agents, their corresponding
        # reward and whether their are done
        env_renderer.render_env(show=True, show_observations=False, show_predictions=False)
        env_renderer.gl.close_window()

    # extract trainrun
    elapsed_time = perf_counter() - start_time

    initial_positions = {agent.handle: agent.initial_position for agent in env.agents}
    initial_directions = {agent.handle: agent.initial_direction for agent in env.agents}
    targets = {agent.handle: agent.target for agent in env.agents}
    minimum_runningtime_dict = {agent.handle: int(1 // env.agents[agent.handle].speed_data['speed']) for agent in env.agents}
    trainrun_dict_from_flatland_positions = extract_trainrun_dict_from_flatland_positions(
        initial_directions=initial_directions,
        initial_positions=initial_positions,
        schedule=flatland_positions,
        targets=targets)

    if rendering:
        env.reset(False, False, False, random_seed=random_seed)
        render_trainruns(
            rail_env=env,
            trainruns=trainrun_dict_from_flatland_positions,
            data_folder="blabla22",
            convert_to_mpeg=True
        )

    trainrun_dict_from_selected_ways = extract_trainrun_dict_from_selected_ways(flatland_controller)

    # TODO SIM-494 trainrun_dict_from_flatland_positions has cycles:
    # TODO SIM-495 why are the two not equal?
    verify_trainrun_dict_simple(trainrun_dict=trainrun_dict_from_flatland_positions,
                                initial_positions=initial_positions,
                                initial_directions=initial_directions,
                                targets=targets,
                                minimum_runningtime_dict=minimum_runningtime_dict)
    verify_trainrun_dict_simple(trainrun_dict=trainrun_dict_from_selected_ways,
                                initial_positions=initial_positions,
                                initial_directions=initial_directions,
                                targets=targets,
                                minimum_runningtime_dict=minimum_runningtime_dict)

    print("verification done")
    return trainrun_dict_from_selected_ways, elapsed_time


def extract_trainrun_dict_from_selected_ways(flatland_controller) -> TrainrunDict:
    trainrun_dict_from_selected_ways = {}
    for agent_id, selected_way in flatland_controller.dispatcher.controllers.items():
        cell_graph_agent = flatland_controller.dispatcher.controllers[agent_id]
        cell_graph = cell_graph_agent.graph
        selected_way: List[AgentWayStep] = selected_way
        trainrun_from_selected_way_tail = [
            TrainrunWaypoint(waypoint=Waypoint(
                position=cell_graph.position_from_vertexid(agent_way_step.vertex_idx),
                direction=agent_way_step.direction),
                scheduled_at=agent_way_step.arrival_time)
            for agent_way_step in reversed(selected_way)
        ]

        trainrun_from_selected_way_tail = trainrun_from_selected_way_tail[1:]
        first_agent_way_step: AgentWayStep = selected_way[-1]
        trainrun_from_selected_way_tweaked_head = [
            TrainrunWaypoint(
                waypoint=Waypoint(
                    position=cell_graph.position_from_vertexid(first_agent_way_step.vertex_idx),
                    direction=MAGIC_DIRECTION_FOR_SOURCE_TARGET),
                scheduled_at=first_agent_way_step.arrival_time),
            TrainrunWaypoint(
                waypoint=Waypoint(
                    position=cell_graph.position_from_vertexid(first_agent_way_step.vertex_idx),
                    direction=first_agent_way_step.direction),
                scheduled_at=first_agent_way_step.arrival_time + 1),

        ]
        trainrun_from_selected_way = trainrun_from_selected_way_tweaked_head + trainrun_from_selected_way_tail

        trainrun_dict_from_selected_ways[agent_id] = trainrun_from_selected_way
    return trainrun_dict_from_selected_ways
