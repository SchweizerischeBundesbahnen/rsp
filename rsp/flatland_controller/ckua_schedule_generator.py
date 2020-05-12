import pprint
from time import perf_counter

import numpy as np
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint
from libs.cell_graph_agent import AgentWayStep

from rsp.experiment_solvers.trainrun_utils import verify_trainrun_dict
from rsp.experiment_solvers.trainrun_utils import verify_trainrun_dict_simple
from rsp.flatland_controller.ckua_flatland_controller import CkUaController
from rsp.flatland_integration.flatland_conversion import extract_trainrun_dict_from_flatland_positions
from rsp.flatland_integration.flatland_conversion import FLATlandPositionsPerTimeStep
from rsp.logger import rsp_logger
from rsp.route_dag.route_dag import MAGIC_DIRECTION_FOR_SOURCE_TARGET

_pp = pprint.PrettyPrinter(indent=4)


# TODO SIM-434 remove noqa
# flake8: noqa

def ckua_generate_schedule(  # noqa:C901
        env: RailEnv,
        random_seed: int,
        rendering: bool = False,
        show: bool = False,
        max_steps: int = np.inf) -> [TrainrunDict, int]:
    # setup the env
    observation, info = env.reset(False, False, False, random_seed=random_seed)
    rendering=True
    show=True

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
        # input()

    if max_steps == np.inf:
        max_steps = env._max_episode_steps

    schedule: FLATlandPositionsPerTimeStep = {}

    # TODO SIM-443 extract without stepping
    selected_ways = {}
    if False:
        print("(A) without FLATland interaction")
        start_time = perf_counter()
        flatland_controller.setup(env)
        steps = 0
        # flatland_controller.controller(env, observation, info, env.get_num_agents())
        while steps < max_steps:
            # TODO SIM-443 call only those agent controllers that need to be called
            action_dict = flatland_controller.dispatcher.step(steps)
            # TODO can we get rid of passing this to env?
            _, _, done, _ = env.step(action_dict)
            # for agent in env.agents:
            #     if len(flatland_controller.dispatcher.controllers[agent.handle].selected_way) == 0:
            #         flatland_controller.dispatcher.controllers[agent.handle].act(agent, steps)
            for agent in env.agents:
                if agent.handle not in selected_ways and len(flatland_controller.dispatcher.controllers[agent.handle].selected_way) > 0:
                    selected_ways[agent.handle] = [el for el in flatland_controller.dispatcher.controllers[agent.handle].selected_way]

            has_selected_way = [len(flatland_controller.dispatcher.controllers[agent.handle].selected_way) > 0 for agent in env.agents]
            # for agent in env.agents:
            #     if len(flatland_controller.dispatcher.controllers[agent.handle].selected_way) > 0:
            #         # TODO hacky, hacky
            #         agent.status = RailAgentStatus.DONE
            if False:
                print(f"{steps} #has_selected_way {sum(has_selected_way)}/{len(env.agents)}")
            ready_to_depart = sum([1 for agent in env.agents if agent.status == RailAgentStatus.READY_TO_DEPART])
            # assert sum(has_selected_way) == (len(env.agents)-ready_to_depart), f"has_selected_way={sum(has_selected_way)}, ready_to_depart={ready_to_depart}, agents={len(env.agents)}"
            steps += 1
            if False:
                print(f"[{steps}] " +
                      " / ".join(
                          [f"{str(RailAgentStatus(status))}={sum([1 for agent in env.agents if agent.status == status])} " for status in RailAgentStatus]))
            if np.alltrue(has_selected_way):
                break
            # if done['__all__']:
            #     break

        print(f"done after {steps}")

        # selected_ways = {agent.handle: flatland_controller.dispatcher.controllers[agent.handle].selected_way for agent in env.agents}
        #
        # # gather all positions for all agents
        # positions = {
        #     agent.handle: _extract_agent_positions_from_selected_ckua_way(
        #         selected_way=flatland_controller.dispatcher.controllers[agent.handle].selected_way,
        #         cell_graph=flatland_controller.dispatcher.graph)
        #     for agent in env.agents
        # }

        # change columns: first indexed by time_step and then by agent_id
        # time_steps = [list(positions[agent.handle].keys()) for agent in env.agents]
        # maxes = [max(list(positions[agent.handle].keys())) for agent in env.agents]
        # max_steps = max(maxes)
        # schedule = {i: {agent.handle: {}
        #                 for agent in env.agents}
        #             for i in range(max_steps + 1)}
        # for agent_id, positions in positions.items():
        #     for time_step, waypoint in positions:
        #         schedule[time_step][agent_id] = waypoint

        print(f"took {perf_counter() - start_time:5.2f}")
    print("(B) with FLATland interaction")

    start_time = perf_counter()
    env.reset(False, False, False, random_seed=random_seed)
    steps = 0
    flatland_controller = CkUaController()
    flatland_controller.setup(env)
    actions_per_step = {}

    while steps < max_steps:
        # print(steps)
        if steps == 0:
            schedule[0] = {}
            for agent in env.agents:
                schedule[steps][agent.handle] = Waypoint(position=agent.position,
                                                         direction=agent.direction)
            if False:
                print(f"[{steps}] {schedule[steps]}")

        action_dict = flatland_controller.controller(env, observation, info, env.get_num_agents())
        actions_per_step[steps] = action_dict

        # , agent.speed_data['position_fraction'])
        if False:
            print(f"[{steps + 1}] {schedule[steps + 1]}")
        _, _, done, _ = env.step(action_dict)
        # if steps > 200 and steps < 240:
        #     print(f"before step {steps + 1}")
        #     for agent in env.agents:
        #         print(f" [{steps + 1}] agent [{agent.handle}] position={agent.position} direction={agent.direction} speed_data={agent.speed_data}")
        # for agent in env.agents:
        #     if agent.handle not in selected_ways and len(flatland_controller.dispatcher.controllers[agent.handle].selected_way) > 0:
        #         selected_ways[agent.handle] = [el for el in flatland_controller.dispatcher.controllers[agent.handle].selected_way]
        if False:
            ready_to_depart_ = [agent.status for agent in env.agents if agent.status == RailAgentStatus.READY_TO_DEPART]
            print(len(ready_to_depart_))

        if rendering:
            # Environment step which returns the observations for all agents, their corresponding
            # reward and whether their are done
            env_renderer.render_env(show=show, show_observations=False, show_predictions=False)
            print(steps)
            import time
            time.sleep(0.2)

        has_selected_way = [len(flatland_controller.dispatcher.controllers[agent.handle].selected_way) > 0 for agent in env.agents]
        # print(f"{steps} done {sum(has_selected_way)}/{len(env.agents)}")
        steps += 1
        schedule[steps] = {}
        for agent in env.agents:
            schedule[steps][agent.handle] = Waypoint(position=agent.position,
                                                     direction=agent.direction)
        if False:
            print(f"[{steps}] " +
                  " / ".join([f"{str(RailAgentStatus(status))}={sum([1 for agent in env.agents if agent.status == status])} " for status in RailAgentStatus]))
        ready_to_depart = sum([1 for agent in env.agents if agent.status == RailAgentStatus.READY_TO_DEPART])
        # if ready_to_depart ==0:
        #     print(f"done after {steps}")
        #     break

        if done['__all__']:
            print(f"_max_episode_steps={env._max_episode_steps}, _elapsed_steps={env._elapsed_steps}" + str(
                (env._max_episode_steps is not None) and (env._elapsed_steps >= env._max_episode_steps)))

            # assert sum([1 for agent in env.agents if agent.position == None or agent.position == agent.target]) == 0, "not all agents at target or no position"
            if sum([1 for agent in env.agents if (agent.status == RailAgentStatus.READY_TO_DEPART or agent.status == RailAgentStatus.ACTIVE)]) > 0:
                rsp_logger.warn(
                    f"not all agents done: {[agent for agent in env.agents if (agent.status == RailAgentStatus.READY_TO_DEPART or agent.status == RailAgentStatus.ACTIVE)]}")
            # assert sum([1 for agent in env.agents if
            #             (agent.status == RailAgentStatus.READY_TO_DEPART or agent.status == RailAgentStatus.ACTIVE)]) == 0, "not all agents done"
            print(f"done after {steps}")
            # schedule[steps + 1] = {}
            # for agent in env.agents:
            #     schedule[steps + 1][agent.handle] = Waypoint(position=agent.position,
            #                                                  direction=agent.direction)  # , agent.speed_data['position_fraction'])
            if False:
                print(f"[{steps + 1}] {schedule[steps]}")
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
    elapsed_time = perf_counter() - start_time
    resource_occupations = {}
    for time_step, waypoint_dict in schedule.items():
        for agent_id, waypoint in waypoint_dict.items():
            resource = waypoint.position
            if resource is None:
                continue
            # TODO SIM-443 global switch release time
            for tt in [time_step, time_step + 1]:
                occupation = (resource, tt)
                if occupation in resource_occupations:
                    assert agent_id == resource_occupations[occupation], \
                        f"conflicting resource occuptions {occupation} for agents {agent_id} and {resource_occupations[occupation]}"
                resource_occupations[occupation] = agent_id

    initial_positions = {agent.handle: agent.initial_position for agent in env.agents}
    initial_directions = {agent.handle: agent.initial_direction for agent in env.agents}
    targets = {agent.handle: agent.target for agent in env.agents}
    minimum_runningtime_dict = {agent.handle: int(1 // env.agents[agent.handle].speed_data['speed']) for agent in env.agents}
    trainrun_dict = extract_trainrun_dict_from_flatland_positions(initial_directions, initial_positions, schedule,
                                                                  targets)

    trainrun_dict_from_selected_ways = {}
    for agent_id, trainrun in trainrun_dict.items():
        # print(f" == agent {agent_id}  ==")
        # print(_pp.pformat(trainrun))
        cell_graph_agent = flatland_controller.dispatcher.controllers[agent_id]
        cell_graph = cell_graph_agent.graph

        selected_way: List[AgentWayStep] = selected_ways[agent_id]
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


        # TODO change first synchronization step with magic direction
        # TODO why do we lag behind one step?
        # TODO change to MAGIC direction at target?

        # trainrun_from_selected_way.append(TrainrunWaypoint(waypoint=Waypoint(position=cell_graph.position_from_vertexid(selected_way[0].vertex_idx), direction=MAGIC_DIRECTION_FOR_SOURCE_TARGET), scheduled_at=selected_way[0].departure_time))
        trainrun_dict_from_selected_ways[agent_id] = trainrun_from_selected_way
        # print(_pp.pformat(trainrun_from_selected_way))
        if False:
            if trainrun_from_selected_way != trainrun:
                differences = set(trainrun_from_selected_way).symmetric_difference(set(trainrun))
                print(f"target agent {agent_id}: {env.agents[agent_id].target}")
                print(
                    f"agent {agent_id}: \ntrainrun={_pp.pformat(trainrun)},\ntrainrun_from_selected_way={_pp.pformat(trainrun_from_selected_way)},\ndifference={differences}")
                rsp_logger.warn(
                    f"agent {agent_id}: \ntrainrun={_pp.pformat(trainrun)},\ntrainrun_from_selected_way={_pp.pformat(trainrun_from_selected_way)},\ndifference={differences}")

    # TODO why does this not work?
    env.reset(False, False, False, random_seed=random_seed)
    verify_trainrun_dict(env=env,
                         trainrun_dict=trainrun_dict_from_selected_ways)
    verify_trainrun_dict_simple(trainrun_dict=trainrun_dict_from_selected_ways,
                                initial_positions=initial_positions,
                                initial_directions=initial_directions,
                                targets=targets,
                                minimum_runningtime_dict=minimum_runningtime_dict)
    print("verification done")
    return trainrun_dict_from_selected_ways, elapsed_time
