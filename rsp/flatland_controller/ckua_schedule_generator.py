import pprint
from time import perf_counter

import numpy as np
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.experiment_solvers.trainrun_utils import verify_trainrun_dict
from rsp.flatland_controller.ckua_flatland_controller import CkUaController
from rsp.flatland_integration.flatland_conversion import extract_trainrun_dict_from_flatland_positions
from rsp.flatland_integration.flatland_conversion import FLATlandPositionsPerTimeStep

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

        env_renderer.render_env(show=False, show_observations=False, show_predictions=False)

    if max_steps == np.inf:
        max_steps = env._max_episode_steps

    schedule: FLATlandPositionsPerTimeStep = {}

    # TODO SIM-443 extract without stepping
    if False:
        print("(A) without FLATland interaction")
        start_time = perf_counter()
        flatland_controller.setup(env)
        steps=0
        # flatland_controller.controller(env, observation, info, env.get_num_agents())
        while steps < max_steps:
            # TODO SIM-443 call only those agent controllers that need to be called
            flatland_controller.dispatcher.step(steps)
            for agent in env.agents:
                if len(flatland_controller.dispatcher.controllers[agent.handle].selected_way) == 0:
                    flatland_controller.dispatcher.controllers[agent.handle].act(agent, steps)

            has_selected_way = [len(flatland_controller.dispatcher.controllers[agent.handle].selected_way) > 0 for agent in env.agents]
            if np.alltrue(has_selected_way):
                break
            print(f"{steps} done {sum(has_selected_way)}/{len(env.agents)}")
            steps += 1
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

        action = flatland_controller.controller(env, observation, info, env.get_num_agents())
        actions_per_step[steps] = action

        schedule[steps + 1] = {}
        for agent in env.agents:
            schedule[steps + 1][agent.handle] = Waypoint(position=agent.position,
                                                         direction=agent.direction)  # , agent.speed_data['position_fraction'])
        if False:
            print(f"[{steps + 1}] {schedule[steps + 1]}")
        observation, all_rewards, done, _ = env.step(action)
        if steps > 200 and steps < 240:
            print(f"before step {steps + 1}")
            for agent in env.agents:
                print(f" [{steps + 1}] agent [{agent.handle}] position={agent.position} direction={agent.direction} speed_data={agent.speed_data}")

        if False:
            ready_to_depart_ = [agent.status for agent in env.agents if agent.status == RailAgentStatus.READY_TO_DEPART]
            print(len(ready_to_depart_))

        if rendering:
            # Environment step which returns the observations for all agents, their corresponding
            # reward and whether their are done
            env_renderer.render_env(show=show, show_observations=False, show_predictions=False)
        has_selected_way = [len(flatland_controller.dispatcher.controllers[agent.handle].selected_way) > 0 for agent in env.agents]
        # print(f"{steps} done {sum(has_selected_way)}/{len(env.agents)}")
        steps += 1
        if done['__all__']:
            print(f"done after {steps}")
            schedule[steps + 1] = {}
            for agent in env.agents:
                schedule[steps + 1][agent.handle] = Waypoint(position=agent.position,
                                                             direction=agent.direction)  # , agent.speed_data['position_fraction'])
            if False:
                print(f"[{steps + 1}] {schedule[steps]}")
            if not ((env._max_episode_steps is not None) and (
                    env._elapsed_steps >= env._max_episode_steps)):
                break
    print(f"took {perf_counter() - start_time:5.2f}")
    for _, actions in actions_per_step.items():
        for agent_id, action in actions.items():
            actions[agent_id] = int(action)
    # import pickle
    # with open(f"tests/01_unit_tests/data/ckua/actions_per_time_step.pkl", "wb") as out:
    #     pickle.dump(actions_per_step, out, protocol=pickle.HIGHEST_PROTOCOL)
    # print("actions_per_step")
    # # print(_pp.pformat(actions_per_step))
    # print(actions_per_step)
    # print("schedule")
    # with open(f"tests/01_unit_tests/data/ckua/schedule.pkl", "wb") as out:
    #     pickle.dump(schedule, out, protocol=pickle.HIGHEST_PROTOCOL)
    # print(_pp.pformat(schedule))

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

    trainrun_dict = extract_trainrun_dict_from_flatland_positions(initial_directions, initial_positions, schedule,
                                                                  targets)
    # TODO why does this not work?
    env.reset(False, False, False, random_seed=random_seed)
    verify_trainrun_dict(env=env, trainrun_dict=trainrun_dict)
    print("verification done")
    return trainrun_dict, elapsed_time
