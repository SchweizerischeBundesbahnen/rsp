import pprint
import time

import numpy as np
from flatland.action_plan.action_plan import ControllerFromTrainruns
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import AgentRenderVariant
from flatland.utils.rendertools import RenderTool

from rsp.flatland_controller.ckua_flatland_controller import CkUaController
from rsp.route_dag.route_dag import MAGIC_DIRECTION_FOR_SOURCE_TARGET
from rsp.utils.flatland_replay_utils import create_controller_from_trainruns_and_malfunction
from rsp.utils.flatland_replay_utils import replay

_pp = pprint.PrettyPrinter(indent=4)


def ckua_generate_schedule(  # noqa:C901
        env: RailEnv,
        random_seed: int,
        rendering: bool = False,
        show: bool = False,
        max_steps: int = np.inf) -> TrainrunDict:
    steps = 0
    total_reward = 0

    # setup the env
    observation, info = env.reset(False, False, False, random_seed=random_seed)

    # Setup the controller (solver)
    flatland_controller = CkUaController()
    flatland_controller.setup(env)

    do_rendering = rendering
    do_rendering_final = rendering
    do_rendering_first = rendering

    if do_rendering or do_rendering_final or do_rendering_final:
        env_renderer = RenderTool(env=env,
                                  gl="PILSVG",
                                  agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS,
                                  show_debug=True,
                                  screen_height=1600,
                                  screen_width=1600)

        env_renderer.render_env(show=False, show_observations=False, show_predictions=False)

    if max_steps == np.inf:
        max_steps = env._max_episode_steps

    schedule = {}
    while steps < max_steps:
        if steps == 0:
            schedule[0] = {}
            for agent in env.agents:
                schedule[steps][agent.handle] = Waypoint(position=agent.position, direction=agent.direction)  # , agent.speed_data['position_fraction'])
            print(f"[{steps}] {schedule[steps]}")

        action = flatland_controller.controller(env, observation, info, env.get_num_agents())
        time.sleep(1)

        schedule[steps + 1] = {}
        for agent in env.agents:
            schedule[steps + 1][agent.handle] = Waypoint(position=agent.position, direction=agent.direction)  # , agent.speed_data['position_fraction'])
        print(f"[{steps + 1}] {schedule[steps + 1]}")
        observation, all_rewards, done, _ = env.step(action)

        if do_rendering or (do_rendering_first and steps == 0):
            # Environment step which returns the observations for all agents, their corresponding
            # reward and whether their are done
            env_renderer.render_env(show=show, show_observations=False, show_predictions=False)

        steps += 1
        total_reward += sum(list(all_rewards.values()))
        if done['__all__']:
            for agent in env.agents:
                schedule[steps][agent.handle] = Waypoint(position=agent.position, direction=agent.direction)  # , agent.speed_data['position_fraction'])
            print(f"[{steps}] {schedule[steps]}")
            if not ((env._max_episode_steps is not None) and (
                    env._elapsed_steps >= env._max_episode_steps)):
                break

    if do_rendering_final:
        # Environment step which returns the observations for all agents, their corresponding
        # reward and whether their are done
        env_renderer.render_env(show=True, show_observations=False, show_predictions=False)
    if do_rendering or do_rendering_first or do_rendering_final:
        env_renderer.gl.close_window()

    print(schedule)
    resource_occupations = {}
    for time_step, waypoint_dict in schedule.items():
        for agent_id, waypoint in waypoint_dict.items():
            resource = waypoint.position
            if resource is None:
                continue
            # TODO release time
            for tt in [time_step, time_step + 1]:
                occupation = (resource, tt)
                if occupation in resource_occupations:
                    assert agent_id == resource_occupations[occupation], \
                        f"conflicting resource occuptions {occupation} for agents {agent_id} and {resource_occupations[occupation]}"
                resource_occupations[occupation] = agent_id

    initial_positions = {agent.handle: agent.initial_position for agent in env.agents}
    initial_directions = {agent.handle: agent.initial_direction for agent in env.agents}
    targets = {agent.handle: agent.target for agent in env.agents}

    trainrun_dict = _extract_trainrun_dict_from_flatland_positions(env, initial_directions, initial_positions, schedule, targets)
    print(_pp.pformat(trainrun_dict))
    verify_trainrun_dict(env, random_seed, trainrun_dict)

    return trainrun_dict


def verify_trainrun_dict(env, random_seed, trainrun_dict):
    env.reset(random_seed=random_seed)
    controller_from_train_runs: ControllerFromTrainruns = create_controller_from_trainruns_and_malfunction(
        trainrun_dict=trainrun_dict,
        env=env)
    # TODO SIM-443 hard replay or implement release time in FLATland:
    #     expected_flatland_positions=convert_trainrundict_to_positions_after_flatland_timestep(trainrun_dict), # noqa: E800
    replay(
        controller_from_train_runs=controller_from_train_runs,
        env=env,
        rendering=True,
        show=True,
        solver_name="CkUaController")


def _extract_trainrun_dict_from_flatland_positions(env, initial_directions, initial_positions, schedule, targets):
    trainrun_dict = {agent.handle: [] for agent in env.agents}
    for agent_id in trainrun_dict:

        curr_pos = None
        for time_step in schedule:

            next_waypoint = schedule[time_step][agent_id]
            if next_waypoint.position is not None:
                if next_waypoint.position != curr_pos:
                    if curr_pos is None:
                        assert time_step >= 1
                        assert next_waypoint.position == initial_positions[agent_id]
                        assert next_waypoint.direction == initial_directions[agent_id]
                        trainrun_dict[agent_id].append(
                            TrainrunWaypoint(waypoint=Waypoint(
                                position=next_waypoint.position,
                                direction=MAGIC_DIRECTION_FOR_SOURCE_TARGET),
                                scheduled_at=time_step - 1))
                    if next_waypoint.position is not None:
                        trainrun_dict[agent_id].append(TrainrunWaypoint(waypoint=next_waypoint, scheduled_at=time_step))
            if next_waypoint.position is None and curr_pos is not None:
                trainrun_dict[agent_id].append(
                    TrainrunWaypoint(
                        waypoint=Waypoint(
                            position=targets[agent_id], direction=MAGIC_DIRECTION_FOR_SOURCE_TARGET
                        ),
                        scheduled_at=time_step))
                assert abs(curr_pos[0] - targets[agent_id][0]) + abs(
                    curr_pos[1] - targets[agent_id][1]) == 1, f"agent {agent_id}: curr_pos={curr_pos} - target={targets[agent_id]}"
            curr_pos = next_waypoint.position
    return trainrun_dict


def dummy_rail_env(observation_builder: ObservationBuilder, number_of_agents: int = 100, random_seed: int = 14) -> RailEnv:
    # Different agent types (trains) with different speeds.
    speed_ration_map = {1.: 0.5,  # Fast passenger train
                        1. / 2.: 0.3,  # Slow passenger train
                        1. / 4.: 0.2}  # Slow freight train

    env = RailEnv(width=100,
                  height=60,
                  rail_generator=sparse_rail_generator(max_num_cities=20,
                                                       # Number of cities in map (where train stations are)
                                                       seed=random_seed,  # Random seed
                                                       grid_mode=False,
                                                       max_rails_between_cities=2,
                                                       max_rails_in_city=8,
                                                       ),
                  schedule_generator=sparse_schedule_generator(speed_ration_map),
                  number_of_agents=number_of_agents,
                  obs_builder_object=observation_builder,
                  remove_agents_at_target=True,
                  random_seed=random_seed,
                  record_steps=True
                  )
    return env


def main():
    ckua_generate_schedule(
        env=dummy_rail_env(observation_builder=DummyObservationBuilder()),
        random_seed=94,
        rendering=True
    )


if __name__ == '__main__':
    main()

# TODO SIM-443 integration schedule generation
# TODO SIM-443 release time in FLATland for replay or set position in FLATland instead? --> check
# TODO SIM-443 refactor ckua_schedule_generator.py
# TODO general: actually, as long as we can use this schedule generation, our problems are not big enough??!!
# TODO SIM-443: understand the scheudling heuristic better
