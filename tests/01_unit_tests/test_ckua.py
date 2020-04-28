import pickle

from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator

from rsp.experiment_solvers.trainrun_utils import verify_trainruns_dict
from rsp.flatland_controller.ckua_schedule_generator import _extract_trainrun_dict_from_flatland_positions


def dummy_rail_env(observation_builder: ObservationBuilder,
                   number_of_agents: int = 22,
                   random_seed: int = 133) -> RailEnv:
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


def test_ckua_conversion():
    env = dummy_rail_env(observation_builder=DummyObservationBuilder())
    env.reset(False, False, False, random_seed=133)

    with open(f"tests/01_unit_tests/data/ckua/actions_per_time_step.pkl", "rb") as out:
        actions_per_step = pickle.load(out)
    print("actions_per_step")
    print(actions_per_step)
    print("schedule")
    with open(f"tests/01_unit_tests/data/ckua/schedule.pkl", "rb") as out:
        schedule = pickle.load(out)
    print(schedule)

    initial_positions = {agent.handle: agent.initial_position for agent in env.agents}
    initial_directions = {agent.handle: agent.initial_direction for agent in env.agents}
    targets = {agent.handle: agent.target for agent in env.agents}

    trainrun_dict = _extract_trainrun_dict_from_flatland_positions(env, initial_directions, initial_positions, schedule,
                                                                   targets)

    verify_trainruns_dict(env=env, trainrun_dict=trainrun_dict)
