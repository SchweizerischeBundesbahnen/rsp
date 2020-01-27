"""
Create flatland environment for specific experiments

Methods
-------
create_flatland_environment:
    Create a Flatland environment without any dynamic events such as malfunctions

create_flatland_environment_with_malfunction:
    Create a Flatland environment with one single malfunction.

"""

# ----------------------------- Flatland ------------------------------------

from typing import Mapping

from flatland.envs.malfunction_generators import single_malfunction_generator
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator

from rsp.utils.data_types import SpeedData


def create_flatland_environment(number_of_agents: int,
                                width: int,
                                height: int,
                                seed_value: int,
                                max_num_cities: int,
                                grid_mode: bool,
                                max_rails_between_cities: int,
                                max_rails_in_city: int,
                                speed_data: SpeedData
                                ) -> (RailEnv, int):
    """
    Generates sparse envs WITHOUT malfunctions for our research experiments

    Parameters
    ----------
    number_of_agents
    width
    height
    seed_value
    max_num_cities
    grid_mode
    max_rails_between_cities
    max_rails_in_city
    speed_data

    Returns
    -------

    """
    rail_generator = sparse_rail_generator(max_num_cities=max_num_cities,
                                           grid_mode=grid_mode,
                                           max_rails_between_cities=max_rails_between_cities,
                                           max_rails_in_city=max_rails_in_city,
                                           seed=seed_value  # Random seed
                                           )
    schedule_generator = sparse_schedule_generator(speed_data)

    environment = RailEnv(width=width,
                          height=height,
                          rail_generator=rail_generator,
                          number_of_agents=number_of_agents,
                          schedule_generator=schedule_generator,
                          remove_agents_at_target=True
                          )
    environment.reset(random_seed=seed_value)

    return environment


def create_flatland_environment_with_malfunction(number_of_agents: int,
                                                 width: int,
                                                 height: int,
                                                 seed_value: int,
                                                 max_num_cities: int,
                                                 grid_mode: bool,
                                                 max_rails_between_cities: int,
                                                 max_rails_in_city: int,
                                                 earliest_malfunction: int,
                                                 malfunction_duration: int,
                                                 speed_data: SpeedData
                                                 ) -> (RailEnv, int):
    """
    Generates sparse envs WITH malfunctions for our research experiments

    Parameters
    ----------
    number_of_agents
    width
    height
    seed_value
    max_num_cities
    grid_mode
    max_rails_between_cities
    max_rails_in_city
    earliest_malfunction
    malfunction_duration
    speed_data

    Returns
    -------

    """
    rail_generator = sparse_rail_generator(max_num_cities=max_num_cities,
                                           grid_mode=grid_mode,
                                           max_rails_between_cities=max_rails_between_cities,
                                           max_rails_in_city=max_rails_in_city,
                                           seed=seed_value  # Random seed
                                           )
    schedule_generator = sparse_schedule_generator(speed_data)

    environment = RailEnv(width=width,
                          height=height,
                          rail_generator=rail_generator,
                          number_of_agents=number_of_agents,
                          schedule_generator=schedule_generator,
                          remove_agents_at_target=True,
                          malfunction_generator_and_process_data=single_malfunction_generator(
                              earlierst_malfunction=earliest_malfunction,
                              malfunction_duration=malfunction_duration)
                          )
    environment.reset(random_seed=seed_value)

    return environment
