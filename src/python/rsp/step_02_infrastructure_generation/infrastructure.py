import os
from typing import Dict
from typing import NamedTuple
from typing import Tuple

import networkx as nx
import numpy as np
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_shortest_paths import get_k_shortest_paths

from rsp.rspflatland.experiment_env_generators import create_flatland_environment
from rsp.global_data_configuration import EXPERIMENT_INFRA_SUBDIRECTORY_NAME
from rsp.scheduling.scheduling_problem import _get_topology_from_agents_path_dict
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import InfrastructureParameters
from rsp.utils.pickle_helper import _pickle_dump
from rsp.utils.pickle_helper import _pickle_load
from rsp.utils.rsp_logger import rsp_logger

# TODO SIM-661 we should separate grid generation from agent placement, speed generation and topo_dict extraction (shortest paths);
#  however, we do not pass the city information out of FLATland to place agents.

Infrastructure = NamedTuple("Infrastructure", [("topo_dict", Dict[int, nx.DiGraph]), ("minimum_travel_time_dict", Dict[int, int]), ("max_episode_steps", int)])


def create_env_from_experiment_parameters(params: InfrastructureParameters) -> RailEnv:
    """
    Parameters
    ----------
    params: ExperimentParameters2
        Parameter set that we pass to the constructor of the RailEenv
    Returns
    -------
    RailEnv
        Static environment where no malfunction occurs
    """

    number_of_agents = params.number_of_agents
    width = params.width
    height = params.height
    flatland_seed_value = int(params.flatland_seed_value)
    max_num_cities = params.max_num_cities
    grid_mode = params.grid_mode
    max_rails_between_cities = params.max_rail_between_cities
    max_rails_in_city = params.max_rail_in_city
    speed_data = params.speed_data

    # Generate static environment for initial schedule generation
    env_static = create_flatland_environment(
        number_of_agents=number_of_agents,
        width=width,
        height=height,
        flatland_seed_value=flatland_seed_value,
        max_num_cities=max_num_cities,
        grid_mode=grid_mode,
        max_rails_between_cities=max_rails_between_cities,
        max_rails_in_city=max_rails_in_city,
        speed_data=speed_data,
    )
    return env_static


def create_infrastructure_from_rail_env(env: RailEnv, k: int):
    rsp_logger.info("create_infrastructure_from_rail_env")
    agents_paths_dict = {
        # TODO https://gitlab.aicrowd.com/flatland/flatland/issues/302: add method to FLATland to create of k shortest paths for all agents
        i: get_k_shortest_paths(env, agent.initial_position, agent.initial_direction, agent.target, k)
        for i, agent in enumerate(env.agents)
    }
    rsp_logger.info("create_infrastructure_from_rail_env: shortest paths done")
    minimum_travel_time_dict = {agent.handle: int(np.ceil(1 / agent.speed_data["speed"])) for agent in env.agents}
    topo_dict = _get_topology_from_agents_path_dict(agents_paths_dict)
    return Infrastructure(topo_dict=topo_dict, minimum_travel_time_dict=minimum_travel_time_dict, max_episode_steps=env._max_episode_steps)


def gen_infrastructure(infra_parameters: InfrastructureParameters) -> Infrastructure:
    """A.1.1 infrastructure generation."""
    rsp_logger.info(f"gen_infrastructure {infra_parameters}")
    infra = create_infrastructure_from_rail_env(
        env=create_env_from_experiment_parameters(infra_parameters), k=infra_parameters.number_of_shortest_paths_per_agent
    )
    rsp_logger.info(f"done gen_infrastructure {infra_parameters}")
    return infra


def exists_infrastructure(base_directory: str, infra_id: int) -> bool:
    """Does a persisted `Infrastructure` exist?"""
    file_name = os.path.join(base_directory, EXPERIMENT_INFRA_SUBDIRECTORY_NAME, f"{infra_id:03d}", "infrastructure.pkl")
    return os.path.isfile(file_name)


def load_infrastructure(base_directory: str, infra_id: int) -> Tuple[Infrastructure, InfrastructureParameters]:
    """Load a persisted `Infrastructure` from a file.
    Parameters
    ----------
    base_directory
    infra_id


    Returns
    -------
    """
    folder = os.path.join(base_directory, EXPERIMENT_INFRA_SUBDIRECTORY_NAME, f"{infra_id:03d}")
    infra = _pickle_load(folder=folder, file_name=f"infrastructure.pkl")
    infra_parameters = _pickle_load(folder=folder, file_name=f"infrastructure_parameters.pkl")
    return infra, infra_parameters


def save_infrastructure(infrastructure: Infrastructure, infrastructure_parameters: InfrastructureParameters, base_directory: str):
    """Persist `Infrastructure` to a file.
    Parameters
    ----------
    infrastructure_parameters
    infrastructure
    base_directory
    """
    folder = os.path.join(base_directory, EXPERIMENT_INFRA_SUBDIRECTORY_NAME, f"{infrastructure_parameters.infra_id:03d}")
    _pickle_dump(obj=infrastructure, folder=folder, file_name="infrastructure.pkl")
    _pickle_dump(obj=infrastructure_parameters, folder=folder, file_name="infrastructure_parameters.pkl")
