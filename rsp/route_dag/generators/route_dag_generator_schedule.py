from typing import Dict

import numpy as np
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_shortest_paths import get_k_shortest_paths
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint

from rsp.route_dag.generators.route_dag_generator_utils import propagate_earliest
from rsp.route_dag.route_dag import _get_topology_with_dummy_nodes_from_agent_paths_dict
from rsp.route_dag.route_dag import AgentsPathsDict
from rsp.route_dag.route_dag import ScheduleProblemDescription
from rsp.utils.data_types import RouteDAGConstraints
from rsp.utils.data_types import RouteDAGConstraintsDict


def schedule_problem_description_from_rail_env(env: RailEnv, k: int) -> ScheduleProblemDescription:
    agents_paths_dict = {
        # TODO https://gitlab.aicrowd.com/flatland/flatland/issues/302: add method to FLATland to create of k shortest paths for all agents
        i: get_k_shortest_paths(env,
                                agent.initial_position,
                                agent.initial_direction,
                                agent.target,
                                k) for i, agent in enumerate(env.agents)
    }

    minimum_travel_time_dict = {agent.handle: int(np.ceil(1 / agent.speed_data['speed']))
                                for agent in env.agents}
    _, topo_dict = _get_topology_with_dummy_nodes_from_agent_paths_dict(agents_paths_dict=agents_paths_dict)
    schedule_problem_description = ScheduleProblemDescription(
        route_dag_constraints_dict=_get_freeze_for_scheduling(minimum_travel_time_dict=minimum_travel_time_dict,
                                                              agents_paths_dict=agents_paths_dict,
                                                              latest_arrival=env._max_episode_steps),
        minimum_travel_time_dict=minimum_travel_time_dict,
        topo_dict=topo_dict,
        max_episode_steps=env._max_episode_steps,
        route_section_penalties={agent.handle: {} for agent in env.agents})
    return schedule_problem_description


def _get_freeze_for_scheduling(
        minimum_travel_time_dict: Dict[int, int],
        agents_paths_dict: AgentsPathsDict,
        latest_arrival: int
) -> RouteDAGConstraintsDict:
    dummy_source_dict, topo_dict = _get_topology_with_dummy_nodes_from_agent_paths_dict(agents_paths_dict)

    return {
        agent_id: RouteDAGConstraints(
            freeze_visit=[],
            freeze_earliest=propagate_earliest(
                banned_set=[],
                earliest_dict={dummy_source_dict[agent_id]: 0},
                minimum_travel_time=minimum_travel_time_dict[agent_id],
                force_freeze_dict={},
                subdag_source=TrainrunWaypoint(waypoint=dummy_source_dict[agent_id], scheduled_at=0),
                topo=topo_dict[agent_id],
            ),
            # TODO SIM-239 deactivate for backward compatibility?
            freeze_latest={waypoint: latest_arrival for waypoint in topo_dict[agent_id].nodes},
            freeze_banned=[],
        )
        for agent_id in agents_paths_dict}
