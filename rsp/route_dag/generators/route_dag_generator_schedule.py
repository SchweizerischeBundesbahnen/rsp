import networkx as nx
import numpy as np
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_shortest_paths import get_k_shortest_paths
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.route_dag.generators.route_dag_generator_utils import propagate_earliest
from rsp.route_dag.generators.route_dag_generator_utils import propagate_latest
from rsp.route_dag.route_dag import _get_topology_with_dummy_nodes_from_agent_paths_dict
from rsp.route_dag.route_dag import get_sinks_for_topo
from rsp.route_dag.route_dag import ScheduleProblemDescription
from rsp.utils.data_types import RouteDAGConstraints


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
    dummy_source_dict, topo_dict = _get_topology_with_dummy_nodes_from_agent_paths_dict(agents_paths_dict)
    schedule_problem_description = ScheduleProblemDescription(
        route_dag_constraints_dict={
            agent_id: _get_route_dag_constraints_for_scheduling(
                minimum_travel_time=minimum_travel_time_dict[agent_id],
                topo=topo_dict[agent_id],
                dummy_source=dummy_source_dict[agent_id],
                latest_arrival=env._max_episode_steps)
            for agent_id, topo in topo_dict.items()},
        minimum_travel_time_dict=minimum_travel_time_dict,
        topo_dict=topo_dict,
        max_episode_steps=env._max_episode_steps,
        route_section_penalties={agent.handle: {} for agent in env.agents},
        weight_lateness_seconds=1
    )

    return schedule_problem_description


def _get_route_dag_constraints_for_scheduling(
        topo: nx.DiGraph,
        dummy_source: Waypoint,
        minimum_travel_time: int,
        latest_arrival: int
) -> RouteDAGConstraints:
    return RouteDAGConstraints(
        freeze_visit=[],
        freeze_earliest=propagate_earliest(
            banned_set=set(),
            earliest_dict={dummy_source: 0},
            minimum_travel_time=minimum_travel_time,
            force_freeze_dict={},
            subdag_source=TrainrunWaypoint(waypoint=dummy_source, scheduled_at=0),
            topo=topo,
        ),
        freeze_latest=propagate_latest(
            banned_set=set(),
            latest_dict={sink: latest_arrival - 1 for sink in get_sinks_for_topo(topo)},
            latest_arrival=latest_arrival,
            minimum_travel_time=minimum_travel_time,
            force_freeze_dict={},
            topo=topo,
        ),
        freeze_banned=[],
    )
