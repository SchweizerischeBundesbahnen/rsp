"""
Data types used in the experiment for the real time rescheduling research project

"""
import pprint
from typing import NamedTuple, List, Dict, Mapping, Set, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint, TrainrunDict, Waypoint
from flatland.utils.ordered_set import OrderedSet

ExperimentFreeze = NamedTuple('ExperimentFreeze', [
    ('freeze_visit', List[TrainrunWaypoint]),
    ('freeze_earliest', Dict[Waypoint, int]),
    ('freeze_latest', Dict[Waypoint, int]),
    ('freeze_banned', List[Waypoint])
])
ExperimentFreezeDict = Dict[int, ExperimentFreeze]

AgentPaths = List[List[Waypoint]]
AgentsPathsDict = Dict[int, AgentPaths]


def experiment_freeze_dict_from_list_of_train_run_waypoint(l: List[TrainrunWaypoint]) -> Dict[TrainrunWaypoint, int]:
    """
    Generate dictionary of scheduled time at waypoint.

    Parameters
    ----------
    l train run waypoints

    Returns
    -------

    """
    return {trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at for trainrun_waypoint in l}


SpeedData = Mapping[float, float]
ExperimentParameters = NamedTuple('ExperimentParameters',
                                  [('experiment_id', str),
                                   ('trials_in_experiment', int),
                                   ('number_of_agents', int),
                                   ('speed_data', SpeedData),
                                   ('width', int),
                                   ('height', int),
                                   ('seed_value', int),
                                   ('max_num_cities', int),
                                   ('grid_mode', bool),
                                   ('max_rail_between_cities', int),
                                   ('max_rail_in_city', int),
                                   ('earliest_malfunction', int),
                                   ('malfunction_duration', int)])

ExperimentAgenda = NamedTuple('ExperimentAgenda', [('experiment_name', str),
                                                   ('experiments', List[ExperimentParameters])])

ExperimentMalfunction = NamedTuple('ExperimentMalfunction', [
    ('time_step', int),
    ('agent_id', int),
    ('malfunction_duration', int)
])

ExperimentResults = NamedTuple('ExperimentResults', [
    ('time_full', float),
    ('time_full_after_malfunction', float),
    ('time_delta_after_malfunction', float),
    ('solution_full', TrainrunDict),
    ('solution_full_after_malfunction', TrainrunDict),
    ('solution_delta_after_malfunction', TrainrunDict),
    ('costs_full', float),  # sum of travelling times in scheduling solution
    ('costs_full_after_malfunction', float),  # total delay at target over all agents with respect to schedule
    ('costs_delta_after_malfunction', float),  # total delay at target over all agents with respect to schedule
    ('experiment_freeze', ExperimentFreezeDict),
    ('malfunction', ExperimentMalfunction),
    ('agent_paths_dict', AgentsPathsDict)
])

ParameterRanges = NamedTuple('ParameterRanges', [('size_range', List[int]),
                                                 ('agent_range', List[int]),
                                                 ('in_city_rail_range', List[int]),
                                                 ('out_city_rail_range', List[int]),
                                                 ('city_range', List[int]),
                                                 ('earliest_malfunction', List[int]),
                                                 ('malfunction_duration', List[int])
                                                 ])

_pp = pprint.PrettyPrinter(indent=4)


def experimentFreezeDictPrettyPrint(d: ExperimentFreezeDict):
    for agent_id, experiment_freeze in d.items():
        prefix = f"agent {agent_id} "
        experimentFreezePrettyPrint(experiment_freeze, prefix)


def experimentFreezePrettyPrint(experiment_freeze: ExperimentFreeze, prefix: str = ""):
    print(f"{prefix}freeze_visit={_pp.pformat(experiment_freeze.freeze_visit)}")
    print(f"{prefix}freeze_earliest={_pp.pformat(experiment_freeze.freeze_earliest)}")
    print(f"{prefix}freeze_latest={_pp.pformat(experiment_freeze.freeze_latest)}")
    print(f"{prefix}freeze_banned={_pp.pformat(experiment_freeze.freeze_banned)}")


def visualize_experiment_freeze(agent_paths: AgentPaths,
                                f: ExperimentFreeze,
                                file_name: str,
                                title: Optional[str] = None,
                                scale: int = 2,
                                ):
    """
    Draws an agent's route graph with constraints into a file.

    Parameters
    ----------
    agent_paths
        the agent's paths spanning its routes graph
    f
        constraints for this agent
    file_name
        save graph to this file
    title
        title in the picture
    scale
        scale in or out

    """
    # N.B. FLATland uses row-column indexing, plt uses x-y (horizontal,vertical with vertical axis going bottom-top)

    # nx directed graph
    topo = nx.DiGraph()
    all_waypoints: Set[Waypoint] = OrderedSet()
    for path in agent_paths:
        for wp1, wp2 in zip(path, path[1:]):
            topo.add_edge(wp1, wp2)
            all_waypoints.add(wp1)
            all_waypoints.add(wp2)

    # figsize
    flatland_positions = np.array([waypoint.position for waypoint in all_waypoints])
    flatland_figsize = np.max(flatland_positions, axis=0) - np.min(flatland_positions, axis=0)
    plt.figure(figsize=(flatland_figsize[1] * scale, flatland_figsize[0] * scale))

    # plt title
    if title:
        plt.title(title)

    # positions with offsets for the pins
    offset = 0.25
    flatland_offset_pattern = {
        # heading north = coming from south: +row
        0: np.array([offset, 0]),
        # heading east = coming from west: -col
        1: np.array([0, -offset]),
        # heading south = coming from north: -row
        2: np.array([-offset, 0]),
        # heading west = coming from east: +col
        3: np.array([0, offset]),
    }
    flatland_pos_with_offset = {wp: np.array(wp.position) + flatland_offset_pattern[wp.direction] for wp in
                                all_waypoints}

    plt_pos = {wp: np.array([p[1], p[0]]) for wp, p in flatland_pos_with_offset.items()}

    def constraint_for_waypoint(waypoint: Waypoint) -> str:
        if waypoint in f.freeze_banned:
            return "X"
        s: str = "["
        if waypoint in f.freeze_visit:
            s = "! ["
        s += str(f.freeze_earliest[waypoint])
        s += ","
        s += str(f.freeze_latest[waypoint])
        s += "]"
        return s

    def color_for_node(n: Waypoint):
        # https://matplotlib.org/examples/color/named_colors.html
        if n in f.freeze_banned:
            return 'salmon'
        elif n in f.freeze_visit:
            return 'orange'
        else:
            return 'lightgreen'

    plt_color_map = [color_for_node(node) for node in topo.nodes()]

    plt_labels = {wp: f"{wp.position[0]},{wp.position[1]},{wp.direction}\n{constraint_for_waypoint(wp)}" for wp in
                  all_waypoints}
    nx.draw(topo, plt_pos,
            labels=plt_labels,
            edge_color='black',
            width=1,
            linewidths=1,
            node_size=1500,
            node_color=plt_color_map,
            alpha=0.9)
    plt.gca().invert_yaxis()
    plt.savefig(file_name)
