from collections import OrderedDict
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

import networkx as nx
import numpy as np
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_shortest_paths import get_k_shortest_paths
from flatland.envs.rail_trainrun_data_structures import Trainrun
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.route_dag.route_dag import AgentsPathsDict
from rsp.route_dag.route_dag import get_sinks_for_topo
from rsp.route_dag.route_dag import get_sources_for_topo
from rsp.route_dag.route_dag import MAGIC_DIRECTION_FOR_SOURCE_TARGET
from rsp.route_dag.route_dag import topo_from_agent_paths
from rsp.route_dag.route_dag import TopoDict
from rsp.scheduling.scheduling_data_types import ScheduleProblemDescription
from rsp.utils.data_types import ExperimentFreeze
from rsp.utils.data_types import ExperimentFreezeDict
from rsp.utils.data_types import ExperimentMalfunction


# TODO SIM-239 separate into scheduling, re-scheduling, delta and helpers! make UT for helpers
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
        experiment_freeze_dict=_get_freeze_for_scheduling(minimum_travel_time_dict=minimum_travel_time_dict,
                                                          agents_paths_dict=agents_paths_dict,
                                                          latest_arrival=env._max_episode_steps),
        minimum_travel_time_dict=minimum_travel_time_dict,
        topo_dict=topo_dict,
        max_episode_steps=env._max_episode_steps)
    return schedule_problem_description


def _get_freeze_for_scheduling(
        minimum_travel_time_dict: Dict[int, int],
        agents_paths_dict: AgentsPathsDict,
        latest_arrival: int
) -> ExperimentFreezeDict:
    dummy_source_dict, topo_dict = _get_topology_with_dummy_nodes_from_agent_paths_dict(agents_paths_dict)

    return {
        agent_id: ExperimentFreeze(
            freeze_visit=[],
            freeze_earliest=_propagate_earliest(
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


def _get_topology_with_dummy_nodes_from_agent_paths_dict(agents_paths_dict):
    # get topology from agent paths
    topo_dict = {agent_id: topo_from_agent_paths(agents_paths_dict[agent_id])
                 for agent_id in agents_paths_dict}
    # add dummy nodes
    dummy_source_dict: Dict[int, Waypoint] = {}
    dummy_sink_dict: Dict[int, Waypoint] = {}
    for agent_id, topo in topo_dict.items():
        sources = list(get_sources_for_topo(topo))
        sinks = list(get_sinks_for_topo(topo))

        dummy_sink_waypoint = Waypoint(position=agents_paths_dict[agent_id][0][-1].position,
                                       direction=MAGIC_DIRECTION_FOR_SOURCE_TARGET)
        dummy_sink_dict[agent_id] = dummy_sink_waypoint
        dummy_source_waypoint = Waypoint(position=agents_paths_dict[agent_id][0][0].position,
                                         direction=MAGIC_DIRECTION_FOR_SOURCE_TARGET)
        dummy_source_dict[agent_id] = dummy_source_waypoint
        for source in sources:
            topo.add_edge(dummy_source_waypoint, source)
        for sink in sinks:
            topo.add_edge(sink, dummy_sink_waypoint)
    return dummy_source_dict, topo_dict


def generic_experiment_freeze_for_rescheduling(
        schedule_trainruns: TrainrunDict,
        minimum_travel_time_dict: Dict[int, int],
        topo_dict: TopoDict,
        force_freeze: Dict[int, List[TrainrunWaypoint]],
        malfunction: ExperimentMalfunction,
        latest_arrival: int
) -> ScheduleProblemDescription:
    """Derives the experiment freeze given the malfunction and optionally a
    force freeze from an Oracle. The node after the malfunction time has to be
    visited with an earliest constraint.

    Parameters
    ----------
    schedule_trainruns
        the schedule before the malfunction happened
    minimum_travel_time_dict
        the agent's speed (constant for every agent, different among agents)
    topo_dict
        the topos for the agents
    force_freeze
        waypoints the oracle told to pass by
    malfunction
        malfunction
    latest_arrival
        end of the global time window

    Returns
    -------
    """
    experiment_freeze_dict = {
        agent_id: _generic_experiment_freeze_for_rescheduling_agent_while_running(
            minimum_travel_time=minimum_travel_time_dict[agent_id],
            topo=topo_dict[agent_id],
            force_freeze=force_freeze[agent_id],
            subdag_source=_get_delayed_trainrun_waypoint_after_malfunction(
                agent_id=agent_id,
                trainrun=schedule_trainruns[agent_id],
                malfunction=malfunction
            ),
            latest_arrival=latest_arrival

        )
        for agent_id, schedule_trainrun in schedule_trainruns.items()
        # ---> handle them special
        #  - if not started -> everything open
        #  - if already done -> everything remains the same
        if (malfunction.time_step >= schedule_trainrun[0].scheduled_at and  # noqa: W504
            malfunction.time_step < schedule_trainrun[-1].scheduled_at) or force_freeze[agent_id]

    }

    # inconsistent data if malfunction agent is not impacted by the malfunction!
    if malfunction.agent_id not in experiment_freeze_dict:
        raise Exception(f"agent {malfunction.agent_id} has malfunction {malfunction} "
                        f"before scheduled start {schedule_trainruns[malfunction.agent_id] if malfunction.agent_id in schedule_trainruns else None}. ")

    # handle the special case of malfunction before scheduled start or after scheduled arrival of agent
    for agent_id, schedule_trainrun in schedule_trainruns.items():
        if agent_id not in experiment_freeze_dict:
            if malfunction.time_step < schedule_trainrun[0].scheduled_at:

                experiment_freeze_dict[agent_id] = ExperimentFreeze(
                    freeze_visit=[],
                    freeze_earliest=_propagate_earliest(
                        banned_set=[],
                        earliest_dict={schedule_trainrun[0].waypoint: schedule_trainrun[0].scheduled_at},
                        minimum_travel_time=minimum_travel_time_dict[agent_id],
                        force_freeze_dict={},
                        subdag_source=schedule_trainrun[0],
                        topo=topo_dict[agent_id],
                    ),
                    freeze_latest=_propagate_latest(
                        banned_set=[],
                        latest_dict={sink: latest_arrival - 1 for sink in get_sinks_for_topo(topo_dict[agent_id])},
                        latest_arrival=latest_arrival,
                        minimum_travel_time=minimum_travel_time_dict[agent_id],
                        force_freeze_dict={},
                        topo=topo_dict[agent_id],
                    ),
                    freeze_banned=[],
                )
                freeze: ExperimentFreeze = experiment_freeze_dict[agent_id]
                # N.B. copy keys into new list (cannot delete keys while looping concurrently looping over them)
                waypoints: List[Waypoint] = list(freeze.freeze_earliest.keys())
                for waypoint in waypoints:
                    if freeze.freeze_earliest[waypoint] > freeze.freeze_latest[waypoint]:
                        del freeze.freeze_latest[waypoint]
                        del freeze.freeze_earliest[waypoint]
                        freeze.freeze_banned.append(waypoint)
            elif malfunction.time_step >= schedule_trainrun[-1].scheduled_at:
                visited = {trainrun_waypoint.waypoint for trainrun_waypoint in schedule_trainrun}
                all_waypoints = topo_dict[agent_id].nodes
                experiment_freeze_dict[agent_id] = ExperimentFreeze(
                    freeze_visit=[trainrun_waypoint.waypoint for trainrun_waypoint in schedule_trainrun],
                    freeze_earliest={trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at
                                     for trainrun_waypoint in schedule_trainrun},
                    freeze_latest={trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at
                                   for trainrun_waypoint in schedule_trainrun},
                    freeze_banned=[waypoint
                                   for waypoint in all_waypoints
                                   if waypoint not in visited],
                )
    # TODO SIM-239 should not be here
    for agent_id in experiment_freeze_dict:
        verify_experiment_freeze_for_agent(
            agent_id=agent_id,
            topo=topo_dict[agent_id],
            experiment_freeze=experiment_freeze_dict[agent_id],
            force_freeze=force_freeze[agent_id],
            malfunction=malfunction if malfunction.agent_id == agent_id else None,
            scheduled_trainrun=list(
                filter(lambda trainrun_waypoint: trainrun_waypoint.scheduled_at <= malfunction.time_step,
                       schedule_trainruns[agent_id]))
        )
    return ScheduleProblemDescription(
        experiment_freeze_dict=experiment_freeze_dict,
        topo_dict=topo_dict,
        minimum_travel_time_dict=minimum_travel_time_dict,
        max_episode_steps=latest_arrival
    )


def get_freeze_for_full_rescheduling(malfunction: ExperimentMalfunction,
                                     schedule_trainruns: TrainrunDict,
                                     minimum_travel_time_dict: Dict[int, int],
                                     topo_dict: Dict[int, nx.DiGraph],
                                     latest_arrival: int
                                     ) -> ScheduleProblemDescription:
    """Returns the experiment freeze for the full re-scheduling problem. Wraps
    the generic freeze by freezing everything up to and including the
    malfunction.

    See param description there.
    """
    return generic_experiment_freeze_for_rescheduling(
        malfunction=malfunction,
        schedule_trainruns=schedule_trainruns,
        minimum_travel_time_dict=minimum_travel_time_dict,
        force_freeze={agent_id: [trainrun_waypoint
                                 for trainrun_waypoint in schedule_trainrun
                                 if trainrun_waypoint.scheduled_at <= malfunction.time_step
                                 ]
                      for agent_id, schedule_trainrun in schedule_trainruns.items()
                      },
        topo_dict=topo_dict,
        latest_arrival=latest_arrival
    )


def _generic_experiment_freeze_for_rescheduling_agent_while_running(
        minimum_travel_time: int,
        topo: nx.DiGraph,
        force_freeze: List[TrainrunWaypoint],
        subdag_source: TrainrunWaypoint,
        latest_arrival: int

) -> ExperimentFreeze:
    """Construct route DAG constraints for this agent. Consider only case where
    malfunction happens during schedule or if there is a (force freeze from the
    oracle).

    Parameters
    ----------

    minimum_travel_time
        the constant cell running time of trains
    agent_paths
        the paths spanning the agent's route DAG.
    force_freeze
        vertices that need be visited and be visited at the given time
    subdag_source
        the entry point into the dag that needs to be visited (the vertex after malfunction that is delayed)
    subdag_targets

    Returns
    -------
    """

    # force freeze in Delta re-scheduling
    force_freeze_dict = {trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at
                         for trainrun_waypoint in force_freeze}
    force_freeze_waypoints_set = {trainrun_waypoint.waypoint for trainrun_waypoint in force_freeze}

    # remove duplicates but deterministc (hashes of dict)
    all_waypoints: List[Waypoint] = topo.nodes

    # span a sub-dag for the problem
    # - for full scheduling (TODO SIM-239), this is source vertex and time 0
    # - for full re-scheduling, this is the next waypoint after the malfunction (delayed for the agent in malfunction)
    # - for delta re-scheduling, if the Oracle tells that more can be freezed than up to malfunction, we use this!
    #   If the force freeze is not contiguous, we need to consider what can be reached given the freezes.

    freeze_visit = []
    freeze_visit_waypoint_set: Set[Waypoint] = set()

    reachable_earliest_dict: [Waypoint, int] = OrderedDict()
    reachable_latest_dict: [Waypoint, int] = OrderedDict()

    def _remove_from_reachable(waypoint):
        # design choice: we give no earliest/latest for banned!
        if waypoint in reachable_earliest_dict:
            reachable_earliest_dict.pop(waypoint)
        if waypoint in reachable_latest_dict:
            reachable_latest_dict.pop(waypoint)

    # sub dag source must be visited (point after malfunction)
    freeze_visit.append(subdag_source.waypoint)
    freeze_visit_waypoint_set.add(subdag_source.waypoint)
    reachable_earliest_dict[subdag_source.waypoint] = subdag_source.scheduled_at

    # there may be multiple vertices by which the last cell may be entered!
    sinks = get_sinks_for_topo(topo)
    for sink in sinks:
        # TODO SIM-239 we should remove this logic here!
        # -1 for occupying the cell for one time step!
        reachable_latest_dict[sink] = latest_arrival - 1

    for trainrun_waypoint in force_freeze:
        reachable_earliest_dict[trainrun_waypoint.waypoint] = trainrun_waypoint.scheduled_at
        reachable_latest_dict[trainrun_waypoint.waypoint] = trainrun_waypoint.scheduled_at
        freeze_visit.append(trainrun_waypoint.waypoint)
        freeze_visit_waypoint_set.add(trainrun_waypoint.waypoint)

    reachable_set = _get_reachable_given_frozen_set(force_freeze=force_freeze, topo=topo)
    reachable_set.add(subdag_source.waypoint)
    for trainrun_waypoint in force_freeze:
        reachable_set.add(trainrun_waypoint.waypoint)

    # ban all that are not reachable in topology
    banned, banned_set = _collect_banned_as_not_reached(all_waypoints, force_freeze_waypoints_set, reachable_set)
    # design choice: we give no earliest/latest for banned!
    for waypoint in banned_set:
        _remove_from_reachable(waypoint)

    # collect earliest and latest in the sub-DAG
    # N.B. we cannot move along paths since this we the order would play a role (SIM-260)
    _propagate_earliest(banned_set, reachable_earliest_dict, force_freeze_dict, minimum_travel_time, subdag_source,
                        topo)
    _propagate_latest(banned_set, force_freeze_dict, latest_arrival, reachable_latest_dict, minimum_travel_time, topo)

    # ban all waypoints that are reachable in the toplogy but not in time (i.e. where earliest > latest)
    for waypoint in all_waypoints:
        if (waypoint not in reachable_earliest_dict or waypoint not in reachable_latest_dict or  # noqa: W504
            reachable_earliest_dict[waypoint] > reachable_latest_dict[waypoint]) \
                and waypoint not in banned_set:
            banned.append(waypoint)
            banned_set.add(waypoint)
            _remove_from_reachable(waypoint)

    return ExperimentFreeze(
        freeze_visit=freeze_visit,
        freeze_earliest=reachable_earliest_dict,
        freeze_banned=banned,
        freeze_latest=reachable_latest_dict
    )


def _collect_banned_as_not_reached(all_waypoints: List[Waypoint],
                                   force_freeze_waypoints_set: Set[Waypoint],
                                   reachable_set: Set[Waypoint]):
    """Bans all that are not either in the forward or backward funnel of the
    freezed ones.

    Returns them as list for iteration and as set for containment test.
    """
    banned: List[Waypoint] = []
    banned_set: Set[Waypoint] = set()
    for waypoint in all_waypoints:
        if waypoint not in reachable_set:
            banned.append(waypoint)
            banned_set.add(waypoint)
            assert waypoint not in force_freeze_waypoints_set, f"{waypoint}"

    return banned, banned_set


def _get_reachable_given_frozen_set(topo: nx.DiGraph,
                                    force_freeze: List[TrainrunWaypoint]) -> Set[Waypoint]:
    """Determines which vertices can still be reached given the frozen set. We
    take all funnels forward and backward from these points and then the
    intersection of those. A source and sink node only have a forward and
    backward funnel, respectively. In FLATland, the source node is always
    unique, the sink node is made unique by a dummy node at the end (the agent
    may enter from more than one direction ino the target cell.)

    Parameters
    ----------
    topo
        directed graph
    force_freeze
        the waypoints that must be visited

    Returns
    -------
    """
    forward_reachable = {waypoint: set() for waypoint in topo.nodes}
    backward_reachable = {waypoint: set() for waypoint in topo.nodes}

    # collect descendants and ancestors of freeze
    for trainrun_waypoint in force_freeze:
        forward_reachable[trainrun_waypoint.waypoint] = set(nx.descendants(topo, trainrun_waypoint.waypoint))
        backward_reachable[trainrun_waypoint.waypoint] = set(nx.ancestors(topo, trainrun_waypoint.waypoint))

    # reflexivity: add waypoint to its own closure (needed for building reachable_set below)
    for waypoint in topo.nodes:
        forward_reachable[waypoint].add(waypoint)
        backward_reachable[waypoint].add(waypoint)

    # reachable are only those that are either in the forward or backward "funnel" of all force freezes!
    reachable_set = set(topo.nodes)
    for trainrun_waypoint in force_freeze:
        waypoint = trainrun_waypoint.waypoint
        forward_and_backward_reachable = forward_reachable[waypoint].union(backward_reachable[waypoint])
        reachable_set.intersection_update(forward_and_backward_reachable)
    return reachable_set


def _propagate_latest(banned_set: Set[Waypoint],
                      force_freeze_dict: Dict[Waypoint, int],
                      latest_arrival: int,
                      latest_dict: Dict[Waypoint, int],
                      minimum_travel_time: int,
                      topo: nx.DiGraph):
    """Extract latest by moving backwards from sinks. Latest time for the agent
    to pass here in order to reach the target in time.

    Parameters
    ----------
    banned_set
    force_freeze_dict
    latest_arrival
    latest_dict
    minimum_travel_time
    topo
    """
    # iterate as long as there are updates (not optimized!)
    # update max(latest_at_next_node-minimum_travel_time,current_latest) until fixed point reached
    done = False
    while not done:
        done = True
        for waypoint in topo.nodes:
            # https://networkx.github.io/documentation/stable/reference/classes/generated/networkx.DiGraph.predecessors.html
            for predecessor in topo.predecessors(waypoint):
                if predecessor in force_freeze_dict or predecessor in banned_set:
                    continue
                else:
                    # TODO SIM-239 make ticket for hard-coded parts
                    # minimum travel time is 1 (synchronization step) if we're goint to the sink
                    minimum_travel_time_corrected = minimum_travel_time
                    if topo.out_degree[waypoint] == 0:
                        minimum_travel_time_corrected = 1

                    path_latest = latest_dict.get(waypoint, -np.inf) - minimum_travel_time_corrected
                    latest = max(path_latest, latest_dict.get(predecessor, -np.inf))
                    if latest < latest_arrival and latest > -np.inf:
                        latest = int(latest)
                        if latest_dict.get(predecessor, None) != latest:
                            done = False
                        latest_dict[predecessor] = latest
    return latest_dict


def _propagate_earliest(banned_set: Set[Waypoint],
                        earliest_dict: Dict[Waypoint, int],
                        force_freeze_dict: Dict[Waypoint, int],
                        minimum_travel_time: int,
                        subdag_source: TrainrunWaypoint,
                        topo: nx.DiGraph):
    """Extract earliest times at nodes by moving forward from source. Earliest
    time for the agent to reach this vertex given the freezed times.

    Parameters
    ----------
    banned_set
    earliest_dict
    force_freeze_dict
    minimum_travel_time
    subdag_source
    topo
    """
    # iterate as long as there are updates (not optimized!)
    # update as min(earliest_at_predecessor+minimum_travel_time,current_earliest) until fixed point reached
    done = False
    while not done:
        done = True
        for waypoint in topo.nodes:
            # https://networkx.github.io/documentation/stable/reference/classes/generated/networkx.DiGraph.predecessors.html
            for successor in topo.successors(waypoint):
                if successor in force_freeze_dict or successor in banned_set:
                    continue
                else:
                    # TODO SIM-239 ticket machen oder erledigen
                    # TODO SIM-XXX put minimum_travel_time per edge and add dumy edges for source and sink as input
                    # add +1 for
                    minimum_travel_time_here = minimum_travel_time

                    # minimum travel time is 1 (synchronization step) to one if we're coming from the source
                    if topo.in_degree[waypoint] == 0:
                        minimum_travel_time_here = 1
                    path_earliest = earliest_dict.get(waypoint, np.inf) + minimum_travel_time_here
                    earliest = min(path_earliest, earliest_dict.get(successor, np.inf))
                    if earliest > subdag_source.scheduled_at and earliest < np.inf:
                        earliest = int(earliest)
                        if earliest_dict.get(successor, None) != earliest:
                            done = False
                        earliest_dict[successor] = earliest
    return earliest_dict


def _get_delayed_trainrun_waypoint_after_malfunction(
        agent_id: int,
        trainrun: Trainrun,
        malfunction: ExperimentMalfunction) -> TrainrunWaypoint:
    """Returns the trainrun waypoint after the malfunction that needs to be re.

    Parameters
    ----------
    agent_id
    trainrun
    malfunction

    Returns
    -------
    """
    previous_scheduled = 0
    for trainrun_waypoint in trainrun:
        if trainrun_waypoint.scheduled_at > malfunction.time_step:
            if agent_id == malfunction.agent_id:
                return TrainrunWaypoint(
                    waypoint=trainrun_waypoint.waypoint,
                    # TODO + 1?
                    scheduled_at=previous_scheduled + malfunction.malfunction_duration + 1)
            else:
                # TODO may this be to pessimistic? should earliest be previous_scheduled + minimum_running_time?
                return trainrun_waypoint
        previous_scheduled = trainrun_waypoint.scheduled_at
    return trainrun[-1]


def verify_experiment_freeze_for_agent(
        agent_id: int,
        experiment_freeze: ExperimentFreeze,
        topo: nx.DiGraph,
        force_freeze: Optional[List[TrainrunWaypoint]] = None,
        malfunction: Optional[ExperimentMalfunction] = None,
        scheduled_trainrun: Optional[Trainrun] = None
):
    """Does the experiment_freeze reflect the force freeze, route DAG and
    malfunctions correctly?

    Parameters
    ----------
    scheduled_trainrun
    experiment_freeze
        the experiment freeze to be verified.
    topo
        the route DAG
    force_freeze
        the trainrun waypoints that must as given (consistency is not checked!)
    malfunction
        if it's the agent in malfunction, the experiment freeze should put a visit and earliest constraint
    scheduled_trainrun
        verify that this whole train run is part of the solution space.
        With malfunctions, caller must ensure that only relevant part is passed to be verified!

    Returns
    -------
    """

    all_waypoints = topo.nodes
    for waypoint in all_waypoints:
        # if waypoint is banned -> must not earliest/latest/visit
        if waypoint in experiment_freeze.freeze_banned:
            assert waypoint not in experiment_freeze.freeze_earliest, \
                f"agent {agent_id}: {waypoint} banned, should have no earliest"
            assert waypoint not in experiment_freeze.freeze_latest, \
                f"agent {agent_id}: {waypoint} banned, should have no latest"
            assert waypoint not in experiment_freeze.freeze_visit, \
                f"agent {agent_id}: {waypoint} banned, should have no visit"
        else:
            # waypoint must have earliest and latest s.t. earliest <= latest
            assert waypoint in experiment_freeze.freeze_earliest, \
                f"agent {agent_id} has no earliest for {waypoint}"
            assert waypoint in experiment_freeze.freeze_latest, \
                f"agent {agent_id} has no latest for {waypoint}"
            assert experiment_freeze.freeze_earliest[waypoint] <= experiment_freeze.freeze_latest[waypoint], \
                f"agent {agent_id} at {waypoint}: earliest should be less or equal to latest, " + \
                f"found {experiment_freeze.freeze_earliest[waypoint]} <= {experiment_freeze.freeze_latest[waypoint]}"

    # verify that force is implemented correctly
    if force_freeze:
        for trainrun_waypoint in force_freeze:
            assert experiment_freeze.freeze_latest[trainrun_waypoint.waypoint] == trainrun_waypoint.scheduled_at, \
                f"agent {agent_id}: should have latest requirement " \
                f"for {trainrun_waypoint.waypoint} at {trainrun_waypoint.scheduled_at} - " \
                f"found {experiment_freeze.freeze_latest[trainrun_waypoint.waypoint]}"
            assert experiment_freeze.freeze_earliest[trainrun_waypoint.waypoint] == trainrun_waypoint.scheduled_at, \
                f"agent {agent_id}: should have earliest requirement " \
                f"for {trainrun_waypoint.waypoint} at {trainrun_waypoint.scheduled_at} - " \
                f"found {experiment_freeze.freeze_earliest[trainrun_waypoint.waypoint]}"
            assert trainrun_waypoint.waypoint in experiment_freeze.freeze_visit, \
                f"agent {agent_id}: should have visit requirement " \
                f"for {trainrun_waypoint.waypoint}"
            assert trainrun_waypoint.waypoint not in experiment_freeze.freeze_banned, \
                f"agent {agent_id}: should have no banned requirement " \
                f"for {trainrun_waypoint.waypoint}"

    # verify that all points up to malfunction are forced to be visited
    if malfunction:
        for waypoint, earliest in experiment_freeze.freeze_earliest.items():
            # everything before malfunction must be the same
            if earliest <= malfunction.time_step:
                assert experiment_freeze.freeze_latest[waypoint] == earliest
                assert waypoint in experiment_freeze.freeze_visit
            else:
                assert earliest >= malfunction.time_step + malfunction.malfunction_duration, f"{agent_id} {malfunction}"

    # verify that scheduled train run is in the solution space
    if scheduled_trainrun:
        scheduled_dict = {trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at
                          for trainrun_waypoint in scheduled_trainrun}
        for waypoint, scheduled_at in scheduled_dict.items():
            assert waypoint not in experiment_freeze.freeze_banned, \
                f"agent {agent_id}: the known solution has " \
                f"schedule {waypoint} at {scheduled_at} - " \
                f"but banned constraint"
            assert waypoint in experiment_freeze.freeze_earliest, \
                f"agent {agent_id}: the known solution has " \
                f"schedule {waypoint} at {scheduled_at} - " \
                f"but no earliest constraint"
            assert scheduled_at >= experiment_freeze.freeze_earliest[waypoint], \
                f"agent {agent_id}: the known solution has " \
                f"schedule {waypoint} at {scheduled_at} - " \
                f"but found earliest {experiment_freeze.freeze_latest[waypoint]}"
            assert waypoint in experiment_freeze.freeze_latest, \
                f"agent {agent_id}: the known solution has " \
                f"schedule {waypoint} at {scheduled_at} - " \
                f"but no latest constraint"
            assert scheduled_at <= experiment_freeze.freeze_latest[waypoint], \
                f"agent {agent_id}: the known solution has " \
                f"schedule {waypoint} at {scheduled_at} - " \
                f"but found latest {experiment_freeze.freeze_latest[waypoint]}"
            assert waypoint not in experiment_freeze.freeze_banned
