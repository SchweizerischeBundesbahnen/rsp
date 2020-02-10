from collections import OrderedDict
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

import numpy
from flatland.envs.rail_trainrun_data_structures import Trainrun
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.utils.data_types import AgentPaths
from rsp.utils.data_types import AgentsPathsDict
from rsp.utils.data_types import ExperimentFreeze
from rsp.utils.data_types import ExperimentFreezeDict
from rsp.utils.data_types import experimentFreezePrettyPrint
from rsp.utils.data_types import ExperimentMalfunction


def generic_experiment_freeze_for_rescheduling(
        schedule_trainruns: TrainrunDict,
        minimum_travel_time_dict: Dict[int, int],
        agents_path_dict: AgentsPathsDict,
        force_freeze: Dict[int, List[TrainrunWaypoint]],
        malfunction: ExperimentMalfunction,
        latest_arrival: int
) -> ExperimentFreezeDict:
    """Derives the experiment freeze given the malfunction and optionally a
    force freeze from an Oracle.

    Parameters
    ----------
    schedule_trainruns
        the schedule before the malfunction
    minimum_travel_time_dict
        the agent's speed (constant for every agent, different among agents)
    agents_path_dict
        the paths spanning the agents' route DAG
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
            agent_paths=agents_path_dict[agent_id],
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
                    freeze_earliest=_get_earliest_entries_for_full_route_dag(
                        minimum_travel_time=minimum_travel_time_dict[agent_id],
                        agent_paths=agents_path_dict[agent_id],
                        earliest=schedule_trainrun[0].scheduled_at
                    ),
                    freeze_latest=_get_latest_entries_for_full_route_dag(
                        minimum_travel_time=minimum_travel_time_dict[agent_id],
                        agent_paths=agents_path_dict[agent_id],
                        latest=latest_arrival
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
                all_waypoints = [
                    waypoint
                    for agent_path in agents_path_dict[agent_id]
                    for waypoint in agent_path
                ]
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

    for agent_id in experiment_freeze_dict:
        verify_experiment_freeze_for_agent(
            agent_id=agent_id,
            agent_paths=agents_path_dict[agent_id],
            experiment_freeze=experiment_freeze_dict[agent_id],
            force_freeze=force_freeze[agent_id],
            malfunction=malfunction if malfunction.agent_id == agent_id else None,
            scheduled_trainrun=list(
                filter(lambda trainrun_waypoint: trainrun_waypoint.scheduled_at <= malfunction.time_step,
                       schedule_trainruns[agent_id]))
        )
        # TODO SIM-241 remove tweaky debg snippet as soon as pipeline is stable
        # uncomment the following lines for debugging purposes
        if False:
            print("experimentFreezePrettyPrint(experiment_freeze_dict[8]) generic rsp")
            experimentFreezePrettyPrint(experiment_freeze_dict[8])

    return experiment_freeze_dict


def _generic_experiment_freeze_for_rescheduling_agent_while_running(
        minimum_travel_time: int,
        agent_paths: AgentPaths,
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

    Returns
    -------
    """

    # force freeze in Delta re-scheduling
    force_freeze_dict = {trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at
                         for trainrun_waypoint in force_freeze}
    force_freeze_waypoints_set = {trainrun_waypoint.waypoint for trainrun_waypoint in force_freeze}

    # remove duplicates but deterministc (hashes of dict)
    all_waypoints: List[Waypoint] = {waypoint: waypoint for agent_path in agent_paths for waypoint in agent_path}.keys()

    # span a sub-dag for the problem
    # - for full scheduling (TODO SIM-173), this is source vertex and time 0
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
    for agent_path in agent_paths:
        # -1 for occupying the cell for one time step!
        reachable_latest_dict[agent_path[-1]] = latest_arrival - 1

    for trainrun_waypoint in force_freeze:
        reachable_earliest_dict[trainrun_waypoint.waypoint] = trainrun_waypoint.scheduled_at
        reachable_latest_dict[trainrun_waypoint.waypoint] = trainrun_waypoint.scheduled_at
        freeze_visit.append(trainrun_waypoint.waypoint)
        freeze_visit_waypoint_set.add(trainrun_waypoint.waypoint)

    # forward and backward funnels
    # We take all funnels forward and backward from these points and then the intersection of those.
    # A source and sink node only have a forward and backward funnel, respectively.
    # In FLATland, the source node is always unique, the sink node is made unique by a dummy node at the end
    # (the agent may enter from more than one direction ino the target cell.)
    reachable_set = _get_reachable_given_frozen_set(agent_paths, all_waypoints, force_freeze)

    # ban all that are not reachable
    banned, banned_set = _collect_banned_as_not_reached(all_waypoints, force_freeze_waypoints_set, reachable_set)
    # design choice: we give no earliest/latest for banned!
    for waypoint in banned_set:
        _remove_from_reachable(waypoint)

    # build predecessor and successor nodes
    predecessors, successors = _build_predecessors_successors(agent_paths)
    # collect earliest and latest in the sub-DAG
    # N.B. we cannot move along paths since this we the order would play a role (SIM-260)
    _add_agent_for_get_freeze_for_delta(
        predecessors,
        successors,
        reachable_earliest_dict,
        reachable_latest_dict,
        force_freeze_dict,
        banned_set,
        subdag_source,
        latest_arrival,
        minimum_travel_time,
    )
    # banned all waypoints not reachable from both source and sink or where earliest > latest
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


def _build_predecessors_successors(agent_paths: AgentPaths):
    predecessors = {waypoint: set() for agent_path in agent_paths for waypoint in agent_path}
    successors = {waypoint: set() for agent_path in agent_paths for waypoint in agent_path}
    for agent_path in agent_paths:
        for first, second in zip(agent_path, agent_path[1:]):
            predecessors[second].add(first)
            successors[first].add(second)
    return predecessors, successors


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


def _get_reachable_given_frozen_set(agent_paths: AgentPaths,
                                    all_waypoints: List[Waypoint],
                                    force_freeze: List[TrainrunWaypoint]) -> Set[Waypoint]:
    """Determines which vertices can still be reached given the frozen set.

    Parameters
    ----------
    agent_paths
        paths that span the agent's route DAG
    all_waypoints
        all waypoints in the route DAG
    force_freeze
        the waypoints that must be visited

    Returns
    -------
    """
    # collect all direct neighbors
    forward_reachable: Dict[Waypoint, Set[Waypoint]] = {waypoint: set() for agent_path in agent_paths for waypoint in
                                                        agent_path}
    backward_reachable: Dict[Waypoint, Set[Waypoint]] = {waypoint: set() for agent_path in agent_paths for waypoint in
                                                         agent_path}
    for agent_path in agent_paths:
        for first, second in zip(agent_path, agent_path[1:]):
            forward_reachable[first].add(second)
            backward_reachable[second].add(first)

    # transitive closure of forward and backward neighbors
    # i.e. iteratively add the neighbors, then neighbors' neighbors, then the neighbors' neighbors' neighbors...
    done = False
    forward_neighbors_count: Dict[Waypoint, int] = {waypoint: len(forward_reachable[waypoint]) for agent_path in
                                                    agent_paths for waypoint in agent_path}
    backward_neighbors_count: Dict[Waypoint, int] = {waypoint: len(backward_reachable[waypoint]) for agent_path in
                                                     agent_paths for waypoint in agent_path}
    while not done:
        for waypoint in all_waypoints:
            # put into frozenset to prevent concurrent update
            frozen_forward_neighbors = frozenset(forward_reachable[waypoint])
            for forward_neighbor in frozen_forward_neighbors:
                forward_reachable[waypoint].update(forward_reachable[forward_neighbor])

        for waypoint in all_waypoints:
            # put into frozenset to prevent concurrent update
            frozen_backward_neighbors = frozenset(backward_reachable[waypoint])
            for backward_neighbor in frozen_backward_neighbors:
                backward_reachable[waypoint].update(backward_reachable[backward_neighbor])

        next_forward_neighbors_count: Dict[Waypoint:int] = {waypoint: len(forward_reachable[waypoint]) for agent_path in
                                                            agent_paths for waypoint in agent_path}
        next_backward_neighbors_count: Dict[Waypoint:int] = {waypoint: len(backward_reachable[waypoint]) for agent_path
                                                             in agent_paths for waypoint in agent_path}
        # done if no updates
        done = forward_neighbors_count == next_forward_neighbors_count and backward_neighbors_count == next_backward_neighbors_count
        forward_neighbors_count = next_forward_neighbors_count
        backward_neighbors_count = next_backward_neighbors_count

    # reflexivity: add waypoint to its own closure (needed for building reachable_set below)
    for waypoint in all_waypoints:
        forward_reachable[waypoint].add(waypoint)
        backward_reachable[waypoint].add(waypoint)

    # reachable are only those that are either in the forward or backward "funnel" of all force freezes!
    reachable_set = set(all_waypoints)
    for trainrun_waypoint in force_freeze:
        waypoint = trainrun_waypoint.waypoint
        forward_and_backward_reachable = forward_reachable[waypoint].union(backward_reachable[waypoint])
        reachable_set.intersection_update(forward_and_backward_reachable)
    return reachable_set


def _search_last_contiguously_freezed_from_start(
        agent_paths: AgentPaths,
        force_freeze: List[TrainrunWaypoint],
        force_freeze_waypoints_set: Set[TrainrunWaypoint],
) -> Optional[TrainrunWaypoint]:
    """Searches the last freezed waypoint such that all waypoints leading there
    are all freezed.

    Parameters
    ----------
    agent_paths
    force_freeze
    force_freeze_waypoints_set

    Returns
    -------
    """
    last_forced_from_start: Optional[TrainrunWaypoint] = None
    for agent_path in agent_paths:
        # if not all freezed are on this path, this path cannot be chosen!
        agent_path_set = set(agent_path)

        if not force_freeze_waypoints_set.issubset(agent_path_set):
            continue
        for index, force_freeze_waypoint in enumerate(force_freeze):
            if force_freeze_waypoint.waypoint != agent_path[index]:
                break  # go to next agent_path
            last_forced_from_start = force_freeze_waypoint
    return last_forced_from_start


def _add_agent_for_get_freeze_for_delta(predecessors: Dict[Waypoint, Set[Waypoint]],
                                        successors: Dict[Waypoint, Set[Waypoint]],
                                        earliest_dict: Dict[Waypoint, int],
                                        latest_dict: Dict[Waypoint, int],
                                        force_freeze_dict: Dict[Waypoint, int],
                                        banned_set: Set[Waypoint],
                                        subdag_source: TrainrunWaypoint,
                                        latest_arrival: int,
                                        minimum_travel_time):
    """Traverse the sub-DAG along one of the paths generating the route DAG and
    update earliest and latest times. Used within
    _generic_experiment_freeze_for_rescheduling_agent_while_running.

    Parameters
    ----------
    predecessors: Dict[Waypoint,Set[Waypoint]]
        predecessor of the agent's waypoints
    successors: Dict[Waypoint,Set[Waypoint]]
        successors of the agent's waypoints
    earliest_dict
        earliest time for the agent to reach this vertex given the freezed times
    latest_dict
        latest time for the agent to pass here in order to reach the target in time
    force_freeze_dict
        vertices that must be visited at the given time
    minimum_travel_time
        constant travel time per edge for the agent (may be different for other agents)
    banned_set
        waypoints that the agent must not visit (as told by the oracle)
    subdag_source
        the waypoint after the malfunction and when it can be there at the earliest
        (everything before this point must be in the force_freeze)
    latest_arrival
        latest arrival at the sink

    Returns
    -------
        updates earliest_dict and latest_dict
    """
    _extract_earliest(banned_set, earliest_dict, force_freeze_dict, minimum_travel_time, subdag_source, successors)
    _extract_latest(banned_set, force_freeze_dict, latest_arrival, latest_dict, minimum_travel_time, predecessors,
                    successors)


def _extract_latest(banned_set, force_freeze_dict, latest_arrival, latest_dict, minimum_travel_time, predecessors,
                    successors):
    # iterate as long as there are updates (not optimized!)
    # update max(latest_at_next_node-minimum_travel_time,current_latest) until fixed point reached
    done = False
    while not done:
        done = True
        for waypoint in successors:
            for predecessor in predecessors[waypoint]:
                if predecessor in force_freeze_dict or predecessor in banned_set:
                    continue
                else:
                    path_latest = latest_dict.get(waypoint, -numpy.inf) - minimum_travel_time
                    latest = max(path_latest, latest_dict.get(predecessor, -numpy.inf))
                    if latest < latest_arrival and latest > -numpy.inf:
                        latest = int(latest)
                        if latest_dict.get(predecessor, None) != latest:
                            done = False
                        latest_dict[predecessor] = latest


def _extract_earliest(banned_set, earliest_dict, force_freeze_dict, minimum_travel_time, subdag_source, successors):
    # iterate as long as there are updates (not optimized!)
    # update as min(earliest_at_predecessor+minimum_travel_time,current_earliest) until fixed point reached
    done = False
    while not done:
        done = True
        for waypoint in successors:
            for successor in successors[waypoint]:
                if successor in force_freeze_dict or successor in banned_set:
                    continue
                else:
                    path_earliest = earliest_dict.get(waypoint, numpy.inf) + minimum_travel_time
                    earliest = min(path_earliest, earliest_dict.get(successor, numpy.inf))
                    if earliest > subdag_source.scheduled_at and earliest < numpy.inf:
                        earliest = int(earliest)
                        if earliest_dict.get(successor, None) != earliest:
                            done = False
                        earliest_dict[successor] = earliest


def _get_delayed_trainrun_waypoint_after_malfunction(
        agent_id: int,
        trainrun: Trainrun,
        malfunction: ExperimentMalfunction) -> TrainrunWaypoint:
    for trainrun_waypoint in trainrun:
        if trainrun_waypoint.scheduled_at > malfunction.time_step:
            if agent_id == malfunction.agent_id:
                return TrainrunWaypoint(
                    waypoint=trainrun_waypoint.waypoint,
                    scheduled_at=trainrun_waypoint.scheduled_at + malfunction.malfunction_duration)
            else:
                return trainrun_waypoint
    return trainrun[-1]


def get_freeze_for_full_rescheduling(malfunction: ExperimentMalfunction,
                                     schedule_trainruns: TrainrunDict,
                                     minimum_travel_time_dict: Dict[int, int],
                                     agents_path_dict: AgentsPathsDict,
                                     latest_arrival: int
                                     ) -> ExperimentFreezeDict:
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
        agents_path_dict=agents_path_dict,
        latest_arrival=latest_arrival
    )


# TODO SIM-173 use this for scheduling together with generic entry point!
def _get_earliest_entries_for_full_route_dag(
        minimum_travel_time: int,
        agent_paths: AgentPaths,
        earliest: int
) -> Dict[Waypoint, int]:
    """Given the minimum travel time per cell (constant per agent), derive the
    earliest times in the route DAG spanned by the k shortest paths.

    If a vertex can be reached by multiple paths, take the earliest time it can be reached according to
      earliest(train,vertex) = minimum-number-of-hops-from-source(train,vertex) * minimum_running_time + 1

    N.B. +1 is for the time spent in FLATland in the first cell upon entering the grid.

    Parameters
    ----------
    minimum_travel_time
    agent_paths
    earliest

    Returns
    -------
    """
    earliest_dict: Dict[Waypoint, int] = {}
    for agent_path in agent_paths:
        for waypoint_index, waypoint in enumerate(agent_path):
            if waypoint_index == 0:
                earliest_dict[waypoint] = earliest
            else:
                earliest_dict[waypoint] = int(min(earliest_dict.get(waypoint, numpy.inf),
                                                  earliest + waypoint_index * minimum_travel_time + 1))
    return earliest_dict


def _get_latest_entries_for_full_route_dag(
        minimum_travel_time: int,
        agent_paths: AgentPaths,
        latest: int
) -> Dict[Waypoint, int]:
    """Given the minimum travel time per cell (constant per agent), derive the
    latest times in the route DAG spanned by the k shortest paths.

    If a vertex can be reached backwards by multiple paths, take the latest time it can be reached according to
      latest(train,vertex) = minimum-number-of-hops-to-sink(train,vertex) * minimum_running_time
    """
    latest_dict: Dict[Waypoint, int] = {}
    for agent_path in agent_paths:
        reversed_agent_path = list(reversed(list(agent_path)))
        for waypoint_index, waypoint in enumerate(reversed_agent_path):
            if waypoint_index == 0:
                latest_dict[waypoint] = latest
            else:
                latest_dict[waypoint] = int(max(latest_dict.get(waypoint, -numpy.inf),
                                                latest - waypoint_index * minimum_travel_time))
    return latest_dict


def verify_experiment_freeze_for_agent(
        agent_id: int,
        experiment_freeze: ExperimentFreeze,
        agent_paths: AgentPaths,
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
    agent_paths
        the paths spanning the train's route DAG.
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

    all_waypoints = {waypoint for agent_path in agent_paths for waypoint in agent_path}
    for waypoint in all_waypoints:
        # if waypoint is banned -> must not earliest/latest/visit
        if waypoint in experiment_freeze.freeze_banned:
            assert waypoint not in experiment_freeze.freeze_earliest, \
                f"{agent_id}: {waypoint} banned, should have no earliest"
            assert waypoint not in experiment_freeze.freeze_latest, \
                f"{agent_id}: {waypoint} banned, should have no latest"
            assert waypoint not in experiment_freeze.freeze_visit, \
                f"{agent_id}: {waypoint} banned, should have no visit"
        else:
            # waypoint must have earliest and latest s.t. earliest <= latest
            assert waypoint in experiment_freeze.freeze_earliest, \
                f"{agent_id} has no earliest for {waypoint}"
            assert waypoint in experiment_freeze.freeze_latest, \
                f"{agent_id} has no latest for {waypoint}"
            assert experiment_freeze.freeze_earliest[waypoint] <= experiment_freeze.freeze_latest[waypoint], \
                f"{agent_id} at {waypoint}: earliest should be less or equal to latest, " + \
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
