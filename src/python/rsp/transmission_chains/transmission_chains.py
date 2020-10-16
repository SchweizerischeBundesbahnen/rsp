from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Tuple

import numpy as np
from rsp.step_02_setup.data_types import ExperimentMalfunction
from rsp.utils.data_types import ResourceOccupation
from rsp.utils.data_types import ScheduleAsResourceOccupations
from rsp.utils.data_types import SchedulingProblemInTimeWindows
from rsp.utils.rsp_logger import rsp_logger

TransmissionLeg = NamedTuple("TransmissionLeg", [("hop_on", ResourceOccupation), ("hop_off", ResourceOccupation), ("delay_time", int)])
TransmissionChain = List[TransmissionLeg]

# hop-on resource-occupations reaching at this depth (int key = depth)
WAVE_PER_DEPTH = Dict[int, List[ResourceOccupation]]
# waves reaching per agent (int key = agent_id)
WAVE_PER_AGENT_AND_DEPTH = Dict[int, WAVE_PER_DEPTH]


# TODO remove, not needed currently?


def extract_transmission_chains_from_time_windows(  # noqa: C901
    malfunction: ExperimentMalfunction, time_windows: SchedulingProblemInTimeWindows
) -> List[TransmissionChain]:
    """Derive transmission chains happening by time window overlap.

    Parameters
    ----------
    malfunction
    time_windows

    Returns
    -------
    """
    time_windows_per_agent = time_windows.time_windows_per_agent_sorted_by_lower_bound
    time_windows_per_resource_and_time_step = time_windows.time_windows_per_resource_and_time_step
    malfunction_agent_id = malfunction.agent_id
    open_wave_front: List[TransmissionChain] = []
    transmission_chains: List[TransmissionChain] = []
    closed_wave_front: List[ResourceOccupation] = []
    malfunction_time_window = next(ro for ro in time_windows_per_agent[malfunction_agent_id] if malfunction.time_step < ro.interval.to_excl)
    for hop_on in time_windows_per_agent[malfunction_agent_id]:
        # TODO is this correct?
        if malfunction.time_step < hop_on.interval.to_excl:
            chain = [TransmissionLeg(malfunction_time_window, hop_on, delay_time=-1)]
            open_wave_front.append(chain)
            transmission_chains.append(chain)
    assert len(open_wave_front) > 0
    loop_count = 0
    while len(open_wave_front) > 0:
        loop_count += 1
        if loop_count % 1000 == 0:
            rsp_logger.info(f"{loop_count}: queue has length {len(open_wave_front)}, {len(transmission_chains)} transmission chains already")
        history = open_wave_front.pop()
        wave_front = history[-1].hop_off
        wave_front_resource = wave_front.resource
        if wave_front in closed_wave_front:
            continue
        validate_transmission_chain_time_window(history)
        closed_wave_front.append(wave_front)

        agents_in_chain_so_far = {leg.hop_on.agent_id for leg in history}

        # all trains with overlapping time window may be impacted!
        hop_ons = set()
        for time_step in range(wave_front.interval.from_incl, wave_front.interval.to_excl):
            for hop_on in time_windows_per_resource_and_time_step[(wave_front_resource, time_step)]:
                # cycle detection
                if hop_on.agent_id not in agents_in_chain_so_far:
                    hop_ons.add(hop_on)
        for hop_on in hop_ons:
            for hop_off in time_windows_per_agent[hop_on.agent_id]:
                if hop_off in closed_wave_front:
                    continue
                if hop_off.interval.from_incl <= hop_on.interval.from_incl:
                    continue
                chain = history + [TransmissionLeg(hop_on, hop_off, delay_time=-1)]
                open_wave_front.append(chain)
                transmission_chains.append(chain)
    return transmission_chains


def extract_transmission_chains_from_schedule(malfunction: ExperimentMalfunction, occupations: ScheduleAsResourceOccupations) -> List[TransmissionChain]:
    """Propagation of delay.
    TODO optimize (for instance, we do not check whether that only the largest delay per resource and agent is in the open list.
    Parameters
    ----------

    Returns
    -------
    List of transmission chains
    """

    malfunction_agent_id = malfunction.agent_id
    delay_time = malfunction.malfunction_duration

    resource_occupations_per_agent = occupations.sorted_resource_occupations_per_agent
    resource_occupations_per_resource = occupations.sorted_resource_occupations_per_resource

    open_wave_front: List[Tuple[ResourceOccupation, TransmissionChain]] = []
    transmission_chains: List[TransmissionChain] = []
    closed_wave_front: Dict[ResourceOccupation, int] = {}
    malfunction_occupation = next(ro for ro in resource_occupations_per_agent[malfunction_agent_id] if malfunction.time_step < ro.interval.to_excl)
    for ro in resource_occupations_per_agent[malfunction_agent_id]:
        # N.B. intervals include release times, therefore we can be strict at upper bound!
        if malfunction.time_step < ro.interval.to_excl:
            # TODO should the interval extended by the malfunction duration?
            chain = [TransmissionLeg(malfunction_occupation, ro, delay_time)]
            open_wave_front.append(chain)
            transmission_chains.append(chain)
    assert len(open_wave_front) > 0
    while len(open_wave_front) > 0:
        history = open_wave_front.pop()
        wave_front = history[-1].hop_off
        delay_time = history[-1].delay_time
        wave_front_resource = wave_front.resource

        if delay_time <= closed_wave_front.get(wave_front, 0):
            continue
        closed_wave_front[wave_front] = delay_time

        # the next scheduled train at the same resource may be impacted!
        ro = next(iter([ro for ro in resource_occupations_per_resource[wave_front_resource] if ro.interval.from_incl >= wave_front.interval.to_excl]), None)
        if ro is not None:
            time_between_agents = ro.interval.from_incl - wave_front.interval.to_excl

            if time_between_agents < delay_time:
                remaining_delay_time = delay_time - time_between_agents
                assert remaining_delay_time >= 0
                impact_distance_from_wave_front = delay_time - remaining_delay_time
                assert impact_distance_from_wave_front >= 0
                # the propagation may flow backwards!
                for subsequent_ro in [
                    subsequent_ro
                    for subsequent_ro in resource_occupations_per_agent[ro.agent_id]
                    if subsequent_ro.interval.from_incl >= ro.interval.from_incl - remaining_delay_time
                ]:
                    chain = history + [TransmissionLeg(ro, subsequent_ro, remaining_delay_time)]
                    assert subsequent_ro not in history
                    if ro in closed_wave_front and closed_wave_front[ro] > remaining_delay_time:
                        # already processed with larger delay
                        continue
                    open_wave_front.append(chain)
                    transmission_chains.append(chain)
    return transmission_chains


def validate_transmission_chains(transmission_chains: List[TransmissionChain]):
    """Check that.

    - transmission legs of same agent
    - hop_off after hop_on
    - consecutive legs between different agents
    - consecutive legs meet at the same resource
    # TODO check for non-circularity?
    Parameters
    ----------
    transmission_chains
    """
    for transmission_chain in transmission_chains:
        for transmission in transmission_chain:
            # leg must be at the same agent
            assert transmission.hop_on.agent_id == transmission.hop_off.agent_id, transmission

            # N.B. because of backwards propagation, the hop_on interval can to the left
        for tr1, tr2 in zip(transmission_chain, transmission_chain[1:]):
            # transmission between different agents
            assert tr1.hop_off.agent_id != tr2.hop_off.agent_id, (tr1, tr2)

            # N.B. because of backwards propagation, next hop_on needs not be at same resource and next hop_on needs not be later


def validate_transmission_chain_time_window(transmission_chain: TransmissionChain):
    for transmission in transmission_chain:
        # leg must be at the same agent
        assert transmission.hop_on.agent_id == transmission.hop_off.agent_id, transmission

        # interval of second interval must start later
        assert transmission.hop_off.interval.from_incl >= transmission.hop_on.interval.from_incl, transmission
    for tr1, tr2 in zip(transmission_chain, transmission_chain[1:]):
        # transmission between different agents
        assert tr1.hop_off.agent_id != tr2.hop_off.agent_id, (tr1, tr2)

        # transmission via same resource
        assert tr2.hop_on.resource == tr1.hop_off.resource, (tr1, tr2)

        # transmission via time window overlap at the same resource
        assert (
            (tr1.hop_off.interval.from_incl <= tr2.hop_on.interval.from_incl <= tr1.hop_off.interval.to_excl)
            or (tr2.hop_on.interval.from_incl <= tr1.hop_off.interval.from_incl <= tr2.hop_on.interval.to_excl)
            or (tr2.hop_on.interval.from_incl <= tr1.hop_off.interval.from_incl and tr2.hop_on.interval.to_excl >= tr1.hop_off.interval.to_excl)
            or (tr2.hop_on.interval.from_incl >= tr1.hop_off.interval.from_incl and tr2.hop_on.interval.to_excl <= tr1.hop_off.interval.to_excl)
        ), (tr1, tr2)


def distance_matrix_from_tranmission_chains(
    number_of_trains: int, transmission_chains: List[TransmissionChain]
) -> Tuple[np.ndarray, Dict[int, int], Dict[int, Dict[int, List[ResourceOccupation]]]]:
    """

    Parameters
    ----------
    number_of_trains
    transmission_chains

    Returns
    -------
    distance_matrix, minimal_depth, wave_fronts_reaching_other_agent

    """

    # non-symmetric distance_matrix: insert distance between two consecutive trains at a resource; if no direct encounter, distance is inf
    distance_matrix = np.full(shape=(number_of_trains, number_of_trains), fill_value=np.inf)
    distance_first_reaching = np.full(shape=(number_of_trains, number_of_trains), fill_value=np.inf)
    # for each agent, wave_front reaching the other agent from malfunction agent
    wave_reaching_other_agent: WAVE_PER_AGENT_AND_DEPTH = {agent_id: {} for agent_id in range(number_of_trains)}
    # for each agent, minimum transmission length reaching the other agent from malfunction agent
    minimal_depth: Dict[int, int] = {}
    minimal_depth[transmission_chains[0][0].hop_off.agent_id] = 0
    for transmission_chain in transmission_chains:
        if len(transmission_chain) < 2:
            # skip transmission chains consisting of only one leg (along the malfunction agent's path)
            continue
        from_ro = transmission_chain[-2].hop_off
        to_ro = transmission_chain[-1].hop_on
        hop_on_depth = len(transmission_chain) - 1
        wave_reaching_other_agent[to_ro.agent_id].setdefault(hop_on_depth, []).append(to_ro)
        minimal_depth.setdefault(to_ro.agent_id, hop_on_depth)
        minimal_depth[to_ro.agent_id] = min(minimal_depth[to_ro.agent_id], hop_on_depth)

        distance = to_ro.interval.from_incl - from_ro.interval.to_excl
        from_agent_id = from_ro.agent_id
        to_agent_id = to_ro.agent_id

        distance_before = distance_matrix[from_agent_id, to_agent_id]

        if distance_before > distance:
            distance_matrix[from_agent_id, to_agent_id] = distance
            distance_first_reaching[from_agent_id, to_agent_id] = to_ro.interval.from_incl
        elif distance_before == distance:
            distance_first_reaching[from_agent_id, to_agent_id] = min(distance_first_reaching[from_agent_id, to_agent_id], to_ro.interval.from_incl)

    return distance_matrix, minimal_depth, wave_reaching_other_agent
