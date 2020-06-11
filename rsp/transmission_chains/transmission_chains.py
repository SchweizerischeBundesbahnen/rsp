from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Tuple

import numpy as np

from rsp.utils.data_types import ResourceOccupation
from rsp.utils.plotting_data_types import SchedulePlotting

TransmissionLeg = NamedTuple('TransmissionLeg', [
    ('hop_on', ResourceOccupation),
    ('hop_off', ResourceOccupation)
])
TransmissionChain = List[TransmissionLeg]

# hop-on resource-occupations reaching at this depth (int key = depth)
WAVE_PER_DEPTH = Dict[int, List[ResourceOccupation]]
# waves reaching per agent (int key = agent_id)
WAVE_PER_AGENT_AND_DEPTH = Dict[int, WAVE_PER_DEPTH]


# TODO we probably too much work here!
def extract_transmission_chains_from_schedule(schedule_plotting: SchedulePlotting) -> List[TransmissionChain]:
    """Propagation of delay.

    Parameters
    ----------
    schedule_plotting

    Returns
    -------
    List of transmission chains
    """

    malfunction = schedule_plotting.malfunction
    malfunction_agent_id = malfunction.agent_id
    delay_time = malfunction.malfunction_duration
    resource_occupations_per_agent = schedule_plotting.schedule_as_resource_occupations.sorted_resource_occupations_per_agent
    resource_occupations_per_resource = schedule_plotting.schedule_as_resource_occupations.sorted_resource_occupations_per_resource

    open_wave_front: List[Tuple[ResourceOccupation, TransmissionChain]] = []
    transmission_chains: List[TransmissionChain] = []
    closed_wave_front: List[ResourceOccupation] = []
    malfunction_occupation = next(ro for ro in resource_occupations_per_agent[malfunction_agent_id] if malfunction.time_step < ro.interval.to_excl)
    for ro in resource_occupations_per_agent[malfunction_agent_id]:
        # N.B. intervals include release times, therefore we can be strict at upper bound!
        if malfunction.time_step < ro.interval.to_excl:
            # TODO should the interval extended by the malfunction duration?
            chain = [TransmissionLeg(malfunction_occupation, ro)]
            open_wave_front.append((ro, chain, delay_time))
            transmission_chains.append(chain)
    assert len(open_wave_front) > 0
    while len(open_wave_front) > 0:
        wave_front, history, delay_time = open_wave_front.pop()
        wave_front_resource = wave_front.resource
        if wave_front in closed_wave_front:
            continue
        closed_wave_front.append(wave_front)

        # the next scheduled train may be impacted!
        for ro in resource_occupations_per_resource[wave_front_resource]:
            time_between_agents = ro.interval.from_incl - wave_front.interval.to_excl
            if ro.interval.from_incl >= wave_front.interval.to_excl and time_between_agents < delay_time:
                delay_time = delay_time - time_between_agents
                for subsequent_ro in resource_occupations_per_agent[ro.agent_id]:
                    # hop_on and hop_off may be at the same resource
                    if subsequent_ro.interval.from_incl >= ro.interval.from_incl:
                        chain = history + [TransmissionLeg(ro, subsequent_ro)]
                        el = (subsequent_ro, chain, delay_time)
                        assert subsequent_ro not in history
                        if ro in closed_wave_front:
                            continue
                        open_wave_front.append(el)
                        transmission_chains.append(chain)
                break
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
            assert transmission.hop_on.agent_id == transmission.hop_off.agent_id, transmission
            # N.B: hop_on and hop_off may be at the same resource occupation!
            assert transmission.hop_off.interval.from_incl >= transmission.hop_on.interval.from_incl, transmission
            assert transmission.hop_off.interval.to_excl >= transmission.hop_on.interval.to_excl, transmission
        for tr1, tr2 in zip(transmission_chain, transmission_chain[1:]):
            assert tr1.hop_off.agent_id != tr2.hop_off.agent_id, (tr1, tr2)
            # transmission via a resource
            assert tr2.hop_on.resource == tr1.hop_off.resource, (tr1, tr2)
            assert tr2.hop_on.interval.from_incl >= tr1.hop_off.interval.to_excl, (tr1, tr2)


def distance_matrix_from_tranmission_chains(
        number_of_trains: int,
        transmission_chains: List[TransmissionChain]) -> Tuple[np.ndarray, Dict[int, int], Dict[int, Dict[int, List[ResourceOccupation]]]]:
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
        assert distance >= 0
        from_agent_id = from_ro.agent_id
        to_agent_id = to_ro.agent_id

        distance_before = distance_matrix[from_agent_id, to_agent_id]

        if distance_before > distance:
            distance_matrix[from_agent_id, to_agent_id] = distance
            distance_first_reaching[from_agent_id, to_agent_id] = to_ro.interval.from_incl
        elif distance_before == distance:
            distance_first_reaching[from_agent_id, to_agent_id] = min(distance_first_reaching[from_agent_id, to_agent_id], to_ro.interval.from_incl)

    return distance_matrix, minimal_depth, wave_reaching_other_agent
