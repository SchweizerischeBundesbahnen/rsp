from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Tuple

import numpy as np

from rsp.experiment_solvers.data_types import ExperimentMalfunction
from rsp.logger import rsp_logger
from rsp.utils.data_types import ResourceOccupation
from rsp.utils.data_types import SchedulingProblemInTimeWindows
from rsp.utils.data_types import SortedResourceOccupationsPerAgent
from rsp.utils.data_types import SortedResourceOccupationsPerResource

TransmissionLeg = NamedTuple('TransmissionLeg', [
    ('hop_on', ResourceOccupation),
    ('hop_off', ResourceOccupation)
])
TransmissionChain = List[TransmissionLeg]

# hop-on resource-occupations reaching at this depth (int key = depth)
WAVE_PER_DEPTH = Dict[int, List[ResourceOccupation]]
# waves reaching per agent (int key = agent_id)
WAVE_PER_AGENT_AND_DEPTH = Dict[int, WAVE_PER_DEPTH]


def extract_transmission_chains_from_time_windows(  # noqa: C901
        malfunction: ExperimentMalfunction,
        time_windows: SchedulingProblemInTimeWindows
) -> List[TransmissionChain]:
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
            chain = [TransmissionLeg(malfunction_time_window, hop_on)]
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
                chain = history + [TransmissionLeg(hop_on, hop_off)]
                open_wave_front.append(chain)
                transmission_chains.append(chain)
    return transmission_chains


# TODO we probably too much work here!
def extract_transmission_chains(
        malfunction: ExperimentMalfunction,
        resource_occupations_per_agent: SortedResourceOccupationsPerAgent,
        resource_occupations_per_resource: SortedResourceOccupationsPerResource
) -> List[TransmissionChain]:
    """Propagation of delay.

    Parameters
    ----------
    malfunction
    resource_occupations_per_agent
    resource_occupations_per_resource

    Returns
    -------
    """
    malfunction_agent_id = malfunction.agent_id
    open_wave_front: List[TransmissionChain] = []
    transmission_chains: List[TransmissionChain] = []
    closed_wave_front: List[ResourceOccupation] = []
    malfunction_occupation = next(ro for ro in resource_occupations_per_agent[malfunction_agent_id] if malfunction.time_step < ro.interval.to_excl)
    for ro in resource_occupations_per_agent[malfunction_agent_id]:
        # N.B. intervals include release times, therefore we can be strict at upper bound!
        if malfunction.time_step < ro.interval.to_excl:
            # TODO should the interval extended by the malfunction duration?
            chain = [TransmissionLeg(malfunction_occupation, ro)]
            open_wave_front.append(chain)
            transmission_chains.append(chain)
    assert len(open_wave_front) > 0
    while len(open_wave_front) > 0:
        history = open_wave_front.pop()
        wave_front = history[-1].hop_off
        wave_front_resource = wave_front.resource
        if wave_front in closed_wave_front:
            continue
        closed_wave_front.append(wave_front)

        # the next scheduled train may be impacted!
        # TODO we do not consider decelerate to let pass yet! search backward in radius? would this be another type of chain?
        for ro in resource_occupations_per_resource[wave_front_resource]:
            if ro.interval.from_incl >= wave_front.interval.to_excl:
                for subsequent_ro in resource_occupations_per_agent[ro.agent_id]:
                    # hop_on and hop_off may be at the same resource
                    if subsequent_ro.interval.from_incl >= ro.interval.from_incl:
                        chain = history + [TransmissionLeg(ro, subsequent_ro)]
                        assert subsequent_ro not in history
                        if ro in closed_wave_front:
                            continue
                        open_wave_front.append(chain)
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
            # leg must be at the same agent
            assert transmission.hop_on.agent_id == transmission.hop_off.agent_id, transmission

            # hop off interval must not be earlier than hop on interval, but may overlap
            # (in particular, hop_on and hop_off may be at the same resource occupation)
            assert transmission.hop_off.interval.from_incl >= transmission.hop_on.interval.from_incl, transmission
            assert transmission.hop_off.interval.to_excl >= transmission.hop_on.interval.to_excl, transmission
        for tr1, tr2 in zip(transmission_chain, transmission_chain[1:]):
            # transmission between different agents
            assert tr1.hop_off.agent_id != tr2.hop_off.agent_id, (tr1, tr2)

            # transmission via same resource
            assert tr2.hop_on.resource == tr1.hop_off.resource, (tr1, tr2)

            # transmission via occupation afterwards
            assert tr2.hop_on.interval.from_incl >= tr1.hop_off.interval.to_excl, (tr1, tr2)


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
        assert (tr1.hop_off.interval.from_incl <= tr2.hop_on.interval.from_incl <= tr1.hop_off.interval.to_excl) or \
               (tr2.hop_on.interval.from_incl <= tr1.hop_off.interval.from_incl <= tr2.hop_on.interval.to_excl) or \
               (tr2.hop_on.interval.from_incl <= tr1.hop_off.interval.from_incl and
                tr2.hop_on.interval.to_excl >= tr1.hop_off.interval.to_excl) or \
               (tr2.hop_on.interval.from_incl >= tr1.hop_off.interval.from_incl and
                tr2.hop_on.interval.to_excl <= tr1.hop_off.interval.to_excl), \
            (tr1, tr2)


def distance_matrix_from_tranmission_chains(
        number_of_trains: int,
        transmission_chains: List[TransmissionChain],
        cutoff: int = None
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], Dict[int, Dict[int, List[ResourceOccupation]]]]:
    """

    Parameters
    ----------
    number_of_trains
    transmission_chains
    cutoff

    Returns
    -------
    distance_matrix, weights_matrix, minimal_depth, wave_fronts_reaching_other_agent

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
        # TODO SIM-511 weighting/damping: take into account number of intermediate steps and their distance!
        if cutoff is not None:
            distances = [to_leg.hop_on.interval.from_incl - from_leg.hop_off.interval.to_excl for from_leg, to_leg in
                         zip(transmission_chain, transmission_chain[1:])]
            for distance in distances:
                assert distance >= 0
            max_distance = np.max(distances)
            if max_distance >= cutoff:
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
    # almost-inverse: distance -> weights
    weights_matrix: np.ndarray = (1 / distance_matrix) + 0.000001
    # normalize
    np_max = np.max(weights_matrix)
    weights_matrix /= np_max
    return distance_matrix, weights_matrix, minimal_depth, wave_reaching_other_agent
