from typing import Dict
from typing import List
from typing import NamedTuple

from rsp.step_02_setup.data_types import ExperimentMalfunction
from rsp.utils.resource_occupation import ResourceOccupation
from rsp.utils.resource_occupation import SchedulingProblemInTimeWindows
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
