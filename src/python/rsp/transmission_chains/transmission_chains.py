from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Tuple

from rsp.resource_occupation.resource_occupation import ResourceOccupation
from rsp.resource_occupation.resource_occupation import ScheduleAsResourceOccupations
from rsp.step_05_experiment_run.experiment_malfunction import ExperimentMalfunction

TransmissionLeg = NamedTuple("TransmissionLeg", [("hop_on", ResourceOccupation), ("hop_off", ResourceOccupation), ("delay_time", int)])
TransmissionChain = List[TransmissionLeg]


def extract_transmission_chains_from_schedule(malfunction: ExperimentMalfunction, occupations: ScheduleAsResourceOccupations) -> List[TransmissionChain]:
    """Propagation of delay.compute_disturbance_propagation_graph
    TODO optimize (for instance, we do not check whether that only the largest delay per resource and agent is in the open list.
    Parameters
    ----------

    Returns
    -------
    List of transmission chains
    """
    # malfunction_agent_id <=> m_{agent}
    malfunction_agent_id = malfunction.agent_id

    # delay_time <=> m_{duration}
    delay_time = malfunction.malfunction_duration

    resource_occupations_per_agent = occupations.sorted_resource_occupations_per_agent
    resource_occupations_per_resource = occupations.sorted_resource_occupations_per_resource

    open_wave_front: List[Tuple[ResourceOccupation, TransmissionChain]] = []
    transmission_chains: List[TransmissionChain] = []
    closed_wave_front: Dict[ResourceOccupation, int] = {}
    malfunction_occupation = next(ro for ro in resource_occupations_per_agent[malfunction_agent_id] if malfunction.time_step < ro.interval.to_excl)

    # here we assume green wave of a: it cannot absorb (some of) d'
    for ro in resource_occupations_per_agent[malfunction_agent_id]:
        # N.B. intervals include release times, therefore we can be strict at upper bound!
        if ro.interval.to_excl > malfunction.time_step:
            # TODO should the interval extended by the malfunction duration?
            chain = [TransmissionLeg(malfunction_occupation, ro, delay_time)]
            open_wave_front.append(chain)
            transmission_chains.append(chain)
    assert len(open_wave_front) > 0
    while len(open_wave_front) > 0:
        history = open_wave_front.pop()

        # wave_front <=> D_{S_0}(a,v)
        wave_front = history[-1].hop_off

        # delay_time <=> d
        delay_time = history[-1].delay_time

        # wave_front_resource <=> r
        wave_front_resource = wave_front.resource

        if delay_time <= closed_wave_front.get(wave_front, 0):
            continue
        closed_wave_front[wave_front] = delay_time

        # subsequent resource occupations: next agent in the delay window at the resource
        # a' <=> ro.agent_id: the next scheduled train at r!
        ro = next(iter([ro for ro in resource_occupations_per_resource[wave_front_resource] if ro.interval.from_incl >= wave_front.interval.to_excl]), None)

        # is a' defined?
        if ro is not None:

            # distance between agents
            gap = ro.interval.from_incl - wave_front.interval.to_excl

            # gap < d
            if gap < delay_time:
                # remaining_delay_time <=> d'
                remaining_delay_time = delay_time - gap
                assert remaining_delay_time >= 0
                impact_distance_from_wave_front = delay_time - remaining_delay_time
                assert impact_distance_from_wave_front >= 0

                # argmax_{v''} { A_{S_0}(a',v''): A_{S_0}(a',v'') < A_{S_0}(a',v') }
                start_ro = next(
                    reversed(
                        [
                            subsequent_ro
                            for subsequent_ro in resource_occupations_per_agent[ro.agent_id]
                            # the propagation may flow backwards in space: agents may already wait one cell in advance!
                            # A_{S_0}(a',v'') < A_{S_0}(a',v')
                            # ro.interval.from_incl <=> A_{S_0}(a',v')
                            # subsequent_ro.interval.from_incl <=> A_{S_0}(a',v'')
                            if subsequent_ro.interval.from_incl < ro.interval.from_incl
                        ]
                    ),
                    None,
                )

                # if argmax_{v''} is not defined, take ro.interval.from_incl <=> A_{S_0}(a',v')
                start = start_ro.interval.from_incl if start_ro is not None else ro.interval.from_incl
                # here we assume green wave of a': it cannot absorb (some of) d'
                for subsequent_ro in [
                    subsequent_ro for subsequent_ro in resource_occupations_per_agent[ro.agent_id] if subsequent_ro.interval.from_incl >= start
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
