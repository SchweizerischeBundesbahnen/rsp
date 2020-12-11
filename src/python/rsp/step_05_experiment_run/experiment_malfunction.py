from typing import NamedTuple

from flatland.envs.rail_trainrun_data_structures import TrainrunDict

ExperimentMalfunction = NamedTuple("ExperimentMalfunction", [("time_step", int), ("agent_id", int), ("malfunction_duration", int)])


def gen_malfunction(
    earliest_malfunction: int, malfunction_duration: int, schedule_trainruns: TrainrunDict, malfunction_agent_id: int,
) -> ExperimentMalfunction:
    """A.2.2. Create malfunction.

    Parameters
    ----------
    earliest_malfunction
    malfunction_duration
    malfunction_agent_id
    schedule_trainruns

    Returns
    -------
    """
    # --------------------------------------------------------------------------------------
    # 1. Generate malfuntion
    # --------------------------------------------------------------------------------------
    # The malfunction is chosen to start relative to the start time of the malfunction_agent_id
    # This relative malfunction time makes it easier to run malfunciton-time variation experiments
    # The malfunction must happen during the scheduled run of the agent, therefore it must happen before the scheduled end!
    malfunction_start = min(
        schedule_trainruns[malfunction_agent_id][0].scheduled_at + earliest_malfunction, schedule_trainruns[malfunction_agent_id][-1].scheduled_at - 1
    )
    malfunction = ExperimentMalfunction(time_step=malfunction_start, malfunction_duration=malfunction_duration, agent_id=malfunction_agent_id)
    return malfunction
