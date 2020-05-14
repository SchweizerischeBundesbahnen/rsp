from typing import Optional
from typing import Tuple

from flatland.envs.agent_utils import EnvAgent
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.malfunction_generators import Malfunction
from flatland.envs.malfunction_generators import MalfunctionGenerator
from flatland.envs.malfunction_generators import MalfunctionProcessData
from numpy.random.mtrand import RandomState


def single_agent_malfunction_generator(malfunction_time: int, malfunction_duration: int, agent_id: int) \
        -> Tuple[MalfunctionGenerator, MalfunctionProcessData]:
    """Malfunction generator which guarantees exactly one malfunction during an
    episode of an ACTIVE agent. The malfunctino occurs at malfunction_time
    after the start of the agent in the environment.

    Parameters
    ----------
    malfunction_time: Earliest possible malfunction onset
    malfunction_duration: The duration of the single malfunction

    Returns
    -------
    generator, Tuple[float, int, int] with mean_malfunction_rate, min_number_of_steps_broken, max_number_of_steps_broken
    """
    # Mean malfunction in number of time steps
    mean_malfunction_rate = 0.

    # Uniform distribution parameters for malfunction duration
    min_number_of_steps_broken = 0
    max_number_of_steps_broken = 0

    # Keep track of the total number of malfunctions in the env
    global_nr_malfunctions = 0

    # Malfunction calls per agent
    malfunction_calls = dict()

    def generator(agent: EnvAgent = None, np_random: RandomState = None, reset=False) -> Optional[Malfunction]:
        # We use the global variable to assure only a single malfunction in the env
        nonlocal global_nr_malfunctions
        nonlocal malfunction_calls

        # Reset malfunciton generator
        if reset:
            nonlocal global_nr_malfunctions
            nonlocal malfunction_calls
            global_nr_malfunctions = 0
            malfunction_calls = dict()
            return Malfunction(0)

        # No more malfunctions if we already had one, ignore all updates
        if global_nr_malfunctions > 0:
            return Malfunction(0)

        # Update number of calls per agent that are active
        if agent.status == RailAgentStatus.ACTIVE:
            if agent.handle in malfunction_calls:
                malfunction_calls[agent.handle] += 1
            else:
                malfunction_calls[agent.handle] = 1

        # Break an agent that is active at the time of the malfunction
        if agent.status == RailAgentStatus.ACTIVE and \
                malfunction_calls[agent.handle] >= malfunction_time and agent.handle == agent_id:
            global_nr_malfunctions += 1
            return Malfunction(malfunction_duration)
        else:
            return Malfunction(0)

    return generator, MalfunctionProcessData(mean_malfunction_rate, min_number_of_steps_broken,
                                             max_number_of_steps_broken)


def single_malfunction_generator(earlierst_malfunction: int, malfunction_duration: int) -> Tuple[
    MalfunctionGenerator, MalfunctionProcessData]:
    """Malfunction generator which guarantees exactly one malfunction during an
    episode of an ACTIVE agent.

    Parameters
    ----------
    earlierst_malfunction: Earliest possible malfunction onset
    malfunction_duration: The duration of the single malfunction

    Returns
    -------
    generator, Tuple[float, int, int] with mean_malfunction_rate, min_number_of_steps_broken, max_number_of_steps_broken
    """
    # Mean malfunction in number of time steps
    mean_malfunction_rate = 0.

    # Uniform distribution parameters for malfunction duration
    min_number_of_steps_broken = 0
    max_number_of_steps_broken = 0

    # Keep track of the total number of malfunctions in the env
    global_nr_malfunctions = 0

    # Malfunction calls per agent
    malfunction_calls = dict()

    def generator(agent: EnvAgent = None, np_random: RandomState = None, reset=False) -> Optional[Malfunction]:
        # We use the global variable to assure only a single malfunction in the env
        nonlocal global_nr_malfunctions
        nonlocal malfunction_calls

        # Reset malfunciton generator
        if reset:
            nonlocal global_nr_malfunctions
            nonlocal malfunction_calls
            global_nr_malfunctions = 0
            malfunction_calls = dict()
            return Malfunction(0)

        # No more malfunctions if we already had one, ignore all updates
        if global_nr_malfunctions > 0:
            return Malfunction(0)

        # Update number of calls per agent
        if agent.handle in malfunction_calls:
            malfunction_calls[agent.handle] += 1
        else:
            malfunction_calls[agent.handle] = 1

        # Break an agent that is active at the time of the malfunction
        if agent.status == RailAgentStatus.ACTIVE and malfunction_calls[agent.handle] >= earlierst_malfunction:
            global_nr_malfunctions += 1
            return Malfunction(malfunction_duration)
        else:
            return Malfunction(0)

    return generator, MalfunctionProcessData(mean_malfunction_rate, min_number_of_steps_broken,
                                             max_number_of_steps_broken)
