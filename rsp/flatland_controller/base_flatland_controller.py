from typing import Dict

from flatland.envs.rail_env import RailEnv


class BaseFlatlandController:
    """Abstract class for the user Flatland controller (my_controller)"""

    def get_observation_builder(self):
        """Create and return the observation builder instance.

        :return:  observation builder instance
        """
        raise NotImplementedError

    def setup(self, env: RailEnv):
        """Setups the controller (implement solver,...)

        :param env: Rail environment instance
        """
        raise NotImplementedError

    def controller(self, env: RailEnv, observation: Dict, info: Dict, num_agents: int) -> Dict:
        """FLATland controller which has to be implemented by the user
        dispatching algorithm.

        :param env: the rail environment
        :param observation: the observation object returned from env.reset(..) and env.step(...)
        :param info: the additional information object returned from env.reset(..) and env.step(...)
        :param num_agents: the number of agents
        :return: action_dict : returns the agent's actions
        """
        raise NotImplementedError
