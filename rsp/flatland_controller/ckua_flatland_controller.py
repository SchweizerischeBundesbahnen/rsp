from typing import Dict

from flatland.envs.rail_env import RailEnv
from libs.cell_graph_dispatcher import CellGraphDispatcher
from libs.dummy_observation import DummyObservationBuilder

from rsp.flatland_controller.base_flatland_controller import BaseFlatlandController


class CkUaController(BaseFlatlandController):
    def get_observation_builder(self):
        return DummyObservationBuilder()

    def setup(self, env: RailEnv):
        self.dispatcher = CellGraphDispatcher(env)

    def controller(self, env: RailEnv, observation: Dict, info: Dict, num_agents: int) -> Dict:
        """FLATland controller which has to be implemented by the user
        dispatching algorithm.

        :param env: the rail environment
        :param observation: the observation object returned from env.reset(..) and env.step(...)
        :param info: the additional information object returned from env.reset(..) and env.step(...)
        :param num_agents: the number of agents
        :return: action_dict : returns the agent's actions
        """
        return self.dispatcher.step(env._elapsed_steps)
