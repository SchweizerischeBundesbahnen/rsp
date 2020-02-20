"""Abstract class for solvers.

These are used to solve the experiments surrounding the real time
rescheduling research project.
"""
import abc
from typing import Any
from typing import Callable

from flatland.envs.rail_env import RailEnv

from rsp.utils.data_types import ExperimentResults

# abstract from rendering to have unit tests without dependendance on FLATland rendering
RendererForEnvInit = Callable[[RailEnv, bool], Any]
RendererForEnvRender = Callable[[Any, int, str, int, str], None]
RendererForEnvCleanup = Callable[[Any], None]


class AbstractSolver:

    @abc.abstractmethod
    def run_experiment_trial(
            self,
            static_rail_env: RailEnv,
            malfunction_rail_env: RailEnv,
            malfunction_env_reset,
            k: int = 10,
            disable_verification_by_replay: bool = False,
            verbose: bool = False,
            debug: bool = False,
            rendering: bool = False
    ) -> ExperimentResults:
        """Runs the experiment.

        Parameters
        ----------
        static_rail_env: RailEnv
            this is the environment without any malfunction
        malfunction_rail_env: RailEnv
            this is the enviroment with a malfunction
        k: int
            number of routing alternatives to generate

        Returns
        -------
        """
