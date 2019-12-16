"""
Abstract class for solvers. These are used to solve the experiments surrounding the real time rescheduling research project.

"""

import abc
from typing import Callable, Any

import numpy as np
from flatland.envs.rail_env import RailEnv

from rsp.utils.data_types import ExperimentResults, ExperimentMalfunction

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
        """
        Runs the experiment.

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


class DummySolver(AbstractSolver):
    """
    Dummy solver used to run experiment pipeline

    Methods
    -------
    run_experiment_trial:
        Returns the correct data format to run tests on full research pipeline
    """

    def run_experiment_trial(
            self,
            static_rail_env: RailEnv,
            malfunction_rail_env: RailEnv,
            malfunction_env_reset,
            k: int = 10,
            disable_verification_by_replay: bool = False,
            verbose: bool = False,
            debug: bool = False,
            rendering: bool = False,
            init_renderer_for_env: RendererForEnvInit = lambda *args, **kwargs: None,
            render_renderer_for_env: RendererForEnvRender = lambda *args, **kwargs: None,
            cleanup_renderer_for_env: RendererForEnvCleanup = lambda *args, **kwargs: None,
    ) -> ExperimentResults:
        """
        Runs the experiment.

        Parameters
        ----------
        static_rail_env: RailEnv
            Rail environment without any malfunction
        malfunction_rail_env: RailEnv
            Rail environment with one single malfunction

        Returns
        -------
        ExperimentResults
        """
        current_results = ExperimentResults(
            time_full=static_rail_env.width + static_rail_env.get_num_agents() + np.random.randint(-5, 5),
            time_full_after_malfunction=static_rail_env.width + static_rail_env.get_num_agents() + np.random.randint(-5,
                                                                                                                     5),
            time_delta_after_malfunction=static_rail_env.width + static_rail_env.get_num_agents() + np.random.randint(
                -5, 5),
            solution_full=[(1, 3), (2, 4)],
            solution_delta=[(1, 3), (2, 4)],
            delta=[1, 2, 3, 4, 5, 7],
            malfunction=ExperimentMalfunction(-1, -1 - 1),
            agent_paths_dict={}
        )

        return current_results
