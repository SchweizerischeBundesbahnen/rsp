"""
Abstract class for solvers. These are used to solve the experiments surrounding the real time rescheduling research project.

"""

import abc

import numpy as np
from flatland.envs.rail_env import RailEnv

from rsp.utils.data_types import ExperimentResults


class AbstractSolver:

    @abc.abstractmethod
    def run_experiment_trial(
            self,
            static_rail_env: RailEnv,
            malfunction_rail_env: RailEnv,
            malfunction_env_reset,
            k: int = 10,
            disable_verification_by_replay: bool = False
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
            disable_verification_by_replay: bool = False
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
            delta=[1, 2, 3, 4, 5, 7])

        return current_results
