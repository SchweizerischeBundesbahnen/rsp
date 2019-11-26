from __future__ import print_function

import abc
from typing import Dict, Any

from flatland.action_plan.action_plan import ActionPlanDict, ControllerFromTrainruns
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_trainrun_data_structures import Waypoint, Trainrun


class AbstractSolutionDescription:

    def __init__(self, env: RailEnv):
        self._env: RailEnv = env
        self._action_plan_element_last_removed: int = 0
        self._action_plan: ActionPlanDict = None

    def create_action_plan(self) -> ControllerFromTrainruns:
        """
        Creates an action plan (fix schedule, when an action has which positions and when it has to take which actions).

        Returns
        -------
        ActionPlanDict
        """
        if self._action_plan is not None:
            return self._action_plan

        solution_trainruns = self.get_trainruns_dict()

        self._action_plan = ControllerFromTrainruns(self._env, solution_trainruns)
        return self._action_plan

    def get_objective_value(self):
        return self._solver.get_objective_value()

    @abc.abstractmethod
    def _get_solver_variable_value(self, var_name: str) -> Any:
        """Get value by variable name. Return type is implmentation specific
           (strings might need to be converted to numbers)."""

    @abc.abstractmethod
    def is_solved(self):
        """Is the model satisfiable, is there any solution?"""

    @abc.abstractmethod
    def is_optimal_solution(self):
        """Is an optimal solution found?"""

    @abc.abstractmethod
    def get_entry_time(self, agent_id: int, v: Waypoint) -> int:
        """Get entry time for an agent at a waypoint."""

    @abc.abstractmethod
    def get_model_latest_arrival_time(self) -> int:
        """Get latest entry time for an agent at a waypoint over all agents and all their non-dummy waypoints."""

    @abc.abstractmethod
    def get_sum_running_times(self) -> int:
        """Get the model's cost of the solution with to its minimization objective
        (which might be slightly different from the FLATland rewards)."""

    @abc.abstractmethod
    def get_trainrun_for_agent(self, agent_id: int) -> Trainrun:
        """Get train run of the agent in the solution."""

    def get_trainruns_dict(self) -> Dict[int, Trainrun]:
        """Get train runs for all agents: waypoints and entry times."""
        return {agent_id: self.get_trainrun_for_agent(agent_id) for agent_id in range(len(self._env.agents))}
