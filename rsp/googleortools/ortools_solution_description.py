from typing import Dict, Optional, List

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_trainrun_data_structures import Waypoint, Trainrun, TrainrunWaypoint

from rsp.abstract_solution_description import AbstractSolutionDescription
from rsp.googleortools.abstract_ortools_solver import AbstractORToolsSolver
from rsp.googleortools.ortools_utils import make_variable_name_agent_at_waypoint


class ORToolsSolutionDescription(AbstractSolutionDescription):

    def __init__(self, env: RailEnv, solver: AbstractORToolsSolver,
                 agents_path: Dict[int, Optional[List[List[Waypoint]]]]):
        super(self.__class__, self).__init__(env)
        self._solver: AbstractORToolsSolver = solver
        self._agents_path: Dict[int, Optional[List[List[Waypoint]]]] = agents_path

    def _get_solver_variable_value(self, var_name) -> int:
        return self._solver.get_solver_variable_value(var_name)

    def is_solved(self):
        """Is the model satisfiable, is there any solution?"""
        return self._solver.is_solved()

    def is_optimal_solution(self):
        """Is an optimal solution found?"""
        return self._solver.is_optimal_solution()

    def get_entry_time(self, agent_id: int, wp: Waypoint) -> int:
        _solver_var = make_variable_name_agent_at_waypoint(agent_id, wp)
        _, scheduled_at = self._get_solver_variable_value(_solver_var)
        return scheduled_at

    def get_trainrun_for_agent(self, agent_id: int) -> Trainrun:
        """Get train run of the agent in the solution."""
        # TODO ortools has only fixed paths - should we work on this?
        agent_path = self._agents_path[agent_id][0]
        path = []
        for wp in agent_path:
            entry_time = self.get_entry_time(agent_id, wp)
            path.append(TrainrunWaypoint(scheduled_at=entry_time, waypoint=wp))
        return path

    def get_model_latest_arrival_time(self) -> int:
        """Get latest entry time for an agent at a waypoint over all agents and all their non-dummy waypoints."""
        # TODO implement for ortools
        return -1

    def get_sum_running_times(self) -> int:
        """Get the model's cost of the solution with to its minimization objective (which might be slightly different from the FLATland rewards)."""
        # TODO implement for ortools
        return -1
