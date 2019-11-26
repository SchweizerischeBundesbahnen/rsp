from __future__ import print_function

from typing import Dict, Optional, List, Tuple

from flatland.envs.rail_env import RailEnv

from rsp.abstract_problem_description import AbstractProblemDescription, Waypoint
from rsp.googleortools.abstract_ortools_solver import AbstractORToolsSolver
from rsp.googleortools.cp_sat_solver import CPSATSolver
from rsp.googleortools.mip_solver import MIPSolver
from rsp.googleortools.ortools_solution_description import ORToolsSolutionDescription
from rsp.googleortools.ortools_utils import make_variable_name_agent_at_waypoint


class ORToolsProblemDescription(AbstractProblemDescription):
    # big value but not INF, otherwise the MIPSolver or CPSATSolver can't handle the value
    PSEUDO_INFINITY = 10000000

    DEBUG = False

    def __init__(self,
                 env: RailEnv,
                 solver: AbstractORToolsSolver,
                 agents_path_dict: Dict[int, Optional[List[List[Waypoint]]]]):
        self._solver = solver
        super().__init__(env, agents_path_dict)
        self._solver = solver

    def get_solver_name(self) -> str:
        """Return the solver name for printing."""
        # TODO refactor: delegate to solver to return its name?
        if isinstance(self._solver, CPSATSolver):
            return "ortools_CPSAT"
        elif isinstance(self._solver, MIPSolver):
            return "ortools_MIP"

    def _lookup_or_create_int_var(self, var_name: str, lower_bound, upper_bound):
        var = self._solver.LookupVariable(var_name)
        if var is None:
            return self._solver.IntVar(lower_bound, upper_bound, var_name)
        return var

    def _implement_train(self, agent_id: int, start_vertices: List[Waypoint], target_vertices: List[Waypoint],
                         minimum_running_time: int):
        """See `AbstractProblemDescription`.

        Parameters
        ----------
        minimum_running_time
        """
        # nothing to do here
        pass

    def _implement_agent_earliest(self, agent_id: int, waypoint: Waypoint, time):
        """See `AbstractProblemDescription`."""
        infinity = self._solver.infinity()
        t_agent_entry = self._lookup_or_create_int_var(make_variable_name_agent_at_waypoint(agent_id, waypoint),
                                                       -infinity, infinity)
        if self.DEBUG:
            print("Rule 102 Time windows for earliest-requirements.")
        self._solver.Add(t_agent_entry >= int(time))

    def _implement_agent_latest(self, agent_id: int, waypoint: Waypoint, time):
        """See `AbstractProblemDescription`."""
        infinity = self._solver.infinity()
        t_agent = self._lookup_or_create_int_var(make_variable_name_agent_at_waypoint(agent_id, waypoint),
                                                 -infinity, infinity)
        if self.DEBUG:
            print("Rule 101 Time windows for latest-requirements.")
        self._solver.Add(t_agent <= int(time))

    def _implement_route_section(self, agent_id: int, entry_waypoint: Waypoint, exit_waypoint: Waypoint,
                                 resource_id: Tuple[int, int], minimum_travel_time: int = 1, penalty=0):
        """See `AbstractProblemDescription`.

        Parameters
        ----------
        penalty
        """
        infinity = self._solver.infinity()

        t_agent_entry = self._lookup_or_create_int_var(
            make_variable_name_agent_at_waypoint(agent_id, entry_waypoint),
            -infinity, infinity)
        t_agent_exit = self._lookup_or_create_int_var(
            make_variable_name_agent_at_waypoint(agent_id, exit_waypoint),
            -infinity, infinity)

        if self.DEBUG:
            print("Rule 103 Minimum section time")
        self._solver.Add(t_agent_exit - t_agent_entry >= int(minimum_travel_time))

    def _implement_resource_mutual_exclusion(self,
                                             agent_1_id: int,
                                             agent_1_entry_waypoint: Waypoint,
                                             agent_1_exit_waypoint: Waypoint,
                                             agent_2_id: int,
                                             agent_2_entry_waypoint: Waypoint,
                                             agent_2_exit_waypoint: Waypoint,
                                             resource_id: Tuple[int, int]
                                             ):
        """See `AbstractProblemDescription`."""
        infinity = self._solver.infinity()

        variable_name = 'res_{}_{}_agents_{}_{}'.format(*resource_id, agent_1_id, agent_2_id)

        t_agent_1_res_x_entry = self._lookup_or_create_int_var(
            make_variable_name_agent_at_waypoint(agent_1_id, agent_1_entry_waypoint),
            -infinity, infinity)
        t_agent_1_res_x_exit = self._lookup_or_create_int_var(
            make_variable_name_agent_at_waypoint(agent_1_id, agent_1_exit_waypoint),
            -infinity, infinity)
        t_agent_2_res_x_entry = self._lookup_or_create_int_var(
            make_variable_name_agent_at_waypoint(agent_2_id, agent_2_entry_waypoint),
            -infinity, infinity)
        t_agent_2_res_x_exit = self._lookup_or_create_int_var(
            make_variable_name_agent_at_waypoint(agent_2_id, agent_2_exit_waypoint),
            -infinity, infinity)

        agent_1_before_agent_2_name = 'agent_1_before_agent_2_{}'.format(variable_name)
        agent_1_before_agent_2_res_x = self._lookup_or_create_int_var(agent_1_before_agent_2_name, 0.0, 1.0)

        agent_2_before_agent_1_name = 'agent_2_before_agent_1_{}'.format(variable_name)
        agent_2_before_agent_1_res_x = self._lookup_or_create_int_var(agent_2_before_agent_1_name, 0.0, 1.0)

        if self.DEBUG:
            print("Rule 104 Resource Occupations.")
        # release time = 1 to allow for synchronization in FLATland
        self._solver.Add(
            t_agent_2_res_x_entry - t_agent_1_res_x_exit + self.PSEUDO_INFINITY * agent_2_before_agent_1_res_x >= 1)
        self._solver.Add(
            t_agent_2_res_x_exit - t_agent_1_res_x_entry - self.PSEUDO_INFINITY * agent_1_before_agent_2_res_x <= -1)

        self._solver.Add(agent_1_before_agent_2_res_x + agent_2_before_agent_1_res_x == 1)

    def _create_objective_function_minimize(self, variables: List[Waypoint]):
        """Creates an objective function to minimize in the solver.

        Parameters
        ----------
        variables
            the target waypoints of the agents
        """
        self._solver.build_objective_function(variables)

    def solve(self) -> ORToolsSolutionDescription:
        self._status = self._solver.Solve()
        # TODO bad code smell: we should not pass the solver instance,
        #  but deduce the result in order to make solution description stateless
        return ORToolsSolutionDescription(self.env, self._solver, self.agents_path_dict)
