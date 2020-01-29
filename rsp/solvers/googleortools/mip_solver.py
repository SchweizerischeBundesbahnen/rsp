from ortools.linear_solver import pywraplp

from rsp.solvers.googleortools.abstract_ortools_solver import AbstractORToolsSolver


class MIPSolver(AbstractORToolsSolver):
    def __init__(self):
        # Create the mip solver with the CBC backend.
        self._solver = pywraplp.Solver('simple_mip_program', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        self._objective = self._solver.Objective()
        self._status = -1

    def name(self):
        return "MIPSolver"

    def infinity(self):
        return self._solver.infinity()

    def IntVar(self, from_int_value, to_int_value, name):
        # Adding a new key value pair
        v = self._solver.IntVar(from_int_value, to_int_value, name)
        return v

    def SetTimeLimit(self, time_limit_milliseconds):
        self._solver.SetTimeLimit(time_limit_milliseconds)

    def Add(self, ct):
        self._solver.Add(ct)

    def Solve(self):
        self._status = self._solver.Solve()

    def get_objective_value(self):
        return self._objective.Value()

    def is_optimal_solution(self):
        return self._status == pywraplp.Solver.OPTIMAL

    def is_solved(self):
        if self._status == pywraplp.Solver.NOT_SOLVED:
            return False
        if self._status == pywraplp.Solver.FEASIBLE:
            return True
        return self.is_optimal_solution()

    def LookupVariable(self, name):
        return self._solver.LookupVariable(name)

    def get_solver_variable_value(self, var_name):
        var = self._solver.LookupVariable(var_name)
        return var, var.solution_value()

    def build_objective_function(self, variables):
        """Minimizes the sum of arrival times."""
        self._objective.Clear()
        self._objective.SetMinimization()
        for v in variables:
            c = 1.0
            self._objective.SetCoefficient(v, float(c))

    def print_info(self):
        print("***********************************************************************************")
        print('Number of variables =', self._solver.NumVariables())
        print('Number of constraints =', self._solver.NumConstraints())
        print("-----------------------------------------------------------------------------------")
        print('Solution:')
        print('Objective value =', self._solver.Objective().Value())
        print("-----------------------------------------------------------------------------------")
        print('\nAdvanced usage:')
        print('Problem solved in %f milliseconds' % self._solver.wall_time())
        print('Problem solved in %d iterations' % self._solver.iterations())
        print('Problem solved in %d branch-and-bound nodes' % self._solver.nodes())
        print("***********************************************************************************")

    def print_all_variables(self):
        for var in self._solver.variables():
            print(var, var.solution_value())
