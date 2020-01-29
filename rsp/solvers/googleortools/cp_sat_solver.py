from ortools.sat.python import cp_model

from rsp.solvers.googleortools.abstract_ortools_solver import AbstractORToolsSolver


class CPSATSolver(AbstractORToolsSolver):
    DEBUG = False

    def __init__(self, enable_objective_function=False):
        # Creates the model.
        self._model = cp_model.CpModel()
        self._solver = cp_model.CpSolver()
        self._status = -1
        self._variable_dict = {}
        self._enable_objective_function = enable_objective_function

    def name(self):
        return "CPSATSolver"

    def infinity(self):
        return 9999

    def IntVar(self, from_int_value, to_int_value, name):
        # Adding a new key value pair
        v = self._model.NewIntVar(int(from_int_value), int(to_int_value), name)
        self._variable_dict.update({name: v})
        return v

    def SetTimeLimit(self, time_limit_milliseconds):
        pass

    def Add(self, ct):
        if self.DEBUG:
            print("  {}".format(ct))
        self._model.Add(ct)

    def build_objective_function(self, variables):
        """Minimizes the sum of arrival times."""
        if not self._enable_objective_function:
            return
        if len(variables) > 0:
            x = '1*variables[0]'
            if len(variables) > 1:
                for i in range(1, len(variables)):
                    x = x + '+ 1*variables[{}]'.format(i)
            self._model.Minimize(eval(x))

    def Solve(self):
        # Creates a solver and solves the model.
        self._status = self._solver.Solve(self._model)

    def is_optimal_solution(self):
        return self._status == cp_model.OPTIMAL

    def is_solved(self):
        return self._status == cp_model.FEASIBLE

    def LookupVariable(self, name):
        return self._variable_dict.get(name)

    def get_solver_variable_value(self, var_name):
        var = self.LookupVariable(var_name)
        if self._status == cp_model.FEASIBLE:
            return var, self._solver.Value(var)
        return var, 0
