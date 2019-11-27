# TODO ortools refactor use @abc.abstractmethod
class AbstractORToolsSolver:

    def name(self):
        return "AbstractSolver"

    def infinity(self):
        return 0.0

    def IntVar(self, from_int_value, to_int_value, name):
        pass

    def SetTimeLimit(self, time_limit_milliseconds):
        pass

    def Add(self, ct):
        pass

    def Solve(self):
        pass

    def get_objective_value(self):
        pass

    def LookupVariable(self, name):
        pass

    def get_solver_variable_value(self, var_name):
        pass

    def build_objective_function(self, variables):
        """Minimizes the sum of arrival times."""
        pass

    def is_optimal_solution(self):
        pass

    def is_solved(self):
        pass

    def print_info(self):
        pass

    def print_all_variables(self):
        pass
