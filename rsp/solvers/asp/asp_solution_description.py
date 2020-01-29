import re
from typing import Set

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_trainrun_data_structures import Trainrun
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint

from rsp.abstract_problem_description import AbstractProblemDescription
from rsp.abstract_problem_description import Waypoint
from rsp.abstract_solution_description import AbstractSolutionDescription
from rsp.solvers.asp.asp_solver import FluxHelperResult


class ASPSolutionDescription(AbstractSolutionDescription):

    def __init__(self, env: RailEnv, asp_solution: FluxHelperResult):
        super(self.__class__, self).__init__(env)
        self.asp_solution: FluxHelperResult = asp_solution
        self.answer_set: Set[str] = self.asp_solution.answer_sets[0]

    def _get_solver_variable_value(self, var_name) -> str:
        return list(filter(lambda s: s.startswith(str(var_name)), self.answer_set))[0]

    def is_solved(self):
        """Is the model satisfiable, is there any solution?"""
        # take stats of last multi-shot call
        return self.asp_solution.stats['summary']['models']['enumerated'] > 0

    def is_optimal_solution(self):
        """Is an optimal solution found?"""
        # take stats of last multi-shot call
        return self.asp_solution.stats['summary']['models']['optimal'] > 0

    @staticmethod
    def _parse_dl_fact(value: str) -> TrainrunWaypoint:
        # dl((t0,((3,5),3)),5) # NOQA
        p = re.compile(r'dl\(\(t[^,]+,\(\(([0-9]+),([0-9]+)\),(.+)\)\),([0-9]+)\)')
        m = p.match(value)
        r = int(m.group(1))
        c = int(m.group(2))
        d = int(m.group(3))
        entry = int(m.group(4))

        return TrainrunWaypoint(scheduled_at=entry, waypoint=Waypoint(position=(r, c), direction=d))

    def get_entry_time(self, agent_id: int, v: Waypoint) -> int:
        # TODO SIM-121 asp_solver should use proper data structures instead of strings to represent answer sets
        # hack since Python's tuple representations has spaces, but ASP gives us them without.
        position_part = str(tuple(v)).replace(" ", "")
        var_prefix = "dl((t{},{}),".format(agent_id, position_part)
        value: str = self._get_solver_variable_value(var_prefix)
        return int(value.replace(var_prefix, "").replace(")", ""))

    def _print_entry_times(self, answer_set):
        print(self._get_entry_times_from_string_answer_set(answer_set))

    def _get_entry_times_from_string_answer_set(self, answer_set):
        return list(filter(lambda s: s.startswith(str("dl")), answer_set))

    def get_trainrun_for_agent(self, agent_id: int) -> Trainrun:
        """Get train run of the agent in the solution."""
        return self._get_solution_trainrun(agent_id)

    def _get_solution_trainrun(self, agent_id) -> Trainrun:
        var_prefix = "dl((t{},".format(agent_id)
        agent_facts = filter(lambda s: s.startswith(str(var_prefix)), self.answer_set)
        agent = self._env.agents[agent_id]
        start_waypoint = AbstractProblemDescription.convert_position_and_entry_direction_to_waypoint(
            *agent.initial_position,
            agent.initial_direction)
        # filter out dl entries that are zero and not relevant to us
        path = list(filter(lambda pse: pse.scheduled_at > 0 or pse.waypoint == start_waypoint,
                           map(self.__class__._parse_dl_fact, agent_facts)))
        path.sort(key=lambda p: p.scheduled_at)
        # remove the transition from the target waypoint to the dummy
        assert path[-1].waypoint.direction == AbstractProblemDescription.MAGIC_DIRECTION_FOR_TARGET
        path = path[:-1]
        return path

    def get_model_latest_arrival_time(self) -> int:
        """Get latest entry time for an agent at a waypoint over all agents and
        all their non-dummy waypoints."""
        latest_entry = 0
        for agent_id in range(len(self._env.agents)):
            latest_entry = max(latest_entry, self.get_trainrun_for_agent(agent_id)[-1].scheduled_at)
        return latest_entry

    def get_sum_running_times(self) -> int:
        """Get the model's cost of the solution with to its minimization
        objective (which might be slightly different from the FLATland
        rewards)."""
        costs = 0
        for agent_id in range(len(self._env.agents)):
            solution_trainrun = self.get_trainrun_for_agent(agent_id)
            costs += solution_trainrun[-1].scheduled_at - solution_trainrun[0].scheduled_at
        return costs

    def get_objective_value(self) -> float:
        return self.asp_solution.stats['summary']['costs'][0]

    def extract_nb_resource_conflicts(self) -> int:
        return len(list(filter(lambda s: s.startswith('shared('), self.answer_set)))
