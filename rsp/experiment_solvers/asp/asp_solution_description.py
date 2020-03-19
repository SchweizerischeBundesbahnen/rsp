import re
from typing import List
from typing import Set

from flatland.envs.rail_trainrun_data_structures import Trainrun
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.experiment_solvers.asp.asp_helper import FluxHelperResult
from rsp.route_dag.route_dag import get_sources_for_topo
from rsp.route_dag.route_dag import MAGIC_DIRECTION_FOR_SOURCE_TARGET
from rsp.route_dag.route_dag import ScheduleProblemDescription


class ASPSolutionDescription():

    def __init__(self,
                 asp_solution: FluxHelperResult,
                 tc: ScheduleProblemDescription
                 ):
        self.asp_solution: FluxHelperResult = asp_solution
        # self.answer_set: Set[str] = self.asp_solution.answer_sets[0]
        # self._action_plan = None
        # self.tc: ScheduleProblemDescription = tc

    def get_trainruns_dict(self) -> TrainrunDict:
        """Get train runs for all agents: waypoints and entry times."""
        return {agent_id: self.get_trainrun_for_agent(agent_id) for agent_id in self.tc.topo_dict}

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
        start_waypoint = list(get_sources_for_topo(self.tc.topo_dict[agent_id]))[0]
        # filter out dl entries that are zero and not relevant to us
        path = list(filter(lambda pse: pse.scheduled_at > 0 or pse.waypoint == start_waypoint,
                           map(self.__class__._parse_dl_fact, agent_facts)))
        path.sort(key=lambda p: p.scheduled_at)
        # remove the transition from the target waypoint to the dummy
        assert path[-1].waypoint.direction == MAGIC_DIRECTION_FOR_SOURCE_TARGET, \
            f"{path[-1]}"
        # TODO SIM-3222 hard-coded assumption that last segment is 1
        assert path[-1].scheduled_at - path[-2].scheduled_at == 1, f"{path[-2:]}"
        path = path[:-1]
        return path

    def get_objective_value(self) -> float:
        return self.asp_solution.stats['summary']['costs'][0]

    def get_solve_time(self) -> float:
        """only solve time of the solver."""
        return self.asp_solution.stats["summary"]["times"]["solve"]

    def get_total_time(self) -> float:
        """total timeo of the solver."""
        return self.asp_solution.stats["summary"]["times"]["total"]

    def get_preprocessing_time(self) -> float:
        """total time minus solve time of the solver."""
        return self.get_total_time() - self.get_solve_time()

    def extract_list_of_lates(self) -> List[str]:
        return list(filter(lambda s: s.startswith('late('), self.answer_set))

    def extract_list_of_active_penalty(self) -> List[str]:
        return list(filter(lambda s: s.startswith('active_penalty('), self.answer_set))

    def extract_nb_resource_conflicts(self) -> int:
        return len(list(filter(lambda s: s.startswith('shared('), self.answer_set)))
