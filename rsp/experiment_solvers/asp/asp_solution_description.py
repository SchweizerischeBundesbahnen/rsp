import re
from typing import List
from typing import Set

from flatland.envs.rail_trainrun_data_structures import Trainrun
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.experiment_solvers.asp.asp_helper import FluxHelperResult
from rsp.route_dag.route_dag import get_sinks_for_topo
from rsp.route_dag.route_dag import get_sources_for_topo
from rsp.route_dag.route_dag import MAGIC_DIRECTION_FOR_SOURCE_TARGET
from rsp.route_dag.route_dag import ScheduleProblemDescription


class ASPSolutionDescription():

    def __init__(self,
                 asp_solution: FluxHelperResult,
                 tc: ScheduleProblemDescription
                 ):
        self.asp_solution: FluxHelperResult = asp_solution
        self.answer_set: Set[str] = self.asp_solution.answer_sets[0]
        self._action_plan = None
        self.tc: ScheduleProblemDescription = tc

    def verify_correctness(self):
        self.__class__.verify_correctness_helper(self.tc, self.answer_set)

    @staticmethod  # noqa: C901
    def verify_correctness_helper(tc: ScheduleProblemDescription, answer_set: Set[str]):
        """Verify that solution is consistent."""

        trainrun_dict = {}

        for agent_id in tc.topo_dict:
            var_prefix = "dl((t{},".format(agent_id)
            agent_facts = filter(lambda s: s.startswith(str(var_prefix)), answer_set)
            source_waypoints = list(get_sources_for_topo(tc.topo_dict[agent_id]))
            sink_waypoints = list(get_sinks_for_topo(tc.topo_dict[agent_id]))
            route_dag_constraints = tc.route_dag_constraints_dict[agent_id]

            minimum_running_time = tc.minimum_travel_time_dict[agent_id]
            topo = tc.topo_dict[agent_id]

            # filter out dl entries that are zero and not relevant to us
            trainrun_waypoints = list(map(ASPSolutionDescription._parse_dl_fact, agent_facts))
            trainrun_waypoints.sort(key=lambda p: p.scheduled_at)
            trainrun_dict[agent_id] = trainrun_waypoints
            waypoints = {trainrun_waypoint.waypoint for trainrun_waypoint in trainrun_waypoints}
            schedule = {trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at for trainrun_waypoint in trainrun_waypoints}

            # 1. verify consistency of solution: there is a single path satisfying minimum_run_time

            # 1.1 verify trainrun is *strictly* increasing
            for wp_1, wp_2 in zip(trainrun_waypoints, trainrun_waypoints[1:]):
                assert wp_2.scheduled_at > wp_1.scheduled_at, \
                    f"(1.1) [{agent_id}] times are not strictly increasing: {wp_1} {wp_2}"

            # 1.2 verify trainrun goes from a source to a sink
            assert trainrun_waypoints[0].waypoint in source_waypoints, \
                f"(1.2) [{agent_id}] unexpected source: {trainrun_waypoints[0].waypoint} not in {source_waypoints}"
            assert trainrun_waypoints[-1].waypoint in sink_waypoints, \
                f"(1.2) [{agent_id}] unexpected sink: {trainrun_waypoints[-1].waypoint} not in {sink_waypoints}"

            # 1.3 verify that the dummy sections (first and last) have time 1
            assert trainrun_waypoints[1].scheduled_at - trainrun_waypoints[0].scheduled_at == 1, \
                f"(1.3) [{agent_id}] dummy section at source should take exactly 1: " \
                f"found {trainrun_waypoints[0].scheduled_at} - {trainrun_waypoints[1].scheduled_at}"
            assert trainrun_waypoints[-1].scheduled_at - trainrun_waypoints[-2].scheduled_at == 1, \
                f"(1.3) [{agent_id}] dummy section at sink should take exactly 1: " \
                f"found {trainrun_waypoints[-2].scheduled_at} - {trainrun_waypoints[-1].scheduled_at}"

            # 1.4 verify minimimum_running_time is respected for all but first and last segment
            for wp_1, wp_2 in zip(trainrun_waypoints[1:], trainrun_waypoints[2:-1]):
                assert wp_2.scheduled_at - wp_1.scheduled_at >= minimum_running_time, \
                    f"(1.4) [{agent_id}] minimum running time not respected: " \
                    f"found {wp_1} - {wp_2}, but minimum_running_time={minimum_running_time}"

            # 1.5 verify that trainrun satisfies topology
            for wp_1, wp_2 in zip(trainrun_waypoints, trainrun_waypoints[1:]):
                assert (wp_1.waypoint, wp_2.waypoint) in topo.edges, \
                    f"(1.5) [{agent_id}] no edge for {wp_1} - {wp_2}"

            # 1.6 verify path has no cycles
            assert len(set(waypoints)) == len(waypoints), \
                f"(1.6) [{agent_id}] cycle"

            # 2. verify solution satisfies constraints:
            for waypoint in route_dag_constraints.freeze_visit:
                assert waypoint in waypoints, \
                    f"(2) [{agent_id}] freeze_visit violated: " \
                    f"{waypoint} must be visited"
            for waypoint, earliest in route_dag_constraints.freeze_earliest.items():
                if waypoint in schedule:
                    assert schedule[waypoint] >= earliest, \
                        f"(2) [{agent_id}] freeze_earliest violated: " \
                        f"{waypoint} must be not be visited before {earliest}, found {schedule[waypoint]}"
            for waypoint, latest in route_dag_constraints.freeze_latest.items():
                if waypoint in schedule:
                    assert schedule[waypoint] <= latest, \
                        f"(2) [{agent_id}] freeze_latest violated: " \
                        f"{waypoint} must be not be visited after {latest}, found {schedule[waypoint]}"
            for waypoint in route_dag_constraints.freeze_banned:
                assert waypoint not in waypoints, \
                    f"(2) [{agent_id}] freeze_banned violated: " \
                    f"{waypoint} must be not be visited"

        # 3. verify mututal exclusion and release time
        resource_occupations = {}
        for agent_id, trainrun in trainrun_dict.items():
            for wp1, wp2 in zip(trainrun, trainrun[1:]):
                resource = wp1.waypoint.position
                # TODO SIM-129 release time 1 hard-coded
                for time in range(wp1.scheduled_at, wp2.scheduled_at + 1):
                    occupation = (resource, time)
                    if occupation in resource_occupations:
                        assert agent_id == resource_occupations[occupation], \
                            f"(3) conflicting resource occuptions {occupation} for {agent_id} and {resource_occupations[occupation]}"
                    resource_occupations[occupation] = agent_id

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
        # TODO SIM-322 hard-coded assumption that last segment is 1
        assert path[-1].scheduled_at - path[-2].scheduled_at == 1, f"{path[-2:]}"
        path = path[:-1]
        return path

    def get_objective_value(self) -> float:
        costs_ = self.asp_solution.stats['summary']['costs']
        return costs_[0] if len(costs_) == 1 else -1

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
