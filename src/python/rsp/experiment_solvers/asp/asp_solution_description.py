import re
from typing import List
from typing import Set

from flatland.envs.rail_trainrun_data_structures import Trainrun
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint
from rsp.experiment_solvers.asp.asp_helper import FluxHelperResult
from rsp.schedule_problem_description.data_types_and_utils import get_sinks_for_topo
from rsp.schedule_problem_description.data_types_and_utils import get_sources_for_topo
from rsp.schedule_problem_description.data_types_and_utils import ScheduleProblemDescription


class ASPSolutionDescription:
    def __init__(self, asp_solution: FluxHelperResult, schedule_problem_description: ScheduleProblemDescription):
        self.asp_solution: FluxHelperResult = asp_solution
        self.answer_set: Set[str] = self.asp_solution.answer_sets[0]
        self.schedule_problem_description: ScheduleProblemDescription = schedule_problem_description

    def verify_correctness(self):
        self.__class__.verify_correctness_helper(self.schedule_problem_description, self.answer_set)

    # TODO SIM-517 harmonize with verify trainruns?
    @staticmethod  # noqa: C901
    def verify_correctness_helper(schedule_problem_description: ScheduleProblemDescription, answer_set: Set[str]):
        """Verify that solution is consistent."""

        trainrun_dict = {}

        for agent_id in schedule_problem_description.topo_dict:
            var_prefix = "dl((t{},".format(agent_id)
            agent_facts = filter(lambda s: s.startswith(str(var_prefix)), answer_set)
            source_waypoints = list(get_sources_for_topo(schedule_problem_description.topo_dict[agent_id]))
            sink_waypoints = list(get_sinks_for_topo(schedule_problem_description.topo_dict[agent_id]))
            route_dag_constraints = schedule_problem_description.route_dag_constraints_dict[agent_id]

            minimum_running_time = schedule_problem_description.minimum_travel_time_dict[agent_id]
            topo = schedule_problem_description.topo_dict[agent_id]

            # filter out dl entries that are zero and not relevant to us
            trainrun_waypoints = list(map(ASPSolutionDescription._parse_dl_fact, agent_facts))
            trainrun_waypoints.sort(key=lambda p: p.scheduled_at)
            trainrun_dict[agent_id] = trainrun_waypoints
            waypoints = {trainrun_waypoint.waypoint for trainrun_waypoint in trainrun_waypoints}
            schedule = {trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at for trainrun_waypoint in trainrun_waypoints}

            # 1. verify consistency of solution: there is a single path satisfying minimum_run_time

            # 1.1 verify trainrun is *strictly* increasing
            for wp_1, wp_2 in zip(trainrun_waypoints, trainrun_waypoints[1:]):
                assert wp_2.scheduled_at > wp_1.scheduled_at, f"(1.1) [{agent_id}] times are not strictly increasing: {wp_1} {wp_2}"

            # 1.2 verify trainrun goes from a source to a sink
            assert (
                trainrun_waypoints[0].waypoint in source_waypoints
            ), f"(1.2) [{agent_id}] unexpected source: {trainrun_waypoints[0].waypoint} not in {source_waypoints}"
            assert (
                trainrun_waypoints[-1].waypoint in sink_waypoints
            ), f"(1.2) [{agent_id}] unexpected sink: {trainrun_waypoints[-1].waypoint} not in {sink_waypoints}"

            # 1.3 verify minimimum_running_time is respected
            for wp_1, wp_2 in zip(trainrun_waypoints, trainrun_waypoints[1:]):
                assert wp_2.scheduled_at - wp_1.scheduled_at >= minimum_running_time, (
                    f"(1.4) [{agent_id}] minimum running time not respected: " f"found {wp_1} - {wp_2}, but minimum_running_time={minimum_running_time}"
                )

            # 1.4 verify that trainrun satisfies topology
            for wp_1, wp_2 in zip(trainrun_waypoints, trainrun_waypoints[1:]):
                assert (wp_1.waypoint, wp_2.waypoint) in topo.edges, f"(1.5) [{agent_id}] no edge for {wp_1} - {wp_2}"

            # 1.5 verify path has no cycles
            assert len(set(waypoints)) == len(waypoints), f"(1.6) [{agent_id}] cycle"

            # 2. verify solution satisfies constraints:
            for waypoint, earliest in route_dag_constraints.earliest.items():
                if waypoint in schedule:
                    assert schedule[waypoint] >= earliest, (
                        f"(2) [{agent_id}] earliest violated: " f"{waypoint} must be not be visited before {earliest}, found {schedule[waypoint]}"
                    )
            for waypoint, latest in route_dag_constraints.latest.items():
                if waypoint in schedule:
                    assert schedule[waypoint] <= latest, (
                        f"(2) [{agent_id}] latest violated: " f"{waypoint} must be not be visited after {latest}, found {schedule[waypoint]}"
                    )

        # 3. verify mututal exclusion and release time
        resource_occupations = {}
        for agent_id, trainrun in trainrun_dict.items():
            for wp1, wp2 in zip(trainrun, trainrun[1:]):
                resource = wp1.waypoint.position
                # TODO SIM-129 release time 1 hard-coded
                for time in range(wp1.scheduled_at, wp2.scheduled_at + 1):
                    occupation = (resource, time)
                    if occupation in resource_occupations:
                        assert (
                            agent_id == resource_occupations[occupation]
                        ), f"(3) conflicting resource occuptions {occupation} for {agent_id} and {resource_occupations[occupation]}"
                    resource_occupations[occupation] = agent_id

    def get_trainruns_dict(self) -> TrainrunDict:
        """Get train runs for all agents: waypoints and entry times."""
        return {agent_id: self.get_trainrun_for_agent(agent_id) for agent_id in self.schedule_problem_description.topo_dict}

    def is_solved(self):
        """Is the model satisfiable, is there any solution?"""
        # take stats of last multi-shot call
        return self.asp_solution.stats["summary"]["models"]["enumerated"] > 0

    @staticmethod
    def _parse_dl_fact(value: str) -> TrainrunWaypoint:
        # dl((t0,((3,5),3)),5) # NOQA
        p = re.compile(r"dl\(\(t[^,]+,\(\(([0-9]+),([0-9]+)\),(.+)\)\),([0-9]+)\)")
        m = p.match(value)
        r = int(m.group(1))
        c = int(m.group(2))
        d = int(m.group(3))
        entry = int(m.group(4))

        return TrainrunWaypoint(scheduled_at=entry, waypoint=Waypoint(position=(r, c), direction=d))

    def get_trainrun_for_agent(self, agent_id: int) -> Trainrun:
        """Get train run of the agent in the solution."""
        return self._get_solution_trainrun(agent_id)

    def _get_solution_trainrun(self, agent_id) -> Trainrun:
        var_prefix = "dl((t{},".format(agent_id)
        agent_facts = filter(lambda s: s.startswith(str(var_prefix)), self.answer_set)
        start_waypoint = list(get_sources_for_topo(self.schedule_problem_description.topo_dict[agent_id]))[0]
        # filter out dl entries that are zero and not relevant to us
        path = list(filter(lambda pse: pse.scheduled_at > 0 or pse.waypoint == start_waypoint, map(self.__class__._parse_dl_fact, agent_facts)))
        path.sort(key=lambda p: p.scheduled_at)
        return path

    def get_objective_value(self) -> float:
        costs_ = self.asp_solution.stats["summary"]["costs"]
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
        return list(filter(lambda s: s.startswith("late("), self.answer_set))

    def extract_list_of_active_penalty(self) -> List[str]:
        return list(filter(lambda s: s.startswith("active_penalty("), self.answer_set))

    def extract_nb_resource_conflicts(self) -> int:
        return len(list(filter(lambda s: s.startswith("shared("), self.answer_set)))
