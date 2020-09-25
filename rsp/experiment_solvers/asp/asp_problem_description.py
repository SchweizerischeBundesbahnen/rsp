from __future__ import print_function

from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import networkx as nx
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.experiment_solvers.asp.asp_helper import flux_helper
from rsp.experiment_solvers.asp.asp_solution_description import ASPSolutionDescription
from rsp.experiment_solvers.asp.data_types import ASPHeuristics
from rsp.experiment_solvers.asp.data_types import ASPObjective
from rsp.schedule_problem_description.data_types_and_utils import get_sinks_for_topo
from rsp.schedule_problem_description.data_types_and_utils import get_sources_for_topo
from rsp.schedule_problem_description.data_types_and_utils import ScheduleProblemDescription
from rsp.schedule_problem_description.route_dag_constraints.route_dag_constraints_schedule import RouteDAGConstraints
from rsp.utils.global_constants import DELAY_MODEL_PENALTY_AFTER_LINEAR
from rsp.utils.global_constants import DELAY_MODEL_RESOLUTION
from rsp.utils.global_constants import DELAY_MODEL_UPPER_BOUND_LINEAR_PENALTY
from rsp.utils.global_constants import RESCHEDULE_HEURISTICS
from rsp.utils.global_constants import SCHEDULE_HEURISTICS


class ASPProblemDescription:

    def __init__(self,
                 schedule_problem_description: ScheduleProblemDescription,
                 asp_objective: ASPObjective = ASPObjective.MINIMIZE_SUM_RUNNING_TIMES,
                 asp_heuristics: Optional[List[ASPHeuristics]] = None,
                 asp_seed_value: Optional[int] = None,
                 nb_threads: int = 2,
                 no_optimize: bool = False
                 ):
        self.schedule_problem_description = schedule_problem_description
        self.asp_seed_value = asp_seed_value
        self.asp_objective: ASPObjective = asp_objective
        self.nb_threads = nb_threads
        self.no_optimize = no_optimize
        self.asp_heuristics: Optional[List[ASPHeuristics]] = asp_heuristics

    @staticmethod
    def factory_rescheduling(
            schedule_problem_description: ScheduleProblemDescription,
            additional_costs_at_targets: Dict[int, Dict[Waypoint, int]] = None,
            asp_seed_value: Optional[int] = None
    ) -> 'ASPProblemDescription':
        asp_problem = ASPProblemDescription(
            schedule_problem_description=schedule_problem_description,
            asp_objective=ASPObjective.MINIMIZE_DELAY_ROUTES_COMBINED,
            asp_heuristics=RESCHEDULE_HEURISTICS,
            asp_seed_value=asp_seed_value,
            no_optimize=False  # Optimize if set to False
        )
        asp_problem._build_asp_program(
            schedule_problem_description=schedule_problem_description,
            add_minimumrunnigtime_per_agent=False,
            additional_costs_at_targets=additional_costs_at_targets
        )
        return asp_problem

    @staticmethod
    def factory_scheduling(
            schedule_problem_description: ScheduleProblemDescription,
            asp_seed_value: Optional[int] = None,
            no_optimize: bool = False
    ) -> 'ASPProblemDescription':
        asp_problem = ASPProblemDescription(
            schedule_problem_description=schedule_problem_description,
            asp_objective=ASPObjective.MINIMIZE_SUM_RUNNING_TIMES,
            asp_heuristics=SCHEDULE_HEURISTICS,
            asp_seed_value=asp_seed_value,
            no_optimize=no_optimize,
            nb_threads=2  # not deterministic any more!
        )
        asp_problem._build_asp_program(
            schedule_problem_description=schedule_problem_description,
            # minimize_total_sum_of_running_times.lp requires minimumrunningtime(agent_id,<minimumrunningtime)
            add_minimumrunnigtime_per_agent=True
        )
        return asp_problem

    @staticmethod
    def _sanitize_waypoint(waypoint: Waypoint):
        return tuple([tuple(waypoint.position), int(waypoint.direction)])

    def _implement_train(self, agent_id: int, start_vertices: List[Waypoint], target_vertices: List[Waypoint]):
        """Rule 2 each train is scheduled.

        Parameters
        ----------
        agent_id
        start_vertices
        target_vertices
        """

        self.asp_program.append("train(t{}).".format(agent_id))
        for start_waypoint in start_vertices:
            self.asp_program.append("start(t{}, {}).".format(agent_id, self._sanitize_waypoint(start_waypoint)))
        for target_waypoint in target_vertices:
            self.asp_program.append("end(t{}, {}).".format(agent_id, self._sanitize_waypoint(target_waypoint)))

    def _implement_route_section(self, agent_id: int,
                                 entry_waypoint: Waypoint,
                                 exit_waypoint: Waypoint,
                                 resource_id: Tuple[int, int],
                                 minimum_travel_time: int = 1,
                                 route_section_penalty=0):
        """Rule 103 Minimum section time For each trainrun_section the
        following holds:

                 texit - tentry >= minimum_running_time + min_stopping_time, where
                 tentry, texit are the entry and exit times into this trainrun_section,
                 minimum_running_time is given by the route_section corresponding to this trainrun_section and
                 min_stopping_time is given by the section_requirement corresponding to this trainrun_section or equal
                 to 0 (zero) if no section_requirement with a min_stopping_time is associated to this trainrun_section.

        Rule 104 Resource Occupations
                 In prose, this means that if a train T1 starts occupying a resource R before train T2, then T2 has to
                 wait until T1 releases it (plus the release time of the resource) before it can start to occupy it.

                 This rule explicitly need not hold between trainrun_sections of the same train.
                 The problem instances are infeasible if you require this separation of occupations also
                 among trainrun_sections of the same train.

        Parameters
        ----------
        """

        # add edge: edge(T,V,V')
        self.asp_program.append("edge(t{}, {},{}).".format(agent_id, self._sanitize_waypoint(entry_waypoint),
                                                           self._sanitize_waypoint(exit_waypoint)))

        # minimum waiting time: w(T,E,W)
        # TODO workaround we use waiting times to model train-specific minimum travel time;
        #      instead we should use train-specific route graphs which are linked by resources only!
        self.asp_program.append(
            "w(t{}, ({},{}),{}).".format(agent_id, self._sanitize_waypoint(entry_waypoint),
                                         self._sanitize_waypoint(exit_waypoint),
                                         int(minimum_travel_time)))
        # minimum running time: m(E,M)
        self.asp_program.append("m(({},{}),{}).".format(self._sanitize_waypoint(entry_waypoint),
                                                        self._sanitize_waypoint(exit_waypoint), 0))

        # declare resource
        # TODO SIM-144 resource may be declared multiple times, maybe we should declare resource by
        #      loop over grid cells and their transitions
        self.asp_program.append(
            "resource(resource_{}_{},({},{})).".format(*resource_id, self._sanitize_waypoint(entry_waypoint),
                                                       self._sanitize_waypoint(exit_waypoint)))

        # TODO SIM-129: release time = 1 to allow for synchronization in FLATland - can we get rid of it?
        self.asp_program.append("b(resource_{}_{},1).".format(*resource_id))

        # add train-specific route penalty (T,E,P) for minimize_delay_and_routes_combined.lp
        # N.B. we do not use penalty(E,P) as for objective minimize_routes.lp and heuristic_ROUTES.lp
        if route_section_penalty > 0:
            # penalty(T,E,P) # noqa: E800
            self.asp_program.append("penalty(t{},({},{}),{})."
                                    .format(agent_id, self._sanitize_waypoint(entry_waypoint),
                                            self._sanitize_waypoint(exit_waypoint), route_section_penalty))

    def solve(self, verbose: bool = False) -> ASPSolutionDescription:
        """Return the solver and return solver-specific solution
        description."""
        asp_solution = flux_helper(self.asp_program,
                                   asp_objective=self.asp_objective,
                                   asp_heuristics=self.asp_heuristics,
                                   asp_seed_value=self.asp_seed_value,
                                   nb_threads=self.nb_threads,
                                   no_optimize=self.no_optimize,
                                   verbose=verbose)
        return ASPSolutionDescription(asp_solution=asp_solution, schedule_problem_description=self.schedule_problem_description)

    @staticmethod
    def convert_position_and_entry_direction_to_waypoint(r: int, c: int, d: int) -> Waypoint:
        """
        Parameters
        ----------
        r
            row
        c
            column
        d
            direction we are facing in the cell (and have entered by)

        Returns
        -------
        row, column, entry direction
        """
        return Waypoint(position=(r, c),
                        direction=int(d))  # convert Grid4TransitionsEnum to int so it can be used as int in ASP!

    def _build_asp_program(self,
                           schedule_problem_description: ScheduleProblemDescription,
                           add_minimumrunnigtime_per_agent: bool = False,
                           additional_costs_at_targets: Dict[int, Dict[Waypoint, int]] = None
                           ):
        # preparation
        _new_asp_program = []
        self.asp_program = _new_asp_program

        # dirty work around to silence ASP complaining "info: atom does not occur in any rule head"
        # (we don't use all features in encoding.lp)
        self.asp_program.append("bridge(0,0,0).")
        self.asp_program.append("edge(0,0,0,0).")
        self.asp_program.append("relevant(0,0,0).")
        self.asp_program.append("m(0,1).")
        self.asp_program.append("connection(0,(0,0),0,(0,0),0).")
        self.asp_program.append("penalty(0,0).")
        self.asp_program.append("penalty(0,0,0).")
        # initially, not act_penalty_for_train activated
        self.asp_program.append("act_penalty_for_train(0,0,0).")

        # add trains
        for agent_id, topo in schedule_problem_description.topo_dict.items():
            sources = get_sources_for_topo(topo)
            sinks = get_sinks_for_topo(topo)
            freeze = schedule_problem_description.route_dag_constraints_dict[agent_id]

            self._implement_train(agent_id=agent_id,
                                  start_vertices=sources,
                                  target_vertices=sinks
                                  )

            for (entry_waypoint, exit_waypoint) in topo.edges:
                self._implement_route_section(
                    agent_id=agent_id,
                    entry_waypoint=entry_waypoint,
                    exit_waypoint=exit_waypoint,
                    resource_id=entry_waypoint.position,
                    minimum_travel_time=schedule_problem_description.minimum_travel_time_dict[agent_id],
                    route_section_penalty=schedule_problem_description.route_section_penalties[agent_id].get((entry_waypoint, exit_waypoint), 0))

            _new_asp_program += self._translate_route_dag_constraints_to_asp(
                agent_id=agent_id,
                topo=schedule_problem_description.topo_dict[agent_id],
                freeze=freeze)

        if add_minimumrunnigtime_per_agent:
            for agent_id in self.schedule_problem_description.minimum_travel_time_dict:
                agent_sink = list(get_sinks_for_topo(self.schedule_problem_description.topo_dict[agent_id]))[0]
                agent_source = list(get_sources_for_topo(self.schedule_problem_description.topo_dict[agent_id]))[0]
                earliest_arrival = self.schedule_problem_description.route_dag_constraints_dict[agent_id].earliest[agent_sink]
                earliest_departure = self.schedule_problem_description.route_dag_constraints_dict[agent_id].earliest[agent_source]
                minimum_running_time = earliest_arrival - earliest_departure
                self.asp_program.append("minimumrunningtime(t{},{}).".format(agent_id, minimum_running_time))
        if additional_costs_at_targets is not None:
            for agent_id, target_penalties in additional_costs_at_targets.items():
                for waypoint, penalty in target_penalties.items():
                    vertex = self._sanitize_waypoint(waypoint)
                    self.asp_program.append("targetpenalty(t{},{},{}).".format(agent_id, vertex, penalty))

        # inject weight lateness
        if self.asp_objective == ASPObjective.MINIMIZE_DELAY_ROUTES_COMBINED:
            _new_asp_program.append(f"#const weight_lateness_seconds = {schedule_problem_description.weight_lateness_seconds}.")

        # inject delay model parameterization
        _new_asp_program.append(f"#const upper_bound_linear_penalty = {DELAY_MODEL_UPPER_BOUND_LINEAR_PENALTY}.")
        _new_asp_program.append(f"#const penalty_after_linear = {DELAY_MODEL_PENALTY_AFTER_LINEAR}.")
        _new_asp_program.append(f"#const resolution = {DELAY_MODEL_RESOLUTION}.")

        # cleanup
        return _new_asp_program

    def _translate_route_dag_constraints_to_asp(self,  # noqa: C901
                                                agent_id: int,
                                                topo: nx.DiGraph,
                                                freeze: RouteDAGConstraints):
        """The model is freezed by translating the ExperimentFreeze into ASP:

        - for all schedule_time_(train,vertex) <= malfunction.time_step:
           - freeze visit(train,vertex) == True
           - dl(train,vertex) == schedule_time_(train,vertex)
        - for all trains train for first vertex s.t. schedule_time_(train,vertex) > malfunction.time_step
           - freeze visit(train,vertex) == True
           - if train == malfunction.agent_id, add constraint
             -  dl(train,vertex) >=  schedule_time(train,previous_vertex) + 1/agent_speed + malfunction.malfunction_duration
        - for all trains and all their vertices s.t. schedule_time_(train,vertex) > malfunction.time_step or schedule_time_(train,vertex) == None
           -  dl(train,vertex) >= malfunction.time_step
           (in particular, if a train has not entered yet, it must not enter in the re-scheduling before the malfunction)

        Parameters
        ----------
        agent_id
        freeze
        """
        frozen: List[TrainrunWaypoint] = []
        train = "t{}".format(agent_id)

        # 2019-12-03 discussion with Potsdam (SIM-146)
        # - no diff-constraints in addition to earliest/latest -> should be added immediately
        # - no route constraints in addition to visit -> should be added immediately
        for waypoint, scheduled_at in freeze.latest.items():
            vertex = self._sanitize_waypoint(waypoint)
            time = scheduled_at

            # add earliest constraint
            # l(t1,1,2).
            frozen.append(f"l({train},{vertex},{time}).")

        for waypoint, scheduled_at in freeze.earliest.items():
            vertex = self._sanitize_waypoint(waypoint)
            time = scheduled_at

            # add earliest constraint
            # e(t1,1,2).
            frozen.append(f"e({train},{vertex},{time}).")

        return frozen
