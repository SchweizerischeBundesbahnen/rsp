from __future__ import print_function

from typing import List
from typing import Tuple

from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.experiment_solvers.asp.asp_helper import ASPHeuristics
from rsp.experiment_solvers.asp.asp_helper import ASPObjective
from rsp.experiment_solvers.asp.asp_helper import flux_helper
from rsp.experiment_solvers.asp.asp_solution_description import ASPSolutionDescription
from rsp.route_dag.generators.route_dag_generator_schedule import RouteDAGConstraints
from rsp.route_dag.route_dag import get_sinks_for_topo
from rsp.route_dag.route_dag import get_sources_for_topo
from rsp.route_dag.route_dag import MAGIC_DIRECTION_FOR_SOURCE_TARGET
from rsp.route_dag.route_dag import ScheduleProblemDescription


class ASPProblemDescription():

    def __init__(self,
                 tc: ScheduleProblemDescription,
                 asp_objective: ASPObjective = ASPObjective.MINIMIZE_SUM_RUNNING_TIMES,
                 asp_heuristics: List[ASPHeuristics] = None
                 ):
        self.tc = tc
        self.asp_objective: ASPObjective = asp_objective
        if asp_heuristics is None:
            self.asp_heuristics: List[ASPHeuristics] = [ASPHeuristics.HEURISIC_ROUTES, ASPHeuristics.HEURISTIC_SEQ]
        else:
            self.asp_heuristics: List[ASPHeuristics] = asp_heuristics

    @staticmethod
    def factory_rescheduling(
            tc: ScheduleProblemDescription,
    ) -> 'ASPProblemDescription':
        asp_problem = ASPProblemDescription(
            tc=tc,
            asp_objective=ASPObjective.MINIMIZE_DELAY_ROUTES_COMBINED,
            # TODO SIM-167 switch on heuristics
            asp_heuristics=[ASPHeuristics.HEURISIC_ROUTES, ASPHeuristics.HEURISTIC_SEQ, ASPHeuristics.HEURISTIC_DELAY]
        )
        asp_problem.asp_program: List[str] = asp_problem._build_asp_program(
            tc=tc,
            add_minimumrunnigtime_per_agent=False
        )
        return asp_problem

    @staticmethod
    def factory_scheduling(
            tc: ScheduleProblemDescription
    ) -> 'ASPProblemDescription':
        asp_problem = ASPProblemDescription(
            tc=tc,
            asp_objective=ASPObjective.MINIMIZE_SUM_RUNNING_TIMES,
            # TODO SIM-167 switch on heuristics
            asp_heuristics=[ASPHeuristics.HEURISIC_ROUTES, ASPHeuristics.HEURISTIC_SEQ, ASPHeuristics.HEURISTIC_DELAY]
        )
        asp_problem.asp_program: List[str] = asp_problem._build_asp_program(
            tc=tc,
            # minimize_total_sum_of_running_times.lp requires minimumrunningtime(agent_id,<minimumrunningtime)
            add_minimumrunnigtime_per_agent=True
        )
        return asp_problem

    def get_solver_name(self) -> str:
        """Return the solver name for printing."""
        return "ASP"

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
        minimum_running_time
        """

        self.asp_program.append("train(t{}).".format(agent_id))
        for start_waypoint in start_vertices:
            self.asp_program.append("start(t{}, {}).".format(agent_id, self._sanitize_waypoint(start_waypoint)))
        for target_waypoint in target_vertices:
            self.asp_program.append("end(t{}, {}).".format(agent_id, self._sanitize_waypoint(target_waypoint)))

    def _implement_agent_earliest(self, agent_id: int, waypoint: Waypoint, time: int):
        """Rule 102 Time windows for earliest-requirements. If a
        section_requirement specifies an entry_earliest and/or exit_earliest
        time, then the event times for the entry_event and/or exit_event on the
        corresponding trainrun_section MUST be >= the specified time.

        Parameters
        ----------
        agent_id
        waypoint
        time
        """
        # ASP fact e(T,V,E)
        self.asp_program.append("e(t{},{},{}).".format(agent_id, self._sanitize_waypoint(waypoint), int(time)))

    def _implement_agent_latest(self, agent_id: int, waypoint: Waypoint, time: int):
        """Rule 101 Time windows for latest-requirements. If a
        section_requirement specifies a entry_latest and/or exit_latest time
        then the event times for the entry_event and/or exit_event on the
        corresponding trainrun_section SHOULD be <= the specified time. If the
        scheduled time is later than required, the solution will still be
        accepted, but it will be penalized by the objective function, see
        below. Important Remark: Among the 12 business rules, this is the only
        'soft' constraint, i.e. a rule that may be violated and the solution
        still accepted.

        Parameters
        ----------
        agent_id
        waypoint
        time
        """

        # ASP fact l(T,V,E)
        self.asp_program.append("l(t{},{},{}).".format(agent_id, self._sanitize_waypoint(waypoint), int(time)))

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

        # ensure that dummy edge at source and target take exactly 1 by forcing also <= 1 (>= is enforced by minimum running time)
        # TODO SIM-322 hard-coded value 1
        self.asp_program.append("&diff{ (T,V')-(T,V) }  <= 1:- start(T,V), visit(T,V), visit(T,V'), edge(T,V,V').")
        self.asp_program.append("&diff{ (T,V')-(T,V) }  <= 1:- end(T,V'), visit(T,V), visit(T,V'), edge(T,V,V').")

        asp_solution = flux_helper(self.asp_program,
                                   bound_all_events=self.tc.max_episode_steps,
                                   asp_objective=self.asp_objective,
                                   asp_heurisics=self.asp_heuristics,
                                   verbose=verbose)
        return ASPSolutionDescription(asp_solution=asp_solution, tc=self.tc)

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
                           tc: ScheduleProblemDescription,
                           add_minimumrunnigtime_per_agent: bool = False,
                           ):
        # preparation
        _new_asp_program = []
        self.asp_program = _new_asp_program

        # add trains
        for agent_id, topo in tc.topo_dict.items():
            sources = get_sources_for_topo(topo)
            sinks = get_sinks_for_topo(topo)

            self._implement_train(agent_id=agent_id,
                                  start_vertices=sources,
                                  target_vertices=sinks
                                  )

            for (entry_waypoint, exit_waypoint) in topo.edges:
                # TODO SIM-322 hard-coded assumptions on dummy edges
                is_dummy_edge = (
                        entry_waypoint.direction == MAGIC_DIRECTION_FOR_SOURCE_TARGET or exit_waypoint.direction ==
                        MAGIC_DIRECTION_FOR_SOURCE_TARGET)
                self._implement_route_section(
                    agent_id=agent_id,
                    entry_waypoint=entry_waypoint,
                    exit_waypoint=exit_waypoint,
                    resource_id=entry_waypoint.position,
                    minimum_travel_time=(1
                                         if is_dummy_edge
                                         else tc.minimum_travel_time_dict[agent_id]),
                    route_section_penalty=tc.route_section_penalties[agent_id].get(
                        (entry_waypoint, exit_waypoint), 0) * tc.weight_route_change if not is_dummy_edge else 0
                )

            _new_asp_program += self._translate_route_dag_constraints_to_ASP(agent_id=agent_id,
                                                                             freeze=tc.route_dag_constraints_dict[
                                                                                 agent_id])

        if add_minimumrunnigtime_per_agent:
            for agent_id in self.tc.minimum_travel_time_dict:
                agent_sink = list(get_sinks_for_topo(self.tc.topo_dict[agent_id]))[0]
                agent_source = list(get_sources_for_topo(self.tc.topo_dict[agent_id]))[0]
                earliest_arrival = self.tc.route_dag_constraints_dict[agent_id].freeze_earliest[agent_sink]
                earliest_departure = self.tc.route_dag_constraints_dict[agent_id].freeze_earliest[agent_source]
                minimum_running_time = earliest_arrival - earliest_departure
                self.asp_program.append("minimumrunningtime(t{},{}).".format(agent_id, minimum_running_time))

        # inject weight lateness
        if self.asp_objective == ASPObjective.MINIMIZE_DELAY_ROUTES_COMBINED:
            _new_asp_program.append(f"#const weight_lateness_seconds = {tc.weight_lateness_seconds}.")

        # cleanup
        return _new_asp_program

    def _translate_route_dag_constraints_to_ASP(self,
                                                agent_id: int,
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

        for trainrun_waypoint in freeze.freeze_visit:
            vertex = self._sanitize_waypoint(trainrun_waypoint)

            # add visit constraint
            # visit(t1,1).
            frozen.append(f"visit({train},{vertex}).")

        for waypoint, scheduled_at in freeze.freeze_latest.items():
            vertex = self._sanitize_waypoint(waypoint)
            time = scheduled_at

            # add earliest constraint
            # l(t1,1,2).
            frozen.append(f"l({train},{vertex},{time}).")

        for waypoint, scheduled_at in freeze.freeze_earliest.items():
            vertex = self._sanitize_waypoint(waypoint)
            time = scheduled_at

            # add earliest constraint
            # e(t1,1,2).
            frozen.append(f"e({train},{vertex},{time}).")

        for waypoint in freeze.freeze_banned:
            vertex = self._sanitize_waypoint(waypoint)
            #  vertex must not be visited
            # visit(t1,1).
            frozen.append(f":- visit({train},{vertex}).")

        return frozen
