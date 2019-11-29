from __future__ import print_function

from typing import Dict, Optional, List, Tuple, Set

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from overrides import overrides

from rsp.abstract_problem_description import AbstractProblemDescription, Waypoint
from rsp.asp.asp_solution_description import ASPSolutionDescription
from rsp.asp.asp_solver import flux_helper, ASPObjective, ASPHeuristics
from rsp.utils.data_types import ExperimentMalfunction


class ASPProblemDescription(AbstractProblemDescription):

    def __init__(self,
                 env: RailEnv,
                 agents_path_dict: Dict[int, Optional[List[Tuple[Waypoint]]]],
                 asp_objective: ASPObjective = ASPObjective.MINIMIZE_SUM_RUNNING_TIMES,
                 asp_heuristics: List[ASPHeuristics] = None
                 ):

        self.asp_program: List[str] = []
        self.env: RailEnv = env
        self.asp_objective: ASPObjective = asp_objective
        if asp_heuristics is None:
            self.asp_heuristics: List[ASPHeuristics] = [ASPHeuristics.HEURISIC_ROUTES, ASPHeuristics.HEURISTIC_SEQ]
        else:
            self.asp_heuristics: List[ASPHeuristics] = asp_heuristics
        self.dummy_target_vertices: Optional[Dict[int, List[Waypoint]]] = None

        self.agents_waypoints: Dict[int, Set[Waypoint]] = {
            agent.handle: {waypoint
                           for agent_path in agents_path_dict[agent.handle]
                           for waypoint in agent_path}
            for agent in env.agents}

        super().__init__(env, agents_path_dict, skip_mutual_exclusion=True)

    @overrides
    def get_solver_name(self) -> str:
        """See :class:`AbstractProblemDescription`."""

        # determine number of routing alternatives
        k = max([len(agent_paths) for agent_paths in self.agents_path_dict.values()])
        return "ASP_{}".format(k)

    @overrides
    def _implement_train(self, agent_id: int, start_vertices: List[Waypoint], target_vertices: List[Waypoint],
                         minimum_running_time: int):
        """See :class:`AbstractProblemDescription`."""

        self.asp_program.append("train(t{}).".format(agent_id))
        self.asp_program.append("minimumrunningtime(t{},{}).".format(agent_id, minimum_running_time))
        for start_waypoint in start_vertices:
            self.asp_program.append("start(t{}, {}).".format(agent_id, tuple(start_waypoint)))
        for target_waypoint in target_vertices:
            self.asp_program.append("end(t{}, {}).".format(agent_id, tuple(target_waypoint)))

    @overrides
    def _implement_agent_earliest(self, agent_id: int, waypoint: Waypoint, time: int):
        """See :class:`AbstractProblemDescription`."""

        # ASP fact e(T,V,E)
        self.asp_program.append("e(t{},{},{}).".format(agent_id, tuple(waypoint), int(time)))

    @overrides
    def _implement_agent_latest(self, agent_id: int, waypoint: Waypoint, time: int):
        """See :class:`AbstractProblemDescription`."""

        # ASP fact l(T,V,E)
        self.asp_program.append("l(t{},{},{}).".format(agent_id, tuple(waypoint), int(time)))

    @overrides
    def _implement_route_section(self, agent_id: int, entry_waypoint: Waypoint, exit_waypoint: Waypoint,
                                 resource_id: Tuple[int, int], minimum_travel_time: int = 1, route_section_penalty=0):
        """See :class:`AbstractProblemDescription`."""

        # add edge: edge(T,V,V')
        self.asp_program.append("edge(t{}, {},{}).".format(agent_id, tuple(entry_waypoint), tuple(exit_waypoint)))

        # minimum waiting time: w(T,E,W)
        # TODO workaround we use waiting times to model train-specific minimum travel time;
        #      instead we should use train-specific route graphs which are linked by resources only!
        self.asp_program.append(
            "w(t{}, ({},{}),{}).".format(agent_id, tuple(entry_waypoint), tuple(exit_waypoint),
                                         int(minimum_travel_time)))
        # minimum running time: m(E,M)
        self.asp_program.append("m({}, ({},{}),{}).".format(agent_id, tuple(entry_waypoint), tuple(exit_waypoint), 0))

        # declare resource
        # TODO SIM-144 resource may be declared multiple times, maybe we should declare resource by
        #      loop over grid cells and their transitions
        self.asp_program.append(
            "resource(resource_{}_{},({},{})).".format(*resource_id, tuple(entry_waypoint), tuple(exit_waypoint)))

        # TODO SIM-129: release time = 1 to allow for synchronization in FLATland - can we get rid of it?
        self.asp_program.append("b(resource_{}_{},1).".format(*resource_id))

        # penalty for objective minimize_routes.lp and heuristic_ROUTES.lp (used only if activated)
        if route_section_penalty > 0:
            # penalty(E,P) # noqa: E800
            self.asp_program.append("penalty(({},{}),{})."
                                    .format(tuple(entry_waypoint), tuple(exit_waypoint), route_section_penalty))

    @overrides
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

        # nothing to do here since derived from data
        pass

    @overrides
    def _create_objective_function_minimize(self, variables: Dict[int, List[Waypoint]]):
        """See :class:`AbstractProblemDescription`."""
        # TODO SIM-137 SIM-113 bad code smell: the chosen minimization objective is passed when the ASP solver is called, but added here in ortools
        self.dummy_target_vertices = variables

    @overrides
    def solve(self, verbose: bool = False) -> ASPSolutionDescription:
        """See :class:`AbstractProblemDescription`."""
        # dirty work around to silence ASP complaining "info: atom does not occur in any rule head"
        # (we don't use all features in encoding.lp)
        self.asp_program.append("bridge(0,0,0).")
        self.asp_program.append("edge(0,0,0,0).")
        self.asp_program.append("relevant(0,0,0).")
        self.asp_program.append("m(0,1).")
        self.asp_program.append("connection(0,(0,0),0,(0,0),0).")
        self.asp_program.append("penalty(0,0).")
        # initially, not act_penalty_for_train activated
        self.asp_program.append("act_penalty_for_train(0,0,0).")

        asp_solution = flux_helper(self.asp_program,
                                   bound_all_events=self.env._max_episode_steps,
                                   asp_objective=self.asp_objective,
                                   asp_heurisics=self.asp_heuristics,
                                   verbose=verbose)
        return ASPSolutionDescription(env=self.env, asp_solution=asp_solution)

    def get_freezed_copy_for_rescheduling_full_after_malfunction(self,
                                                                 malfunction: ExperimentMalfunction,
                                                                 freeze: Dict[int, Set[TrainrunWaypoint]],
                                                                 schedule_trainruns: Dict[int, List[TrainrunWaypoint]],
                                                                 verbose: bool = False) -> 'ASPProblemDescription':
        """
        Returns a problem description with additional constraints to freeze all the variables up to malfunction.

        The freeze comes from :meth:`rsp.asp.ASPExperimentSolver._get_freeze_for_trainrun`.


        The ASP constraints are derived by :meth:`rsp.asp.asp_problem_description.ASPProblemDescription._translate_freeze_full_after_malfunction_to_ASP`.

        Parameters
        ----------
        malfunction
            the malfunction
        trainruns_dict
            the schedule
        verbose
            print debug information

        Returns
        -------
        ASPProblemDescription

        """
        freezed_copy = self._prepare_freezed_copy(schedule_trainruns, malfunction)

        for agent_id, trainrun_waypoints in freeze.items():
            f = self._translate_freeze_full_after_malfunction_to_ASP(agent_id, freeze[agent_id], malfunction)
            freezed_copy.asp_program += f

        return freezed_copy

    def get_freezed_copy_for_rescheduling_delta_after_malfunction(self,
                                                                  malfunction: ExperimentMalfunction,
                                                                  freeze: Dict[int, Set[TrainrunWaypoint]],
                                                                  schedule_trainruns: Dict[int, List[TrainrunWaypoint]],
                                                                  verbose: bool = False) -> 'ASPProblemDescription':
        """
        Returns a problem description with additional constraints to freeze all the variables up to malfunction.

        The freeze comes from :meth:`rsp.asp.ASPExperimentSolver._get_freeze_for_trainrun`.


        The ASP constraints are derived by :meth:`rsp.asp.asp_problem_description.ASPProblemDescription._translate_freeze_full_after_malfunction_to_ASP`.

        Parameters
        ----------
        malfunction
            the malfunction
        trainruns_dict
            the schedule
        verbose
            print debug information

        Returns
        -------
        ASPProblemDescription

        """
        freezed_copy = self._prepare_freezed_copy(schedule_trainruns=schedule_trainruns, malfunction=malfunction)

        for agent_id, trainrun_waypoints in freeze.items():
            f = self._translate_freeze_delta_after_malfunction_to_ASP(agent_id, freeze[agent_id], malfunction)
            freezed_copy.asp_program += f
        return freezed_copy

    def _prepare_freezed_copy(self, schedule_trainruns: Dict[int, List[TrainrunWaypoint]],
                              malfunction: ExperimentMalfunction):
        env = self.env
        freezed_copy = ASPProblemDescription(
            env,
            agents_path_dict=self.agents_path_dict,
            asp_objective=ASPObjective.MINIMIZE_DELAY,
            # TODO SIM-146 no effect yet!
            asp_heuristics=[ASPHeuristics.HEURISIC_ROUTES, ASPHeuristics.HEURISTIC_SEQ, ASPHeuristics.HEURISTIC_DELAY]
        )
        freezed_copy.asp_program = self.asp_program.copy()
        # remove all earliest constraints

        freezed_copy.asp_program = list(filter(lambda s: not s.startswith("e("), freezed_copy.asp_program))
        asp_program = freezed_copy.asp_program
        # add constraints for dummy target nodes
        for agent_id, dummy_target_waypoints in self.dummy_target_vertices.items():
            train = "t{}".format(agent_id)
            for dummy_target_waypoint in dummy_target_waypoints:
                vertex = tuple(dummy_target_waypoint)
                # add + 1 for dummy target edge within target cell
                asp_program.append(
                    f"e({train},{vertex},{schedule_trainruns[agent_id][-1].scheduled_at + 1}).")
        # linear penalties up to upper_bound_linear_penalty and then penalty_after_linear
        # penalize +1 at each time step after the scheduled time up to upper_bound_linear_penalty
        # TODO SIM-146 ASP performance enhancement: possibly only penalize only in intervals > 1 for speed-up
        asp_program.append("#const upper_bound_linear_penalty = 60.")
        asp_program.append("#const penalty_after_linear = 5000000.")
        asp_program.append("linear_range(1..upper_bound_linear_penalty).")
        asp_program.append("potlate(T,V,E+S,1) :- e(T,V,E), linear_range(S), end(T,V).")
        asp_program.append(
            "potlate(T,V,E+upper_bound_linear_penalty+1,penalty_after_linear) :- e(T,V,E), end(T,V).")

        return freezed_copy

    def _translate_freeze_full_after_malfunction_to_ASP(self, agent_id: int, freeze: List[TrainrunWaypoint],
                                                        malfunction: ExperimentMalfunction):
        """
        The model is freezed by adding constraints in the following way:
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
        """
        frozen: List[TrainrunWaypoint] = []
        done: Set[Waypoint] = set()
        previous_vertex = None
        train = "t{}".format(agent_id)

        # constraint all times
        for trainrun_waypoint in freeze:
            vertex = tuple(trainrun_waypoint.waypoint)
            time = trainrun_waypoint.scheduled_at
            # all times up to the malfunction are constrained to stay the same
            if trainrun_waypoint.scheduled_at <= malfunction.time_step:
                # 1. freeze times up to malfunction
                # (train,vertex) <= time --> &diff{ (train,vertex)-0 } <= time.
                # (train,vertex) >= time --> &diff{ 0-(train,vertex) }  <= -time.
                frozen.append(f"&diff{{ ({train},{vertex})-0 }} <= {time}.")
                frozen.append(f"&diff{{ 0-({train},{vertex}) }}  <= -{time}.")

                # 2. in order to speed up search (probably), add (maybe) redundant constraints on time windows
                # e(t1,1,2).
                frozen.append(f"e({train},{vertex},{time}).")
                # l(t1,1,2).
                frozen.append(f"l({train},{vertex},{time}).")

                # 3. in order to speed up search (probably), add (maybe) redundant constraints on routes
                # visit(t1,1).
                frozen.append(f"visit({train},{vertex}).")

            # all times after the malfunction are constrained to be larger or equal to our input
            else:
                # 1. freeze times up to malfunction
                # (train,vertex) <= time --> &diff{ (train,vertex)-0 } <= time.
                frozen.append(f"&diff{{ 0-({train},{vertex}) }}  <= -{time}.")

                # 2. in order to speed up search (probably), add (maybe) redundant constraints on time windows
                # e(t1,1,2).
                frozen.append(f"e({train},{vertex},{time}).")

                # 3. in order to speed up search (probably), add (maybe) redundant constraints on routes
                # visit(t1,1).
                frozen.append(f"visit({train},{vertex}).")

            # do we have to add route(....)?
            if previous_vertex:
                # route(T,(V,V')).
                frozen.append(f"route({train},({previous_vertex},{vertex})).")

            previous_vertex = vertex
            done.add(trainrun_waypoint.waypoint)

        for waypoint in {waypoint for waypoint in self.agents_waypoints[agent_id] if waypoint not in done}:
            vertex = tuple(waypoint)
            # (train,vertex) >= time --> &diff{ 0-(train,vertex) }  <= -time.
            frozen.append(f"e({train},{vertex},{malfunction.time_step}).")

        # TODO SIM-146 ASP performance enhancement: discuss with Potsdam whether excluding routes that can never be entered would speed solution time as well.

        return frozen

    def _translate_freeze_delta_after_malfunction_to_ASP(self, agent_id: int, freeze: List[TrainrunWaypoint],
                                                         malfunction: ExperimentMalfunction):
        """
        The model is freezed by adding constraints in the following way:
        - for all train run waypoints in the freeze, add
          - dl(train,vertex) == trainrun_waypoint.scheduled_at
        - for all other possible route waypoints, add
           -  dl(train,vertex) >= malfunction.time_step
        """
        frozen: List[TrainrunWaypoint] = []
        done: Set[Waypoint] = set()
        previous_vertex = None
        train = "t{}".format(agent_id)

        # constraint all times in the freeze to stay the same
        for trainrun_waypoint in freeze:
            vertex = tuple(trainrun_waypoint.waypoint)
            time = trainrun_waypoint.scheduled_at
            # constraint times to stay the same
            # 1. freeze times up to malfunction
            # (train,vertex) <= time --> &diff{ (train,vertex)-0 } <= time.
            # (train,vertex) >= time --> &diff{ 0-(train,vertex) }  <= -time.
            frozen.append(f"&diff{{ ({train},{vertex})-0 }} <= {time}.")
            frozen.append(f"&diff{{ 0-({train},{vertex}) }}  <= -{time}.")

            # 2. in order to speed up search (probably), add (maybe) redundant constraints on time windows
            # e(t1,1,2).
            frozen.append(f"e({train},{vertex},{time}).")
            # l(t1,1,2).
            frozen.append(f"l({train},{vertex},{time}).")

            # 3. in order to speed up search (probably), add (maybe) redundant constraints on routes
            # visit(t1,1).
            frozen.append(f"visit({train},{vertex}).")

            # do we have to add route(....)?
            if previous_vertex:
                # route(T,(V,V')).
                frozen.append(f"route({train},({previous_vertex},{vertex})).")

            previous_vertex = vertex
            done.add(trainrun_waypoint.waypoint)

        # constraint all other times in the freeze to be >= malfunction
        for waypoint in {waypoint for waypoint in self.agents_waypoints[agent_id] if waypoint not in done}:
            vertex = tuple(waypoint)
            # (train,vertex) >= time --> &diff{ 0-(train,vertex) }  <= -time.
            frozen.append(f"e({train},{vertex},{malfunction.time_step}).")

        # TODO SIM-146 ASP performance enhancement: discuss with Potsdam whether excluding routes that can never be entered would speed solution time as well.

        return frozen
