from __future__ import print_function

from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from overrides import overrides

from rsp.abstract_problem_description import AbstractProblemDescription
from rsp.abstract_problem_description import Waypoint
from rsp.rescheduling.rescheduling_utils import ExperimentFreeze
from rsp.rescheduling.rescheduling_utils import ExperimentFreezeDict
from rsp.solvers.asp.asp_solution_description import ASPSolutionDescription
from rsp.solvers.asp.asp_solver import ASPHeuristics
from rsp.solvers.asp.asp_solver import ASPObjective
from rsp.solvers.asp.asp_solver import flux_helper


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
        self.experiment_freeze_dict = None
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

    def get_copy_for_experiment_freeze(self,
                                       experiment_freeze_dict: ExperimentFreezeDict,
                                       schedule_trainruns: Dict[int, List[TrainrunWaypoint]],
                                       verbose: bool = False) -> 'ASPProblemDescription':
        """Returns a problem description with additional constraints to freeze
        all the variables up to malfunction.

        The freeze comes from :meth:`rsp.asp.ASPExperimentSolver._get_freeze_for_malfunction_per_train`.


        The ASP constraints are derived by :meth:`rsp.solvers.asp.asp_problem_description.ASPProblemDescription._translate_experiment_freeze_to_ASP`.

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

        freezed_copy = self._prepare_freezed_copy(schedule_trainruns)
        freezed_copy.experiment_freeze_dict = experiment_freeze_dict

        for agent_id, experiment_freeze in experiment_freeze_dict.items():
            f = self._translate_experiment_freeze_to_ASP(agent_id, experiment_freeze)
            freezed_copy.asp_program += f

        return freezed_copy

    def _prepare_freezed_copy(self, schedule_trainruns: Dict[int, List[TrainrunWaypoint]]):
        env = self.env
        freezed_copy = ASPProblemDescription(
            env,
            agents_path_dict=self.agents_path_dict,
            asp_objective=ASPObjective.MINIMIZE_DELAY,
            # TODO SIM-167 switch on heuristics
            asp_heuristics=[ASPHeuristics.HEURISIC_ROUTES, ASPHeuristics.HEURISTIC_SEQ, ASPHeuristics.HEURISTIC_DELAY]
        )
        freezed_copy.asp_program = self.asp_program.copy()

        # remove all earliest constraints
        # 2019-12-03 discussion with Potsdam (SIM-146): adding multiple earliest constraints for the same vertex could have side-effects
        freezed_copy.asp_program = list(filter(lambda s: not s.startswith("e("), freezed_copy.asp_program))
        asp_program = freezed_copy.asp_program

        # TODO SIM-173 is this correct????? this seems like a dirty hack. move to where earliest are produces
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
        # TODO SIM-171 ASP performance enhancement: possibly only penalize only in intervals > 1 for speed-up
        asp_program.append("#const upper_bound_linear_penalty = 60.")
        asp_program.append("#const penalty_after_linear = 5000000.")
        asp_program.append("linear_range(1..upper_bound_linear_penalty).")
        asp_program.append("potlate(T,V,E+S,1) :- e(T,V,E), linear_range(S), end(T,V).")
        asp_program.append(
            "potlate(T,V,E+upper_bound_linear_penalty+1,penalty_after_linear) :- e(T,V,E), end(T,V).")

        return freezed_copy

    def _translate_experiment_freeze_to_ASP(self,
                                            agent_id: int,
                                            freeze: ExperimentFreeze):
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
            vertex = tuple(trainrun_waypoint)

            # add visit constraint
            # visit(t1,1).
            frozen.append(f"visit({train},{vertex}).")

        for waypoint, scheduled_at in freeze.freeze_latest.items():
            vertex = tuple(waypoint)
            time = scheduled_at

            # add earliest constraint
            # l(t1,1,2).
            frozen.append(f"l({train},{vertex},{time}).")

        for waypoint, scheduled_at in freeze.freeze_earliest.items():
            vertex = tuple(waypoint)
            time = scheduled_at

            # add earliest constraint
            # e(t1,1,2).
            frozen.append(f"e({train},{vertex},{time}).")

        for waypoint in freeze.freeze_banned:
            vertex = tuple(waypoint)
            #  vertex must not be visited
            # visit(t1,1).
            frozen.append(f":- visit({train},{vertex}).")

        return frozen
