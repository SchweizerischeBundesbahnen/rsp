from __future__ import print_function

from typing import Dict, Optional, List, Tuple, Set

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_trainrun_data_structures import Trainrun

from solver.abstract_problem_description import AbstractProblemDescription, Waypoint
from solver.asp.asp_solution_description import ASPSolutionDescription
from solver.asp.asp_solver import flux_helper, ASPObjective
from solver.utils.data_types import Malfunction


class ASPProblemDescription(AbstractProblemDescription):

    def __init__(self,
                 env: RailEnv,
                 agents_path_dict: Dict[int, Optional[List[Tuple[Waypoint]]]],
                 asp_objective: ASPObjective = ASPObjective.MINIMIZE_SUM_RUNNING_TIMES
                 ):

        self.asp_program: List[str] = []
        self.env: RailEnv = env
        self.asp_objective: ASPObjective = asp_objective

        self.agents_waypoints: Dict[int, Set[Waypoint]] = {
            agent.handle: {waypoint
                           for agent_path in agents_path_dict[agent.handle]
                           for waypoint in agent_path}
            for agent in env.agents}

        super().__init__(env, agents_path_dict, skip_mutual_exclusion=True)

    def get_solver_name(self) -> str:
        """Return the solver name for printing."""

        # determine number of routing alternatives
        k = max([len(agent_paths) for agent_paths in self.agents_path_dict.values()])
        return "ASP_{}".format(k)

    def _implement_train(self, agent_id: int, start_vertices: List[Waypoint], target_vertices: List[Waypoint],
                         minimum_running_time: int):
        """See `AbstractProblemDescription`.

        Parameters
        ----------
        minimum_running_time
        """

        self.asp_program.append("train(t{}).".format(agent_id))
        self.asp_program.append("minimumrunningtime(t{},{}).".format(agent_id, minimum_running_time))
        for start_waypoint in start_vertices:
            self.asp_program.append("start(t{}, {}).".format(agent_id, tuple(start_waypoint)))
        for target_waypoint in target_vertices:
            self.asp_program.append("end(t{}, {}).".format(agent_id, tuple(target_waypoint)))

    def _implement_agent_earliest(self, agent_id: int, waypoint: Waypoint, time):
        """See `AbstractProblemDescription`."""

        # ASP fact e(T,V,E)
        self.asp_program.append("e(t{},{},{}).".format(agent_id, tuple(waypoint), int(time)))

    def _implement_agent_latest(self, agent_id: int, waypoint: Waypoint, time: int):
        """See `AbstractProblemDescription`."""

        # ASP fact l(T,V,E)
        self.asp_program.append("l(t{},{},{}).".format(agent_id, tuple(waypoint), int(time)))

    def _implement_route_section(self, agent_id: int, entry_waypoint: Waypoint, exit_waypoint: Waypoint,
                                 resource_id: Tuple[int, int], minimum_travel_time: int = 1, route_section_penalty=0):
        """See `AbstractProblemDescription`.

        Parameters
        ----------
        route_section_penalty
        """

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

        # TODO SIM-129: release time = 1 to allow for synchronization in FLATland
        self.asp_program.append("b(resource_{}_{},1).".format(*resource_id))

        # penalty for objective minimize_routes.lp and heuristic_ROUTES.lp (used only if activated)
        if route_section_penalty > 0:
            # penalty(E,P)
            self.asp_program.append("penalty(({},{}),{})."
                                    .format(tuple(entry_waypoint), tuple(exit_waypoint), route_section_penalty))

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

    # create objective function to minimize
    def _create_objective_function_minimize(self, variables: List[Waypoint]):
        # TODO SIM-137 SIM-113 bad code smell: the chosen minimization objective is passed when the ASP solver is called, but added here in ortools
        pass

    def solve(self) -> ASPSolutionDescription:
        # dirty work around to silence ASP complaining "info: atom does not occur in any rule head"
        # (we don't use all features in encoding.lp)
        self.asp_program.append("bridge(0,0,0).")
        self.asp_program.append("edge(0,0,0,0).")
        self.asp_program.append("relevant(0,0,0).")
        self.asp_program.append("m(0,1).")
        self.asp_program.append("connection(0,(0,0),0,(0,0),0).")
        self.asp_program.append("penalty(0,0).")

        asp_solution = flux_helper(self.asp_program,
                                   bound_all_events=self.env._max_episode_steps,
                                   asp_objective=self.asp_objective)
        # TODO SIM-105 bad code smell: we should not pass the ctl instance,
        #  but deduce the result in order to make solution description stateless
        return ASPSolutionDescription(env=self.env, asp_solution=asp_solution)

    def get_freezed_copy_for_rescheduling(self,
                                          malfunction: Malfunction,
                                          trainruns_dict: Dict[int, Trainrun],
                                          verbose: bool = False) -> 'ASPProblemDescription':
        """
        Returns a problem description with additional constraints to freeze all the variables up to malfunction.

        See :meth:`solver.asp.asp_problem_description.ASPProblemDescription._get_freeze_for_trainrun` for implementation details.

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
        env = self.env
        freezed_copy = ASPProblemDescription(env, self.agents_path_dict, ASPObjective.MINIMIZE_DELAY)
        freezed_copy.asp_program = self.asp_program.copy()
        # silence "info: atom does not occur in any rule head:" without silencing all warnings
        freezed_copy.asp_program.append("potlate(0,0,0,0).")
        for agent in self.env.agents:
            if verbose:
                print(f"freezing agent {agent.handle}")
                print(f"waypoints={self.agents_waypoints[agent.handle]}")
                print(f"paths={self.agents_path_dict[agent.handle]}")
            freeze = self._get_freeze_for_trainrun(agent.handle, trainruns_dict[agent.handle], malfunction)
            freezed_copy.asp_program += freeze
        return freezed_copy

    def _get_freeze_for_trainrun(self, agent_id: int, agent_solution_trainrun: Trainrun, malfunction: Malfunction,
                                 verbose: bool = False) -> \
            List[str]:
        """
        Keep everything the same up to malfunction and delay the malfunctioning train by the malfunction duration.

        Input: Schedule gives us schedules times schedule_time_(train,vertex) and routes by visit(train,vertex) == True.

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

        Parameters
        ----------
        agent_id
        agent_solution_trainrun
        malfunction

        Returns
        -------

        """
        frozen = []
        time = 0
        done: Set[Waypoint] = set()
        previous_vertex = None
        for waypoint_index, trainrun_waypoint in enumerate(agent_solution_trainrun):
            train = "t{}".format(agent_id)
            vertex = tuple(trainrun_waypoint.waypoint)

            if trainrun_waypoint.scheduled_at <= malfunction.time_step:
                done.add(trainrun_waypoint.waypoint)
                time = trainrun_waypoint.scheduled_at

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

                # TODO discuss with Potsdam whether excluding routes that can never be entered would speed solution time as well.
            else:
                # we're at the first vertex after the freeeze;
                # the train has already entered the edge leading to this vertex (speed fraction >= 0);
                # therefore, freeze this vertex as well since the train cannot "beam" to another edge
                if waypoint_index > 0:
                    done.add(trainrun_waypoint.waypoint)
                    frozen.append(f"visit({train},{vertex}).")

                    if malfunction.agent_id == agent_id:
                        # TODO is this safe because of rounding errors?
                        minimum_travel_time = int(1 / self.env.agents[agent_id].speed_data['speed'])
                        # (train,vertex) >= time --> &diff{ 0-(train,vertex) }  <= -time.
                        earliest_time = time + minimum_travel_time + malfunction.malfunction_duration

                        if verbose:
                            print(f"agent {agent_id} at {vertex} earliest_time={earliest_time}")

                        frozen.append(f"&diff{{ 0-({train},{vertex}) }}  <= -{earliest_time}.")

                        frozen.append(f"e({train},{vertex},{earliest_time}).")

                        frozen.append(f"visit({train},{vertex}).")

                # do not consider remainder of this train!
                break
            previous_vertex = vertex
        for waypoint in {waypoint for waypoint in self.agents_waypoints[agent_id] if waypoint not in done}:
            vertex = tuple(waypoint)
            # (train,vertex) >= time --> &diff{ 0-(train,vertex) }  <= -time.
            frozen.append(f"e({train},{vertex},{malfunction.time_step}).")
        return frozen
