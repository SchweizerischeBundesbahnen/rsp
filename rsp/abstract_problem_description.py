import abc
from typing import List, Optional, Dict
from typing import Tuple

import numpy as np
from flatland.core.grid.grid_utils import Vec2dOperations as Vec2d
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.abstract_solution_description import AbstractSolutionDescription


class AbstractProblemDescription:
    MAGIC_DIRECTION_FOR_TARGET = 5

    def __init__(self,
                 env: RailEnv,
                 agents_path_dict: Optional[Dict[int, List[List[Waypoint]]]],
                 skip_mutual_exclusion: bool = False):
        self.env = env
        self.agents_path_dict: Dict[int, Optional[List[Waypoint]]] = agents_path_dict
        self._create_along_paths(skip_mutual_exclusion)

    @abc.abstractmethod
    def get_solver_name(self) -> str:
        """Return the solver name for printing."""

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

    def convert_agent_target_to_dummy_target_waypoint(self, agent) -> Waypoint:
        return Waypoint(position=agent.target, direction=self.__class__.MAGIC_DIRECTION_FOR_TARGET)

    @abc.abstractmethod
    def _implement_train(self, agent_id: int, start_vertices: List[Waypoint], target_vertices: List[Waypoint],
                         minimum_running_time: int):
        """
        Rule 2 each train is scheduled.

        Parameters
        ----------
        agent_id
        start_vertices
        target_vertices
        minimum_running_time
        """

    @abc.abstractmethod
    def _implement_agent_latest(self, agent_id: int, waypoint: Waypoint, time: int):
        """
        Rule 101 Time windows for latest-requirements.
                 If a section_requirement specifies a entry_latest and/or exit_latest time then the event times for the
                 entry_event and/or exit_event on the corresponding trainrun_section SHOULD be <= the specified time.
                 If the scheduled time is later than required, the solution will still be accepted,
                 but it will be penalized by the objective function, see below.
                 Important Remark: Among the 12 business rules, this is the only 'soft' constraint,
                 i.e. a rule that may be violated and the solution still accepted.

        Parameters
        ----------
        agent_id
        waypoint
        time

        """

    @abc.abstractmethod
    def _implement_agent_earliest(self, agent_id: int, waypoint: Waypoint, time):
        """
        Rule 102 Time windows for earliest-requirements.
                 If a section_requirement specifies an entry_earliest and/or exit_earliest time,
                 then the event times for the entry_event and/or exit_event on the corresponding trainrun_section
                 MUST be >= the specified time

        Parameters
        ----------
        agent_id
        waypoint
        time

        """

    @abc.abstractmethod
    def _implement_route_section(self, agent_id: int, entry_waypoint: Waypoint, exit_waypoint: Waypoint,
                                 resource_id: Tuple[int, int], minimum_travel_time: int = 1, penalty=0):
        """
        Rule 103 Minimum section time
                 For each trainrun_section the following holds:
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
        penalty
        penalty

        """

    @abc.abstractmethod
    def _implement_resource_mutual_exclusion(self,
                                             agent_1_id: int,
                                             agent_1_entry_waypoint: Waypoint,
                                             agent_1_exit_waypoint: Waypoint,
                                             agent_2_id: int,
                                             agent_2_entry_waypoint: Waypoint,
                                             agent_2_exit_waypoint: Waypoint,
                                             resource_id: Tuple[int, int]
                                             ):
        """
        Rule 104 Resource Occupations
                 In prose, this means that if a train T1 starts occupying a resource R before train T2,
                 then T2 has to wait until T1 releases it (plus the release time of the resource)
                 before it can start to occupy it.

                 This rule explicitly need not hold between trainrun_sections of the same train.
                 The problem instances are infeasible if you require this separation of occupations
                 also among trainrun_sections of the same train.

        Parameters
        ----------
        agent_1_id
        agent_1_entry_waypoint
        agent_1_exit_waypoint
        agent_2_id
        agent_2_entry_waypoint
        agent_2_exit_waypoint
        resource_id

        """

    @abc.abstractmethod
    def _create_objective_function_minimize(self, variables: Dict[int, List[Waypoint]]):
        """
        Add the objective to our model

        Parameters
        ----------
        variables
            dummy target waypoints (in order to have an edge covering the cell and to enforce that the cell has to be occupied in order to reach the target)


        Returns
        -------

        """

    def _create_along_paths(self, skip_mutual_exclusion=False):
        max_agents_steps_allowed: int = self.env._max_episode_steps
        dummy_target_vertices_dict: Dict[int, List[Waypoint]] = {agent.handle: [] for agent in self.env.agents}
        already_added = set()
        for agent_id, agent in enumerate(self.env.agents):
            dummy_target_vertices = dummy_target_vertices_dict[agent_id]

            agent_paths = self.agents_path_dict[agent_id]
            agent_minimum_running_time = self.get_agent_minimum_running_time(agent, agent_paths)

            for path_index, agent_path in enumerate(agent_paths):
                source_waypoint = self.convert_position_and_entry_direction_to_waypoint(*agent_path[0].position,
                                                                                        agent_path[0].direction)
                dummy_target_waypoint = self.convert_agent_target_to_dummy_target_waypoint(agent)
                dummy_target_vertices.append(dummy_target_waypoint)

                self._implement_train(agent_id,
                                      [source_waypoint],
                                      [dummy_target_waypoint],
                                      agent_minimum_running_time)

                self._add_agent_waypoints(agent, agent_id, agent_path, already_added, dummy_target_waypoint,
                                          max_agents_steps_allowed, path_index)

                dummy_target_entry = (agent_id, dummy_target_waypoint)
                if dummy_target_entry not in already_added:
                    self._implement_agent_earliest(*dummy_target_entry, 0)
                    self._implement_agent_latest(*dummy_target_entry, max_agents_steps_allowed)

        # Mutual exclusion (required only for ortools, would not be required for ASP,
        # since the grounding phase thus this for us)
        if not skip_mutual_exclusion:
            self._add_mutual_exclusion_ortools()

        self._create_objective_function_minimize(dummy_target_vertices_dict)

    @staticmethod
    def get_agent_minimum_running_time(agent: EnvAgent, agent_paths: List[Waypoint]):
        """Get minimum number of steps taken in FLATland """

        # agent paths do not last dummy synchronization segment
        # number of steps is number of vertices minus 1!
        agent_min_number_of_steps = min([len(agent_path) - 1 for agent_path in agent_paths])

        # we assume activate_agents=False, therefore add 1 step!
        agent_minimum_running_time = int(agent_min_number_of_steps // agent.speed_data['speed'] + 1)
        return agent_minimum_running_time

    def _add_agent_waypoints(self,
                             agent: EnvAgent,
                             agent_id: int,
                             agent_path,
                             already_added,
                             dummy_target_waypoint,
                             max_agents_steps_allowed,
                             path_index):

        minimum_running_time_p_cell = int(np.ceil(1.0 / agent.speed_data['speed']))

        # agents paths are ordered by shortest path -> we add constraints for the occurence in the first path encountered only.
        for waypoint_index, waypoint in enumerate(agent_path[:-1]):

            current_position = waypoint.position
            current_direction = waypoint.direction

            entry_waypoint = self.convert_position_and_entry_direction_to_waypoint(*current_position,
                                                                                   current_direction)

            # add time window [0 * waypoint_index * minimum_running_time_p_cell+1,max_agents_steps_allowed] for entry event
            # 0 for first
            agent_entry = (agent_id, entry_waypoint)
            if agent_entry not in already_added:
                self._implement_agent_earliest(
                    *agent_entry, 0 * waypoint_index * minimum_running_time_p_cell + 1
                    if waypoint_index > 0
                    else 0)
                self._implement_agent_latest(*agent_entry, max_agents_steps_allowed)

            next_waypoint: Waypoint = agent_path[waypoint_index + 1]
            next_position = next_waypoint.position

            # add minimum running time etc for this section
            agent_section = (agent_id, entry_waypoint, next_waypoint)
            if agent_section not in already_added:
                already_added.add(agent_section)
                # we use the path_index to penalize taking this route section
                # N.B. the penalty is the index of the first occurrence of this route section
                #      in the given k paths.
                self._implement_route_section(*agent_section,
                                              current_position,
                                              # FLATland agents stay one tick in the first cell when they are place before they move
                                              minimum_running_time_p_cell if waypoint_index > 0 else minimum_running_time_p_cell + 1,
                                              path_index)

            if Vec2d.is_equal(agent.target, next_position):
                agent_entry_target = (agent_id, next_waypoint, dummy_target_waypoint)
                if agent_entry_target not in already_added:
                    # TODO do we have to check that initial_position != target?
                    self._implement_agent_earliest(*agent_entry,
                                                   0 * (waypoint_index + 1) * minimum_running_time_p_cell + 1)
                    self._implement_agent_latest(*agent_entry, max_agents_steps_allowed)

                    # stay in the last cell for one tick to allow for synchronization
                    # TODO SIM-129 can we with this but without additional release time to be consistent with FLATland?
                    self._implement_route_section(*agent_entry_target, next_position, 1)

    def _add_mutual_exclusion_ortools(self):
        for agent_id, agent in enumerate(self.env.agents):
            for agent_path in self.agents_path_dict[agent_id]:
                for waypoint_index, waypoint in enumerate(agent_path):
                    position = waypoint.position
                    entry_waypoint = self.convert_position_and_entry_direction_to_waypoint(*waypoint.position,
                                                                                           waypoint.direction)
                    if Vec2d.is_equal(position, agent_path[-1].position):
                        exit_waypoint = self.convert_agent_target_to_dummy_target_waypoint(agent)
                    else:
                        next_action_element: Waypoint = agent_path[waypoint_index + 1]
                        next_position = next_action_element.position
                        next_direction = next_action_element.direction

                        exit_waypoint = self.convert_position_and_entry_direction_to_waypoint(*next_position,
                                                                                              next_direction)

                    self._add_implement_resource_mutual_exclusion_over_opposite_agents(agent_id, entry_waypoint,
                                                                                       exit_waypoint,
                                                                                       position)

    def _add_implement_resource_mutual_exclusion_over_opposite_agents(self, agent_id, entry_waypoint, exit_waypoint,
                                                                      position):
        for opp_agent_id, opp_agent in enumerate(self.env.agents):
            if agent_id == opp_agent_id:
                continue
            for opp_agent_path in self.agents_path_dict[opp_agent_id]:
                for opp_path_loop, opp_p_data in enumerate(opp_agent_path):
                    opp_position = opp_p_data.position

                    opp_entry_waypoint = self.convert_position_and_entry_direction_to_waypoint(
                        *opp_p_data.position,
                        opp_p_data.direction)

                    if Vec2d.is_equal(opp_p_data.position, opp_agent_path[-1].position):
                        opp_exit_waypoint = self.convert_agent_target_to_dummy_target_waypoint(opp_agent)

                    else:
                        opp_data_next_action_element = opp_agent_path[opp_path_loop + 1]
                        opp_exit_waypoint = self.convert_position_and_entry_direction_to_waypoint(
                            *opp_data_next_action_element.position,
                            opp_data_next_action_element.direction)

                    if Vec2d.is_equal(position, opp_position):
                        # Agent [1] and [2] : mutual exclusive resource allocation
                        self._implement_resource_mutual_exclusion(
                            agent_id,
                            entry_waypoint,
                            exit_waypoint,
                            opp_agent_id,
                            opp_entry_waypoint,
                            opp_exit_waypoint,
                            position
                        )

    @abc.abstractmethod
    def solve(self, verbose: bool = False) -> AbstractSolutionDescription:
        """
        Used in `AbstractSolutionDescription.solve_problem()` to trigger the solver and
        wrap the problem description in a solution description.
        """
