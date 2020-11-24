import timeit
from functools import partial
from typing import Dict

import numpy as np
from IPython.core.magics.execution import Timer, TimeitResult
from flatland.envs.rail_trainrun_data_structures import Waypoint
from numba import vectorize, prange, njit

from rsp.scheduling.scheduling_problem import get_paths_in_route_dag
from rsp.step_03_run.experiments import load_infrastructure, create_env_from_experiment_parameters
from rsp.utils.global_constants import GLOBAL_CONSTANTS


@vectorize(nopython=True)
def ufunc_zero_clamp(x, threshold):
    if np.abs(x) > threshold:
        return x
    else:
        return 0


def test():
    n = 10000
    a_int16 = np.arange(n).astype(np.int16)
    a_float32 = np.linspace(0, 1, n, dtype=np.float32)
    a_float32_strided = np.linspace(0, 1, 2 * n, dtype=np.float32)[::2]  # view of every other element
    timeit_ipython_magic_wrapper(partial(ufunc_zero_clamp, a_int16, 1600))
    timeit_ipython_magic_wrapper(partial(ufunc_zero_clamp, a_float32, 0.3))
    timeit_ipython_magic_wrapper(partial(ufunc_zero_clamp, a_float32_strided, 0.3))


def timeit_ipython_magic_wrapper(f,
                                 timefunc=timeit.default_timer,
                                 number: int = 0,
                                 default_repeat=7 if timeit.default_repeat < 7 else timeit.default_repeat,
                                 repeat: int = None,
                                 precision: int = 3,
                                 quiet: bool = False,
                                 return_result: bool = False,
                                 # Minimum time above which compilation time will be reported
                                 tc_min=0.1

                                 ):
    if repeat is None:
        repeat = default_repeat

    timer = Timer(timer=timefunc, stmt=f)
    t0 = timefunc()

    tc = timefunc() - t0

    # This is used to check if there is a huge difference between the
    # best and worst timings.
    # Issue: https://github.com/ipython/ipython/issues/6471
    if number == 0:
        # determine number so that 0.2 <= total time < 2.0
        for index in range(0, 10):
            number = 10 ** index
            time_number = timer.timeit(number)
            if time_number >= 0.2:
                break

    all_runs = timer.repeat(repeat, number)
    best = min(all_runs) / number
    worst = max(all_runs) / number
    timeit_result = TimeitResult(number, repeat, best, worst, all_runs, tc, precision)
    if not quiet:
        # Check best timing is greater than zero to avoid a
        # ZeroDivisionError.
        # In cases where the slowest timing is lesser than a microsecond
        # we assume that it does not really matter if the fastest
        # timing is 4 times faster than the slowest timing or not.
        if worst > 4 * best and best > 0 and worst > 1e-6:
            print("The slowest run took %0.2f times longer than the "
                  "fastest. This could mean that an intermediate result "
                  "is being cached." % (worst / best))

        print(timeit_result)

        if tc > tc_min:
            print("Compiler time: %.2f s" % tc)
    if return_result:
        return timeit_result


@njit(parallel=True)
def check_agent_at(occupations, unshifted_bit_pattern, agent_id: int, start_time: int):
    conflicts = 0
    for resource_index in prange(occupations.shape[0]):
        from_t_incl, to_t_excl = unshifted_bit_pattern[agent_id][resource_index]
        for t in prange(from_t_incl + start_time, to_t_excl + start_time):
            conflicts += occupations[resource_index][t]
    return conflicts


@njit(parallel=True)
def apply_agent_at(occupations, unshifted_bit_pattern, agent_id: int, start_time: int, flag: bool):
    conflicts = 0
    for resource_index in prange(occupations.shape[0]):
        from_t_incl, to_t_excl = unshifted_bit_pattern[agent_id][resource_index]
        for t in range(from_t_incl + start_time, to_t_excl + start_time):
            occupations[resource_index][t] = flag


def main():
    infra, infra_parameters = load_infrastructure(
        base_directory="../rsp-data/PUBLICATION_DATA",
        infra_id=0
    )
    # take shortest path for each agent
    topo_dict = infra.topo_dict
    number_of_shortest_paths_per_agent_schedule = 1
    shortest_path_per_agent = {}
    for agent_id, topo in topo_dict.items():
        paths = get_paths_in_route_dag(topo)
        shortest_path_per_agent[agent_id] = paths[0]
        paths = paths[:number_of_shortest_paths_per_agent_schedule]
        remaining_vertices = {vertex for path in paths for vertex in path}
        topo.remove_nodes_from(set(topo.nodes).difference(remaining_vertices))

    # collect resources
    resources = set()
    for _, topo in topo_dict.items():
        for v in topo.nodes:
            wp: Waypoint = v
            resources.add(wp.position)

    # make coordinate -> index mapping (used resources only)
    # index -> resource
    resource_index_to_position: Dict[int, Waypoint] = dict(enumerate(resources))
    position_to_resource_index: Dict[Waypoint, int] = {resource: index for index, resource in resource_index_to_position.items()}
    number_agents = len(topo_dict)
    number_resources = len(resources)

    #
    env = create_env_from_experiment_parameters(infra_parameters)
    max_episode_steps = env._max_episode_steps

    # empty resource-time-expansion
    # TODO can we make this more compact, 1 bit per timestep?
    occupations = np.zeros(shape=(number_resources, max_episode_steps), dtype=np.bool)

    # for each agent, make bit pattern if they started at zero

    unshifted_bit_pattern = np.zeros(shape=(number_agents, number_resources, 2), dtype=np.int64)
    for agent_id, shortest_path in shortest_path_per_agent.items():
        mrt = infra.minimum_travel_time_dict[agent_id]
        for index, wp in enumerate(shortest_path):
            from_t_incl = index * mrt
            to_t_excl = (index + 1) * mrt + GLOBAL_CONSTANTS.RELEASE_TIME
            resource = position_to_resource_index[wp.position]
            unshifted_bit_pattern[agent_id][resource] = (from_t_incl, to_t_excl)

    if False:
        # warm up jit: TODO make this better
        timeit_ipython_magic_wrapper(partial(check_agent_at, occupations, unshifted_bit_pattern, agent_id=0, start_time=0))
        conflicts = check_agent_at(occupations, unshifted_bit_pattern, agent_id=0, start_time=0)
        assert conflicts == 0
        apply_agent_at(occupations, unshifted_bit_pattern, agent_id=0, start_time=0, flag=True)
        timeit_ipython_magic_wrapper(partial(apply_agent_at, occupations, unshifted_bit_pattern, agent_id=0, start_time=0, flag=True))
        conflicts = check_agent_at(occupations, unshifted_bit_pattern, agent_id=0, start_time=0)
        assert conflicts > 0
        apply_agent_at(occupations, unshifted_bit_pattern, agent_id=0, start_time=0, flag=False)
        timeit_ipython_magic_wrapper(partial(apply_agent_at, occupations, unshifted_bit_pattern, agent_id=0, start_time=0, flag=False))
        conflicts = check_agent_at(occupations, unshifted_bit_pattern, agent_id=0, start_time=0)
        assert conflicts == 0

    # TODO sort agents descending by running time
    # TODO put loop into numba and parallelize
    # TODO take random increments instead (with exponentially decreasing probability)
    # TODO take random backtrackings instead (with exponentially decreasing probability)
    # TODO visualization of unshifted and shifted

    print({
        mrt * len(shortest_path_per_agent[agent_id]) for agent_id, mrt in infra.minimum_travel_time_dict.items()
    })

    elapsed_start_time = timeit.default_timer()
    start_times = {}
    agent_id = 0
    outer_count = 0
    inner_count = 0
    while agent_id < number_agents:
        outer_count += 1
        if outer_count % 1000 == 0:
            elapsed = timeit.default_timer()
            print(
                f"{agent_id}/{number_agents}@{start_times.get(agent_id, -1) + 1}/{max_episode_steps} after {elapsed - elapsed_start_time:10.3f}s = {(elapsed - elapsed_start_time) / outer_count:10.3f}s/it {(elapsed - elapsed_start_time)*1000 / inner_count:10.3f}ms/it ({outer_count} / {inner_count})")
        mrt = infra.minimum_travel_time_dict[agent_id]
        # backtracking?
        if agent_id in start_times:
            apply_agent_at(occupations=occupations, unshifted_bit_pattern=unshifted_bit_pattern, agent_id=agent_id, start_time=start_times[agent_id],
                           flag=False)

        found = False
        # if not backtracking, start at 0
        for start_time in range(start_times.get(agent_id, -1) + 1, max_episode_steps - mrt * len(shortest_path_per_agent[agent_id])):
            inner_count += 1
            if check_agent_at(occupations=occupations, unshifted_bit_pattern=unshifted_bit_pattern, agent_id=agent_id, start_time=start_time) == 0:
                apply_agent_at(occupations=occupations, unshifted_bit_pattern=unshifted_bit_pattern, agent_id=agent_id, start_time=start_time, flag=True)
                start_times[agent_id] = start_time
                if agent_id < number_agents / 3:
                    print(f"{agent_id}/{number_agents}@{start_times[agent_id]}/{max_episode_steps}")
                agent_id += 1
                found = True
                break
        if not found:
            # remove trace before backtracing
            if agent_id in start_times:
                apply_agent_at(occupations=occupations, unshifted_bit_pattern=unshifted_bit_pattern, agent_id=agent_id, start_time=start_times[agent_id],
                               flag=False)
                del start_times[agent_id]
            # backtracing
            agent_id -= 1
    print(f"elapsed {timeit.default_timer() - elapsed_start_time}")
    print(start_times)


if __name__ == '__main__':
    main()
