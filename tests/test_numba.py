import timeit
from functools import partial

import numpy as np
from IPython.core.magics.execution import Timer, TimeitResult
from numba import vectorize

from rsp.step_03_run.experiments import list_infrastructure_and_schedule_params_from_base_directory


@vectorize(nopython=True)
def ufunc_zero_clamp(x, threshold):
    if np.abs(x) > threshold:
        return x
    else:
        return 0

def make_time_expansion(nb_resources, time_horizon):
    assert time_horizon <= 2**32
    return np.ndarray(nb_resources, dtype=np.int32)

def make_bit_pattern_for_trainrun() -> Dict[]:

# make coordinate -> index mapping (used resources only)
# for each agent, make bit pattern
# to check, shift bit pattern by start time
# implement insertion and removal at time t for agent a



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

def make_resources(infra_parameters: InfraParameters):
    infra_parameters

if __name__ == '__main__':
    infra_parameters_list, infra_schedule_dict = list_infrastructure_and_schedule_params_from_base_directory(
        base_directory="../rsp-data/PUBLICATION_DATA/PUBLICATION_DATA_baseline_2020_11_17T23_40_54"
    )
    infra_parameters = infra_parameters_list[0]
